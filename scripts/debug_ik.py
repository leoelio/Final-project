from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from widowx_env import TASKS, WidowXTabletopEnv
from widowx_env.ik_controller import DampedLeastSquaresIK, target_above_object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug WidowX IK to move above a target object.")
    parser.add_argument("--target", default="red_cube")
    parser.add_argument("--task", choices=sorted(TASKS), default="pick_red_cube")
    parser.add_argument("--complexity", choices=("easy", "medium", "hard", "language"), default="medium")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--z-offset", type=float, default=0.12)
    parser.add_argument("--ik-tolerance", type=float, default=0.015)
    parser.add_argument("--success-tolerance", "--tolerance", dest="success_tolerance", type=float, default=0.03)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0, help="Viewer hold time after reaching the target. 0 means keep open.")
    parser.add_argument("--speed", type=float, default=1.0, help="Viewer playback speed multiplier.")
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--arm-kp", type=float, default=150.0)
    parser.add_argument("--arm-force", type=float, default=100.0)
    return parser.parse_args()


def make_goal_action(env: WidowXTabletopEnv, qpos: np.ndarray) -> np.ndarray:
    action = env.home_ctrl.copy()
    action[:6] = qpos[:6]
    action[6] = 0.037
    return action


def track_goal(env: WidowXTabletopEnv, goal_action: np.ndarray, steps: int, viewer=None, speed: float = 1.0) -> None:
    start_ctrl = env.data.ctrl.copy()
    dt = float(env.model.opt.timestep)
    for step in range(steps):
        blend = min(1.0, (step + 1) / max(1, steps))
        action = (1.0 - blend) * start_ctrl + blend * goal_action
        env.step(action)
        if viewer is not None:
            viewer.sync()
            if speed > 0:
                time.sleep(dt / speed)


def run_once(args: argparse.Namespace, viewer_enabled: bool) -> dict[str, float | bool]:
    env = WidowXTabletopEnv(seed=args.seed)
    env.set_arm_actuator_strength(kp=args.arm_kp, force_limit=args.arm_force)
    obs = env.reset(task=args.task, complexity=args.complexity, seed=args.seed)
    if args.target not in obs["active_objects"]:
        raise ValueError(f"target {args.target!r} is not active in this reset: {obs['active_objects']}")

    initial_tcp = env.tcp_position().copy()
    target = target_above_object(env, args.target, z_offset=args.z_offset)
    ik = DampedLeastSquaresIK(env, tolerance=args.ik_tolerance)
    result = ik.solve(target)
    goal_action = make_goal_action(env, result.qpos)

    print(f"instruction: {obs['instruction']}", flush=True)
    print(f"active_objects: {', '.join(obs['active_objects'])}", flush=True)
    print(f"target_object: {args.target}", flush=True)
    print(f"target_xyz: {np.round(result.target, 4).tolist()}", flush=True)
    print(f"initial_tcp_xyz: {np.round(initial_tcp, 4).tolist()}", flush=True)
    print(f"ik_tcp_xyz: {np.round(result.tcp_position, 4).tolist()}", flush=True)
    print(f"ik_error_m: {result.error_norm:.4f}", flush=True)
    print(f"ik_iterations: {result.iterations}", flush=True)
    print(f"ik_converged: {result.converged}", flush=True)

    # Reset to the original scene so the viewer shows the motion, not only the solved pose.
    env.reset(task=args.task, complexity=args.complexity, seed=args.seed)

    if viewer_enabled:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            track_goal(env, goal_action, steps=args.steps, viewer=viewer, speed=args.speed)
            start = time.time()
            while viewer.is_running():
                env.step(goal_action)
                viewer.sync()
                if args.duration and time.time() - start > args.duration:
                    break
                time.sleep(0.01)
    else:
        track_goal(env, goal_action, steps=args.steps)

    final_tcp = env.tcp_position().copy()
    final_error = float(np.linalg.norm(final_tcp - target))
    summary = {
        "ik_converged": result.converged,
        "ik_error_m": result.error_norm,
        "final_tracking_error_m": final_error,
        "success": final_error < args.success_tolerance,
    }
    print("summary:", summary, flush=True)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_once(args, viewer_enabled=args.viewer)
    if not summary["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
