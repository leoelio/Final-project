from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
import sys

import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from widowx_env import TASKS, WidowXTabletopEnv
from widowx_env.scripted_expert import PickConfig, PickOnlyExpert, PickPlaceConfig, PickPlaceExpert


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scripted expert rollouts.")
    parser.add_argument("--task", choices=sorted(TASKS), default="pick_red_cube")
    parser.add_argument("--complexity", choices=("easy", "medium", "hard", "language"), default="easy")
    parser.add_argument("--workspace-profile", choices=("legacy", "core_v2"), default="legacy")
    parser.add_argument("--target", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--min-success-rate", type=float, default=0.7)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0, help="Viewer hold time after rollout. 0 means keep open.")
    parser.add_argument("--speed", type=float, default=1.0, help="Viewer playback speed multiplier.")
    parser.add_argument("--arm-kp", type=float, default=150.0)
    parser.add_argument("--arm-force", type=float, default=100.0)
    parser.add_argument("--gripper-kp", type=float, default=800.0)
    parser.add_argument("--gripper-force", type=float, default=140.0)
    parser.add_argument("--friction", type=float, default=3.0)
    parser.add_argument("--place-tcp-z", type=float, default=0.055)
    parser.add_argument("--approach-z", type=float, default=0.12)
    parser.add_argument("--grasp-z", type=float, default=0.008)
    parser.add_argument("--lift-z", type=float, default=0.18)
    parser.add_argument("--mode", choices=("auto", "pick", "place"), default="auto")
    parser.add_argument("--retries", type=int, default=None)
    return parser.parse_args()


def print_plan(plan: dict, mode: str, attempt: int, total_attempts: int) -> None:
    print(f"attempt: {attempt}/{total_attempts}", flush=True)
    plan_steps = ("approach", "grasp", "lift", "transfer", "place", "retreat") if mode == "place" else ("approach", "grasp", "lift")
    for name in plan_steps:
        result = plan[name]
        print(
            f"{name}_target={np.round(result.target, 4).tolist()} "
            f"{name}_tcp={np.round(result.tcp_position, 4).tolist()} "
            f"{name}_ik_error={result.error_norm:.4f} "
            f"{name}_converged={result.converged}",
            flush=True,
        )


def run_episode(args: argparse.Namespace, seed: int, viewer_enabled: bool = False) -> dict[str, float | bool | int]:
    env = WidowXTabletopEnv(seed=seed, workspace_profile=args.workspace_profile)
    env.set_arm_actuator_strength(kp=args.arm_kp, force_limit=args.arm_force)
    env.set_gripper_actuator_strength(kp=args.gripper_kp, force_limit=args.gripper_force)
    env.set_grasp_contact_friction(sliding=args.friction)
    obs = env.reset(task=args.task, complexity=args.complexity, seed=seed)
    object_name = args.target or obs["target_object"]
    if object_name not in obs["active_objects"]:
        raise ValueError(f"target {object_name!r} is not active in this reset: {obs['active_objects']}")

    mode = args.mode
    if mode == "auto":
        mode = "place" if env.task.kind == "place" and env.task.target_geom else "pick"

    if mode == "place":
        if not env.task.target_geom:
            raise ValueError(f"task {args.task!r} does not define a target geom for place mode")
        config = PickPlaceConfig(
            approach_z_offset=args.approach_z,
            grasp_z_offset=args.grasp_z,
            lift_z_offset=args.lift_z,
            place_tcp_z=args.place_tcp_z,
        )
        expert = PickPlaceExpert(env, config)
        retries = 2 if args.retries is None else args.retries
    else:
        config = PickConfig(
            approach_z_offset=args.approach_z,
            grasp_z_offset=args.grasp_z,
            lift_z_offset=args.lift_z,
        )
        expert = PickOnlyExpert(env, config)
        retries = 0 if args.retries is None else args.retries

    print(f"episode_seed: {seed}", flush=True)
    print(f"instruction: {obs['instruction']}", flush=True)
    print(f"active_objects: {', '.join(obs['active_objects'])}", flush=True)
    print(f"target_object: {object_name}", flush=True)

    def execute_attempts(viewer=None) -> dict[str, float | bool | int]:
        total_attempts = max(0, retries) + 1
        summary: dict[str, float | bool | int] = {"success": False, "attempts": 0}
        for attempt in range(1, total_attempts + 1):
            plan = expert.plan(object_name, env.task.target_geom) if mode == "place" else expert.plan(object_name)
            print_plan(plan, mode, attempt, total_attempts)
            summary = expert.execute(plan, viewer=viewer, speed=args.speed)
            summary["attempts"] = attempt
            print("attempt_summary:", summary, flush=True)
            if summary["success"]:
                break
        return summary

    if viewer_enabled:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            summary = execute_attempts(viewer)
            start = time.time()
            while viewer.is_running():
                viewer.sync()
                if args.duration and time.time() - start > args.duration:
                    break
                time.sleep(0.01)
    else:
        summary = execute_attempts()

    print("summary:", summary, flush=True)
    return summary


def main() -> None:
    args = parse_args()
    successes = 0
    summaries = []
    for episode in range(args.episodes):
        summary = run_episode(args, seed=args.seed + episode, viewer_enabled=args.viewer and episode == 0)
        summaries.append(summary)
        successes += int(bool(summary["success"]))
    success_rate = successes / len(summaries)
    print(f"success_rate: {successes}/{len(summaries)} = {success_rate:.3f}", flush=True)
    if success_rate < args.min_success_rate:
        raise SystemExit(1)
    if args.viewer:
        os._exit(0)


if __name__ == "__main__":
    main()
