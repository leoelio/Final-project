from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from collect_demos import TrajectoryBuffer, json_ready, object_positions  # noqa: E402
from widowx_env import TASKS, WidowXTabletopEnv  # noqa: E402
from widowx_env.scripted_expert import PickPlaceConfig, PickPlaceExpert, interpolate_action  # noqa: E402
from widowx_env.tabletop_env import OBJECTS  # noqa: E402


VERSION = "contact_stage_demo_v1"
STAGES = (
    ("approach", "approach", "approach_steps"),
    ("descend", "grasp_open", "descend_steps"),
    ("close", "grasp_closed", "close_steps"),
    ("lift", "lift_closed", "lift_steps"),
    ("transfer", "transfer_closed", "transfer_steps"),
    ("place_descend", "place_closed", "place_descend_steps"),
    ("release", "place_open", "open_steps"),
    ("retreat", "retreat_open", "retreat_steps"),
    ("hold", "retreat_open", "hold_steps"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect contact-stage scripted demonstrations in the standard demo format.")
    parser.add_argument("--task", choices=sorted(TASKS), default="place_blue_cube_blue_pad")
    parser.add_argument("--complexity", choices=("easy", "medium", "hard", "language"), default="medium")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "demos")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--min-success-rate", type=float, default=1.0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-viewer", dest="viewer", action="store_false")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--speed", type=float, default=0.05)
    parser.add_argument("--arm-kp", type=float, default=150.0)
    parser.add_argument("--arm-force", type=float, default=100.0)
    parser.add_argument("--gripper-kp", type=float, default=900.0)
    parser.add_argument("--gripper-force", type=float, default=180.0)
    parser.add_argument("--friction", type=float, default=3.0)
    parser.add_argument("--approach-z", type=float, default=0.12)
    parser.add_argument("--grasp-z", type=float, default=0.008)
    parser.add_argument("--lift-z", type=float, default=0.18)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--lift-threshold", type=float, default=0.085)
    parser.add_argument("--tcp-lift-threshold", type=float, default=0.12)
    parser.set_defaults(viewer=False)
    return parser.parse_args()


def make_run_dir(args: argparse.Namespace) -> Path:
    run_name = args.run_name or f"contact_stage_demo_{args.task}_{args.complexity}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output / run_name
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    (run_dir / "episodes").mkdir(parents=True)
    return run_dir


def configure_env(args: argparse.Namespace, seed: int) -> tuple[WidowXTabletopEnv, dict]:
    env = WidowXTabletopEnv(seed=seed)
    env.set_arm_actuator_strength(kp=float(args.arm_kp), force_limit=float(args.arm_force))
    env.set_gripper_actuator_strength(kp=float(args.gripper_kp), force_limit=float(args.gripper_force))
    env.set_grasp_contact_friction(sliding=float(args.friction))
    obs = env.reset(task=str(args.task), complexity=str(args.complexity), seed=seed)
    return env, obs


def make_expert(args: argparse.Namespace, env: WidowXTabletopEnv) -> PickPlaceExpert:
    return PickPlaceExpert(
        env,
        PickPlaceConfig(
            approach_z_offset=float(args.approach_z),
            grasp_z_offset=float(args.grasp_z),
            lift_z_offset=float(args.lift_z),
        ),
    )


def track_stage(
    env: WidowXTabletopEnv,
    trajectory: TrajectoryBuffer,
    target_action: np.ndarray,
    steps: int,
    viewer=None,
    speed: float = 0.0,
) -> None:
    start_action = env.data.ctrl.copy()
    dt = float(env.model.opt.timestep)
    for step in range(int(steps)):
        action = interpolate_action(start_action, target_action, step, int(steps))
        env.step(action)
        trajectory.record(action, env)
        if viewer is not None:
            viewer.sync()
            if speed > 0:
                time.sleep(dt / speed)


def lifted_for_transfer(args: argparse.Namespace, env: WidowXTabletopEnv) -> bool:
    metrics = env.metrics()
    object_position = env.object_position(env.episode_target_object)
    tcp_object_distance = float(np.linalg.norm(env.tcp_position() - object_position))
    return bool(float(metrics["object_z"]) >= float(args.lift_threshold) and tcp_object_distance < float(args.tcp_lift_threshold))


def execute_attempt(args: argparse.Namespace, env: WidowXTabletopEnv, trajectory: TrajectoryBuffer, expert: PickPlaceExpert, object_name: str, viewer=None) -> dict:
    if not env.task.target_geom:
        raise ValueError(f"task {env.task.name!r} does not define a target geom")
    plan = expert.plan(object_name, env.task.target_geom)
    actions = plan["actions"]
    cfg = expert.config

    for stage_name, action_key, steps_attr in STAGES[:4]:
        _ = stage_name
        track_stage(env, trajectory, actions[action_key], int(getattr(cfg, steps_attr)), viewer, float(args.speed))
    if not lifted_for_transfer(args, env):
        metrics = env.metrics()
        return {
            "success": False,
            "continued_to_place": False,
            "target_distance": float(metrics["target_distance"]),
            "object_z": float(metrics["object_z"]),
            "contact_count": float(metrics["contact_count"]),
            "out_of_table": bool(metrics["out_of_table"]),
        }

    for stage_name, action_key, steps_attr in STAGES[4:]:
        _ = stage_name
        track_stage(env, trajectory, actions[action_key], int(getattr(cfg, steps_attr)), viewer, float(args.speed))

    metrics = env.metrics()
    return {
        "success": bool(metrics["success"]),
        "continued_to_place": True,
        "target_distance": float(metrics["target_distance"]),
        "object_z": float(metrics["object_z"]),
        "contact_count": float(metrics["contact_count"]),
        "out_of_table": bool(metrics["out_of_table"]),
    }


def collect_episode(args: argparse.Namespace, run_dir: Path, episode_index: int, seed: int, viewer_enabled: bool) -> dict:
    env, obs = configure_env(args, seed)
    if env.task.kind != "place" or not env.task.target_geom:
        raise ValueError("contact-stage demo collection currently requires a place task with target geom")

    object_name = str(obs["target_object"])
    expert = make_expert(args, env)
    trajectory = TrajectoryBuffer(OBJECTS)
    initial_objects = object_positions(env)
    target_position = env.target_position(env.task.target_geom).round(6).tolist()

    def execute_attempts(viewer=None) -> dict:
        summary: dict = {"success": False, "attempts": 0, "continued_to_place": False}
        for attempt in range(1, max(0, int(args.retries)) + 2):
            trajectory.current_attempt = attempt
            trajectory.start_attempt(attempt, env)
            summary = execute_attempt(args, env, trajectory, expert, object_name, viewer)
            summary["attempts"] = attempt
            if bool(summary["success"]):
                break
        return summary

    if viewer_enabled:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            summary = execute_attempts(viewer)
            start = time.time()
            while viewer.is_running():
                viewer.sync()
                if args.duration and time.time() - start > float(args.duration):
                    break
                time.sleep(0.01)
    else:
        summary = execute_attempts()

    episode_file = run_dir / "episodes" / f"episode_{episode_index:06d}_seed_{seed}.npz"
    trajectory.save(episode_file)
    summary = json_ready(summary)
    return {
        "episode_index": int(episode_index),
        "seed": int(seed),
        "version": VERSION,
        "task": str(args.task),
        "complexity": str(args.complexity),
        "mode": "contact_stage_place",
        "instruction": obs["instruction"],
        "target_object": object_name,
        "target_geom": env.task.target_geom,
        "target_position": target_position,
        "active_objects": list(obs["active_objects"]),
        "initial_objects": initial_objects,
        "final_objects": object_positions(env),
        "success": bool(summary["success"]),
        "attempts": int(summary["attempts"]),
        "continued_to_place": bool(summary.get("continued_to_place", False)),
        "summary": summary,
        "trajectory_file": episode_file.relative_to(run_dir).as_posix(),
        "trajectory_steps": len(trajectory.actions),
    }


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir(args)
    metadata_path = run_dir / "metadata.jsonl"

    successes = 0
    with metadata_path.open("w", encoding="utf-8") as file:
        for episode_index in range(int(args.episodes)):
            seed = int(args.seed) + episode_index
            metadata = collect_episode(args, run_dir, episode_index, seed, viewer_enabled=bool(args.viewer and episode_index == 0))
            successes += int(metadata["success"])
            file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            file.flush()
            print(
                f"episode {episode_index}: seed={seed} success={metadata['success']} "
                f"attempts={metadata['attempts']} steps={metadata['trajectory_steps']}",
                flush=True,
            )

    success_rate = successes / max(1, int(args.episodes))
    summary = {
        "version": VERSION,
        "episodes": int(args.episodes),
        "successes": int(successes),
        "success_rate": float(success_rate),
        "metadata": metadata_path.name,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"run_dir: {run_dir}", flush=True)
    print(f"success_rate: {successes}/{args.episodes} = {success_rate:.3f}", flush=True)
    if success_rate < float(args.min_success_rate):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
