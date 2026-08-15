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

from widowx_env import TASKS, WidowXTabletopEnv
from widowx_env.scripted_expert import PickConfig, PickOnlyExpert, PickPlaceConfig, PickPlaceExpert
from widowx_env.tabletop_env import OBJECTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect scripted expert demonstrations.")
    parser.add_argument("--task", choices=sorted(TASKS), default="place_blue_cube_red_pad")
    parser.add_argument("--complexity", choices=("easy", "medium", "hard", "language"), default="medium")
    parser.add_argument("--workspace-profile", choices=("legacy", "core_v2"), default="legacy")
    parser.add_argument("--target", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "demos")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--min-success-rate", type=float, default=0.0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0, help="Viewer hold time after first rollout. 0 means keep open.")
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


def object_positions(env: WidowXTabletopEnv) -> dict[str, list[float]]:
    return {name: env.object_position(name).astype(float).round(6).tolist() for name in OBJECTS}


def json_ready(summary: dict[str, float | bool | int]) -> dict[str, float | bool | int]:
    return {key: value.item() if isinstance(value, np.generic) else value for key, value in summary.items()}


class TrajectoryBuffer:
    def __init__(self, object_names: tuple[str, ...]) -> None:
        self.object_names = object_names
        self.current_attempt = 0
        self.actions: list[np.ndarray] = []
        self.qpos: list[np.ndarray] = []
        self.qvel: list[np.ndarray] = []
        self.ctrl: list[np.ndarray] = []
        self.tcp: list[np.ndarray] = []
        self.object_positions: list[np.ndarray] = []
        self.attempt_ids: list[int] = []
        self.times: list[float] = []
        self.attempt_start_ids: list[int] = []
        self.attempt_start_qpos: list[np.ndarray] = []
        self.attempt_start_qvel: list[np.ndarray] = []
        self.attempt_start_ctrl: list[np.ndarray] = []
        self.attempt_start_tcp: list[np.ndarray] = []
        self.attempt_start_object_positions: list[np.ndarray] = []
        self.attempt_start_times: list[float] = []

    def start_attempt(self, attempt: int, env: WidowXTabletopEnv) -> None:
        self.attempt_start_ids.append(attempt)
        self.attempt_start_qpos.append(env.data.qpos.astype(np.float32).copy())
        self.attempt_start_qvel.append(env.data.qvel.astype(np.float32).copy())
        self.attempt_start_ctrl.append(env.data.ctrl.astype(np.float32).copy())
        self.attempt_start_tcp.append(env.tcp_position().astype(np.float32).copy())
        self.attempt_start_object_positions.append(
            np.stack([env.object_position(name) for name in self.object_names]).astype(np.float32)
        )
        self.attempt_start_times.append(float(env.data.time))

    def record(self, action: np.ndarray, env: WidowXTabletopEnv) -> None:
        self.actions.append(action.astype(np.float32).copy())
        self.qpos.append(env.data.qpos.astype(np.float32).copy())
        self.qvel.append(env.data.qvel.astype(np.float32).copy())
        self.ctrl.append(env.data.ctrl.astype(np.float32).copy())
        self.tcp.append(env.tcp_position().astype(np.float32).copy())
        self.object_positions.append(np.stack([env.object_position(name) for name in self.object_names]).astype(np.float32))
        self.attempt_ids.append(self.current_attempt)
        self.times.append(float(env.data.time))

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            actions=np.asarray(self.actions, dtype=np.float32),
            qpos=np.asarray(self.qpos, dtype=np.float32),
            qvel=np.asarray(self.qvel, dtype=np.float32),
            ctrl=np.asarray(self.ctrl, dtype=np.float32),
            tcp=np.asarray(self.tcp, dtype=np.float32),
            object_positions=np.asarray(self.object_positions, dtype=np.float32),
            object_names=np.asarray(self.object_names),
            attempt_ids=np.asarray(self.attempt_ids, dtype=np.int16),
            times=np.asarray(self.times, dtype=np.float32),
            attempt_start_ids=np.asarray(self.attempt_start_ids, dtype=np.int16),
            attempt_start_qpos=np.asarray(self.attempt_start_qpos, dtype=np.float32),
            attempt_start_qvel=np.asarray(self.attempt_start_qvel, dtype=np.float32),
            attempt_start_ctrl=np.asarray(self.attempt_start_ctrl, dtype=np.float32),
            attempt_start_tcp=np.asarray(self.attempt_start_tcp, dtype=np.float32),
            attempt_start_object_positions=np.asarray(self.attempt_start_object_positions, dtype=np.float32),
            attempt_start_times=np.asarray(self.attempt_start_times, dtype=np.float32),
        )


def resolve_mode(args: argparse.Namespace, env: WidowXTabletopEnv) -> str:
    if args.mode != "auto":
        return args.mode
    if env.task.kind == "pick":
        return "pick"
    if env.task.kind == "place" and env.task.target_geom:
        return "place"
    raise ValueError(f"task {args.task!r} is not supported by the current pick/place expert")


def make_expert(args: argparse.Namespace, env: WidowXTabletopEnv, mode: str) -> tuple[PickOnlyExpert | PickPlaceExpert, int]:
    if mode == "place":
        config = PickPlaceConfig(
            approach_z_offset=args.approach_z,
            grasp_z_offset=args.grasp_z,
            lift_z_offset=args.lift_z,
            place_tcp_z=args.place_tcp_z,
        )
        return PickPlaceExpert(env, config), 2 if args.retries is None else args.retries

    config = PickConfig(
        approach_z_offset=args.approach_z,
        grasp_z_offset=args.grasp_z,
        lift_z_offset=args.lift_z,
    )
    return PickOnlyExpert(env, config), 0 if args.retries is None else args.retries


def collect_episode(args: argparse.Namespace, run_dir: Path, episode_index: int, seed: int, viewer_enabled: bool) -> dict:
    env = WidowXTabletopEnv(seed=seed, workspace_profile=args.workspace_profile)
    env.set_arm_actuator_strength(kp=args.arm_kp, force_limit=args.arm_force)
    env.set_gripper_actuator_strength(kp=args.gripper_kp, force_limit=args.gripper_force)
    env.set_grasp_contact_friction(sliding=args.friction)
    obs = env.reset(task=args.task, complexity=args.complexity, seed=seed)
    object_name = args.target or obs["target_object"]
    if object_name not in obs["active_objects"]:
        raise ValueError(f"target {object_name!r} is not active in this reset: {obs['active_objects']}")

    mode = resolve_mode(args, env)
    if mode == "place" and not env.task.target_geom:
        raise ValueError(f"task {args.task!r} does not define a target geom for place mode")

    expert, retries = make_expert(args, env, mode)
    trajectory = TrajectoryBuffer(OBJECTS)
    initial_objects = object_positions(env)
    target_position = env.target_position(env.task.target_geom).round(6).tolist() if env.task.target_geom else None

    def execute_attempts(viewer=None) -> dict[str, float | bool | int]:
        total_attempts = max(0, retries) + 1
        summary: dict[str, float | bool | int] = {"success": False, "attempts": 0}
        for attempt in range(1, total_attempts + 1):
            trajectory.current_attempt = attempt
            plan = expert.plan(object_name, env.task.target_geom) if mode == "place" else expert.plan(object_name)
            trajectory.start_attempt(attempt, env)
            summary = expert.execute(plan, viewer=viewer, record_step=trajectory.record, speed=args.speed)
            summary["attempts"] = attempt
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

    episode_file = run_dir / "episodes" / f"episode_{episode_index:06d}_seed_{seed}.npz"
    trajectory.save(episode_file)
    summary = json_ready(summary)
    return {
        "episode_index": episode_index,
        "seed": seed,
        "task": args.task,
        "complexity": args.complexity,
        "mode": mode,
        "instruction": obs["instruction"],
        "target_object": object_name,
        "target_geom": env.task.target_geom,
        "target_position": target_position,
        "active_objects": list(obs["active_objects"]),
        "initial_objects": initial_objects,
        "final_objects": object_positions(env),
        "success": bool(summary["success"]),
        "attempts": int(summary["attempts"]),
        "summary": summary,
        "trajectory_file": episode_file.relative_to(run_dir).as_posix(),
        "trajectory_steps": len(trajectory.actions),
    }


def make_run_dir(args: argparse.Namespace) -> Path:
    run_name = args.run_name or f"{args.task}_{args.complexity}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output / run_name
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    (run_dir / "episodes").mkdir(parents=True)
    return run_dir


def sampling_failure_metadata(args: argparse.Namespace, episode_index: int, seed: int, error: RuntimeError) -> dict:
    return {
        "episode_index": episode_index,
        "seed": seed,
        "task": args.task,
        "complexity": args.complexity,
        "success": False,
        "attempts": 0,
        "trajectory_file": None,
        "trajectory_steps": 0,
        "collection_error": str(error),
    }


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir(args)
    metadata_path = run_dir / "metadata.jsonl"

    successes = 0
    with metadata_path.open("w", encoding="utf-8") as file:
        for episode_index in range(args.episodes):
            seed = args.seed + episode_index
            try:
                metadata = collect_episode(
                    args,
                    run_dir,
                    episode_index=episode_index,
                    seed=seed,
                    viewer_enabled=args.viewer and episode_index == 0,
                )
            except RuntimeError as error:
                if "could not sample non-overlapping object positions" not in str(error):
                    raise
                metadata = sampling_failure_metadata(args, episode_index, seed, error)
            successes += int(metadata["success"])
            file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            file.flush()
            print(
                f"episode {episode_index}: seed={seed} "
                f"success={metadata['success']} attempts={metadata['attempts']} "
                f"steps={metadata['trajectory_steps']}",
                flush=True,
            )

    success_rate = successes / max(1, args.episodes)
    summary = {
        "episodes": args.episodes,
        "successes": successes,
        "success_rate": success_rate,
        "metadata": metadata_path.name,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"run_dir: {run_dir}", flush=True)
    print(f"success_rate: {successes}/{args.episodes} = {success_rate:.3f}", flush=True)
    if success_rate < args.min_success_rate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
