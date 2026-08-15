from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from widowx_env import TASKS, WidowXTabletopEnv  # noqa: E402
from widowx_env.demo_dataset import read_metadata  # noqa: E402
from widowx_env.scripted_expert import PickConfig, PickOnlyExpert, PickPlaceConfig, PickPlaceExpert  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a structured single-attempt waypoint policy.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--task", choices=sorted(TASKS), default=None)
    parser.add_argument("--complexity", choices=("easy", "medium", "hard", "language"), default=None)
    parser.add_argument("--workspace-profile", choices=("legacy", "core_v2"), default="legacy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2840)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-viewer", dest="viewer", action="store_false")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--speed", type=float, default=0.05)
    parser.add_argument("--arm-kp", type=float, default=150.0)
    parser.add_argument("--arm-force", type=float, default=100.0)
    parser.add_argument("--gripper-kp", type=float, default=800.0)
    parser.add_argument("--gripper-force", type=float, default=140.0)
    parser.add_argument("--friction", type=float, default=3.0)
    parser.set_defaults(viewer=True)
    return parser.parse_args()


def latest_model() -> Path:
    root = ROOT / "outputs" / "structured_waypoint_policy"
    candidates = sorted(root.glob("structured_waypoint_policy_*.npz"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no structured waypoint policy artifacts found under {root}")
    return candidates[-1]


def load_model(path: Path) -> dict:
    with np.load(path) as data:
        metadata = json.loads(data["metadata"].item())
        params = data["params"].astype(np.float32)
    return {"metadata": metadata, "params": params}


def infer_task_defaults(model: dict) -> tuple[str, str]:
    first = read_metadata(Path(model["metadata"]["run_dir"]))[0]
    return str(first["task"]), str(first["complexity"])


def configure_env(args: argparse.Namespace, seed: int, task: str, complexity: str) -> tuple[WidowXTabletopEnv, dict]:
    env = WidowXTabletopEnv(seed=seed, workspace_profile=args.workspace_profile)
    env.set_arm_actuator_strength(kp=args.arm_kp, force_limit=args.arm_force)
    env.set_gripper_actuator_strength(kp=args.gripper_kp, force_limit=args.gripper_force)
    env.set_grasp_contact_friction(sliding=args.friction)
    obs = env.reset(task=task, complexity=complexity, seed=seed)
    return env, obs


def policy_configs(model: dict) -> tuple[PickConfig, PickPlaceConfig]:
    meta = model["metadata"]
    pick = PickConfig(
        approach_z_offset=float(meta["approach_z"]),
        grasp_z_offset=float(meta["grasp_z"]),
        lift_z_offset=float(meta["lift_z"]),
        open_gripper=float(meta["open_gripper"]),
        close_gripper=float(meta["close_gripper"]),
    )
    place = PickPlaceConfig(
        approach_z_offset=float(meta["approach_z"]),
        grasp_z_offset=float(meta["grasp_z"]),
        lift_z_offset=float(meta["lift_z"]),
        transfer_z_offset=float(meta["transfer_z"]),
        place_tcp_z=float(meta["place_tcp_z"]),
        retreat_z_offset=float(meta["retreat_z"]),
        open_gripper=float(meta["open_gripper"]),
        close_gripper=float(meta["close_gripper"]),
    )
    return pick, place


def rollout_with_env(
    args: argparse.Namespace,
    model: dict,
    env: WidowXTabletopEnv,
    obs: dict,
    seed: int,
    task: str,
    complexity: str,
    viewer=None,
) -> dict:
    pick_cfg, place_cfg = policy_configs(model)
    object_name = obs["target_object"]
    mode = "place" if env.task.kind == "place" and env.task.target_geom else "pick"
    if mode == "place":
        expert = PickPlaceExpert(env, place_cfg)
        plan = expert.plan(object_name, env.task.target_geom)
    else:
        expert = PickOnlyExpert(env, pick_cfg)
        plan = expert.plan(object_name)
    summary = expert.execute(plan, viewer=viewer, speed=args.speed if viewer is not None else 0.0)
    metrics = env.metrics()
    return {
        "seed": seed,
        "task": task,
        "complexity": complexity,
        "instruction": obs["instruction"],
        "active_objects": list(obs["active_objects"]),
        "success": bool(summary["success"]),
        "target_distance": float(summary.get("target_distance", metrics["target_distance"])),
        "object_z": float(summary["object_z"]),
        "grasp_success": bool(metrics["grasp_success"]),
        "out_of_table": bool(summary["out_of_table"]),
        "steps_taken": int(sum((260, 220, 260, 420, 700, 320, 220, 280, 160))) if mode == "place" else int(sum((260, 220, 260, 420, 160))),
        "stop_reason": None,
        "mean_action_norm": None,
        "max_action_norm": None,
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    model_path = args.model or latest_model()
    model = load_model(model_path)
    default_task, default_complexity = infer_task_defaults(model)
    task = args.task or default_task
    complexity = args.complexity or default_complexity

    print(f"model_path: {model_path}", flush=True)
    print(f"model_train_run: {model['metadata']['run_dir']}", flush=True)
    print(f"task: {task}", flush=True)
    print(f"complexity: {complexity}", flush=True)

    summaries = []
    if args.viewer:
        env, obs = configure_env(args, args.seed, task, complexity)
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            summary = rollout_with_env(args, model, env, obs, args.seed, task, complexity, viewer)
            summaries.append(summary)
            print("episode_summary:", summary, flush=True)
            start = time.time()
            while viewer.is_running():
                viewer.sync()
                if args.duration and time.time() - start > args.duration:
                    break
                time.sleep(0.01)
        start_episode = 1
    else:
        start_episode = 0

    for episode in range(start_episode, args.episodes):
        seed = args.seed + episode
        env, obs = configure_env(args, seed, task, complexity)
        summary = rollout_with_env(args, model, env, obs, seed, task, complexity)
        summaries.append(summary)
        print("episode_summary:", summary, flush=True)

    successes = sum(int(item["success"]) for item in summaries)
    print(f"success_rate: {successes}/{len(summaries)} = {successes / max(1, len(summaries)):.3f}", flush=True)


if __name__ == "__main__":
    main()
