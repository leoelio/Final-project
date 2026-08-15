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
sys.path.insert(0, str(ROOT / "scripts"))

from widowx_env import TASKS, WidowXTabletopEnv
from widowx_env.demo_dataset import observation_from_env, read_metadata
from widowx_env.tabletop_env import OBJECTS
from run_bc_policy import postprocess_action, print_step, unsafe_reason


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained NumPy kNN BC policy in closed loop.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--task", choices=sorted(TASKS), default=None)
    parser.add_argument("--complexity", choices=("easy", "medium", "hard", "language"), default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2840)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--phase-window", type=float, default=0.02)
    parser.add_argument("--min-candidates", type=int, default=128)
    parser.add_argument("--qpos-weight", type=float, default=0.25)
    parser.add_argument("--qvel-weight", type=float, default=0.05)
    parser.add_argument("--ctrl-weight", type=float, default=0.25)
    parser.add_argument("--tcp-weight", type=float, default=1.0)
    parser.add_argument("--object-weight", type=float, default=4.0)
    parser.add_argument("--target-weight", type=float, default=1.0)
    parser.add_argument("--phase-weight", type=float, default=2.0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-viewer", dest="viewer", action="store_false")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--arm-kp", type=float, default=150.0)
    parser.add_argument("--arm-force", type=float, default=100.0)
    parser.add_argument("--gripper-kp", type=float, default=800.0)
    parser.add_argument("--gripper-force", type=float, default=140.0)
    parser.add_argument("--friction", type=float, default=3.0)
    parser.add_argument("--clip-actions", action="store_true")
    parser.add_argument("--no-clip-actions", dest="clip_actions", action="store_false")
    parser.add_argument("--action-alpha", type=float, default=1.0)
    parser.add_argument("--max-arm-delta", type=float, default=0.05)
    parser.add_argument("--max-gripper-delta", type=float, default=0.002)
    parser.add_argument("--stop-on-unsafe", action="store_true")
    parser.add_argument("--no-stop-on-unsafe", dest="stop_on_unsafe", action="store_false")
    parser.add_argument("--log-every", type=int, default=500)
    parser.set_defaults(clip_actions=True)
    parser.set_defaults(stop_on_unsafe=True)
    parser.set_defaults(viewer=True)
    return parser.parse_args()


def latest_model() -> Path:
    candidates = sorted((ROOT / "outputs" / "knn_bc").glob("knn_bc_*.npz"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no kNN BC models found under {ROOT / 'outputs' / 'knn_bc'}")
    return candidates[-1]


def load_model(path: Path) -> dict:
    with np.load(path) as data:
        return {
            "observations_norm": data["observations_norm"].astype(np.float32),
            "actions": data["actions"].astype(np.float32),
            "phases": data["phases"].astype(np.float32),
            "x_mean": data["x_mean"].astype(np.float32),
            "x_std": data["x_std"].astype(np.float32),
            "action_min": data["action_min"].astype(np.float32),
            "action_max": data["action_max"].astype(np.float32),
            "metadata": json.loads(data["metadata"].item()),
        }


def infer_task_defaults(model: dict) -> tuple[str, str]:
    first = read_metadata(Path(model["metadata"]["run_dir"]))[0]
    return str(first["task"]), str(first["complexity"])


def feature_weights(args: argparse.Namespace, env: WidowXTabletopEnv) -> np.ndarray:
    weights = np.ones(int(model_observation_dim(env)), dtype=np.float32)
    offset = 0
    for size, weight in (
        (env.model.nq, args.qpos_weight),
        (env.model.nv, args.qvel_weight),
        (env.model.nu, args.ctrl_weight),
        (3, args.tcp_weight),
        (len(OBJECTS) * 3, args.object_weight),
        (3, args.target_weight),
        (3, args.phase_weight),
    ):
        weights[offset: offset + size] = float(weight)
        offset += size
    return weights


def model_observation_dim(env: WidowXTabletopEnv) -> int:
    return env.model.nq + env.model.nv + env.model.nu + 3 + len(OBJECTS) * 3 + 3 + 3


def predict_action(args: argparse.Namespace, model: dict, observation: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x = ((observation.astype(np.float32) - model["x_mean"]) / model["x_std"]).astype(np.float32)
    phase = float(observation[-3])
    phase_delta = np.abs(model["phases"] - phase)
    candidates = np.flatnonzero(phase_delta <= args.phase_window)
    min_candidates = max(int(args.k), int(args.min_candidates))
    if len(candidates) < min_candidates:
        count = min(len(phase_delta), min_candidates)
        candidates = np.argpartition(phase_delta, count - 1)[:count]

    diff = (model["observations_norm"][candidates] - x) * weights
    distances = np.einsum("ij,ij->i", diff, diff)
    k = min(int(args.k), len(candidates))
    nearest_local = np.argpartition(distances, k - 1)[:k]
    nearest = candidates[nearest_local]
    nearest_distances = distances[nearest_local]
    weights = 1.0 / (nearest_distances + 1e-6)
    weights = weights / weights.sum()
    return (weights[:, None] * model["actions"][nearest]).sum(axis=0).astype(np.float32)


def configure_env(args: argparse.Namespace, seed: int, task: str, complexity: str) -> tuple[WidowXTabletopEnv, dict]:
    env = WidowXTabletopEnv(seed=seed)
    env.set_arm_actuator_strength(kp=args.arm_kp, force_limit=args.arm_force)
    env.set_gripper_actuator_strength(kp=args.gripper_kp, force_limit=args.gripper_force)
    env.set_grasp_contact_friction(sliding=args.friction)
    obs = env.reset(task=task, complexity=complexity, seed=seed)
    return env, obs


def rollout_with_env(args: argparse.Namespace, model: dict, env: WidowXTabletopEnv, obs: dict, seed: int, task: str, complexity: str, viewer=None) -> dict:
    target_position = env.target_position(env.task.target_geom).copy() if env.task.target_geom else np.zeros(3, dtype=np.float32)
    ctrl_min = env.model.actuator_ctrlrange[:, 0]
    ctrl_max = env.model.actuator_ctrlrange[:, 1]
    weights = feature_weights(args, env)
    dt = float(env.model.opt.timestep)
    previous_action = env.data.ctrl.copy()
    action_norms = []
    stop_reason = None
    steps_taken = 0
    phase_denom = max(1, args.steps - 1)

    for step in range(args.steps):
        observation = observation_from_env(env, target_position, phase=step / phase_denom)
        raw_action = predict_action(args, model, observation, weights)
        action = postprocess_action(args, model, raw_action, previous_action, ctrl_min, ctrl_max)
        delta = action - previous_action
        env.step(action)
        previous_action = action
        action_norms.append(float(np.linalg.norm(action)))
        steps_taken = step + 1
        metrics = env.metrics()
        if args.log_every > 0 and (step == 0 or (step + 1) % args.log_every == 0):
            print_step(step + 1, metrics, action, delta)
        stop_reason = unsafe_reason(metrics) if args.stop_on_unsafe else None
        if viewer is not None:
            viewer.sync()
            if args.speed > 0:
                time.sleep(dt / args.speed)
        if stop_reason:
            print(f"stopped_early: step={step + 1}, reason={stop_reason}", flush=True)
            break

    metrics = env.metrics()
    return {
        "seed": seed,
        "task": task,
        "complexity": complexity,
        "instruction": obs["instruction"],
        "active_objects": list(obs["active_objects"]),
        "success": bool(metrics["success"]),
        "target_distance": float(metrics["target_distance"]),
        "object_z": float(metrics["object_z"]),
        "grasp_success": bool(metrics["grasp_success"]),
        "out_of_table": bool(metrics["out_of_table"]),
        "steps_taken": steps_taken,
        "stop_reason": stop_reason,
        "mean_action_norm": float(np.mean(action_norms)),
        "max_action_norm": float(np.max(action_norms)),
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
    print(f"samples: {model['metadata']['samples']}", flush=True)
    print(f"k: {args.k}", flush=True)
    print(f"phase_window: {args.phase_window}", flush=True)
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
