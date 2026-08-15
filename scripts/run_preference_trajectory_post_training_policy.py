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

from run_bc_policy import postprocess_action, print_step, unsafe_reason  # noqa: E402
from run_trajectory_knn_policy import (  # noqa: E402
    configure_env,
    history_weights,
    infer_task_defaults,
    model_history,
    prepare_model_input,
    select_action,
)
from widowx_env import TASKS, WidowXTabletopEnv  # noqa: E402
from widowx_env.demo_dataset import observation_from_env  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a preference-weighted trajectory-kNN post-training candidate.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--task", choices=sorted(TASKS), default=None)
    parser.add_argument("--complexity", choices=("easy", "medium", "hard", "language"), default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2840)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--phase-window", type=float, default=0.03)
    parser.add_argument("--min-candidates", type=int, default=256)
    parser.add_argument("--history-decay", type=float, default=0.25)
    parser.add_argument("--preference-power", type=float, default=1.0)
    parser.add_argument("--distance-epsilon", type=float, default=1e-6)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-viewer", dest="viewer", action="store_false")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--speed", type=float, default=0.05)
    parser.add_argument("--arm-kp", type=float, default=150.0)
    parser.add_argument("--arm-force", type=float, default=100.0)
    parser.add_argument("--gripper-kp", type=float, default=800.0)
    parser.add_argument("--gripper-force", type=float, default=140.0)
    parser.add_argument("--friction", type=float, default=3.0)
    parser.add_argument("--clip-actions", action="store_true")
    parser.add_argument("--no-clip-actions", dest="clip_actions", action="store_false")
    parser.add_argument("--action-alpha", type=float, default=0.25)
    parser.add_argument("--max-arm-delta", type=float, default=0.012)
    parser.add_argument("--max-gripper-delta", type=float, default=0.0005)
    parser.add_argument("--replan-interval", type=int, default=1)
    parser.add_argument("--temporal-ensemble", action="store_true")
    parser.add_argument("--no-temporal-ensemble", dest="temporal_ensemble", action="store_false")
    parser.add_argument("--ensemble-decay", type=float, default=0.1)
    parser.add_argument("--stop-on-unsafe", action="store_true")
    parser.add_argument("--no-stop-on-unsafe", dest="stop_on_unsafe", action="store_false")
    parser.add_argument("--log-every", type=int, default=500)
    parser.set_defaults(clip_actions=True)
    parser.set_defaults(stop_on_unsafe=True)
    parser.set_defaults(temporal_ensemble=True)
    parser.set_defaults(viewer=True)
    return parser.parse_args()


def latest_model() -> Path:
    candidates = sorted(
        (ROOT / "outputs" / "preference_post_training").glob("preference_trajectory_post_training_*.npz"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"no preference post-training models found under {ROOT / 'outputs' / 'preference_post_training'}")
    return candidates[-1]


def load_model(path: Path) -> dict:
    with np.load(path) as data:
        return {
            "observations_norm": data["observations_norm"].astype(np.float32),
            "action_chunks": data["action_chunks"].astype(np.float32),
            "phases": data["phases"].astype(np.float32),
            "sample_preferences": data["sample_preferences"].astype(np.float32),
            "x_mean": data["x_mean"].astype(np.float32),
            "x_std": data["x_std"].astype(np.float32),
            "action_min": data["action_min"].astype(np.float32),
            "action_max": data["action_max"].astype(np.float32),
            "metadata": json.loads(data["metadata"].item()),
        }


def predict_chunk(args: argparse.Namespace, model: dict, model_input: np.ndarray, phase: float) -> np.ndarray:
    x = ((model_input.astype(np.float32) - model["x_mean"]) / model["x_std"]).astype(np.float32)
    phase_delta = np.abs(model["phases"] - float(phase))
    candidates = np.flatnonzero(phase_delta <= args.phase_window)
    min_candidates = max(int(args.k), int(args.min_candidates))
    if len(candidates) < min_candidates:
        count = min(len(phase_delta), min_candidates)
        candidates = np.argpartition(phase_delta, count - 1)[:count]

    diff = (model["observations_norm"][candidates] - x) * history_weights(model, args.history_decay)
    distances = np.einsum("ij,ij->i", diff, diff)
    k = min(int(args.k), len(candidates))
    nearest_local = np.argpartition(distances, k - 1)[:k]
    nearest = candidates[nearest_local]
    nearest_distances = distances[nearest_local]
    preference = np.maximum(model["sample_preferences"][nearest], 1e-6) ** float(args.preference_power)
    weights = preference / (nearest_distances + float(args.distance_epsilon))
    weights /= float(weights.sum())
    return (weights[:, None, None] * model["action_chunks"][nearest]).sum(axis=0).astype(np.float32)


def rollout_with_env(args: argparse.Namespace, model: dict, env: WidowXTabletopEnv, obs: dict, seed: int, task: str, complexity: str, viewer=None) -> dict:
    target_position = env.target_position(env.task.target_geom).copy() if env.task.target_geom else np.zeros(3, dtype=np.float32)
    ctrl_min = env.model.actuator_ctrlrange[:, 0]
    ctrl_max = env.model.actuator_ctrlrange[:, 1]
    dt = float(env.model.opt.timestep)
    horizon = int(model["metadata"]["horizon"])
    replan_interval = max(1, int(args.replan_interval))
    previous_action = env.data.ctrl.copy()
    chunks: list[tuple[int, np.ndarray]] = []
    observation_history: list[np.ndarray] = []
    action_norms = []
    stop_reason = None
    steps_taken = 0
    phase_denom = max(1, args.steps - 1)
    lift_threshold = 0.085
    tcp_lift_threshold = 0.12
    max_object_z = -float("inf")
    ever_grasp_success = False
    ever_tcp_lift_success = False
    min_tcp_object_distance = None
    min_tcp_object_distance_while_lifted = None

    for step in range(args.steps):
        phase = step / phase_denom
        observation = observation_from_env(env, target_position, phase=phase)
        observation_history.append(observation)
        observation_history = observation_history[-model_history(model):]
        if step % replan_interval == 0:
            model_input = prepare_model_input(model, observation_history)
            chunks.append((step, predict_chunk(args, model, model_input, phase)))
            chunks = [(start, chunk) for start, chunk in chunks if step - start < horizon]

        raw_action = select_action(args, chunks, step)
        action = postprocess_action(args, model, raw_action, previous_action, ctrl_min, ctrl_max)
        delta = action - previous_action
        env.step(action)
        previous_action = action
        action_norms.append(float(np.linalg.norm(action)))
        steps_taken = step + 1
        metrics = env.metrics()
        object_position = env.object_position(env.episode_target_object)
        tcp_object_distance = float(np.linalg.norm(env.tcp_position() - object_position))
        max_object_z = max(max_object_z, float(metrics["object_z"]))
        ever_grasp_success = bool(ever_grasp_success or metrics["grasp_success"])
        if min_tcp_object_distance is None or tcp_object_distance < min_tcp_object_distance:
            min_tcp_object_distance = tcp_object_distance
        if float(metrics["object_z"]) >= lift_threshold:
            if min_tcp_object_distance_while_lifted is None or tcp_object_distance < min_tcp_object_distance_while_lifted:
                min_tcp_object_distance_while_lifted = tcp_object_distance
            if tcp_object_distance < tcp_lift_threshold:
                ever_tcp_lift_success = True
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
        "max_object_z": float(max_object_z),
        "grasp_success": bool(metrics["grasp_success"]),
        "ever_grasp_success": bool(ever_grasp_success),
        "ever_tcp_lift_success": bool(ever_tcp_lift_success),
        "height_threshold_hit": bool(max_object_z >= lift_threshold),
        "strict_grasp_lift_success": bool(ever_grasp_success and max_object_z >= lift_threshold),
        "tcp_grasp_lift_success": bool(ever_tcp_lift_success and max_object_z >= lift_threshold),
        "min_tcp_object_distance": min_tcp_object_distance,
        "min_tcp_object_distance_while_lifted": min_tcp_object_distance_while_lifted,
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
    print(f"version: {model['metadata'].get('version', 'preference_trajectory_post_training_v1_candidate')}", flush=True)
    print(f"preference_source: {model['metadata'].get('preference_source')}", flush=True)
    print(f"preference_strategy: {model['metadata'].get('preference_strategy')}", flush=True)
    print(f"preference_summary: {model['metadata'].get('preference_summary')}", flush=True)
    print(f"samples: {model['metadata']['samples']}", flush=True)
    print(f"horizon: {model['metadata']['horizon']}", flush=True)
    print(f"history: {model_history(model)}", flush=True)
    print(f"k: {args.k}", flush=True)
    print(f"phase_window: {args.phase_window}", flush=True)
    print(f"preference_power: {args.preference_power}", flush=True)
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
