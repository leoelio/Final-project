from __future__ import annotations

import argparse
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
    infer_task_defaults,
    latest_model,
    load_model,
    model_history,
    predict_chunk,
    prepare_model_input,
    select_action,
)
from widowx_env import TASKS, WidowXTabletopEnv  # noqa: E402
from widowx_env.demo_dataset import observation_from_env  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trajectory-kNN with a diagnostic grasp/release gripper gate.")
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
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-viewer", dest="viewer", action="store_false")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--speed", type=float, default=0.05)
    parser.add_argument("--arm-kp", type=float, default=150.0)
    parser.add_argument("--arm-force", type=float, default=100.0)
    parser.add_argument("--gripper-kp", type=float, default=900.0)
    parser.add_argument("--gripper-force", type=float, default=180.0)
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
    parser.add_argument("--close-phase", type=float, default=0.22)
    parser.add_argument("--release-phase", type=float, default=0.78)
    parser.add_argument("--near-threshold", type=float, default=0.11)
    parser.add_argument("--open-gripper", type=float, default=0.037)
    parser.add_argument("--close-gripper", type=float, default=0.015)
    parser.add_argument("--log-every", type=int, default=500)
    parser.set_defaults(clip_actions=True)
    parser.set_defaults(stop_on_unsafe=True)
    parser.set_defaults(temporal_ensemble=True)
    parser.set_defaults(viewer=True)
    return parser.parse_args()


def gripper_gate(args: argparse.Namespace, env: WidowXTabletopEnv, raw_action: np.ndarray, phase: float) -> np.ndarray:
    action = raw_action.copy()
    metrics = env.metrics()
    near_object = float(metrics["ee_object_distance"]) <= float(args.near_threshold)
    release_ready = bool(env.task.target_geom) and np.isfinite(metrics["target_distance"]) and float(metrics["target_distance"]) < 0.095

    if phase >= float(args.release_phase) and release_ready:
        action[6] = float(args.open_gripper)
    elif phase >= float(args.close_phase) or near_object:
        action[6] = float(args.close_gripper)
    else:
        action[6] = float(args.open_gripper)
    return action.astype(np.float32)


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
    gate_closed_steps = 0
    gate_open_steps = 0
    stop_reason = None
    steps_taken = 0
    phase_denom = max(1, args.steps - 1)

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
        gated_action = gripper_gate(args, env, raw_action, phase)
        if abs(float(gated_action[6]) - float(args.close_gripper)) < abs(float(gated_action[6]) - float(args.open_gripper)):
            gate_closed_steps += 1
        else:
            gate_open_steps += 1
        action = postprocess_action(args, model, gated_action, previous_action, ctrl_min, ctrl_max)
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
        "target_object": env.episode_target_object,
        "success": bool(metrics["success"]),
        "target_distance": float(metrics["target_distance"]),
        "object_z": float(metrics["object_z"]),
        "grasp_success": bool(metrics["grasp_success"]),
        "out_of_table": bool(metrics["out_of_table"]),
        "steps_taken": steps_taken,
        "stop_reason": stop_reason,
        "mean_action_norm": float(np.mean(action_norms)),
        "max_action_norm": float(np.max(action_norms)),
        "gate_closed_steps": gate_closed_steps,
        "gate_open_steps": gate_open_steps,
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
    print("version: grasp_gated_trajectory_knn_v1_candidate", flush=True)
    print(f"samples: {model['metadata']['samples']}", flush=True)
    print(f"horizon: {model['metadata']['horizon']}", flush=True)
    print(f"history: {model_history(model)}", flush=True)
    print(f"k: {args.k}", flush=True)
    print(f"phase_window: {args.phase_window}", flush=True)
    print(f"close_phase: {args.close_phase}", flush=True)
    print(f"release_phase: {args.release_phase}", flush=True)
    print(f"near_threshold: {args.near_threshold}", flush=True)
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
