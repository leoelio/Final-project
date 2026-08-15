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
from run_mlp_policy import forward  # noqa: E402
from train_object_action_head import build_features  # noqa: E402
from train_phase_action_head import PHASE_NAMES  # noqa: E402
from widowx_env import TASKS, WidowXTabletopEnv  # noqa: E402
from widowx_env.demo_dataset import observation_from_env, read_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a phase-conditioned object-language action-head policy.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--task", choices=sorted(TASKS), default=None)
    parser.add_argument("--complexity", choices=("easy", "medium", "hard", "language"), default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2840)
    parser.add_argument("--phase-mode", choices=("progress", "state", "hybrid"), default="progress")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-viewer", dest="viewer", action="store_false")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--speed", type=float, default=0.1)
    parser.add_argument("--arm-kp", type=float, default=150.0)
    parser.add_argument("--arm-force", type=float, default=100.0)
    parser.add_argument("--gripper-kp", type=float, default=800.0)
    parser.add_argument("--gripper-force", type=float, default=140.0)
    parser.add_argument("--friction", type=float, default=3.0)
    parser.add_argument("--clip-actions", action="store_true")
    parser.add_argument("--no-clip-actions", dest="clip_actions", action="store_false")
    parser.add_argument("--action-alpha", type=float, default=0.2)
    parser.add_argument("--max-arm-delta", type=float, default=0.01)
    parser.add_argument("--max-gripper-delta", type=float, default=0.0005)
    parser.add_argument("--stop-on-unsafe", action="store_true")
    parser.add_argument("--no-stop-on-unsafe", dest="stop_on_unsafe", action="store_false")
    parser.add_argument("--log-every", type=int, default=500)
    parser.set_defaults(clip_actions=True)
    parser.set_defaults(stop_on_unsafe=True)
    parser.set_defaults(viewer=True)
    return parser.parse_args()


def latest_model() -> Path:
    root = ROOT / "outputs" / "phase_action_head"
    candidates = sorted(root.glob("phase_conditioned_action_head_lite_*.npz"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no phase action-head models found under {root}")
    return candidates[-1]


def load_model(path: Path) -> dict:
    with np.load(path) as data:
        metadata = json.loads(data["metadata"].item())
        phase_models = []
        for phase_id in range(len(metadata["phase_names"])):
            layers = []
            for index in range(len(metadata["hidden_sizes"]) + 1):
                layers.append(
                    {
                        "w": data[f"phase{phase_id}_w{index}"].astype(np.float32),
                        "b": data[f"phase{phase_id}_b{index}"].astype(np.float32),
                    }
                )
            phase_models.append(layers)
        return {
            "phase_models": phase_models,
            "x_mean": data["x_mean"].astype(np.float32),
            "x_std": data["x_std"].astype(np.float32),
            "y_mean": data["y_mean"].astype(np.float32),
            "y_std": data["y_std"].astype(np.float32),
            "action_min": data["action_min"].astype(np.float32),
            "action_max": data["action_max"].astype(np.float32),
            "metadata": metadata,
        }


def infer_task_defaults(model: dict) -> tuple[str, str]:
    first = read_metadata(Path(model["metadata"]["run_dir"]))[0]
    return str(first["task"]), str(first["complexity"])


def metadata_from_env(env: WidowXTabletopEnv, obs: dict) -> dict:
    return {
        "target_object": obs["target_object"],
        "target_geom": env.task.target_geom,
        "active_objects": list(obs["active_objects"]),
    }


def progress_phase(model: dict, progress: float) -> int:
    thresholds = np.asarray(model["metadata"]["phase_thresholds"], dtype=np.float32)
    return int(np.digitize(np.asarray([progress], dtype=np.float32), thresholds)[0])


def state_phase(env: WidowXTabletopEnv, obs: dict) -> int:
    object_position = env.object_position(obs["target_object"])
    tcp_position = env.tcp_position()
    tcp_object_distance = float(np.linalg.norm(tcp_position - object_position))
    target_distance = float("inf")
    if env.task.target_geom:
        target_position = env.target_position(env.task.target_geom)
        target_distance = float(np.linalg.norm(object_position[:2] - target_position[:2]))

    if object_position[2] > 0.09 and target_distance <= 0.08:
        return 4
    if object_position[2] > 0.085 and target_distance > 0.08:
        return 3
    if object_position[2] > 0.065 or (tcp_object_distance < 0.08 and env.data.ctrl[6] <= 0.022):
        return 2
    if tcp_object_distance < 0.09:
        return 1
    return 0


def choose_phase(args: argparse.Namespace, model: dict, env: WidowXTabletopEnv, obs: dict, progress: float) -> int:
    by_progress = progress_phase(model, progress)
    if args.phase_mode == "progress":
        return by_progress
    by_state = state_phase(env, obs)
    if args.phase_mode == "state":
        return by_state
    return min(by_progress, by_state + 1)


def predict_action(
    args: argparse.Namespace,
    model: dict,
    env: WidowXTabletopEnv,
    target_position: np.ndarray,
    progress: float,
    obs: dict,
) -> tuple[np.ndarray, int]:
    phase_id = choose_phase(args, model, env, obs, progress)
    observation = observation_from_env(env, target_position, phase=progress)[None, :]
    features = build_features(observation, metadata_from_env(env, obs), model["metadata"]["layout"])
    x = ((features[0].astype(np.float32) - model["x_mean"]) / model["x_std"])[None, :]
    y = forward(model["phase_models"][phase_id], x)[0].astype(np.float32)
    return y * model["y_std"] + model["y_mean"], phase_id


def configure_env(args: argparse.Namespace, seed: int, task: str, complexity: str) -> tuple[WidowXTabletopEnv, dict]:
    env = WidowXTabletopEnv(seed=seed)
    env.set_arm_actuator_strength(kp=args.arm_kp, force_limit=args.arm_force)
    env.set_gripper_actuator_strength(kp=args.gripper_kp, force_limit=args.gripper_force)
    env.set_grasp_contact_friction(sliding=args.friction)
    obs = env.reset(task=task, complexity=complexity, seed=seed)
    return env, obs


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
    target_position = env.target_position(env.task.target_geom).copy() if env.task.target_geom else np.zeros(3, dtype=np.float32)
    ctrl_min = env.model.actuator_ctrlrange[:, 0]
    ctrl_max = env.model.actuator_ctrlrange[:, 1]
    dt = float(env.model.opt.timestep)
    previous_action = env.data.ctrl.copy()
    action_norms = []
    phase_counts = {name: 0 for name in model["metadata"]["phase_names"]}
    stop_reason = None
    steps_taken = 0
    phase_denom = max(1, args.steps - 1)
    max_object_z = float(env.object_position(obs["target_object"])[2])
    ever_grasp_success = False
    ever_tcp_lift_success = False
    min_tcp_object_distance = float("inf")
    min_tcp_object_distance_while_lifted = float("inf")
    lift_threshold = 0.08

    for step in range(args.steps):
        progress = step / phase_denom
        raw_action, phase_id = predict_action(args, model, env, target_position, progress, obs)
        phase_counts[model["metadata"]["phase_names"][phase_id]] += 1
        action = postprocess_action(args, model, raw_action, previous_action, ctrl_min, ctrl_max)
        delta = action - previous_action
        env.step(action)
        previous_action = action
        action_norms.append(float(np.linalg.norm(action)))
        steps_taken = step + 1
        metrics = env.metrics()
        object_position = env.object_position(obs["target_object"])
        tcp_object_distance = float(np.linalg.norm(env.tcp_position() - object_position))
        max_object_z = max(max_object_z, float(object_position[2]))
        ever_grasp_success = ever_grasp_success or bool(metrics["grasp_success"])
        min_tcp_object_distance = min(min_tcp_object_distance, tcp_object_distance)
        if float(object_position[2]) >= lift_threshold:
            min_tcp_object_distance_while_lifted = min(min_tcp_object_distance_while_lifted, tcp_object_distance)
            ever_tcp_lift_success = ever_tcp_lift_success or tcp_object_distance <= 0.09
        if args.log_every > 0 and (step == 0 or (step + 1) % args.log_every == 0):
            print_step(step + 1, metrics, action, delta)
            print(f"phase={model['metadata']['phase_names'][phase_id]} mode={args.phase_mode}", flush=True)
        stop_reason = unsafe_reason(metrics) if args.stop_on_unsafe else None
        if viewer is not None:
            viewer.sync()
            if args.speed > 0:
                time.sleep(dt / args.speed)
        if stop_reason:
            print(f"stopped_early: step={step + 1}, reason={stop_reason}", flush=True)
            break

    metrics = env.metrics()
    height_threshold_hit = bool(max_object_z >= lift_threshold)
    return {
        "seed": seed,
        "task": task,
        "complexity": complexity,
        "instruction": obs["instruction"],
        "active_objects": list(obs["active_objects"]),
        "phase_mode": args.phase_mode,
        "phase_counts": phase_counts,
        "success": bool(metrics["success"]),
        "target_distance": float(metrics["target_distance"]),
        "object_z": float(metrics["object_z"]),
        "max_object_z": float(max_object_z),
        "height_threshold_hit": height_threshold_hit,
        "grasp_success": bool(metrics["grasp_success"]),
        "ever_grasp_success": bool(ever_grasp_success),
        "ever_tcp_lift_success": bool(ever_tcp_lift_success),
        "tcp_grasp_lift_success": bool(ever_tcp_lift_success and height_threshold_hit),
        "strict_grasp_lift_success": bool(ever_grasp_success and height_threshold_hit),
        "min_tcp_object_distance": None if not np.isfinite(min_tcp_object_distance) else float(min_tcp_object_distance),
        "min_tcp_object_distance_while_lifted": None
        if not np.isfinite(min_tcp_object_distance_while_lifted)
        else float(min_tcp_object_distance_while_lifted),
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
    print(f"feature_dim: {model['metadata']['feature_dim']}", flush=True)
    print(f"phase_mode: {args.phase_mode}", flush=True)
    print(f"phase_names: {', '.join(model['metadata']['phase_names'])}", flush=True)
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
    raise SystemExit(0)


if __name__ == "__main__":
    main()
