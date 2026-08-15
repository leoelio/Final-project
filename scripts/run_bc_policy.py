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

from widowx_env import TASKS, WidowXTabletopEnv
from widowx_env.demo_dataset import observation_from_env, read_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained NumPy BC policy in closed loop.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--task", choices=sorted(TASKS), default=None)
    parser.add_argument("--complexity", choices=("easy", "medium", "hard", "language"), default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1980)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-viewer", dest="viewer", action="store_false")
    parser.add_argument("--duration", type=float, default=0.0, help="Viewer hold time after first rollout. 0 means keep open.")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--arm-kp", type=float, default=150.0)
    parser.add_argument("--arm-force", type=float, default=100.0)
    parser.add_argument("--gripper-kp", type=float, default=800.0)
    parser.add_argument("--gripper-force", type=float, default=140.0)
    parser.add_argument("--friction", type=float, default=3.0)
    parser.add_argument("--clip-actions", action="store_true")
    parser.add_argument("--no-clip-actions", dest="clip_actions", action="store_false")
    parser.add_argument("--action-alpha", type=float, default=0.05, help="1.0 uses raw policy output; lower values smooth with the previous action.")
    parser.add_argument("--max-arm-delta", type=float, default=0.01)
    parser.add_argument("--max-gripper-delta", type=float, default=0.0005)
    parser.add_argument("--stop-on-unsafe", action="store_true")
    parser.add_argument("--no-stop-on-unsafe", dest="stop_on_unsafe", action="store_false")
    parser.add_argument("--log-every", type=int, default=100)
    parser.set_defaults(clip_actions=True)
    parser.set_defaults(stop_on_unsafe=True)
    parser.set_defaults(viewer=True)
    return parser.parse_args()


def latest_model() -> Path:
    candidates = sorted((ROOT / "outputs" / "bc").glob("bc_linear_*.npz"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no BC models found under {ROOT / 'outputs' / 'bc'}")
    return candidates[-1]


def load_model(path: Path) -> dict:
    with np.load(path) as data:
        model = {
            "weights": data["weights"].astype(np.float32),
            "x_mean": data["x_mean"].astype(np.float32),
            "x_std": data["x_std"].astype(np.float32),
            "metadata": json.loads(data["metadata"].item()),
        }
        if "action_min" in data.files and "action_max" in data.files:
            model["action_min"] = data["action_min"].astype(np.float32)
            model["action_max"] = data["action_max"].astype(np.float32)
        return model


def infer_task_defaults(model: dict) -> tuple[str, str]:
    run_dir = Path(model["metadata"]["run_dir"])
    first = read_metadata(run_dir)[0]
    return str(first["task"]), str(first["complexity"])


def predict_action(model: dict, observation: np.ndarray) -> np.ndarray:
    x = observation[None, :]
    if x.shape[1] != model["x_mean"].shape[0]:
        raise ValueError(f"observation dim {x.shape[1]} does not match model dim {model['x_mean'].shape[0]}")
    x_norm = (x - model["x_mean"]) / model["x_std"]
    x_aug = np.concatenate([x_norm, np.ones((1, 1), dtype=np.float32)], axis=1)
    return (x_aug @ model["weights"])[0].astype(np.float32)


def postprocess_action(
    args: argparse.Namespace,
    model: dict,
    raw_action: np.ndarray,
    previous_action: np.ndarray,
    ctrl_min: np.ndarray,
    ctrl_max: np.ndarray,
) -> np.ndarray:
    action = raw_action
    if args.clip_actions and "action_min" in model:
        action = np.clip(action, model["action_min"], model["action_max"])
    alpha = float(np.clip(args.action_alpha, 0.0, 1.0))
    action = alpha * action + (1.0 - alpha) * previous_action
    delta_limit = np.full_like(action, args.max_arm_delta, dtype=np.float32)
    delta_limit[6] = args.max_gripper_delta
    action = previous_action + np.clip(action - previous_action, -delta_limit, delta_limit)
    if args.clip_actions and "action_min" in model:
        action = np.clip(action, model["action_min"], model["action_max"])
    action = np.clip(action, ctrl_min, ctrl_max)
    return action.astype(np.float32)


def unsafe_reason(metrics: dict[str, float | bool]) -> str | None:
    if bool(metrics["out_of_table"]):
        return "target object left the table workspace"
    if float(metrics["object_z"]) < -0.05:
        return "target object fell below the table"
    if np.isfinite(metrics["target_distance"]) and float(metrics["target_distance"]) > 1.0:
        return "target object moved too far from the goal"
    return None


def print_step(step: int, metrics: dict[str, float | bool], action: np.ndarray, delta: np.ndarray) -> None:
    print(
        f"step={step} success={bool(metrics['success'])} "
        f"target_distance={float(metrics['target_distance']):.4f} "
        f"ee_object_distance={float(metrics['ee_object_distance']):.4f} "
        f"object_z={float(metrics['object_z']):.4f} "
        f"out_of_table={bool(metrics['out_of_table'])} "
        f"max_delta={float(np.max(np.abs(delta))):.5f} "
        f"action={np.round(action, 4).tolist()}",
        flush=True,
    )


def rollout(args: argparse.Namespace, model: dict, seed: int, task: str, complexity: str, viewer=None) -> dict:
    env = WidowXTabletopEnv(seed=seed)
    env.set_arm_actuator_strength(kp=args.arm_kp, force_limit=args.arm_force)
    env.set_gripper_actuator_strength(kp=args.gripper_kp, force_limit=args.gripper_force)
    env.set_grasp_contact_friction(sliding=args.friction)
    obs = env.reset(task=task, complexity=complexity, seed=seed)
    target_position = env.target_position(env.task.target_geom).copy() if env.task.target_geom else np.zeros(3, dtype=np.float32)
    dt = float(env.model.opt.timestep)
    ctrl_min = env.model.actuator_ctrlrange[:, 0]
    ctrl_max = env.model.actuator_ctrlrange[:, 1]

    action_norms = []
    previous_action = env.data.ctrl.copy()
    stop_reason = None
    steps_taken = 0
    phase_denom = max(1, args.steps - 1)
    for step in range(args.steps):
        observation = observation_from_env(env, target_position, phase=step / phase_denom)
        raw_action = predict_action(model, observation)
        action = postprocess_action(args, model, raw_action, previous_action, ctrl_min, ctrl_max)
        delta = action - previous_action
        action_norms.append(float(np.linalg.norm(action)))
        env.step(action)
        previous_action = action
        steps_taken = step + 1
        metrics = env.metrics()
        if args.log_every > 0 and (step == 0 or (step + 1) % args.log_every == 0):
            print_step(step + 1, metrics, action, delta)
        stop_reason = unsafe_reason(metrics) if args.stop_on_unsafe else None
        if stop_reason:
            print(f"stopped_early: step={step + 1}, reason={stop_reason}", flush=True)
            break
        if viewer is not None:
            viewer.sync()
            if args.speed > 0:
                time.sleep(dt / args.speed)

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
    print(f"task: {task}", flush=True)
    print(f"complexity: {complexity}", flush=True)

    summaries = []
    if args.viewer:
        env_for_viewer = WidowXTabletopEnv(seed=args.seed)
        env_for_viewer.set_arm_actuator_strength(kp=args.arm_kp, force_limit=args.arm_force)
        env_for_viewer.set_gripper_actuator_strength(kp=args.gripper_kp, force_limit=args.gripper_force)
        env_for_viewer.set_grasp_contact_friction(sliding=args.friction)
        env_for_viewer.reset(task=task, complexity=complexity, seed=args.seed)
        with mujoco.viewer.launch_passive(env_for_viewer.model, env_for_viewer.data) as viewer:
            summary = rollout_with_env(args, model, env_for_viewer, args.seed, task, complexity, viewer)
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
        summary = rollout(args, model, args.seed + episode, task, complexity)
        summaries.append(summary)
        print("episode_summary:", summary, flush=True)

    successes = sum(int(item["success"]) for item in summaries)
    print(f"success_rate: {successes}/{len(summaries)} = {successes / max(1, len(summaries)):.3f}", flush=True)


def rollout_with_env(
    args: argparse.Namespace,
    model: dict,
    env: WidowXTabletopEnv,
    seed: int,
    task: str,
    complexity: str,
    viewer,
) -> dict:
    obs = env.observation(render=False)
    target_position = env.target_position(env.task.target_geom).copy() if env.task.target_geom else np.zeros(3, dtype=np.float32)
    dt = float(env.model.opt.timestep)
    ctrl_min = env.model.actuator_ctrlrange[:, 0]
    ctrl_max = env.model.actuator_ctrlrange[:, 1]
    action_norms = []
    previous_action = env.data.ctrl.copy()
    stop_reason = None
    steps_taken = 0
    phase_denom = max(1, args.steps - 1)

    for step in range(args.steps):
        observation = observation_from_env(env, target_position, phase=step / phase_denom)
        raw_action = predict_action(model, observation)
        action = postprocess_action(args, model, raw_action, previous_action, ctrl_min, ctrl_max)
        delta = action - previous_action
        action_norms.append(float(np.linalg.norm(action)))
        env.step(action)
        previous_action = action
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


if __name__ == "__main__":
    main()
