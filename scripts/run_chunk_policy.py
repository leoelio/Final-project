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
from grasp_gate import add_grasp_gate_args, apply_grasp_gate, make_gate_stats, update_gate_stats  # noqa: E402
from run_mlp_policy import forward  # noqa: E402
from widowx_env import TASKS, WidowXTabletopEnv  # noqa: E402
from widowx_env.demo_dataset import observation_from_env, read_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a NumPy action-chunk BC / ACT-lite policy in closed loop.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--task", choices=sorted(TASKS), default=None)
    parser.add_argument("--complexity", choices=("easy", "medium", "hard", "language"), default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2840)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-viewer", dest="viewer", action="store_false")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--speed", type=float, default=0.2)
    parser.add_argument("--arm-kp", type=float, default=150.0)
    parser.add_argument("--arm-force", type=float, default=100.0)
    parser.add_argument("--gripper-kp", type=float, default=800.0)
    parser.add_argument("--gripper-force", type=float, default=140.0)
    parser.add_argument("--friction", type=float, default=3.0)
    add_grasp_gate_args(parser)
    parser.add_argument("--clip-actions", action="store_true")
    parser.add_argument("--no-clip-actions", dest="clip_actions", action="store_false")
    parser.add_argument("--action-alpha", type=float, default=0.25)
    parser.add_argument("--max-arm-delta", type=float, default=0.012)
    parser.add_argument("--max-gripper-delta", type=float, default=0.0005)
    parser.add_argument("--replan-interval", type=int, default=1)
    parser.add_argument("--temporal-ensemble", action="store_true")
    parser.add_argument("--no-temporal-ensemble", dest="temporal_ensemble", action="store_false")
    parser.add_argument("--ensemble-decay", type=float, default=0.05)
    parser.add_argument("--stop-on-unsafe", action="store_true")
    parser.add_argument("--no-stop-on-unsafe", dest="stop_on_unsafe", action="store_false")
    parser.add_argument("--log-every", type=int, default=500)
    parser.set_defaults(clip_actions=True)
    parser.set_defaults(stop_on_unsafe=True)
    parser.set_defaults(temporal_ensemble=True)
    parser.set_defaults(viewer=True)
    return parser.parse_args()


def latest_model() -> Path:
    model_dir = ROOT / "outputs" / "chunk_bc"
    candidates = sorted(
        [*model_dir.glob("chunk_bc_*.npz"), *model_dir.glob("trajectory_chunk_bc_*.npz")],
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"no chunk BC models found under {model_dir}")
    return candidates[-1]


def load_model(path: Path) -> dict:
    with np.load(path) as data:
        metadata = json.loads(data["metadata"].item())
        layers = []
        for index in range(len(metadata["hidden_sizes"]) + 1):
            layers.append({"w": data[f"w{index}"].astype(np.float32), "b": data[f"b{index}"].astype(np.float32)})
        return {
            "layers": layers,
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


def model_history(model: dict) -> int:
    return max(1, int(model["metadata"].get("history", 1)))


def prepare_model_input(model: dict, observation_history: list[np.ndarray]) -> np.ndarray:
    history = model_history(model)
    raw_history = np.asarray(observation_history[-history:], dtype=np.float32)
    if len(raw_history) < history:
        pad = np.repeat(raw_history[:1], history - len(raw_history), axis=0)
        raw_history = np.concatenate([pad, raw_history], axis=0)
    if model["metadata"].get("augment_relative", False):
        raw_history = augment_relative_features(raw_history, model["metadata"]["layout"])
    return raw_history.reshape(-1).astype(np.float32)


def predict_chunk(model: dict, model_input: np.ndarray) -> np.ndarray:
    x = ((model_input.astype(np.float32) - model["x_mean"]) / model["x_std"])[None, :]
    y = forward(model["layers"], x)[0].astype(np.float32)
    chunk = y * model["y_std"] + model["y_mean"]
    horizon = int(model["metadata"]["horizon"])
    action_dim = int(model["metadata"]["action_dim"])
    return chunk.reshape(horizon, action_dim).astype(np.float32)


def augment_relative_features(observations: np.ndarray, layout: dict[str, int]) -> np.ndarray:
    tcp_start = int(layout["tcp_start"])
    object_start = tcp_start + 3
    object_end = object_start + int(layout["object_count"]) * 3
    target_start = object_end

    tcp = observations[:, tcp_start: tcp_start + 3]
    objects = observations[:, object_start:object_end].reshape(len(observations), int(layout["object_count"]), 3)
    target = observations[:, target_start: target_start + 3]
    object_to_tcp = (objects - tcp[:, None, :]).reshape(len(observations), -1)
    object_to_target = (objects - target[:, None, :]).reshape(len(observations), -1)
    tcp_to_target = tcp - target
    return np.concatenate([observations, object_to_tcp, object_to_target, tcp_to_target], axis=1).astype(np.float32)


def configure_env(args: argparse.Namespace, seed: int, task: str, complexity: str) -> tuple[WidowXTabletopEnv, dict]:
    env = WidowXTabletopEnv(seed=seed)
    env.set_arm_actuator_strength(kp=args.arm_kp, force_limit=args.arm_force)
    env.set_gripper_actuator_strength(kp=args.gripper_kp, force_limit=args.gripper_force)
    env.set_grasp_contact_friction(sliding=args.friction)
    obs = env.reset(task=task, complexity=complexity, seed=seed)
    return env, obs


def select_action(
    args: argparse.Namespace,
    chunks: list[tuple[int, np.ndarray]],
    step: int,
) -> np.ndarray:
    candidates = []
    weights = []
    for start_step, chunk in chunks:
        offset = step - start_step
        if 0 <= offset < len(chunk):
            candidates.append(chunk[offset])
            weights.append(np.exp(-args.ensemble_decay * offset))

    if not candidates:
        raise RuntimeError("no valid action chunk is available")
    if not args.temporal_ensemble:
        return candidates[-1].astype(np.float32)

    action_stack = np.stack(candidates).astype(np.float32)
    weight_array = np.asarray(weights, dtype=np.float32)
    weight_array /= float(weight_array.sum())
    return (action_stack * weight_array[:, None]).sum(axis=0).astype(np.float32)


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
    max_object_z = -float("inf")
    ever_grasp_success = False
    gate_stats = make_gate_stats()

    for step in range(args.steps):
        phase = step / phase_denom
        observation = observation_from_env(env, target_position, phase=phase)
        observation_history.append(observation)
        observation_history = observation_history[-model_history(model):]
        if step % replan_interval == 0:
            chunks.append((step, predict_chunk(model, prepare_model_input(model, observation_history))))
            chunks = [(start, chunk) for start, chunk in chunks if step - start < horizon]

        raw_action = select_action(args, chunks, step)
        raw_action, gate_state = apply_grasp_gate(args, env, raw_action, phase)
        update_gate_stats(gate_stats, gate_state)
        action = postprocess_action(args, model, raw_action, previous_action, ctrl_min, ctrl_max)
        delta = action - previous_action
        env.step(action)
        previous_action = action
        action_norms.append(float(np.linalg.norm(action)))
        steps_taken = step + 1
        metrics = env.metrics()
        max_object_z = max(max_object_z, float(metrics["object_z"]))
        ever_grasp_success = ever_grasp_success or bool(metrics["grasp_success"])
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
        "out_of_table": bool(metrics["out_of_table"]),
        "steps_taken": steps_taken,
        "stop_reason": stop_reason,
        "mean_action_norm": float(np.mean(action_norms)),
        "max_action_norm": float(np.max(action_norms)),
        "gate_mode": "grasp_gate" if bool(args.grasp_gate) else "policy",
        **gate_stats,
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
    print(f"hidden_sizes: {model['metadata']['hidden_sizes']}", flush=True)
    print(f"horizon: {model['metadata']['horizon']}", flush=True)
    print(f"history: {model_history(model)}", flush=True)
    print(f"grasp_gate: {bool(args.grasp_gate)}", flush=True)
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
