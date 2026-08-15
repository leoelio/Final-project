from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from torch_runtime import ensure_torch_path  # noqa: E402

ensure_torch_path()

import torch  # noqa: E402

from run_bc_policy import postprocess_action, print_step, unsafe_reason  # noqa: E402
from train_torch_act import StateACTPolicy  # noqa: E402
from train_vision_language_action_head import feature_from_env_state  # noqa: E402
from widowx_env import TASKS, WidowXTabletopEnv  # noqa: E402
from widowx_env.demo_dataset import read_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a rendered-visual-feature ACT-lite policy in closed loop.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--task", choices=sorted(TASKS), default=None)
    parser.add_argument("--complexity", choices=("easy", "medium", "hard", "language"), default=None)
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
    parser.add_argument("--clip-actions", action="store_true")
    parser.add_argument("--no-clip-actions", dest="clip_actions", action="store_false")
    parser.add_argument("--action-alpha", type=float, default=0.25)
    parser.add_argument("--max-arm-delta", type=float, default=0.012)
    parser.add_argument("--max-gripper-delta", type=float, default=0.0005)
    parser.add_argument("--replan-interval", type=int, default=4)
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
    root = ROOT / "outputs" / "visual_feature_act"
    candidates = sorted(root.glob("visual_feature_act_lite_*.pt"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no visual feature ACT models found under {root}")
    return candidates[-1]


def load_model(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    policy = StateACTPolicy(
        observation_dim=int(metadata["observation_dim"]),
        action_dim=int(metadata["action_dim"]),
        history=int(metadata["history"]),
        horizon=int(metadata["horizon"]),
        d_model=int(metadata["d_model"]),
        nhead=int(metadata["nhead"]),
        encoder_layers=int(metadata["encoder_layers"]),
        decoder_layers=int(metadata["decoder_layers"]),
        dim_feedforward=int(metadata["dim_feedforward"]),
        dropout=float(metadata["dropout"]),
    )
    policy.load_state_dict(checkpoint["model_state"])
    policy.eval()
    return {
        "policy": policy,
        "x_mean": checkpoint["x_mean"].float().numpy(),
        "x_std": checkpoint["x_std"].float().numpy(),
        "y_mean": checkpoint["y_mean"].float().numpy(),
        "y_std": checkpoint["y_std"].float().numpy(),
        "action_min": checkpoint["action_min"].float().numpy(),
        "action_max": checkpoint["action_max"].float().numpy(),
        "metadata": metadata,
    }


def infer_task_defaults(model: dict) -> tuple[str, str]:
    first = read_metadata(Path(model["metadata"]["run_dir"]))[0]
    return str(first["task"]), str(first["complexity"])


def metadata_from_obs(env: WidowXTabletopEnv, obs: dict) -> dict:
    return {
        "task": obs["task"],
        "target_object": obs["target_object"],
        "target_geom": env.task.target_geom,
        "active_objects": list(obs["active_objects"]),
    }


def model_history(model: dict) -> int:
    return max(1, int(model["metadata"].get("history", 1)))


def current_feature(model: dict, env: WidowXTabletopEnv, renderer: mujoco.Renderer, obs: dict, phase: float) -> np.ndarray:
    return feature_from_env_state(
        env,
        renderer,
        metadata_from_obs(env, obs),
        phase,
        int(model["metadata"]["grid_size"]),
    )


def prepare_model_input(model: dict, feature_history: list[np.ndarray]) -> np.ndarray:
    history = model_history(model)
    raw_history = np.asarray(feature_history[-history:], dtype=np.float32)
    if len(raw_history) < history:
        pad = np.repeat(raw_history[:1], history - len(raw_history), axis=0)
        raw_history = np.concatenate([pad, raw_history], axis=0)
    return ((raw_history - model["x_mean"][None, :]) / model["x_std"][None, :]).astype(np.float32)


@torch.no_grad()
def predict_chunk(model: dict, model_input: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(model_input[None, :, :].astype(np.float32))
    y_norm = model["policy"](x)[0].cpu().numpy().astype(np.float32)
    return y_norm * model["y_std"][None, :] + model["y_mean"][None, :]


def configure_env(args: argparse.Namespace, model: dict, seed: int, task: str, complexity: str) -> tuple[WidowXTabletopEnv, dict]:
    image_size = int(model["metadata"]["image_size"])
    camera = str(model["metadata"]["camera"])
    env = WidowXTabletopEnv(seed=seed, image_size=(image_size, image_size), camera=camera)
    env.set_arm_actuator_strength(kp=args.arm_kp, force_limit=args.arm_force)
    env.set_gripper_actuator_strength(kp=args.gripper_kp, force_limit=args.gripper_force)
    env.set_grasp_contact_friction(sliding=args.friction)
    obs = env.reset(task=task, complexity=complexity, seed=seed)
    return env, obs


def select_action(args: argparse.Namespace, chunks: list[tuple[int, np.ndarray]], step: int) -> np.ndarray:
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
    ctrl_min = env.model.actuator_ctrlrange[:, 0]
    ctrl_max = env.model.actuator_ctrlrange[:, 1]
    dt = float(env.model.opt.timestep)
    horizon = int(model["metadata"]["horizon"])
    replan_interval = max(1, int(args.replan_interval))
    previous_action = env.data.ctrl.copy()
    chunks: list[tuple[int, np.ndarray]] = []
    feature_history: list[np.ndarray] = []
    action_norms = []
    stop_reason = None
    steps_taken = 0
    phase_denom = max(1, args.steps - 1)
    renderer = mujoco.Renderer(env.model, height=int(model["metadata"]["image_size"]), width=int(model["metadata"]["image_size"]))
    try:
        for step in range(args.steps):
            feature_history.append(current_feature(model, env, renderer, obs, step / phase_denom))
            feature_history = feature_history[-model_history(model):]
            if step % replan_interval == 0:
                chunks.append((step, predict_chunk(model, prepare_model_input(model, feature_history))))
                chunks = [(start, chunk) for start, chunk in chunks if step - start < horizon]

            raw_action = select_action(args, chunks, step)
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
    finally:
        renderer.close()

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
    print(f"feature_dim: {model['metadata']['observation_dim']}", flush=True)
    print(f"history: {model_history(model)}", flush=True)
    print(f"horizon: {model['metadata']['horizon']}", flush=True)
    print(f"trainable_params: {model['metadata'].get('trainable_params')}", flush=True)
    print(f"task: {task}", flush=True)
    print(f"complexity: {complexity}", flush=True)

    summaries = []
    if args.viewer:
        env, obs = configure_env(args, model, args.seed, task, complexity)
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
        env, obs = configure_env(args, model, seed, task, complexity)
        summary = rollout_with_env(args, model, env, obs, seed, task, complexity)
        summaries.append(summary)
        print("episode_summary:", summary, flush=True)

    successes = sum(int(item["success"]) for item in summaries)
    print(f"success_rate: {successes}/{len(summaries)} = {successes / max(1, len(summaries)):.3f}", flush=True)


if __name__ == "__main__":
    main()
