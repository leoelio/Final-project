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

from torch_runtime import ensure_torch_path  # noqa: E402

ensure_torch_path()

import torch  # noqa: E402

from run_bc_policy import postprocess_action, print_step, unsafe_reason  # noqa: E402
from run_torch_act_policy import select_action  # noqa: E402
from train_torch_diffusion_policy import StateDiffusionPolicy  # noqa: E402
from widowx_env import TASKS, WidowXTabletopEnv  # noqa: E402
from widowx_env.demo_dataset import observation_from_env, read_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight PyTorch state Diffusion Policy action-chunk baseline.")
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
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--temporal-ensemble", action="store_true")
    parser.add_argument("--no-temporal-ensemble", dest="temporal_ensemble", action="store_false")
    parser.add_argument("--ensemble-decay", type=float, default=0.1)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.add_argument("--stop-on-unsafe", action="store_true")
    parser.add_argument("--no-stop-on-unsafe", dest="stop_on_unsafe", action="store_false")
    parser.add_argument("--log-every", type=int, default=500)
    parser.set_defaults(clip_actions=True)
    parser.set_defaults(stop_on_unsafe=True)
    parser.set_defaults(temporal_ensemble=True)
    parser.set_defaults(deterministic=True)
    parser.set_defaults(viewer=True)
    return parser.parse_args()


def latest_model() -> Path:
    root = ROOT / "outputs" / "torch_diffusion_policy"
    candidates = sorted(root.glob("torch_diffusion_policy_state_chunk_*.pt"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no torch diffusion policy models found under {root}")
    return candidates[-1]


def load_model(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    policy = StateDiffusionPolicy(
        observation_dim=int(metadata["observation_dim"]),
        action_dim=int(metadata["action_dim"]),
        history=int(metadata["history"]),
        horizon=int(metadata["horizon"]),
        diffusion_steps=int(metadata["diffusion_steps"]),
        d_model=int(metadata["d_model"]),
        nhead=int(metadata["nhead"]),
        encoder_layers=int(metadata["encoder_layers"]),
        decoder_layers=int(metadata["decoder_layers"]),
        dim_feedforward=int(metadata["dim_feedforward"]),
        dropout=float(metadata["dropout"]),
    )
    policy.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.eval()
    return {
        "policy": policy,
        "device": device,
        "x_mean": checkpoint["x_mean"].float().numpy(),
        "x_std": checkpoint["x_std"].float().numpy(),
        "y_mean": checkpoint["y_mean"].float().numpy(),
        "y_std": checkpoint["y_std"].float().numpy(),
        "action_min": checkpoint["action_min"].float().numpy(),
        "action_max": checkpoint["action_max"].float().numpy(),
        "betas": checkpoint["betas"].float().to(device),
        "alphas": checkpoint["alphas"].float().to(device),
        "alpha_bars": checkpoint["alpha_bars"].float().to(device),
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
    return ((raw_history - model["x_mean"][None, :]) / model["x_std"][None, :]).astype(np.float32)


@torch.no_grad()
def predict_chunk(args: argparse.Namespace, model: dict, model_input: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    device = model["device"]
    policy = model["policy"]
    horizon = int(model["metadata"]["horizon"])
    action_dim = int(model["metadata"]["action_dim"])
    sample = torch.from_numpy(rng.normal(size=(1, horizon, action_dim)).astype(np.float32)).to(device)
    observations = torch.from_numpy(model_input[None, :, :].astype(np.float32)).to(device)
    alpha_bars = model["alpha_bars"]
    betas = model["betas"]
    diffusion_steps = int(model["metadata"]["diffusion_steps"])
    if args.sample_steps is None or int(args.sample_steps) >= diffusion_steps:
        schedule = list(reversed(range(diffusion_steps)))
    else:
        count = max(2, int(args.sample_steps))
        schedule = sorted(set(int(round(value)) for value in np.linspace(diffusion_steps - 1, 0, count)), reverse=True)

    for index, timestep in enumerate(schedule):
        t = torch.full((1,), timestep, dtype=torch.long, device=device)
        eps = policy(observations, sample, t)
        sqrt_ab = torch.sqrt(alpha_bars[timestep])
        sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bars[timestep])
        x0 = (sample - sqrt_one_minus_ab * eps) / torch.clamp(sqrt_ab, min=1e-6)
        next_timestep = schedule[index + 1] if index + 1 < len(schedule) else -1
        if next_timestep < 0:
            sample = x0
        else:
            prev_ab = alpha_bars[next_timestep]
            sample = torch.sqrt(prev_ab) * x0 + torch.sqrt(1.0 - prev_ab) * eps
            if not args.deterministic:
                noise = torch.from_numpy(rng.normal(size=(1, horizon, action_dim)).astype(np.float32)).to(device)
                sample = sample + torch.sqrt(betas[timestep]) * noise

    chunk_norm = sample[0].detach().cpu().numpy().astype(np.float32)
    return chunk_norm * model["y_std"][None, :] + model["y_mean"][None, :]


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
    rng = np.random.default_rng(seed)

    for step in range(args.steps):
        observation = observation_from_env(env, target_position, phase=step / phase_denom)
        observation_history.append(observation)
        observation_history = observation_history[-model_history(model):]
        if step % replan_interval == 0:
            chunks.append((step, predict_chunk(args, model, prepare_model_input(model, observation_history), rng)))
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
    print(f"torch_version: {torch.__version__}", flush=True)
    print(f"device: {model['device']}", flush=True)
    print(f"horizon: {model['metadata']['horizon']}", flush=True)
    print(f"history: {model_history(model)}", flush=True)
    print(f"diffusion_steps: {model['metadata']['diffusion_steps']}", flush=True)
    print(f"sample_steps: {args.sample_steps or model['metadata']['diffusion_steps']}", flush=True)
    print(f"trainable_params: {model['metadata'].get('trainable_params')}", flush=True)
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
