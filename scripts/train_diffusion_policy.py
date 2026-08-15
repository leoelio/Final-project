from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_chunk_bc import build_chunk_samples  # noqa: E402
from train_mlp_bc import (  # noqa: E402
    adam_update,
    backward,
    clip_grads,
    forward,
    init_model,
    make_adam_states,
    mse,
    parse_hidden_sizes,
    split_by_episode,
)
from widowx_env.demo_dataset import latest_run_dir, load_demo_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NumPy Diffusion Policy-lite action chunk baseline.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "diffusion_policy")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=16)
    parser.add_argument("--hidden-sizes", default="256,256")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--all-attempts", action="store_true")
    return parser.parse_args()


def diffusion_schedule(steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    betas = np.linspace(1e-4, 0.02, int(steps), dtype=np.float32)
    alphas = (1.0 - betas).astype(np.float32)
    alpha_bars = np.cumprod(alphas).astype(np.float32)
    return betas, alphas, alpha_bars


def timestep_features(timesteps: np.ndarray, diffusion_steps: int) -> np.ndarray:
    t = timesteps.astype(np.float32) / max(1.0, float(diffusion_steps - 1))
    return np.stack([t, np.sin(np.pi * t), np.cos(np.pi * t)], axis=1).astype(np.float32)


def diffusion_input(x: np.ndarray, noisy_chunk: np.ndarray, timesteps: np.ndarray, diffusion_steps: int) -> np.ndarray:
    return np.concatenate([x, noisy_chunk, timestep_features(timesteps, diffusion_steps)], axis=1).astype(np.float32)


def build_split_chunks(args: argparse.Namespace, dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_mask, val_mask = split_by_episode(dataset, args.val_fraction)
    train_episodes = set(int(item) for item in np.unique(dataset.episode_indices[train_mask]))
    val_episodes = set(int(item) for item in np.unique(dataset.episode_indices[val_mask]))
    x_train, y_train = build_chunk_samples(
        dataset.observations,
        dataset.actions,
        dataset.episode_indices,
        dataset.segments,
        train_episodes,
        int(args.horizon),
        int(args.sample_stride),
    )
    x_val, y_val = build_chunk_samples(
        dataset.observations,
        dataset.actions,
        dataset.episode_indices,
        dataset.segments,
        val_episodes,
        int(args.horizon),
        int(args.sample_stride),
    )
    return x_train, y_train, x_val, y_val


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    run_dir = args.run_dir or latest_run_dir()
    dataset = load_demo_dataset(
        run_dir,
        successful_only=not args.include_failures,
        successful_attempt_only=not args.all_attempts,
    )
    x_train, y_train, x_val, y_val = build_split_chunks(args, dataset)

    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_std[y_std < 1e-6] = 1.0

    x_train_norm = ((x_train - x_mean) / x_std).astype(np.float32)
    y_train_norm = ((y_train - y_mean) / y_std).astype(np.float32)
    x_val_norm = ((x_val - x_mean) / x_std).astype(np.float32)
    y_val_norm = ((y_val - y_mean) / y_std).astype(np.float32)

    betas, alphas, alpha_bars = diffusion_schedule(args.diffusion_steps)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    input_dim = x_train_norm.shape[1] + y_train_norm.shape[1] + 3
    layers = init_model(input_dim, y_train_norm.shape[1], hidden_sizes, rng)
    states = make_adam_states(layers)

    step = 0
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(x_train_norm))
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = order[start: start + args.batch_size]
            timesteps = rng.integers(0, args.diffusion_steps, size=len(batch), dtype=np.int32)
            noise = rng.normal(size=y_train_norm[batch].shape).astype(np.float32)
            sqrt_ab = np.sqrt(alpha_bars[timesteps])[:, None]
            sqrt_one_minus_ab = np.sqrt(1.0 - alpha_bars[timesteps])[:, None]
            noisy = sqrt_ab * y_train_norm[batch] + sqrt_one_minus_ab * noise
            inputs = diffusion_input(x_train_norm[batch], noisy, timesteps, args.diffusion_steps)

            prediction, activations, preacts = forward(layers, inputs, cache=True)
            losses.append(mse(prediction, noise))
            grads = backward(layers, activations, preacts, prediction, noise, args.weight_decay)
            clip_grads(grads, args.grad_clip)
            step += 1
            adam_update(layers, grads, states, step, args.lr)

        val_count = min(4096, len(x_val_norm))
        val_indices = rng.choice(len(x_val_norm), size=val_count, replace=False)
        val_timesteps = rng.integers(0, args.diffusion_steps, size=val_count, dtype=np.int32)
        val_noise = rng.normal(size=y_val_norm[val_indices].shape).astype(np.float32)
        val_noisy = (
            np.sqrt(alpha_bars[val_timesteps])[:, None] * y_val_norm[val_indices]
            + np.sqrt(1.0 - alpha_bars[val_timesteps])[:, None] * val_noise
        )
        val_inputs = diffusion_input(x_val_norm[val_indices], val_noisy, val_timesteps, args.diffusion_steps)
        val_pred = forward(layers, val_inputs).astype(np.float32)
        print(
            f"epoch={epoch} train_noise_mse={float(np.mean(losses)):.8f} "
            f"val_noise_mse={mse(val_pred, val_noise):.8f}",
            flush=True,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / f"diffusion_policy_lite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    save_items = {
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "y_std": y_std.astype(np.float32),
        "action_min": dataset.actions.min(axis=0).astype(np.float32),
        "action_max": dataset.actions.max(axis=0).astype(np.float32),
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
    }
    for index, layer in enumerate(layers):
        save_items[f"w{index}"] = layer["w"].astype(np.float32)
        save_items[f"b{index}"] = layer["b"].astype(np.float32)

    meta = {
        "method": "diffusion_policy_lite",
        "run_dir": str(run_dir),
        "source_samples": int(len(dataset.actions)),
        "train_chunks": int(len(x_train)),
        "val_chunks": int(len(x_val)),
        "observation_dim": int(x_train.shape[1]),
        "action_dim": int(dataset.actions.shape[1]),
        "chunk_output_dim": int(y_train.shape[1]),
        "horizon": int(args.horizon),
        "sample_stride": int(args.sample_stride),
        "diffusion_steps": int(args.diffusion_steps),
        "hidden_sizes": hidden_sizes,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "successful_only": not args.include_failures,
        "successful_attempt_only": not args.all_attempts,
    }
    save_items["metadata"] = json.dumps(meta, ensure_ascii=False)
    np.savez_compressed(model_path, **save_items)

    print(f"run_dir: {run_dir}", flush=True)
    print(f"model_path: {model_path}", flush=True)
    print(f"train_chunks: {meta['train_chunks']}", flush=True)
    print(f"val_chunks: {meta['val_chunks']}", flush=True)
    print(f"observation_dim: {meta['observation_dim']}", flush=True)
    print(f"action_dim: {meta['action_dim']}", flush=True)
    print(f"horizon: {meta['horizon']}", flush=True)
    print(f"diffusion_steps: {meta['diffusion_steps']}", flush=True)


if __name__ == "__main__":
    main()
