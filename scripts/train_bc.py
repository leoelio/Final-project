from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from widowx_env.demo_dataset import latest_run_dir, load_demo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small NumPy behavior cloning baseline.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "bc")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--all-attempts", action="store_true")
    return parser.parse_args()


def split_by_episode(dataset, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    episodes = np.array(sorted(np.unique(dataset.episode_indices)))
    if len(episodes) <= 1:
        mask = np.ones(len(dataset.actions), dtype=bool)
        return mask, mask.copy()

    val_count = max(1, int(round(len(episodes) * val_fraction)))
    val_episodes = set(int(item) for item in episodes[-val_count:])
    val_mask = np.array([int(item) in val_episodes for item in dataset.episode_indices], dtype=bool)
    train_mask = ~val_mask
    return train_mask, val_mask


def fit_ridge(x: np.ndarray, y: np.ndarray, l2: float) -> dict[str, np.ndarray]:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    x_norm = (x - x_mean) / x_std
    x_aug = np.concatenate([x_norm, np.ones((len(x_norm), 1), dtype=np.float64)], axis=1)

    lhs = x_aug.T @ x_aug
    reg = np.eye(lhs.shape[0], dtype=np.float64) * l2
    rhs = x_aug.T @ y
    try:
        weights = np.linalg.solve(lhs + reg, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(lhs + reg, rhs, rcond=None)[0]
    return {"weights": weights.astype(np.float32), "x_mean": x_mean.astype(np.float32), "x_std": x_std.astype(np.float32)}


def predict(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    x_norm = (x - model["x_mean"]) / model["x_std"]
    x_aug = np.concatenate([x_norm, np.ones((len(x_norm), 1), dtype=np.float32)], axis=1)
    return x_aug @ model["weights"]


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction - target) ** 2))


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or latest_run_dir()
    dataset = load_demo_dataset(
        run_dir,
        successful_only=not args.include_failures,
        successful_attempt_only=not args.all_attempts,
    )
    train_mask, val_mask = split_by_episode(dataset, args.val_fraction)

    model = fit_ridge(dataset.observations[train_mask], dataset.actions[train_mask], args.l2)
    train_mse = mse(predict(model, dataset.observations[train_mask]), dataset.actions[train_mask])
    val_mse = mse(predict(model, dataset.observations[val_mask]), dataset.actions[val_mask])

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / f"bc_linear_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    meta = {
        "run_dir": str(run_dir),
        "samples": int(len(dataset.actions)),
        "observation_dim": int(dataset.observations.shape[1]),
        "action_dim": int(dataset.actions.shape[1]),
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "train_mse": train_mse,
        "val_mse": val_mse,
        "l2": args.l2,
        "successful_only": not args.include_failures,
        "successful_attempt_only": not args.all_attempts,
    }
    np.savez_compressed(
        model_path,
        **model,
        action_min=dataset.actions.min(axis=0).astype(np.float32),
        action_max=dataset.actions.max(axis=0).astype(np.float32),
        metadata=json.dumps(meta, ensure_ascii=False),
    )

    print(f"run_dir: {run_dir}", flush=True)
    print(f"model_path: {model_path}", flush=True)
    print(f"samples: {meta['samples']}", flush=True)
    print(f"observation_dim: {meta['observation_dim']}", flush=True)
    print(f"action_dim: {meta['action_dim']}", flush=True)
    print(f"train_samples: {meta['train_samples']}", flush=True)
    print(f"val_samples: {meta['val_samples']}", flush=True)
    print(f"train_mse: {train_mse:.8f}", flush=True)
    print(f"val_mse: {val_mse:.8f}", flush=True)


if __name__ == "__main__":
    main()
