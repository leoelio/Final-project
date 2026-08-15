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
    parser = argparse.ArgumentParser(description="Train a small NumPy MLP behavior cloning baseline.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "mlp_bc")
    parser.add_argument("--hidden-sizes", default="256,256")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
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


def parse_hidden_sizes(value: str) -> list[int]:
    sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not sizes:
        raise ValueError("--hidden-sizes must contain at least one layer size")
    return sizes


def init_model(input_dim: int, output_dim: int, hidden_sizes: list[int], rng: np.random.Generator) -> list[dict[str, np.ndarray]]:
    dims = [input_dim, *hidden_sizes, output_dim]
    layers = []
    for index, (fan_in, fan_out) in enumerate(zip(dims[:-1], dims[1:])):
        scale = np.sqrt(2.0 / fan_in) if index < len(dims) - 2 else np.sqrt(1.0 / fan_in)
        layers.append(
            {
                "w": rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(np.float32),
                "b": np.zeros(fan_out, dtype=np.float32),
            }
        )
    return layers


def forward(layers: list[dict[str, np.ndarray]], x: np.ndarray, cache: bool = False):
    activations = [x]
    preacts = []
    value = x
    for index, layer in enumerate(layers):
        z = value @ layer["w"] + layer["b"]
        preacts.append(z)
        value = np.maximum(z, 0.0) if index < len(layers) - 1 else z
        activations.append(value)
    if cache:
        return value, activations, preacts
    return value


def backward(
    layers: list[dict[str, np.ndarray]],
    activations: list[np.ndarray],
    preacts: list[np.ndarray],
    prediction: np.ndarray,
    target: np.ndarray,
    weight_decay: float,
) -> list[dict[str, np.ndarray]]:
    grad = (2.0 / prediction.size) * (prediction - target)
    grads: list[dict[str, np.ndarray]] = []
    for index in reversed(range(len(layers))):
        grad_w = activations[index].T @ grad + weight_decay * layers[index]["w"]
        grad_b = grad.sum(axis=0)
        grads.append({"w": grad_w.astype(np.float32), "b": grad_b.astype(np.float32)})
        if index > 0:
            grad = (grad @ layers[index]["w"].T) * (preacts[index - 1] > 0.0)
    grads.reverse()
    return grads


def clip_grads(grads: list[dict[str, np.ndarray]], max_norm: float) -> None:
    if max_norm <= 0:
        return
    total = 0.0
    for grad in grads:
        total += float(np.sum(grad["w"] * grad["w"]) + np.sum(grad["b"] * grad["b"]))
    norm = float(np.sqrt(total))
    if norm <= max_norm:
        return
    scale = max_norm / (norm + 1e-8)
    for grad in grads:
        grad["w"] *= scale
        grad["b"] *= scale


def adam_update(
    layers: list[dict[str, np.ndarray]],
    grads: list[dict[str, np.ndarray]],
    states: list[dict[str, np.ndarray]],
    step: int,
    lr: float,
) -> None:
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    for layer, grad, state in zip(layers, grads, states):
        for key in ("w", "b"):
            state[f"m_{key}"] = beta1 * state[f"m_{key}"] + (1.0 - beta1) * grad[key]
            state[f"v_{key}"] = beta2 * state[f"v_{key}"] + (1.0 - beta2) * (grad[key] * grad[key])
            m_hat = state[f"m_{key}"] / (1.0 - beta1**step)
            v_hat = state[f"v_{key}"] / (1.0 - beta2**step)
            layer[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)


def make_adam_states(layers: list[dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
    return [
        {
            "m_w": np.zeros_like(layer["w"]),
            "v_w": np.zeros_like(layer["w"]),
            "m_b": np.zeros_like(layer["b"]),
            "v_b": np.zeros_like(layer["b"]),
        }
        for layer in layers
    ]


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction - target) ** 2))


def predict_actions(layers: list[dict[str, np.ndarray]], x: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    return forward(layers, x).astype(np.float32) * y_std + y_mean


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    run_dir = args.run_dir or latest_run_dir()
    dataset = load_demo_dataset(
        run_dir,
        successful_only=not args.include_failures,
        successful_attempt_only=not args.all_attempts,
    )
    train_mask, val_mask = split_by_episode(dataset, args.val_fraction)

    x_train = dataset.observations[train_mask].astype(np.float32)
    y_train = dataset.actions[train_mask].astype(np.float32)
    x_val = dataset.observations[val_mask].astype(np.float32)
    y_val = dataset.actions[val_mask].astype(np.float32)

    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_std[y_std < 1e-6] = 1.0

    x_train = ((x_train - x_mean) / x_std).astype(np.float32)
    y_train_norm = ((y_train - y_mean) / y_std).astype(np.float32)
    x_val = ((x_val - x_mean) / x_std).astype(np.float32)
    y_val_norm = ((y_val - y_mean) / y_std).astype(np.float32)

    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    layers = init_model(x_train.shape[1], y_train.shape[1], hidden_sizes, rng)
    states = make_adam_states(layers)

    step = 0
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(x_train))
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = order[start: start + args.batch_size]
            prediction, activations, preacts = forward(layers, x_train[batch], cache=True)
            losses.append(mse(prediction, y_train_norm[batch]))
            grads = backward(layers, activations, preacts, prediction, y_train_norm[batch], args.weight_decay)
            clip_grads(grads, args.grad_clip)
            step += 1
            adam_update(layers, grads, states, step, args.lr)

        val_prediction = forward(layers, x_val).astype(np.float32)
        val_mse_norm = mse(val_prediction, y_val_norm)
        print(
            f"epoch={epoch} train_mse_norm={float(np.mean(losses)):.8f} "
            f"val_mse_norm={val_mse_norm:.8f}",
            flush=True,
        )

    train_pred = predict_actions(layers, x_train, y_mean, y_std)
    val_pred = predict_actions(layers, x_val, y_mean, y_std)
    train_mse = mse(train_pred, y_train)
    val_mse = mse(val_pred, y_val)

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / f"mlp_bc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    save_items = {
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "y_std": y_std.astype(np.float32),
        "action_min": dataset.actions.min(axis=0).astype(np.float32),
        "action_max": dataset.actions.max(axis=0).astype(np.float32),
    }
    for index, layer in enumerate(layers):
        save_items[f"w{index}"] = layer["w"].astype(np.float32)
        save_items[f"b{index}"] = layer["b"].astype(np.float32)

    meta = {
        "run_dir": str(run_dir),
        "samples": int(len(dataset.actions)),
        "observation_dim": int(dataset.observations.shape[1]),
        "action_dim": int(dataset.actions.shape[1]),
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "hidden_sizes": hidden_sizes,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "train_mse": train_mse,
        "val_mse": val_mse,
        "successful_only": not args.include_failures,
        "successful_attempt_only": not args.all_attempts,
    }
    save_items["metadata"] = json.dumps(meta, ensure_ascii=False)
    np.savez_compressed(model_path, **save_items)

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
