from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_chunk_bc import output_weights, weighted_mse  # noqa: E402
from train_mlp_bc import mse, split_by_episode  # noqa: E402
from train_object_action_head import (  # noqa: E402
    TARGET_GEOMS,
    build_training_arrays,
    observation_layout,
    resolve_run_dirs,
    split_by_source_episode,
)
from widowx_env.tabletop_env import OBJECTS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train adapter/LoRA-style residual modules on a frozen object-language action head.")
    parser.add_argument("--base-model", type=Path, default=ROOT / "outputs" / "object_action_head" / "object_action_head_lite_20260720_044703.npz")
    parser.add_argument("--mode", choices=("adapter", "lora"), default="adapter")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--run-dirs", type=Path, nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "peft_action_head")
    parser.add_argument("--model-prefix", default=None)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--gripper-loss-weight", type=float, default=4.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--all-attempts", action="store_true")
    return parser.parse_args()


def load_base_model(path: Path) -> dict:
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


def frozen_forward(layers: list[dict[str, np.ndarray]], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = x
    hidden = x
    for index, layer in enumerate(layers):
        z = value @ layer["w"] + layer["b"]
        value = np.maximum(z, 0.0) if index < len(layers) - 1 else z
        if index == len(layers) - 2:
            hidden = value
    return value.astype(np.float32), hidden.astype(np.float32)


def init_params(mode: str, hidden_dim: int, action_dim: int, rank: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    rank = max(1, int(rank))
    if mode == "adapter":
        return {
            "adapter_down": rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, rank)).astype(np.float32),
            "adapter_down_b": np.zeros(rank, dtype=np.float32),
            "adapter_up": np.zeros((rank, action_dim), dtype=np.float32),
            "adapter_up_b": np.zeros(action_dim, dtype=np.float32),
        }
    return {
        "lora_a": rng.normal(0.0, 0.01, size=(hidden_dim, rank)).astype(np.float32),
        "lora_b": np.zeros((rank, action_dim), dtype=np.float32),
    }


def residual_forward(mode: str, params: dict[str, np.ndarray], hidden: np.ndarray, scale: float) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    if mode == "adapter":
        z = hidden @ params["adapter_down"] + params["adapter_down_b"]
        bottleneck = np.maximum(z, 0.0)
        residual = (bottleneck @ params["adapter_up"] + params["adapter_up_b"]) * scale
        return residual.astype(np.float32), (z, bottleneck)
    mid = hidden @ params["lora_a"]
    residual = (mid @ params["lora_b"]) * scale
    return residual.astype(np.float32), (mid,)


def residual_backward(
    mode: str,
    params: dict[str, np.ndarray],
    hidden: np.ndarray,
    cache: tuple[np.ndarray, ...],
    grad_prediction: np.ndarray,
    scale: float,
    weight_decay: float,
) -> dict[str, np.ndarray]:
    grad = grad_prediction * scale
    if mode == "adapter":
        z, bottleneck = cache
        grad_up = bottleneck.T @ grad + weight_decay * params["adapter_up"]
        grad_up_b = grad.sum(axis=0)
        grad_bottleneck = grad @ params["adapter_up"].T
        grad_z = grad_bottleneck * (z > 0.0)
        grad_down = hidden.T @ grad_z + weight_decay * params["adapter_down"]
        grad_down_b = grad_z.sum(axis=0)
        return {
            "adapter_down": grad_down.astype(np.float32),
            "adapter_down_b": grad_down_b.astype(np.float32),
            "adapter_up": grad_up.astype(np.float32),
            "adapter_up_b": grad_up_b.astype(np.float32),
        }
    (mid,) = cache
    grad_b = mid.T @ grad + weight_decay * params["lora_b"]
    grad_mid = grad @ params["lora_b"].T
    grad_a = hidden.T @ grad_mid + weight_decay * params["lora_a"]
    return {"lora_a": grad_a.astype(np.float32), "lora_b": grad_b.astype(np.float32)}


def make_states(params: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    return {name: {"m": np.zeros_like(value), "v": np.zeros_like(value)} for name, value in params.items()}


def clip_param_grads(grads: dict[str, np.ndarray], max_norm: float) -> None:
    if max_norm <= 0:
        return
    total = sum(float(np.sum(value * value)) for value in grads.values())
    norm = float(np.sqrt(total))
    if norm <= max_norm:
        return
    scale = max_norm / (norm + 1e-8)
    for value in grads.values():
        value *= scale


def adam_update_params(params: dict[str, np.ndarray], grads: dict[str, np.ndarray], states: dict[str, dict[str, np.ndarray]], step: int, lr: float) -> None:
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    for name, value in params.items():
        state = states[name]
        grad = grads[name]
        state["m"] = beta1 * state["m"] + (1.0 - beta1) * grad
        state["v"] = beta2 * state["v"] + (1.0 - beta2) * (grad * grad)
        m_hat = state["m"] / (1.0 - beta1**step)
        v_hat = state["v"] / (1.0 - beta2**step)
        value -= lr * m_hat / (np.sqrt(v_hat) + eps)


def prediction_and_hidden(base: dict, x_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return frozen_forward(base["layers"], x_norm)


def peft_predict_norm(base: dict, params: dict[str, np.ndarray], mode: str, x_norm: np.ndarray, scale: float) -> np.ndarray:
    base_prediction, hidden = prediction_and_hidden(base, x_norm)
    residual, _ = residual_forward(mode, params, hidden, scale)
    return (base_prediction + residual).astype(np.float32)


def trainable_count(params: dict[str, np.ndarray]) -> int:
    return int(sum(value.size for value in params.values()))


def frozen_count(base: dict) -> int:
    return int(sum(layer["w"].size + layer["b"].size for layer in base["layers"]))


def split_masks(args: argparse.Namespace, source_ids: np.ndarray, episode_indices: np.ndarray, actions: np.ndarray, run_dirs: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    if len(run_dirs) > 1:
        return split_by_source_episode(source_ids, episode_indices, args.val_fraction)
    split_dataset = SimpleNamespace(actions=actions, episode_indices=episode_indices)
    return split_by_episode(split_dataset, args.val_fraction)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    started = time.time()
    base = load_base_model(args.base_model)
    run_dirs = resolve_run_dirs(args)
    layout = observation_layout()
    features, actions, episode_indices, source_ids, sources = build_training_arrays(args, run_dirs, layout)
    train_mask, val_mask = split_masks(args, source_ids, episode_indices, actions, run_dirs)

    x_norm = ((features - base["x_mean"]) / base["x_std"]).astype(np.float32)
    y_norm = ((actions - base["y_mean"]) / base["y_std"]).astype(np.float32)
    x_train = x_norm[train_mask]
    y_train = y_norm[train_mask]
    x_val = x_norm[val_mask]
    y_val = y_norm[val_mask]

    _, sample_hidden = prediction_and_hidden(base, x_train[:1])
    params = init_params(args.mode, int(sample_hidden.shape[1]), int(actions.shape[1]), args.rank, rng)
    states = make_states(params)
    loss_weights = output_weights(1, actions.shape[1], args.gripper_loss_weight)
    scale = float(args.adapter_scale if args.mode == "adapter" else args.lora_alpha / max(1, args.rank))

    step = 0
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(x_train))
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            base_prediction, hidden = prediction_and_hidden(base, x_train[batch])
            residual, cache = residual_forward(args.mode, params, hidden, scale)
            prediction = base_prediction + residual
            losses.append(weighted_mse(prediction, y_train[batch], loss_weights))
            grad_prediction = (2.0 / prediction.size) * (prediction - y_train[batch]) * loss_weights[None, :]
            grads = residual_backward(args.mode, params, hidden, cache, grad_prediction, scale, args.weight_decay)
            clip_param_grads(grads, args.grad_clip)
            step += 1
            adam_update_params(params, grads, states, step, args.lr)
        val_prediction = peft_predict_norm(base, params, args.mode, x_val, scale)
        print(
            f"epoch={epoch} train_mse_norm={float(np.mean(losses)):.8f} "
            f"val_mse_norm={weighted_mse(val_prediction, y_val, loss_weights):.8f}",
            flush=True,
        )

    train_pred_norm = peft_predict_norm(base, params, args.mode, x_train, scale)
    val_pred_norm = peft_predict_norm(base, params, args.mode, x_val, scale)
    train_pred = train_pred_norm * base["y_std"] + base["y_mean"]
    val_pred = val_pred_norm * base["y_std"] + base["y_mean"]
    train_mse = mse(train_pred, actions[train_mask])
    val_mse = mse(val_pred, actions[val_mask])
    elapsed = time.time() - started

    args.output.mkdir(parents=True, exist_ok=True)
    default_prefix = f"{args.mode}_action_head_lite"
    model_path = args.output / f"{args.model_prefix or default_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    save_items = {
        "x_mean": base["x_mean"].astype(np.float32),
        "x_std": base["x_std"].astype(np.float32),
        "y_mean": base["y_mean"].astype(np.float32),
        "y_std": base["y_std"].astype(np.float32),
        "action_min": base["action_min"].astype(np.float32),
        "action_max": base["action_max"].astype(np.float32),
    }
    for index, layer in enumerate(base["layers"]):
        save_items[f"base_w{index}"] = layer["w"].astype(np.float32)
        save_items[f"base_b{index}"] = layer["b"].astype(np.float32)
    for name, value in params.items():
        save_items[name] = value.astype(np.float32)

    meta = {
        "method": f"{args.mode}_object_language_action_head_lite",
        "mode": args.mode,
        "base_model": str(args.base_model),
        "run_dir": str(run_dirs[0]),
        "run_dirs": [str(item) for item in run_dirs],
        "sources": sources,
        "samples": int(len(actions)),
        "feature_dim": int(features.shape[1]),
        "action_dim": int(actions.shape[1]),
        "hidden_dim": int(sample_hidden.shape[1]),
        "rank": int(args.rank),
        "scale": float(scale),
        "trainable_params": trainable_count(params),
        "frozen_params": frozen_count(base),
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "gripper_loss_weight": float(args.gripper_loss_weight),
        "train_mse": train_mse,
        "val_mse": val_mse,
        "train_time_seconds": float(elapsed),
        "peak_vram_mb": 0.0,
        "layout": layout,
        "objects": list(OBJECTS),
        "target_geoms": list(TARGET_GEOMS),
        "successful_only": not args.include_failures,
        "successful_attempt_only": not args.all_attempts,
        "note": "Frozen object-language action-head backbone plus local PEFT residual module; not a pretrained VLM/VLA LoRA/Adapter.",
    }
    save_items["metadata"] = json.dumps(meta, ensure_ascii=False)
    np.savez_compressed(model_path, **save_items)

    print(f"run_dirs: {[str(item) for item in run_dirs]}", flush=True)
    print(f"base_model: {args.base_model}", flush=True)
    print(f"model_path: {model_path}", flush=True)
    print(f"mode: {args.mode}", flush=True)
    print(f"samples: {meta['samples']}", flush=True)
    print(f"feature_dim: {meta['feature_dim']}", flush=True)
    print(f"trainable_params: {meta['trainable_params']}", flush=True)
    print(f"frozen_params: {meta['frozen_params']}", flush=True)
    print(f"train_mse: {train_mse:.8f}", flush=True)
    print(f"val_mse: {val_mse:.8f}", flush=True)
    print(f"train_time_seconds: {elapsed:.2f}", flush=True)


if __name__ == "__main__":
    main()
