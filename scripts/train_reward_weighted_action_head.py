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

from train_chunk_bc import output_weights  # noqa: E402
from train_mlp_bc import (  # noqa: E402
    adam_update,
    clip_grads,
    forward,
    init_model,
    make_adam_states,
    mse,
    parse_hidden_sizes,
    split_by_episode,
)
from train_object_action_head import (  # noqa: E402
    TARGET_GEOMS,
    build_dataset_features,
    observation_layout,
)
from widowx_env.demo_dataset import latest_run_dir, load_demo_dataset, read_metadata  # noqa: E402
from widowx_env.tabletop_env import OBJECTS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a reward-weighted object-language action-head BC baseline.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "reward_weighted_action_head")
    parser.add_argument("--model-prefix", default="reward_weighted_action_head_lite")
    parser.add_argument("--hidden-sizes", default="256,256")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--gripper-loss-weight", type=float, default=4.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--final-attempt-only", action="store_true")
    parser.add_argument("--failed-attempt-weight", type=float, default=0.20)
    parser.add_argument("--final-attempt-weight", type=float, default=1.0)
    parser.add_argument("--distance-bonus", type=float, default=1.0)
    parser.add_argument("--lift-bonus", type=float, default=0.5)
    parser.add_argument("--phase-bonus", type=float, default=0.25)
    parser.add_argument("--distance-sigma", type=float, default=0.10)
    parser.add_argument("--lift-height", type=float, default=0.08)
    parser.add_argument("--min-sample-weight", type=float, default=0.10)
    parser.add_argument("--max-sample-weight", type=float, default=4.0)
    return parser.parse_args()


def metadata_by_episode(run_dir: Path) -> dict[int, dict]:
    return {int(item["episode_index"]): item for item in read_metadata(run_dir)}


def reward_weights(dataset, run_dir: Path, layout: dict[str, int], args: argparse.Namespace) -> np.ndarray:
    metadata_map = metadata_by_episode(run_dir)
    object_start = int(layout["object_start"])
    target_start = int(layout["target_start"])
    weights = np.zeros(len(dataset.actions), dtype=np.float32)
    offset = 0

    for segment in dataset.segments:
        length = int(segment["steps"])
        segment_slice = slice(offset, offset + length)
        offset += length
        metadata = metadata_map[int(segment["episode_index"])]
        observations = dataset.observations[segment_slice]
        objects = observations[:, object_start:target_start].reshape(length, len(OBJECTS), 3)
        target_position = observations[:, target_start:target_start + 3]
        target_object_index = OBJECTS.index(str(metadata["target_object"]))
        object_position = objects[:, target_object_index, :]

        distance = np.linalg.norm(object_position[:, :2] - target_position[:, :2], axis=1)
        distance_score = np.exp(-distance / max(1e-6, float(args.distance_sigma)))
        lift_score = np.clip((object_position[:, 2] - 0.026) / max(1e-6, float(args.lift_height)), 0.0, 1.0)
        phase = observations[:, -3]

        final_attempt = int(segment["attempt_id"]) == int(metadata["attempts"])
        attempt_weight = float(args.final_attempt_weight if final_attempt else args.failed_attempt_weight)
        shaped = (
            attempt_weight
            * (
                1.0
                + float(args.distance_bonus) * distance_score
                + float(args.lift_bonus) * lift_score
                + float(args.phase_bonus) * phase
            )
        )
        weights[segment_slice] = shaped.astype(np.float32)

    weights = np.clip(weights, float(args.min_sample_weight), float(args.max_sample_weight))
    mean = float(weights.mean())
    if mean <= 0.0:
        raise ValueError("reward weights have non-positive mean")
    return (weights / mean).astype(np.float32)


def sample_weighted_mse(prediction: np.ndarray, target: np.ndarray, action_weights: np.ndarray, sample_weights: np.ndarray) -> float:
    loss = ((prediction - target) ** 2) * action_weights[None, :] * sample_weights[:, None]
    return float(np.mean(loss))


def sample_weighted_backward(
    layers: list[dict[str, np.ndarray]],
    activations: list[np.ndarray],
    preacts: list[np.ndarray],
    prediction: np.ndarray,
    target: np.ndarray,
    action_weights: np.ndarray,
    sample_weights: np.ndarray,
    weight_decay: float,
) -> list[dict[str, np.ndarray]]:
    grad = (2.0 / prediction.size) * (prediction - target) * action_weights[None, :] * sample_weights[:, None]
    grads: list[dict[str, np.ndarray]] = []
    for index in reversed(range(len(layers))):
        grad_w = activations[index].T @ grad + weight_decay * layers[index]["w"]
        grad_b = grad.sum(axis=0)
        grads.append({"w": grad_w.astype(np.float32), "b": grad_b.astype(np.float32)})
        if index > 0:
            grad = (grad @ layers[index]["w"].T) * (preacts[index - 1] > 0.0)
    grads.reverse()
    return grads


def predict_actions(layers: list[dict[str, np.ndarray]], x: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    return forward(layers, x).astype(np.float32) * y_std + y_mean


def main() -> None:
    args = parse_args()
    started = time.time()
    rng = np.random.default_rng(args.seed)
    run_dir = args.run_dir or latest_run_dir()
    dataset = load_demo_dataset(
        run_dir,
        successful_only=not args.include_failures,
        successful_attempt_only=bool(args.final_attempt_only),
    )
    layout = observation_layout()
    features = build_dataset_features(dataset, run_dir, layout).astype(np.float32)
    actions = dataset.actions.astype(np.float32)
    sample_weights = reward_weights(dataset, run_dir, layout, args)
    train_mask, val_mask = split_by_episode(SimpleNamespace(actions=actions, episode_indices=dataset.episode_indices), args.val_fraction)

    x_train = features[train_mask].astype(np.float32)
    y_train = actions[train_mask].astype(np.float32)
    w_train = sample_weights[train_mask].astype(np.float32)
    x_val = features[val_mask].astype(np.float32)
    y_val = actions[val_mask].astype(np.float32)
    w_val = sample_weights[val_mask].astype(np.float32)

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

    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    layers = init_model(x_train_norm.shape[1], y_train_norm.shape[1], hidden_sizes, rng)
    states = make_adam_states(layers)
    action_weights = output_weights(1, actions.shape[1], args.gripper_loss_weight)

    step = 0
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(x_train_norm))
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = order[start: start + args.batch_size]
            prediction, activations, preacts = forward(layers, x_train_norm[batch], cache=True)
            losses.append(sample_weighted_mse(prediction, y_train_norm[batch], action_weights, w_train[batch]))
            grads = sample_weighted_backward(
                layers,
                activations,
                preacts,
                prediction,
                y_train_norm[batch],
                action_weights,
                w_train[batch],
                args.weight_decay,
            )
            clip_grads(grads, args.grad_clip)
            step += 1
            adam_update(layers, grads, states, step, args.lr)

        val_prediction = forward(layers, x_val_norm).astype(np.float32)
        print(
            f"epoch={epoch} train_weighted_mse_norm={float(np.mean(losses)):.8f} "
            f"val_weighted_mse_norm={sample_weighted_mse(val_prediction, y_val_norm, action_weights, w_val):.8f}",
            flush=True,
        )

    train_pred = predict_actions(layers, x_train_norm, y_mean, y_std)
    val_pred = predict_actions(layers, x_val_norm, y_mean, y_std)
    train_mse = mse(train_pred, y_train)
    val_mse = mse(val_pred, y_val)

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / f"{args.model_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    save_items = {
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "y_std": y_std.astype(np.float32),
        "action_min": actions.min(axis=0).astype(np.float32),
        "action_max": actions.max(axis=0).astype(np.float32),
    }
    for index, layer in enumerate(layers):
        save_items[f"w{index}"] = layer["w"].astype(np.float32)
        save_items[f"b{index}"] = layer["b"].astype(np.float32)

    meta = {
        "method": "reward_weighted_object_language_action_head_lite",
        "run_dir": str(run_dir),
        "samples": int(len(actions)),
        "feature_dim": int(features.shape[1]),
        "action_dim": int(actions.shape[1]),
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "hidden_sizes": hidden_sizes,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "gripper_loss_weight": float(args.gripper_loss_weight),
        "train_mse": float(train_mse),
        "val_mse": float(val_mse),
        "sample_weight_min": float(sample_weights.min()),
        "sample_weight_max": float(sample_weights.max()),
        "sample_weight_mean": float(sample_weights.mean()),
        "failed_attempt_weight": float(args.failed_attempt_weight),
        "final_attempt_weight": float(args.final_attempt_weight),
        "distance_bonus": float(args.distance_bonus),
        "lift_bonus": float(args.lift_bonus),
        "phase_bonus": float(args.phase_bonus),
        "final_attempt_only": bool(args.final_attempt_only),
        "layout": layout,
        "objects": list(OBJECTS),
        "target_geoms": list(TARGET_GEOMS),
        "successful_only": not args.include_failures,
        "successful_attempt_only": bool(args.final_attempt_only),
        "train_time_seconds": float(time.time() - started),
        "note": "Reward-weighted BC/action-head proxy using attempt preference and dense shaping from successful demonstrations.",
    }
    save_items["metadata"] = json.dumps(meta, ensure_ascii=False)
    np.savez_compressed(model_path, **save_items)

    print(f"run_dir: {run_dir}", flush=True)
    print(f"model_path: {model_path}", flush=True)
    print(f"samples: {meta['samples']}", flush=True)
    print(f"feature_dim: {meta['feature_dim']}", flush=True)
    print(f"action_dim: {meta['action_dim']}", flush=True)
    print(f"sample_weight_min: {meta['sample_weight_min']:.4f}", flush=True)
    print(f"sample_weight_max: {meta['sample_weight_max']:.4f}", flush=True)
    print(f"train_mse: {train_mse:.8f}", flush=True)
    print(f"val_mse: {val_mse:.8f}", flush=True)
    print(f"train_time_seconds: {meta['train_time_seconds']:.2f}", flush=True)


if __name__ == "__main__":
    main()
