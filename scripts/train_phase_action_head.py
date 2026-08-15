from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_chunk_bc import output_weights, weighted_backward, weighted_mse  # noqa: E402
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
    build_features,
    observation_layout,
    resolve_run_dirs,
    split_by_source_episode,
)
from widowx_env.demo_dataset import load_demo_dataset, read_metadata  # noqa: E402
from widowx_env.tabletop_env import OBJECTS  # noqa: E402


PHASE_NAMES = ("approach", "grasp", "lift", "transfer", "place_release")
PHASE_THRESHOLDS = np.asarray([0.17, 0.26, 0.41, 0.66], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a phase-conditioned object-language action-head baseline.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--run-dirs", type=Path, nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "phase_action_head")
    parser.add_argument("--model-prefix", default="phase_conditioned_action_head_lite")
    parser.add_argument("--hidden-sizes", default="192,192")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--gripper-loss-weight", type=float, default=4.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--all-attempts", action="store_true")
    return parser.parse_args()


def phase_labels_for_length(length: int) -> np.ndarray:
    progress = np.linspace(0.0, 1.0, length, dtype=np.float32)
    return np.digitize(progress, PHASE_THRESHOLDS).astype(np.int16)


def build_source_features(dataset, run_dir: Path, layout: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    metadata_by_episode = {int(item["episode_index"]): item for item in read_metadata(run_dir)}
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    offset = 0
    for segment in dataset.segments:
        length = int(segment["steps"])
        segment_slice = slice(offset, offset + length)
        offset += length
        metadata = metadata_by_episode[int(segment["episode_index"])]
        features.append(build_features(dataset.observations[segment_slice], metadata, layout))
        labels.append(phase_labels_for_length(length))
    return np.concatenate(features, axis=0).astype(np.float32), np.concatenate(labels, axis=0)


def build_training_arrays(
    args: argparse.Namespace,
    run_dirs: list[Path],
    layout: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    feature_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    phase_parts: list[np.ndarray] = []
    episode_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    sources: list[dict] = []
    episode_offset = 0

    for source_index, run_dir in enumerate(run_dirs):
        dataset = load_demo_dataset(
            run_dir,
            successful_only=not args.include_failures,
            successful_attempt_only=not args.all_attempts,
        )
        features, phase_labels = build_source_features(dataset, run_dir, layout)
        remapped_episodes = dataset.episode_indices.astype(np.int32) + episode_offset

        feature_parts.append(features)
        action_parts.append(dataset.actions.astype(np.float32))
        phase_parts.append(phase_labels.astype(np.int16))
        episode_parts.append(remapped_episodes)
        source_parts.append(np.full(len(dataset.actions), source_index, dtype=np.int16))
        sources.append(
            {
                "run_dir": str(run_dir),
                "samples": int(len(dataset.actions)),
                "episodes": int(len(np.unique(dataset.episode_indices))),
                "segments": int(len(dataset.segments)),
            }
        )
        episode_offset += int(dataset.episode_indices.max()) + 1

    return (
        np.concatenate(feature_parts, axis=0).astype(np.float32),
        np.concatenate(action_parts, axis=0).astype(np.float32),
        np.concatenate(phase_parts, axis=0).astype(np.int16),
        np.concatenate(episode_parts, axis=0).astype(np.int32),
        np.concatenate(source_parts, axis=0).astype(np.int16),
        sources,
    )


def predict_actions(layers: list[dict[str, np.ndarray]], x: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    return forward(layers, x).astype(np.float32) * y_std + y_mean


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    run_dirs = resolve_run_dirs(args)
    layout = observation_layout()
    features, actions, phase_labels, episode_indices, source_ids, sources = build_training_arrays(args, run_dirs, layout)

    if len(run_dirs) > 1:
        train_mask, val_mask = split_by_source_episode(source_ids, episode_indices, args.val_fraction)
    else:
        split_dataset = SimpleNamespace(actions=actions, episode_indices=episode_indices)
        train_mask, val_mask = split_by_episode(split_dataset, args.val_fraction)

    x_train_raw = features[train_mask].astype(np.float32)
    y_train_raw = actions[train_mask].astype(np.float32)
    p_train = phase_labels[train_mask]
    x_val_raw = features[val_mask].astype(np.float32)
    y_val_raw = actions[val_mask].astype(np.float32)
    p_val = phase_labels[val_mask]

    x_mean = x_train_raw.mean(axis=0)
    x_std = x_train_raw.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    y_mean = y_train_raw.mean(axis=0)
    y_std = y_train_raw.std(axis=0)
    y_std[y_std < 1e-6] = 1.0

    x_train = ((x_train_raw - x_mean) / x_std).astype(np.float32)
    y_train = ((y_train_raw - y_mean) / y_std).astype(np.float32)
    x_val = ((x_val_raw - x_mean) / x_std).astype(np.float32)
    y_val = ((y_val_raw - y_mean) / y_std).astype(np.float32)

    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    phase_models = [init_model(x_train.shape[1], y_train.shape[1], hidden_sizes, rng) for _ in PHASE_NAMES]
    phase_states = [make_adam_states(layers) for layers in phase_models]
    phase_steps = [0 for _ in PHASE_NAMES]
    loss_weights = output_weights(1, actions.shape[1], args.gripper_loss_weight)

    for phase_id, phase_name in enumerate(PHASE_NAMES):
        if not np.any(p_train == phase_id):
            raise ValueError(f"no training samples for phase {phase_id}: {phase_name}")

    for epoch in range(1, args.epochs + 1):
        train_losses = []
        val_losses = []
        for phase_id, phase_name in enumerate(PHASE_NAMES):
            train_indices = np.flatnonzero(p_train == phase_id)
            val_indices = np.flatnonzero(p_val == phase_id)
            order = rng.permutation(train_indices)
            phase_losses = []
            for start in range(0, len(order), args.batch_size):
                batch = order[start: start + args.batch_size]
                layers = phase_models[phase_id]
                prediction, activations, preacts = forward(layers, x_train[batch], cache=True)
                phase_losses.append(weighted_mse(prediction, y_train[batch], loss_weights))
                grads = weighted_backward(layers, activations, preacts, prediction, y_train[batch], loss_weights, args.weight_decay)
                clip_grads(grads, args.grad_clip)
                phase_steps[phase_id] += 1
                adam_update(layers, grads, phase_states[phase_id], phase_steps[phase_id], args.lr)
            train_losses.append(float(np.mean(phase_losses)))
            if len(val_indices):
                val_prediction = forward(phase_models[phase_id], x_val[val_indices]).astype(np.float32)
                val_losses.append(float(weighted_mse(val_prediction, y_val[val_indices], loss_weights)))
            print(
                f"epoch={epoch} phase={phase_name} train_mse_norm={train_losses[-1]:.8f} "
                f"val_mse_norm={(val_losses[-1] if len(val_indices) else float('nan')):.8f}",
                flush=True,
            )
        print(
            f"epoch_summary={epoch} train_mse_norm={float(np.mean(train_losses)):.8f} "
            f"val_mse_norm={float(np.mean(val_losses)):.8f}",
            flush=True,
        )

    train_pred = np.zeros_like(y_train_raw)
    val_pred = np.zeros_like(y_val_raw)
    phase_counts: dict[str, dict[str, int]] = {}
    for phase_id, phase_name in enumerate(PHASE_NAMES):
        train_indices = np.flatnonzero(p_train == phase_id)
        val_indices = np.flatnonzero(p_val == phase_id)
        train_pred[train_indices] = predict_actions(phase_models[phase_id], x_train[train_indices], y_mean, y_std)
        if len(val_indices):
            val_pred[val_indices] = predict_actions(phase_models[phase_id], x_val[val_indices], y_mean, y_std)
        phase_counts[phase_name] = {"train": int(len(train_indices)), "val": int(len(val_indices))}

    train_mse = mse(train_pred, y_train_raw)
    val_mse = mse(val_pred, y_val_raw)

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
    for phase_id, layers in enumerate(phase_models):
        for index, layer in enumerate(layers):
            save_items[f"phase{phase_id}_w{index}"] = layer["w"].astype(np.float32)
            save_items[f"phase{phase_id}_b{index}"] = layer["b"].astype(np.float32)

    trainable_params = sum(int(array.size) for key, array in save_items.items() if "_w" in key or "_b" in key)
    meta = {
        "method": "phase_conditioned_object_language_action_head_lite",
        "run_dir": str(run_dirs[0]),
        "run_dirs": [str(item) for item in run_dirs],
        "sources": sources,
        "samples": int(len(actions)),
        "feature_dim": int(features.shape[1]),
        "action_dim": int(actions.shape[1]),
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "hidden_sizes": hidden_sizes,
        "phase_names": list(PHASE_NAMES),
        "phase_thresholds": [float(item) for item in PHASE_THRESHOLDS],
        "phase_counts": phase_counts,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "gripper_loss_weight": float(args.gripper_loss_weight),
        "train_mse": train_mse,
        "val_mse": val_mse,
        "trainable_params": trainable_params,
        "layout": layout,
        "objects": list(OBJECTS),
        "target_geoms": list(TARGET_GEOMS),
        "successful_only": not args.include_failures,
        "successful_attempt_only": not args.all_attempts,
    }
    save_items["metadata"] = json.dumps(meta, ensure_ascii=False)
    np.savez_compressed(model_path, **save_items)

    print(f"run_dirs: {[str(item) for item in run_dirs]}", flush=True)
    print(f"model_path: {model_path}", flush=True)
    print(f"samples: {meta['samples']}", flush=True)
    print(f"feature_dim: {meta['feature_dim']}", flush=True)
    print(f"action_dim: {meta['action_dim']}", flush=True)
    print(f"trainable_params: {meta['trainable_params']}", flush=True)
    print(f"phase_counts: {phase_counts}", flush=True)
    print(f"train_mse: {train_mse:.8f}", flush=True)
    print(f"val_mse: {val_mse:.8f}", flush=True)


if __name__ == "__main__":
    main()
