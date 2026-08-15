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
from train_chunk_bc import output_weights, weighted_backward, weighted_mse  # noqa: E402
from widowx_env import WidowXTabletopEnv  # noqa: E402
from widowx_env.demo_dataset import latest_run_dir, load_demo_dataset, read_metadata  # noqa: E402
from widowx_env.tabletop_env import OBJECTS  # noqa: E402


TARGET_GEOMS = ("target_red_pad", "target_blue_pad", "target_bowl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an object-language feature action-head baseline.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--run-dirs", type=Path, nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "object_action_head")
    parser.add_argument("--model-prefix", default=None)
    parser.add_argument("--hidden-sizes", default="256,256")
    parser.add_argument("--epochs", type=int, default=20)
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


def observation_layout() -> dict[str, int]:
    env = WidowXTabletopEnv(seed=0)
    qpos_dim = int(env.data.qpos.size)
    qvel_dim = int(env.data.qvel.size)
    ctrl_dim = int(env.data.ctrl.size)
    tcp_start = qpos_dim + qvel_dim + ctrl_dim
    object_start = tcp_start + 3
    target_start = object_start + len(OBJECTS) * 3
    return {
        "qpos_dim": qpos_dim,
        "qvel_dim": qvel_dim,
        "ctrl_dim": ctrl_dim,
        "tcp_start": tcp_start,
        "object_start": object_start,
        "target_start": target_start,
    }


def one_hot(name: str | None, names: tuple[str, ...]) -> np.ndarray:
    value = np.zeros(len(names), dtype=np.float32)
    if name in names:
        value[names.index(str(name))] = 1.0
    return value


def build_features(observations: np.ndarray, metadata: dict, layout: dict[str, int]) -> np.ndarray:
    target_object = str(metadata["target_object"])
    target_geom = metadata.get("target_geom")
    active_objects = set(metadata.get("active_objects", []))
    target_object_index = OBJECTS.index(target_object)

    qpos_dim = layout["qpos_dim"]
    qvel_dim = layout["qvel_dim"]
    ctrl_dim = layout["ctrl_dim"]
    tcp_start = layout["tcp_start"]
    object_start = layout["object_start"]
    target_start = layout["target_start"]

    qpos = observations[:, :qpos_dim]
    qvel = observations[:, qpos_dim: qpos_dim + qvel_dim]
    ctrl = observations[:, qpos_dim + qvel_dim: qpos_dim + qvel_dim + ctrl_dim]
    tcp = observations[:, tcp_start: tcp_start + 3]
    objects = observations[:, object_start: target_start].reshape(len(observations), len(OBJECTS), 3)
    target_position = observations[:, target_start: target_start + 3]
    phase = observations[:, -3:]

    object_position = objects[:, target_object_index, :]
    object_to_tcp = object_position - tcp
    object_to_target = object_position - target_position
    tcp_to_target = tcp - target_position
    object_id = np.repeat(one_hot(target_object, OBJECTS)[None, :], len(observations), axis=0)
    target_id = np.repeat(one_hot(target_geom, TARGET_GEOMS)[None, :], len(observations), axis=0)
    active_mask = np.repeat(np.asarray([name in active_objects for name in OBJECTS], dtype=np.float32)[None, :], len(observations), axis=0)

    return np.concatenate(
        [
            qpos,
            qvel,
            ctrl,
            tcp,
            object_position,
            target_position,
            object_to_tcp,
            object_to_target,
            tcp_to_target,
            object_id,
            target_id,
            active_mask,
            phase,
        ],
        axis=1,
    ).astype(np.float32)


def build_dataset_features(dataset, run_dir: Path, layout: dict[str, int]) -> np.ndarray:
    metadata_by_episode = {int(item["episode_index"]): item for item in read_metadata(run_dir)}
    features: list[np.ndarray] = []
    offset = 0
    for segment in dataset.segments:
        length = int(segment["steps"])
        segment_slice = slice(offset, offset + length)
        offset += length
        metadata = metadata_by_episode[int(segment["episode_index"])]
        features.append(build_features(dataset.observations[segment_slice], metadata, layout))
    return np.concatenate(features, axis=0)


def resolve_run_dirs(args: argparse.Namespace) -> list[Path]:
    if args.run_dirs:
        return [Path(item) for item in args.run_dirs]
    return [args.run_dir or latest_run_dir()]


def build_training_arrays(args: argparse.Namespace, run_dirs: list[Path], layout: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    feature_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
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
        features = build_dataset_features(dataset, run_dir, layout)
        remapped_episodes = dataset.episode_indices.astype(np.int32) + episode_offset

        feature_parts.append(features)
        action_parts.append(dataset.actions.astype(np.float32))
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
        np.concatenate(episode_parts, axis=0).astype(np.int32),
        np.concatenate(source_parts, axis=0).astype(np.int16),
        sources,
    )


def split_by_source_episode(source_ids: np.ndarray, episode_indices: np.ndarray, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    val_mask = np.zeros(len(episode_indices), dtype=bool)
    for source_id in sorted(int(item) for item in np.unique(source_ids)):
        source_mask = source_ids == source_id
        episodes = np.array(sorted(np.unique(episode_indices[source_mask])))
        if len(episodes) <= 1:
            val_mask |= source_mask
            continue
        val_count = max(1, int(round(len(episodes) * val_fraction)))
        val_episodes = set(int(item) for item in episodes[-val_count:])
        val_mask |= source_mask & np.array([int(item) in val_episodes for item in episode_indices], dtype=bool)
    train_mask = ~val_mask
    return train_mask, val_mask


def predict_actions(layers: list[dict[str, np.ndarray]], x: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    return forward(layers, x).astype(np.float32) * y_std + y_mean


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    run_dirs = resolve_run_dirs(args)
    layout = observation_layout()
    features, actions, episode_indices, source_ids, sources = build_training_arrays(args, run_dirs, layout)
    if len(run_dirs) > 1:
        train_mask, val_mask = split_by_source_episode(source_ids, episode_indices, args.val_fraction)
    else:
        split_dataset = SimpleNamespace(actions=actions, episode_indices=episode_indices)
        train_mask, val_mask = split_by_episode(split_dataset, args.val_fraction)

    x_train = features[train_mask].astype(np.float32)
    y_train = actions[train_mask].astype(np.float32)
    x_val = features[val_mask].astype(np.float32)
    y_val = actions[val_mask].astype(np.float32)

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
    loss_weights = output_weights(1, actions.shape[1], args.gripper_loss_weight)

    step = 0
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(x_train_norm))
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = order[start: start + args.batch_size]
            prediction, activations, preacts = forward(layers, x_train_norm[batch], cache=True)
            losses.append(weighted_mse(prediction, y_train_norm[batch], loss_weights))
            grads = weighted_backward(layers, activations, preacts, prediction, y_train_norm[batch], loss_weights, args.weight_decay)
            clip_grads(grads, args.grad_clip)
            step += 1
            adam_update(layers, grads, states, step, args.lr)

        val_prediction = forward(layers, x_val_norm).astype(np.float32)
        print(
            f"epoch={epoch} train_mse_norm={float(np.mean(losses)):.8f} "
            f"val_mse_norm={weighted_mse(val_prediction, y_val_norm, loss_weights):.8f}",
            flush=True,
        )

    train_pred = predict_actions(layers, x_train_norm, y_mean, y_std)
    val_pred = predict_actions(layers, x_val_norm, y_mean, y_std)
    train_mse = mse(train_pred, y_train)
    val_mse = mse(val_pred, y_val)

    args.output.mkdir(parents=True, exist_ok=True)
    model_prefix = args.model_prefix or ("multi_task_object_action_head_lite" if len(run_dirs) > 1 else "object_action_head_lite")
    model_path = args.output / f"{model_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
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
        "method": "multi_task_object_language_action_head_lite" if len(run_dirs) > 1 else "object_language_action_head_lite",
        "run_dir": str(run_dirs[0]),
        "run_dirs": [str(item) for item in run_dirs],
        "sources": sources,
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
        "train_mse": train_mse,
        "val_mse": val_mse,
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
    print(f"train_mse: {train_mse:.8f}", flush=True)
    print(f"val_mse: {val_mse:.8f}", flush=True)


if __name__ == "__main__":
    main()
