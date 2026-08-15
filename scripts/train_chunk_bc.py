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
from widowx_env import WidowXTabletopEnv  # noqa: E402
from widowx_env.demo_dataset import latest_run_dir, load_demo_dataset  # noqa: E402
from widowx_env.tabletop_env import OBJECTS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NumPy action-chunk BC / ACT-lite baseline.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "chunk_bc")
    parser.add_argument("--model-prefix", default=None)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--history", type=int, default=1)
    parser.add_argument("--sample-stride", type=int, default=8)
    parser.add_argument("--hidden-sizes", default="256,256")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--gripper-loss-weight", type=float, default=6.0)
    parser.add_argument("--augment-relative", action="store_true")
    parser.add_argument("--no-augment-relative", dest="augment_relative", action="store_false")
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--all-attempts", action="store_true")
    parser.set_defaults(augment_relative=True)
    return parser.parse_args()


def build_chunk_samples(
    observations: np.ndarray,
    actions: np.ndarray,
    episode_indices: np.ndarray,
    segments: list[dict],
    allowed_episodes: set[int],
    horizon: int,
    sample_stride: int,
    history: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    offset = 0
    history = max(1, int(history))
    for segment in segments:
        length = int(segment["steps"])
        segment_slice = slice(offset, offset + length)
        offset += length
        if int(segment["episode_index"]) not in allowed_episodes:
            continue
        if length < horizon + history - 1:
            continue
        segment_observations = observations[segment_slice]
        segment_actions = actions[segment_slice]
        for start in range(history - 1, length - horizon + 1, sample_stride):
            xs.append(segment_observations[start - history + 1: start + 1].reshape(-1))
            ys.append(segment_actions[start: start + horizon].reshape(-1))

    if not xs:
        raise ValueError("no chunk samples were built; reduce --horizon or --sample-stride")

    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def predict_chunks(layers: list[dict[str, np.ndarray]], x: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    return forward(layers, x).astype(np.float32) * y_std + y_mean


def observation_layout() -> dict[str, int]:
    env = WidowXTabletopEnv(seed=0)
    qpos_dim = int(env.data.qpos.size)
    qvel_dim = int(env.data.qvel.size)
    ctrl_dim = int(env.data.ctrl.size)
    return {
        "qpos_dim": qpos_dim,
        "qvel_dim": qvel_dim,
        "ctrl_dim": ctrl_dim,
        "tcp_start": qpos_dim + qvel_dim + ctrl_dim,
        "object_count": len(OBJECTS),
    }


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


def output_weights(horizon: int, action_dim: int, gripper_loss_weight: float) -> np.ndarray:
    weights = np.ones(horizon * action_dim, dtype=np.float32)
    weights[action_dim - 1 :: action_dim] = max(1.0, float(gripper_loss_weight))
    return weights / weights.mean()


def weighted_mse(prediction: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    return float(np.mean(((prediction - target) ** 2) * weights[None, :]))


def weighted_backward(
    layers: list[dict[str, np.ndarray]],
    activations: list[np.ndarray],
    preacts: list[np.ndarray],
    prediction: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    weight_decay: float,
) -> list[dict[str, np.ndarray]]:
    grad = (2.0 / prediction.size) * (prediction - target) * weights[None, :]
    grads: list[dict[str, np.ndarray]] = []
    for index in reversed(range(len(layers))):
        grad_w = activations[index].T @ grad + weight_decay * layers[index]["w"]
        grad_b = grad.sum(axis=0)
        grads.append({"w": grad_w.astype(np.float32), "b": grad_b.astype(np.float32)})
        if index > 0:
            grad = (grad @ layers[index]["w"].T) * (preacts[index - 1] > 0.0)
    grads.reverse()
    return grads


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    run_dir = args.run_dir or latest_run_dir()
    dataset = load_demo_dataset(
        run_dir,
        successful_only=not args.include_failures,
        successful_attempt_only=not args.all_attempts,
    )
    raw_observation_dim = int(dataset.observations.shape[1])
    layout = observation_layout()
    observations = dataset.observations
    if args.augment_relative:
        observations = augment_relative_features(observations, layout)
    train_mask, val_mask = split_by_episode(dataset, args.val_fraction)
    train_episodes = set(int(item) for item in np.unique(dataset.episode_indices[train_mask]))
    val_episodes = set(int(item) for item in np.unique(dataset.episode_indices[val_mask]))

    horizon = max(2, int(args.horizon))
    sample_stride = max(1, int(args.sample_stride))
    history = max(1, int(args.history))
    x_train, y_train = build_chunk_samples(
        observations,
        dataset.actions,
        dataset.episode_indices,
        dataset.segments,
        train_episodes,
        horizon,
        sample_stride,
        history,
    )
    x_val, y_val = build_chunk_samples(
        observations,
        dataset.actions,
        dataset.episode_indices,
        dataset.segments,
        val_episodes,
        horizon,
        sample_stride,
        history,
    )

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
    loss_weights = output_weights(horizon, dataset.actions.shape[1], args.gripper_loss_weight)

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

    train_pred = predict_chunks(layers, x_train_norm, y_mean, y_std)
    val_pred = predict_chunks(layers, x_val_norm, y_mean, y_std)
    train_mse = mse(train_pred, y_train)
    val_mse = mse(val_pred, y_val)

    args.output.mkdir(parents=True, exist_ok=True)
    model_prefix = args.model_prefix or ("trajectory_chunk_bc" if history > 1 else "chunk_bc")
    model_path = args.output / f"{model_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
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
        "method": "trajectory_conditioned_action_chunk_bc" if history > 1 else "action_chunk_bc",
        "run_dir": str(run_dir),
        "source_samples": int(len(dataset.actions)),
        "train_chunks": int(len(x_train)),
        "val_chunks": int(len(x_val)),
        "raw_observation_dim": raw_observation_dim,
        "single_observation_dim": int(observations.shape[1]),
        "observation_dim": int(x_train.shape[1]),
        "action_dim": int(dataset.actions.shape[1]),
        "chunk_output_dim": int(y_train.shape[1]),
        "horizon": horizon,
        "history": history,
        "sample_stride": sample_stride,
        "hidden_sizes": hidden_sizes,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "gripper_loss_weight": float(args.gripper_loss_weight),
        "augment_relative": bool(args.augment_relative),
        "layout": layout,
        "train_mse": train_mse,
        "val_mse": val_mse,
        "successful_only": not args.include_failures,
        "successful_attempt_only": not args.all_attempts,
    }
    save_items["metadata"] = json.dumps(meta, ensure_ascii=False)
    np.savez_compressed(model_path, **save_items)

    print(f"run_dir: {run_dir}", flush=True)
    print(f"model_path: {model_path}", flush=True)
    print(f"source_samples: {meta['source_samples']}", flush=True)
    print(f"train_chunks: {meta['train_chunks']}", flush=True)
    print(f"val_chunks: {meta['val_chunks']}", flush=True)
    print(f"observation_dim: {meta['observation_dim']}", flush=True)
    print(f"action_dim: {meta['action_dim']}", flush=True)
    print(f"horizon: {horizon}", flush=True)
    print(f"history: {history}", flush=True)
    print(f"sample_stride: {sample_stride}", flush=True)
    print(f"train_mse: {train_mse:.8f}", flush=True)
    print(f"val_mse: {val_mse:.8f}", flush=True)


if __name__ == "__main__":
    main()
