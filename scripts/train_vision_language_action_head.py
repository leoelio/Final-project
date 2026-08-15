from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_chunk_bc import output_weights, weighted_backward, weighted_mse  # noqa: E402
from train_mlp_bc import adam_update, clip_grads, forward, init_model, make_adam_states, mse, parse_hidden_sizes  # noqa: E402
from train_object_action_head import TARGET_GEOMS, one_hot  # noqa: E402
from widowx_env import TASKS, WidowXTabletopEnv  # noqa: E402
from widowx_env.demo_dataset import phase_features, read_metadata  # noqa: E402
from widowx_env.tabletop_env import OBJECTS  # noqa: E402


TASK_KINDS = ("pick", "place", "push")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a frozen-visual-feature + language-token action-head baseline.")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data" / "demos" / "place_blue_cube_blue_pad_medium_20260702_051752")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "vision_language_action_head")
    parser.add_argument("--model-prefix", default="vision_language_action_head_lite")
    parser.add_argument("--hidden-sizes", default="256,256")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--gripper-loss-weight", type=float, default=4.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--sample-stride", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=16000)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--camera", default="top_rgb")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--log-every-episodes", type=int, default=10)
    return parser.parse_args()


def color_centroid_features(rgb: np.ndarray) -> np.ndarray:
    image = rgb.astype(np.float32) / 255.0
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    masks = (
        (r > 0.55) & (g < 0.38) & (b < 0.38),
        (b > 0.50) & (r < 0.42) & (g < 0.50),
        (g > 0.45) & (r < 0.45) & (b < 0.45),
        (r > 0.55) & (g > 0.45) & (b < 0.38),
        (r > 0.72) & (g > 0.72) & (b > 0.72),
        (r < 0.28) & (g < 0.28) & (b < 0.28),
    )
    yy, xx = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    x_norm = (xx.astype(np.float32) / max(1, image.shape[1] - 1)) * 2.0 - 1.0
    y_norm = (yy.astype(np.float32) / max(1, image.shape[0] - 1)) * 2.0 - 1.0
    features = []
    for mask in masks:
        count = int(mask.sum())
        coverage = count / mask.size
        if count:
            mx = float(x_norm[mask].mean())
            my = float(y_norm[mask].mean())
            sx = float(x_norm[mask].std())
            sy = float(y_norm[mask].std())
        else:
            mx = my = sx = sy = 0.0
        features.extend([coverage, mx, my, sx, sy])
    return np.asarray(features, dtype=np.float32)


def pooled_rgb_features(rgb: np.ndarray, grid_size: int) -> np.ndarray:
    image = rgb.astype(np.float32) / 255.0
    height, width, channels = image.shape
    cell_h = height // grid_size
    cell_w = width // grid_size
    cropped = image[: cell_h * grid_size, : cell_w * grid_size, :]
    pooled = cropped.reshape(grid_size, cell_h, grid_size, cell_w, channels).mean(axis=(1, 3))
    global_stats = np.concatenate([image.mean(axis=(0, 1)), image.std(axis=(0, 1))])
    return np.concatenate([pooled.reshape(-1), global_stats, color_centroid_features(rgb)]).astype(np.float32)


def language_features(metadata: dict) -> np.ndarray:
    task = TASKS[str(metadata["task"])]
    target_object = str(metadata["target_object"])
    target_geom = metadata.get("target_geom")
    active_objects = set(metadata.get("active_objects", []))
    active_mask = np.asarray([name in active_objects for name in OBJECTS], dtype=np.float32)
    task_kind = one_hot(task.kind, TASK_KINDS)
    relation = np.asarray([1.0 if task.relation == "leftmost" else 0.0], dtype=np.float32)
    return np.concatenate(
        [
            one_hot(target_object, OBJECTS),
            one_hot(target_geom, TARGET_GEOMS),
            active_mask,
            task_kind,
            relation,
        ]
    ).astype(np.float32)


def feature_from_env_state(
    env: WidowXTabletopEnv,
    renderer: mujoco.Renderer,
    metadata: dict,
    phase: float,
    grid_size: int,
) -> np.ndarray:
    renderer.update_scene(env.data, camera=env.camera)
    rgb = renderer.render()
    robot_qpos = env.data.qpos[: env.robot_nq].astype(np.float32)
    robot_qvel = env.data.qvel[: env.robot_nv].astype(np.float32)
    robot_ctrl = env.data.ctrl.astype(np.float32)
    return np.concatenate(
        [
            robot_qpos,
            robot_qvel,
            robot_ctrl,
            pooled_rgb_features(rgb, grid_size),
            language_features(metadata),
            phase_features(phase),
        ]
    ).astype(np.float32)


def pre_step_array(series: np.ndarray, start_value: np.ndarray, indices: np.ndarray) -> np.ndarray:
    values = series[indices]
    if len(indices) == 1:
        return start_value[None, ...].astype(np.float32)
    return np.concatenate([start_value[None, ...], values[:-1]], axis=0).astype(np.float32)


def attempt_start_index(data: np.lib.npyio.NpzFile, attempt_id: int) -> int:
    matches = np.flatnonzero(data["attempt_start_ids"] == attempt_id)
    if len(matches) == 0:
        raise ValueError(f"attempt {attempt_id} has no saved start state")
    return int(matches[0])


def selected_attempts(data: np.lib.npyio.NpzFile, metadata: dict, include_failures: bool) -> list[int]:
    if metadata["success"] and not include_failures:
        return [int(metadata["attempts"])]
    return sorted(int(item) for item in np.unique(data["attempt_ids"]))


def episode_features(
    env: WidowXTabletopEnv,
    renderer: mujoco.Renderer,
    trajectory_path: Path,
    metadata: dict,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    actions = []
    with np.load(trajectory_path) as data:
        attempts = selected_attempts(data, metadata, args.include_failures)
        for attempt_id in attempts:
            indices = np.flatnonzero(data["attempt_ids"] == attempt_id)
            if len(indices) == 0:
                continue
            start = attempt_start_index(data, attempt_id)
            sampled = indices[:: max(1, args.sample_stride)]
            qpos = pre_step_array(data["qpos"], data["attempt_start_qpos"][start], indices)
            qvel = pre_step_array(data["qvel"], data["attempt_start_qvel"][start], indices)
            ctrl = pre_step_array(data["ctrl"], data["attempt_start_ctrl"][start], indices)
            local_phase = np.linspace(0.0, 1.0, len(indices), dtype=np.float32)
            local_lookup = {int(index): position for position, index in enumerate(indices)}

            for source_index in sampled:
                local = local_lookup[int(source_index)]
                env.data.qpos[:] = qpos[local]
                env.data.qvel[:] = qvel[local]
                env.data.ctrl[:] = ctrl[local]
                mujoco.mj_forward(env.model, env.data)
                features.append(feature_from_env_state(env, renderer, metadata, float(local_phase[local]), args.grid_size))
                actions.append(data["actions"][source_index].astype(np.float32))

    if not features:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)
    return np.stack(features).astype(np.float32), np.stack(actions).astype(np.float32)


def build_training_arrays(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    rng = np.random.default_rng(args.seed)
    metadata_rows = [item for item in read_metadata(args.run_dir) if args.include_failures or item["success"]]
    metadata_rows = sorted(metadata_rows, key=lambda item: int(item["episode_index"]))
    if args.max_samples > 0:
        rng.shuffle(metadata_rows)

    env = WidowXTabletopEnv(seed=args.seed, image_size=(args.image_size, args.image_size), camera=args.camera)
    renderer = mujoco.Renderer(env.model, height=args.image_size, width=args.image_size)
    feature_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    episode_parts: list[np.ndarray] = []
    sources: list[dict] = []
    total = 0
    started = time.time()
    try:
        for row_number, metadata in enumerate(metadata_rows, start=1):
            env.reset(task=str(metadata["task"]), complexity=str(metadata["complexity"]), seed=int(metadata["seed"]))
            x, y = episode_features(env, renderer, args.run_dir / metadata["trajectory_file"], metadata, args)
            if len(y) == 0:
                continue
            if args.max_samples > 0 and total + len(y) > args.max_samples:
                keep = max(0, args.max_samples - total)
                x = x[:keep]
                y = y[:keep]
            if len(y) == 0:
                break
            feature_parts.append(x)
            action_parts.append(y)
            episode_parts.append(np.full(len(y), int(metadata["episode_index"]), dtype=np.int32))
            sources.append(
                {
                    "episode_index": int(metadata["episode_index"]),
                    "seed": int(metadata["seed"]),
                    "samples": int(len(y)),
                }
            )
            total += len(y)
            if args.log_every_episodes > 0 and row_number % args.log_every_episodes == 0:
                elapsed = time.time() - started
                print(f"rendered_episodes={row_number} samples={total} elapsed={elapsed:.1f}s", flush=True)
            if args.max_samples > 0 and total >= args.max_samples:
                break
    finally:
        renderer.close()

    if not feature_parts:
        raise ValueError(f"no samples loaded from {args.run_dir}")
    return (
        np.concatenate(feature_parts, axis=0).astype(np.float32),
        np.concatenate(action_parts, axis=0).astype(np.float32),
        np.concatenate(episode_parts, axis=0).astype(np.int32),
        sources,
    )


def split_by_episode(episode_indices: np.ndarray, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    episodes = np.array(sorted(np.unique(episode_indices)))
    if len(episodes) <= 1:
        mask = np.ones(len(episode_indices), dtype=bool)
        return mask, mask.copy()
    val_count = max(1, int(round(len(episodes) * val_fraction)))
    val_episodes = set(int(item) for item in episodes[-val_count:])
    val_mask = np.array([int(item) in val_episodes for item in episode_indices], dtype=bool)
    return ~val_mask, val_mask


def predict_actions(layers: list[dict[str, np.ndarray]], x: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    return forward(layers, x).astype(np.float32) * y_std + y_mean


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    started = time.time()
    features, actions, episode_indices, sources = build_training_arrays(args)
    train_mask, val_mask = split_by_episode(episode_indices, args.val_fraction)

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
            batch = order[start : start + args.batch_size]
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
    elapsed = time.time() - started

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
        "method": "vision_language_action_head_lite",
        "run_dir": str(args.run_dir),
        "samples": int(len(actions)),
        "source_episodes": int(len(np.unique(episode_indices))),
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
        "sample_stride": int(args.sample_stride),
        "image_size": int(args.image_size),
        "grid_size": int(args.grid_size),
        "camera": str(args.camera),
        "train_mse": train_mse,
        "val_mse": val_mse,
        "train_time_seconds": elapsed,
        "peak_vram_mb": 0.0,
        "sources": sources,
        "objects": list(OBJECTS),
        "target_geoms": list(TARGET_GEOMS),
        "task_kinds": list(TASK_KINDS),
        "successful_only": not args.include_failures,
        "note": "Frozen MuJoCo RGB feature proxy plus language/proprioception action head; not a pretrained VLM/VLA.",
    }
    save_items["metadata"] = json.dumps(meta, ensure_ascii=False)
    np.savez_compressed(model_path, **save_items)

    print(f"run_dir: {args.run_dir}", flush=True)
    print(f"model_path: {model_path}", flush=True)
    print(f"samples: {meta['samples']}", flush=True)
    print(f"feature_dim: {meta['feature_dim']}", flush=True)
    print(f"action_dim: {meta['action_dim']}", flush=True)
    print(f"train_mse: {train_mse:.8f}", flush=True)
    print(f"val_mse: {val_mse:.8f}", flush=True)
    print(f"train_time_seconds: {elapsed:.2f}", flush=True)


if __name__ == "__main__":
    main()
