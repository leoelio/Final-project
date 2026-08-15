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

from torch_runtime import ensure_torch_path  # noqa: E402
from vlm_runtime import ensure_vlm_path  # noqa: E402

ensure_torch_path()
ensure_vlm_path()

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import CLIPModel, CLIPProcessor  # noqa: E402

from train_chunk_bc import output_weights, weighted_backward, weighted_mse  # noqa: E402
from train_mlp_bc import adam_update, clip_grads, forward, init_model, make_adam_states, mse, parse_hidden_sizes  # noqa: E402
from train_vision_language_action_head import (  # noqa: E402
    attempt_start_index,
    pre_step_array,
    selected_attempts,
)
from widowx_env import WidowXTabletopEnv  # noqa: E402
from widowx_env.demo_dataset import phase_features, read_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a frozen CLIP image/text encoder + lightweight action head baseline.")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data" / "demos" / "place_blue_cube_blue_pad_medium_20260702_051752")
    parser.add_argument("--run-dirs", type=Path, nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "clip_action_head")
    parser.add_argument("--model-prefix", default="clip_action_head_lite")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--hidden-sizes", default="256,128")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--gripper-loss-weight", type=float, default=4.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--sample-stride", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--camera", default="top_rgb")
    parser.add_argument("--workspace-profile", choices=("legacy", "core_v2"), default="legacy")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--clip-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--log-every-episodes", type=int, default=10)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable")
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_clip(name: str, device: torch.device) -> tuple[CLIPModel, CLIPProcessor]:
    processor = CLIPProcessor.from_pretrained(name)
    model = CLIPModel.from_pretrained(name).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, processor


def clip_tensor(output):
    return output.pooler_output if hasattr(output, "pooler_output") else output


def clip_encode_batches(
    model: CLIPModel,
    processor: CLIPProcessor,
    rgbs: list[np.ndarray],
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    features: list[np.ndarray] = []
    for start in range(0, len(rgbs), batch_size):
        batch_rgbs = rgbs[start : start + batch_size]
        batch_texts = texts[start : start + batch_size]
        images = [Image.fromarray(rgb.astype(np.uint8), mode="RGB") for rgb in batch_rgbs]
        inputs = processor(text=batch_texts, images=images, return_tensors="pt", padding=True)
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            image_features = clip_tensor(model.get_image_features(pixel_values=inputs["pixel_values"]))
            text_features = clip_tensor(model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]))
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            text_features = torch.nn.functional.normalize(text_features, dim=-1)
            combined = torch.cat([image_features, text_features], dim=-1).cpu().numpy().astype(np.float32)
        features.append(combined)
    return np.concatenate(features, axis=0).astype(np.float32)


def split_by_source_episode_indices(source_ids: np.ndarray, episode_indices: np.ndarray, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    val_mask = np.zeros(len(episode_indices), dtype=bool)
    for source_id in sorted(int(item) for item in np.unique(source_ids)):
        source_mask = source_ids == source_id
        episodes = np.array(sorted(np.unique(episode_indices[source_mask])))
        if len(episodes) <= 1:
            continue
        val_count = max(1, int(round(len(episodes) * val_fraction)))
        val_episodes = set(int(item) for item in episodes[-val_count:])
        val_mask |= source_mask & np.array([int(item) in val_episodes for item in episode_indices], dtype=bool)
    return ~val_mask, val_mask


def reset_to_saved_state(env: WidowXTabletopEnv, qpos: np.ndarray, qvel: np.ndarray, ctrl: np.ndarray) -> None:
    env.data.qpos[:] = qpos
    env.data.qvel[:] = qvel
    env.data.ctrl[:] = ctrl
    mujoco.mj_forward(env.model, env.data)


def episode_samples(
    env: WidowXTabletopEnv,
    renderer: mujoco.Renderer,
    trajectory_path: Path,
    metadata: dict,
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], list[str], list[np.ndarray], np.ndarray]:
    rgbs: list[np.ndarray] = []
    texts: list[str] = []
    proprio: list[np.ndarray] = []
    actions: list[np.ndarray] = []
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
                reset_to_saved_state(env, qpos[local], qvel[local], ctrl[local])
                renderer.update_scene(env.data, camera=env.camera)
                rgbs.append(renderer.render().copy())
                texts.append(str(metadata["instruction"]))
                proprio.append(
                    np.concatenate(
                        [
                            env.data.qpos[: env.robot_nq].astype(np.float32),
                            env.data.qvel[: env.robot_nv].astype(np.float32),
                            env.data.ctrl.astype(np.float32),
                            phase_features(float(local_phase[local])),
                        ]
                    ).astype(np.float32)
                )
                actions.append(data["actions"][source_index].astype(np.float32))
                if args.max_samples > 0 and len(actions) >= args.max_samples:
                    break
            if args.max_samples > 0 and len(actions) >= args.max_samples:
                break
    return rgbs, texts, proprio, np.asarray(actions, dtype=np.float32)


def resolve_run_dirs(args: argparse.Namespace) -> list[Path]:
    return [Path(item) for item in args.run_dirs] if args.run_dirs else [args.run_dir]


def build_training_arrays(
    args: argparse.Namespace,
    model: CLIPModel,
    processor: CLIPProcessor,
    run_dirs: list[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    rng = np.random.default_rng(args.seed)
    env = WidowXTabletopEnv(
        seed=args.seed,
        image_size=(args.image_size, args.image_size),
        camera=args.camera,
        workspace_profile=args.workspace_profile,
    )
    renderer = mujoco.Renderer(env.model, height=args.image_size, width=args.image_size)
    feature_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    episode_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    sources: list[dict] = []
    total = 0
    started = time.time()
    episode_offset = 0
    try:
        for source_id, run_dir in enumerate(run_dirs):
            metadata_rows = [item for item in read_metadata(run_dir) if args.include_failures or item["success"]]
            metadata_rows = sorted(metadata_rows, key=lambda item: int(item["episode_index"]))
            rng.shuffle(metadata_rows)
            source_samples = 0
            source_episodes = 0
            for row_number, metadata in enumerate(metadata_rows, start=1):
                env.reset(task=str(metadata["task"]), complexity=str(metadata["complexity"]), seed=int(metadata["seed"]))
                rgbs, texts, proprio, actions = episode_samples(env, renderer, run_dir / metadata["trajectory_file"], metadata, args)
                if len(actions) == 0:
                    continue
                if args.max_samples > 0 and total + len(actions) > args.max_samples:
                    keep = max(0, args.max_samples - total)
                    rgbs, texts, proprio, actions = rgbs[:keep], texts[:keep], proprio[:keep], actions[:keep]
                if len(actions) == 0:
                    break
                clip_features = clip_encode_batches(model, processor, rgbs, texts, args.clip_batch_size)
                feature_parts.append(np.concatenate([clip_features, np.asarray(proprio, dtype=np.float32)], axis=1).astype(np.float32))
                action_parts.append(actions.astype(np.float32))
                episode_parts.append(np.full(len(actions), int(metadata["episode_index"]) + episode_offset, dtype=np.int32))
                source_parts.append(np.full(len(actions), source_id, dtype=np.int16))
                source_samples += len(actions)
                source_episodes += 1
                total += len(actions)
                if args.log_every_episodes > 0 and row_number % args.log_every_episodes == 0:
                    print(f"source={source_id} encoded_episodes={row_number} samples={total} elapsed={time.time() - started:.1f}s", flush=True)
                if args.max_samples > 0 and total >= args.max_samples:
                    break
            sources.append({"run_dir": str(run_dir), "samples": int(source_samples), "episodes": int(source_episodes)})
            episode_offset += max(1, max((int(item["episode_index"]) for item in metadata_rows), default=-1) + 1)
            if args.max_samples > 0 and total >= args.max_samples:
                break
    finally:
        renderer.close()

    if not feature_parts:
        raise ValueError(f"no CLIP samples loaded from {args.run_dir}")
    return (
        np.concatenate(feature_parts, axis=0).astype(np.float32),
        np.concatenate(action_parts, axis=0).astype(np.float32),
        np.concatenate(episode_parts, axis=0).astype(np.int32),
        np.concatenate(source_parts, axis=0).astype(np.int16),
        sources,
    )


def predict_actions(layers: list[dict[str, np.ndarray]], x: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    return forward(layers, x).astype(np.float32) * y_std + y_mean


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    started = time.time()
    run_dirs = resolve_run_dirs(args)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    clip_model, processor = load_clip(args.clip_model, device)
    frozen_encoder_params = int(sum(parameter.numel() for parameter in clip_model.parameters()))
    features, actions, episode_indices, source_ids, sources = build_training_arrays(args, clip_model, processor, run_dirs)
    train_mask, val_mask = split_by_source_episode_indices(source_ids, episode_indices, args.val_fraction)
    if not np.any(train_mask):
        raise ValueError("training split is empty; provide at least one training episode per source")
    if not np.any(val_mask):
        val_mask = train_mask.copy()

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
        "method": "clip_action_head_lite",
        "clip_model": str(args.clip_model),
        "run_dir": str(run_dirs[0]),
        "run_dirs": [str(item) for item in run_dirs],
        "samples": int(len(actions)),
        "source_episodes": int(len(np.unique(episode_indices))),
        "feature_dim": int(features.shape[1]),
        "clip_feature_dim": 1024,
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
        "camera": str(args.camera),
        "workspace_profile": str(args.workspace_profile),
        "device": str(device),
        "clip_batch_size": int(args.clip_batch_size),
        "train_mse": train_mse,
        "val_mse": val_mse,
        "train_time_seconds": float(elapsed),
        "peak_vram_mb": float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else 0.0,
        "frozen_encoder_params": frozen_encoder_params,
        "sources": sources,
        "successful_only": not args.include_failures,
        "note": "Frozen pretrained CLIP image/text encoder plus lightweight action head; CLIP is pretrained, action head is local.",
    }
    save_items["metadata"] = json.dumps(meta, ensure_ascii=False)
    np.savez_compressed(model_path, **save_items)

    print(f"run_dirs: {[str(item) for item in run_dirs]}", flush=True)
    print(f"device: {device}", flush=True)
    print(f"clip_model: {args.clip_model}", flush=True)
    print(f"model_path: {model_path}", flush=True)
    print(f"samples: {meta['samples']}", flush=True)
    print(f"feature_dim: {meta['feature_dim']}", flush=True)
    print(f"frozen_encoder_params: {frozen_encoder_params}", flush=True)
    print(f"train_mse: {train_mse:.8f}", flush=True)
    print(f"val_mse: {val_mse:.8f}", flush=True)
    print(f"train_time_seconds: {elapsed:.2f}", flush=True)


if __name__ == "__main__":
    main()
