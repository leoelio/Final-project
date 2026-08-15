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

from run_clip_action_head import load_clip  # noqa: E402
from train_clip_action_head import clip_encode_batches, reset_to_saved_state  # noqa: E402
from train_vision_language_action_head import attempt_start_index, selected_attempts  # noqa: E402
from widowx_env import WidowXTabletopEnv  # noqa: E402
from widowx_env.demo_dataset import read_metadata  # noqa: E402


TASK_LABELS = (
    "place_blue_cube_blue_pad",
    "place_blue_cube_red_pad",
    "place_red_cube_red_pad",
    "move_leftmost_cube_to_bowl",
)

# These phrases deliberately do not duplicate the held-out OOD sentences.
LANGUAGE_AUGMENTATIONS = {
    "place_blue_cube_blue_pad": (
        "place the navy cuboid on the blue target",
        "transfer the cerulean object to the blue mat",
        "put the blue cube onto its blue platform",
    ),
    "place_blue_cube_red_pad": (
        "place the navy cuboid on the red target",
        "transfer the cerulean object to the scarlet mat",
        "put the blue cube onto the vermilion platform",
    ),
    "place_red_cube_red_pad": (
        "place the ruby cuboid on the red target",
        "transfer the vermilion object to the scarlet mat",
        "put the red cube onto the crimson platform",
    ),
    "move_leftmost_cube_to_bowl": (
        "move the extreme-left cube into the container",
        "transfer the far-left block to the vessel",
        "put the left-side cube inside the receptacle",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a frozen CLIP task-intent adapter for the Core V2 structured waypoint policy.")
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "clip_semantic_waypoint")
    parser.add_argument("--model-prefix", default="clip_semantic_waypoint_core_v2_v1")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--camera", default="top_rgb")
    parser.add_argument("--workspace-profile", choices=("legacy", "core_v2"), default="core_v2")
    parser.add_argument("--max-episodes-per-run", type=int, default=0)
    parser.add_argument("--language-augmentation", choices=("none", "semantic_alias_v1"), default="none")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def split_per_task(labels: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    val = np.zeros(len(labels), dtype=bool)
    for label in sorted(int(item) for item in np.unique(labels)):
        indices = np.flatnonzero(labels == label)
        count = max(1, int(round(len(indices) * fraction)))
        val[indices[-count:]] = True
    return ~val, val


def collect_samples(args: argparse.Namespace, clip_model, processor) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    env = WidowXTabletopEnv(
        seed=args.seed,
        image_size=(args.image_size, args.image_size),
        camera=args.camera,
        workspace_profile=args.workspace_profile,
    )
    renderer = mujoco.Renderer(env.model, height=args.image_size, width=args.image_size)
    images: list[np.ndarray] = []
    texts: list[str] = []
    labels: list[int] = []
    sources: list[dict] = []
    try:
        for run_dir in args.run_dirs:
            task_count = 0
            for metadata in sorted(read_metadata(run_dir), key=lambda row: int(row["episode_index"])):
                if not metadata["success"] or str(metadata["task"]) not in TASK_LABELS:
                    continue
                with np.load(run_dir / metadata["trajectory_file"]) as data:
                    attempts = selected_attempts(data, metadata, include_failures=False)
                    if not attempts:
                        continue
                    start = attempt_start_index(data, int(attempts[0]))
                    reset_to_saved_state(env, data["attempt_start_qpos"][start], data["attempt_start_qvel"][start], data["attempt_start_ctrl"][start])
                renderer.update_scene(env.data, camera=args.camera)
                rendered = renderer.render().copy()
                task = str(metadata["task"])
                instructions = [str(metadata["instruction"])]
                if args.language_augmentation == "semantic_alias_v1":
                    instructions.extend(LANGUAGE_AUGMENTATIONS[task])
                for instruction in instructions:
                    images.append(rendered)
                    texts.append(instruction)
                    labels.append(TASK_LABELS.index(task))
                task_count += 1
                if args.max_episodes_per_run > 0 and task_count >= args.max_episodes_per_run:
                    break
            sources.append({"run_dir": str(run_dir), "episodes": task_count})
    finally:
        renderer.close()
    if not images:
        raise ValueError("no successful Core V2 demonstrations found")
    features = clip_encode_batches(clip_model, processor, images, texts, batch_size=32)
    return features, np.asarray(labels, dtype=np.int64), sources


def main() -> None:
    args = parse_args()
    started = time.time()
    rng = np.random.default_rng(args.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    clip_model, processor = load_clip(args.clip_model)
    features, labels, sources = collect_samples(args, clip_model, processor)
    train_mask, val_mask = split_per_task(labels, args.val_fraction)
    x_train, y_train = features[train_mask], labels[train_mask]
    x_val, y_val = features[val_mask], labels[val_mask]
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    x_train = (x_train - x_mean) / x_std
    x_val = (x_val - x_mean) / x_std
    weights = rng.normal(0.0, 0.01, size=(x_train.shape[1], len(TASK_LABELS))).astype(np.float32)
    bias = np.zeros(len(TASK_LABELS), dtype=np.float32)
    for epoch in range(1, args.epochs + 1):
        for start in range(0, len(x_train), args.batch_size):
            batch = np.arange(start, min(start + args.batch_size, len(x_train)))
            logits = x_train[batch] @ weights + bias
            probabilities = softmax(logits)
            probabilities[np.arange(len(batch)), y_train[batch]] -= 1.0
            grad_w = x_train[batch].T @ probabilities / len(batch) + args.weight_decay * weights
            grad_b = probabilities.mean(axis=0)
            weights -= args.lr * grad_w.astype(np.float32)
            bias -= args.lr * grad_b.astype(np.float32)
        if epoch in {1, args.epochs} or epoch % 50 == 0:
            train_accuracy = float(np.mean(np.argmax(x_train @ weights + bias, axis=1) == y_train))
            val_accuracy = float(np.mean(np.argmax(x_val @ weights + bias, axis=1) == y_val))
            print(f"epoch={epoch} train_accuracy={train_accuracy:.3f} val_accuracy={val_accuracy:.3f}", flush=True)

    metadata = {
        "method": "frozen_clip_semantic_waypoint",
        "clip_model": args.clip_model,
        "frozen_encoder_params": int(sum(parameter.numel() for parameter in clip_model.parameters())),
        "task_labels": list(TASK_LABELS),
        "samples": int(len(labels)),
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "train_accuracy": float(np.mean(np.argmax(x_train @ weights + bias, axis=1) == y_train)),
        "val_accuracy": float(np.mean(np.argmax(x_val @ weights + bias, axis=1) == y_val)),
        "workspace_profile": args.workspace_profile,
        "image_size": args.image_size,
        "camera": args.camera,
        "epochs": args.epochs,
        "max_episodes_per_run": args.max_episodes_per_run,
        "language_augmentation": args.language_augmentation,
        "augmentation_protocol": "Three task-preserving instruction variants per successful demonstration; held-out OOD sentences are not used for training." if args.language_augmentation != "none" else "none",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "sources": sources,
        "train_time_seconds": time.time() - started,
        "peak_vram_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0,
        "note": "Frozen pretrained CLIP task-intent adapter plus structured waypoint executor; not an end-to-end VLA policy.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{args.model_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    np.savez_compressed(path, x_mean=x_mean, x_std=x_std, weights=weights, bias=bias, metadata=json.dumps(metadata, ensure_ascii=False))
    print(f"model_path: {path}", flush=True)
    print(f"samples: {metadata['samples']}", flush=True)
    print(f"train_accuracy: {metadata['train_accuracy']:.3f}", flush=True)
    print(f"val_accuracy: {metadata['val_accuracy']:.3f}", flush=True)


if __name__ == "__main__":
    main()
