from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
import torch.nn as nn  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from run_clip_action_head import load_clip  # noqa: E402
from train_clip_action_head import reset_to_saved_state  # noqa: E402
from train_clip_semantic_waypoint import LANGUAGE_AUGMENTATIONS, TASK_LABELS  # noqa: E402
from train_vision_language_action_head import attempt_start_index, selected_attempts  # noqa: E402
from widowx_env import WidowXTabletopEnv  # noqa: E402
from widowx_env.demo_dataset import read_metadata  # noqa: E402
from widowx_env.vision_grounding import load_calibration  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a frozen-CLIP patch-token language-conditioned 2D pointer head.")
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--calibration", type=Path, default=ROOT / "outputs" / "rgb_grounding" / "top_rgb_core_v2_calibration_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "clip_patch_pointer")
    parser.add_argument("--model-prefix", default="clip_patch_pointer_core_v2_v1")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--workspace-profile", choices=("legacy", "core_v2"), default="core_v2")
    parser.add_argument("--max-episodes-per-run", type=int, default=25)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--camera", default="top_rgb")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--clip-batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


class PatchPointerHead(nn.Module):
    def __init__(self, patch_dim: int, text_dim: int, hidden_size: int, task_count: int) -> None:
        super().__init__()
        self.visual_projection = nn.Linear(patch_dim, hidden_size)
        self.text_projection = nn.Linear(text_dim, hidden_size)
        self.pointer = nn.Linear(hidden_size, 1)
        self.residual = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 2))
        self.intent = nn.Linear(text_dim, task_count)

    def forward(self, patches: torch.Tensor, text: torch.Tensor, grid_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        visual = self.visual_projection(patches)
        text_hidden = self.text_projection(text)
        pointer_logits = self.pointer(torch.tanh(visual + text_hidden[:, None, :])).squeeze(-1)
        weights = torch.softmax(pointer_logits, dim=1)
        axis = (torch.arange(grid_size, device=patches.device, dtype=patches.dtype) + 0.5) / grid_size
        u, v = torch.meshgrid(axis, axis, indexing="xy")
        centers = torch.stack((u.reshape(-1), v.reshape(-1)), dim=1)
        base_uv = weights @ centers
        pooled = torch.sum(weights[:, :, None] * visual, dim=1)
        residual = 0.22 * torch.tanh(self.residual(torch.cat((pooled, text_hidden), dim=1)))
        predicted_uv = torch.clamp(base_uv + residual, 0.0, 1.0)
        return self.intent(text), predicted_uv, pointer_logits


def resolve_device(value: str) -> torch.device:
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def restore_initial_state(env: WidowXTabletopEnv, trajectory_path: Path, metadata: dict) -> None:
    with np.load(trajectory_path) as data:
        attempts = selected_attempts(data, metadata, include_failures=False)
        start = attempt_start_index(data, int(attempts[0]))
        reset_to_saved_state(env, data["attempt_start_qpos"][start], data["attempt_start_qvel"][start], data["attempt_start_ctrl"][start])


def world_to_pixel(matrix: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return np.linalg.solve(matrix[:, :2], np.asarray(xy, dtype=np.float32) - matrix[:, 2]).astype(np.float32)


def collect_examples(args: argparse.Namespace, matrix: np.ndarray) -> tuple[list[np.ndarray], list[str], np.ndarray, np.ndarray, list[dict]]:
    env = WidowXTabletopEnv(seed=args.seed, image_size=(args.image_size, args.image_size), camera=args.camera, workspace_profile=args.workspace_profile)
    renderer = mujoco.Renderer(env.model, height=args.image_size, width=args.image_size)
    images: list[np.ndarray] = []
    texts: list[str] = []
    labels: list[int] = []
    target_uv: list[np.ndarray] = []
    episodes: list[int] = []
    sources: list[dict] = []
    episode_offset = 0
    try:
        for source_id, run_dir in enumerate(args.run_dirs):
            rows = [row for row in read_metadata(run_dir) if row.get("success") and str(row["task"]) in TASK_LABELS]
            rows.sort(key=lambda row: int(row["episode_index"]))
            rows = rows[: args.max_episodes_per_run] if args.max_episodes_per_run > 0 else rows
            for metadata in rows:
                task = str(metadata["task"])
                env.reset(task=task, complexity=str(metadata["complexity"]), seed=int(metadata["seed"]))
                restore_initial_state(env, run_dir / metadata["trajectory_file"], metadata)
                renderer.update_scene(env.data, camera=args.camera)
                image = renderer.render().copy()
                source_xy = np.asarray(metadata["initial_objects"][str(metadata["target_object"])][:2], dtype=np.float32)
                source_uv = world_to_pixel(matrix, source_xy) / float(args.image_size - 1)
                source_uv = np.clip(source_uv, 0.0, 1.0)
                for instruction in (str(metadata["instruction"]), *LANGUAGE_AUGMENTATIONS[task]):
                    images.append(image)
                    texts.append(instruction)
                    labels.append(TASK_LABELS.index(task))
                    target_uv.append(source_uv)
                    episodes.append(episode_offset + int(metadata["episode_index"]))
            sources.append({"run_dir": str(run_dir), "successful_episodes": len(rows), "source_id": source_id})
            episode_offset += max((int(row["episode_index"]) for row in rows), default=-1) + 1
    finally:
        renderer.close()
    if not images:
        raise ValueError("no eligible successful Core V2 demonstrations found")
    return images, texts, np.asarray(labels, dtype=np.int64), np.stack(target_uv).astype(np.float32), np.asarray(episodes, dtype=np.int32), sources


def encode_patch_text(clip_model, processor, images: list[np.ndarray], texts: list[str], batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    device = next(clip_model.parameters()).device
    patch_batches: list[np.ndarray] = []
    text_batches: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        batch_images = [Image.fromarray(image.astype(np.uint8), mode="RGB") for image in images[start : start + batch_size]]
        inputs = processor(text=texts[start : start + batch_size], images=batch_images, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            vision = clip_model.vision_model(pixel_values=inputs["pixel_values"]).last_hidden_state[:, 1:, :]
            text = clip_model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            if hasattr(text, "pooler_output"):
                text = text.pooler_output
            text = torch.nn.functional.normalize(text, dim=-1)
        patch_batches.append(vision.cpu().numpy().astype(np.float32))
        text_batches.append(text.cpu().numpy().astype(np.float32))
    return np.concatenate(patch_batches, axis=0), np.concatenate(text_batches, axis=0)


def split_by_task_episode(labels: np.ndarray, episodes: np.ndarray, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    validation = np.zeros(len(labels), dtype=bool)
    for task_label in np.unique(labels):
        task_episodes = np.unique(episodes[labels == task_label])
        count = max(1, int(round(len(task_episodes) * val_fraction)))
        held_out = task_episodes[-count:]
        validation |= (labels == task_label) & np.isin(episodes, held_out)
    return ~validation, validation


def score(model: PatchPointerHead, patches: torch.Tensor, text: torch.Tensor, labels: torch.Tensor, target_uv: torch.Tensor, matrix: torch.Tensor, image_size: int) -> dict[str, float]:
    grid_size = int(round(patches.shape[1] ** 0.5))
    model.eval()
    with torch.no_grad():
        intent, predicted_uv, _ = model(patches, text, grid_size)
        pixels = predicted_uv * float(image_size - 1)
        world = pixels @ matrix[:, :2].T + matrix[:, 2]
        target_pixels = target_uv * float(image_size - 1)
        target_world = target_pixels @ matrix[:, :2].T + matrix[:, 2]
        error = torch.linalg.vector_norm(world - target_world, dim=1)
        return {
            "intent_accuracy": float((intent.argmax(dim=1) == labels).float().mean().item()),
            "pointer_mae_m": float(error.mean().item()),
            "pointer_rmse_m": float(torch.sqrt(torch.mean(error ** 2)).item()),
        }


def main() -> None:
    args = parse_args()
    started = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    calibration = load_calibration(args.calibration)
    if calibration.image_size != args.image_size or calibration.camera != args.camera or calibration.workspace_profile != args.workspace_profile:
        raise ValueError("calibration must match image-size, camera, and workspace-profile")
    clip_model, processor = load_clip(args.clip_model)
    clip_model = clip_model.to(device)
    images, texts, labels, target_uv, episodes, sources = collect_examples(args, calibration.matrix)
    patches, text_features = encode_patch_text(clip_model, processor, images, texts, args.clip_batch_size)
    train_mask, validation_mask = split_by_task_episode(labels, episodes, args.val_fraction)
    model = PatchPointerHead(patches.shape[2], text_features.shape[1], args.hidden_size, len(TASK_LABELS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(patches[train_mask]), torch.from_numpy(text_features[train_mask]), torch.from_numpy(labels[train_mask]), torch.from_numpy(target_uv[train_mask])),
        batch_size=args.batch_size,
        shuffle=True,
    )
    matrix = torch.from_numpy(calibration.matrix.astype(np.float32)).to(device)
    train_tensors = [torch.from_numpy(array[train_mask]).to(device) for array in (patches, text_features, labels, target_uv)]
    val_tensors = [torch.from_numpy(array[validation_mask]).to(device) for array in (patches, text_features, labels, target_uv)]
    grid_size = int(round(patches.shape[1] ** 0.5))
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_patches, batch_text, batch_labels, batch_uv in train_loader:
            logits, prediction, pointer_logits = model(batch_patches.to(device), batch_text.to(device), grid_size)
            patch_u = torch.clamp((batch_uv[:, 0] * grid_size).long(), 0, grid_size - 1)
            patch_v = torch.clamp((batch_uv[:, 1] * grid_size).long(), 0, grid_size - 1)
            patch_targets = patch_v * grid_size + patch_u
            loss = (
                torch.nn.functional.cross_entropy(logits, batch_labels.to(device))
                + 1.5 * torch.nn.functional.cross_entropy(pointer_logits, patch_targets.to(device))
                + 20.0 * torch.nn.functional.mse_loss(prediction, batch_uv.to(device))
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        if epoch in {1, args.epochs} or epoch % 50 == 0:
            train_metrics = score(model, *train_tensors, matrix, args.image_size)
            validation_metrics = score(model, *val_tensors, matrix, args.image_size)
            history.append({"epoch": epoch, "loss": float(np.mean(losses)), "train": train_metrics, "validation": validation_metrics})
            print(f"epoch={epoch} loss={np.mean(losses):.5f} val_intent={validation_metrics['intent_accuracy']:.3f} val_pointer_rmse_m={validation_metrics['pointer_rmse_m']:.4f}", flush=True)
    train_metrics = score(model, *train_tensors, matrix, args.image_size)
    validation_metrics = score(model, *val_tensors, matrix, args.image_size)
    metadata = {
        "version": "clip_patch_pointer_core_v2_v1",
        "method": "frozen CLIP patch-token and language features + lightweight 2D pointer/action-parameter head + structured executor",
        "method_boundary": "Runtime uses top RGB, instruction, a fixed camera-plane calibration, predicted intent, and predicted pixel pointer. The structured executor performs continuous contact control. This is a lightweight VLM spatial action head, not end-to-end VLA, OpenVLA, or LoRA.",
        "offline_truth_boundary": "MuJoCo object positions supervise initial source-pixel labels and calculate offline error only; runtime does not query object positions.",
        "clip_model": args.clip_model,
        "frozen_encoder_params": int(sum(parameter.numel() for parameter in clip_model.parameters())),
        "trainable_head_params": int(sum(parameter.numel() for parameter in model.parameters())),
        "task_labels": list(TASK_LABELS),
        "samples": int(len(labels)),
        "train_samples": int(train_mask.sum()),
        "validation_samples": int(validation_mask.sum()),
        "workspace_profile": args.workspace_profile,
        "image_size": args.image_size,
        "camera": args.camera,
        "calibration_path": str(args.calibration),
        "calibration_rms_error_m": calibration.rms_error_m,
        "patch_count": int(patches.shape[1]),
        "patch_grid_size": grid_size,
        "patch_dim": int(patches.shape[2]),
        "text_dim": int(text_features.shape[1]),
        "hidden_size": args.hidden_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "sources": sources,
        "train_time_seconds": time.time() - started,
        "peak_vram_mb": float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / f"{args.model_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save({"state_dict": model.state_dict(), "calibration_matrix": calibration.matrix.astype(np.float32), "metadata": metadata}, model_path)
    metrics_path = model_path.with_suffix(".json")
    metrics_path.write_text(json.dumps({"model": str(model_path), "history": history, "metadata": metadata}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"model_path: {model_path}", flush=True)
    print(f"metrics_path: {metrics_path}", flush=True)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
