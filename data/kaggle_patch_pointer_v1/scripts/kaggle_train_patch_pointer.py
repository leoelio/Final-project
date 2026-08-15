from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import numpy as np

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from transformers import CLIPModel, CLIPProcessor


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
        return self.intent(text), torch.clamp(base_uv + residual, 0.0, 1.0), pointer_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle trainer for frozen-CLIP patch-token language-conditioned 2D pointer heads.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/clip_patch_pointer_kaggle_v1.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("/kaggle/working/clip_patch_pointer_kaggle_v1.json"))
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else value if value != "auto" else "cpu")


def load_examples(root: Path) -> tuple[list[dict], dict]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (root / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    examples = []
    for row in rows:
        for instruction in row["instruction_variants"]:
            examples.append({**row, "instruction": instruction})
    return examples, manifest


def encode(clip_model, processor, root: Path, examples: list[dict], batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    device = next(clip_model.parameters()).device
    patch_batches, text_batches, labels, uv = [], [], [], []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        images = [Image.open(root / row["image"]).convert("RGB") for row in batch]
        inputs = processor(text=[row["instruction"] for row in batch], images=images, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            patches = clip_model.vision_model(pixel_values=inputs["pixel_values"]).last_hidden_state[:, 1:, :]
            text = clip_model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            if hasattr(text, "pooler_output"):
                text = text.pooler_output
            text = torch.nn.functional.normalize(text, dim=-1)
        patch_batches.append(patches.cpu().numpy().astype(np.float32))
        text_batches.append(text.cpu().numpy().astype(np.float32))
        labels.extend(int(row["task_label"]) for row in batch)
        uv.extend(row["source_pixel_uv_normalized"] for row in batch)
    return np.concatenate(patch_batches), np.concatenate(text_batches), np.asarray(labels, dtype=np.int64), np.asarray(uv, dtype=np.float32)


def metrics(model: PatchPointerHead, patches: torch.Tensor, text: torch.Tensor, labels: torch.Tensor, target_uv: torch.Tensor, matrix: torch.Tensor, image_size: int) -> dict[str, float]:
    grid = int(round(patches.shape[1] ** 0.5))
    model.eval()
    with torch.no_grad():
        logits, predicted_uv, _ = model(patches, text, grid)
        pixels = predicted_uv * float(image_size - 1)
        target_pixels = target_uv * float(image_size - 1)
        world = pixels @ matrix[:, :2].T + matrix[:, 2]
        target_world = target_pixels @ matrix[:, :2].T + matrix[:, 2]
        error = torch.linalg.vector_norm(world - target_world, dim=1)
        return {
            "intent_accuracy": float((logits.argmax(dim=1) == labels).float().mean().item()),
            "pointer_mae_m": float(error.mean().item()),
            "pointer_rmse_m": float(torch.sqrt(torch.mean(error ** 2)).item()),
        }


def main() -> None:
    args = parse_args()
    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    examples, manifest = load_examples(args.dataset_root)
    train_mask = np.asarray([row["split"] == "train" for row in examples], dtype=bool)
    validation_mask = ~train_mask
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("dataset must contain train and validation examples")
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(device)
    clip_model.eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad_(False)
    patches, text, labels, target_uv = encode(clip_model, processor, args.dataset_root, examples, args.clip_batch_size)
    model = PatchPointerHead(patches.shape[2], text.shape[1], args.hidden_size, len(manifest["task_labels"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(patches[train_mask]), torch.from_numpy(text[train_mask]), torch.from_numpy(labels[train_mask]), torch.from_numpy(target_uv[train_mask])),
        batch_size=args.batch_size,
        shuffle=True,
    )
    matrix = torch.tensor(manifest["calibration"]["matrix"], dtype=torch.float32, device=device)
    train_tensors = [torch.from_numpy(item[train_mask]).to(device) for item in (patches, text, labels, target_uv)]
    validation_tensors = [torch.from_numpy(item[validation_mask]).to(device) for item in (patches, text, labels, target_uv)]
    grid = int(round(patches.shape[1] ** 0.5))
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_patches, batch_text, batch_labels, batch_uv in loader:
            logits, prediction, pointer_logits = model(batch_patches.to(device), batch_text.to(device), grid)
            patch_u = torch.clamp((batch_uv[:, 0] * grid).long(), 0, grid - 1)
            patch_v = torch.clamp((batch_uv[:, 1] * grid).long(), 0, grid - 1)
            pointer_targets = patch_v * grid + patch_u
            loss = (
                torch.nn.functional.cross_entropy(logits, batch_labels.to(device))
                + 1.5 * torch.nn.functional.cross_entropy(pointer_logits, pointer_targets.to(device))
                + 20.0 * torch.nn.functional.mse_loss(prediction, batch_uv.to(device))
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        if epoch in {1, args.epochs} or epoch % 50 == 0:
            train_result = metrics(model, *train_tensors, matrix, manifest["image_size"])
            validation_result = metrics(model, *validation_tensors, matrix, manifest["image_size"])
            history.append({"epoch": epoch, "loss": float(np.mean(losses)), "train": train_result, "validation": validation_result})
            print(f"epoch={epoch} loss={np.mean(losses):.5f} val_intent={validation_result['intent_accuracy']:.3f} val_pointer_rmse_m={validation_result['pointer_rmse_m']:.4f}", flush=True)
    train_result = metrics(model, *train_tensors, matrix, manifest["image_size"])
    validation_result = metrics(model, *validation_tensors, matrix, manifest["image_size"])
    metadata = {
        "version": "clip_patch_pointer_kaggle_v1",
        "method": "frozen CLIP patch-token and language features + lightweight 2D pointer/action-parameter head + structured MuJoCo executor",
        "method_boundary": "The learned head receives top RGB and language. Pixel-to-world conversion uses a fixed camera calibration. MuJoCo object truth is used only for offline labels and scoring, never runtime inference.",
        "dataset_manifest": str(args.dataset_root / "manifest.json"),
        "dataset_content_sha256": manifest["dataset_content_sha256"],
        "clip_model": args.clip_model,
        "frozen_encoder_params": int(sum(parameter.numel() for parameter in clip_model.parameters())),
        "trainable_head_params": int(sum(parameter.numel() for parameter in model.parameters())),
        "task_labels": manifest["task_labels"],
        "workspace_profile": manifest["workspace_profile"],
        "image_size": manifest["image_size"],
        "camera": manifest["camera"],
        "patch_count": int(patches.shape[1]),
        "patch_grid_size": grid,
        "patch_dim": int(patches.shape[2]),
        "text_dim": int(text.shape[1]),
        "hidden_size": args.hidden_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "expanded_samples": len(examples),
        "train_samples": int(train_mask.sum()),
        "validation_samples": int(validation_mask.sum()),
        "train_metrics": train_result,
        "validation_metrics": validation_result,
        "train_time_seconds": time.time() - started,
        "peak_vram_mb": float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "calibration_matrix": np.asarray(manifest["calibration"]["matrix"], dtype=np.float32), "metadata": metadata}, args.output)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps({"model": str(args.output), "history": history, "metadata": metadata}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"model": str(args.output), "metrics": str(args.metrics), "validation": validation_result}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
