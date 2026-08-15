from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

try:
    from torch_runtime import ensure_torch_path  # noqa: E402
    from vlm_runtime import ensure_vlm_path  # noqa: E402
except ModuleNotFoundError:
    def ensure_torch_path() -> None:
        return None

    def ensure_vlm_path() -> None:
        return None

ensure_torch_path()
ensure_vlm_path()

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from clip_lora_utils import inject_vision_lora, load_lora_state_dict, lora_parameters, lora_state_dict  # noqa: E402

try:
    from run_clip_action_head import load_clip  # noqa: E402
except ModuleNotFoundError:
    from transformers import CLIPModel, CLIPProcessor  # noqa: E402

    def load_clip(model_name: str):
        return CLIPModel.from_pretrained(model_name), CLIPProcessor.from_pretrained(model_name)

try:
    from train_clip_patch_pointer import PatchPointerHead  # noqa: E402
except ModuleNotFoundError:
    import torch.nn as nn  # noqa: E402

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
    parser = argparse.ArgumentParser(description="Train a CLIP visual-attention LoRA plus 2D patch-pointer head on the Kaggle spatial pack.")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "kaggle_patch_pointer_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "clip_lora_patch_pointer" / "clip_lora_patch_pointer_core_v2_v1.pt")
    parser.add_argument("--metrics", type=Path, default=ROOT / "outputs" / "clip_lora_patch_pointer" / "clip_lora_patch_pointer_core_v2_v1.json")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--max-train-scenes", type=int, default=0, help="Positive values are smoke-test only and must not be used for final evaluation.")
    parser.add_argument("--max-validation-scenes", type=int, default=0, help="Positive values are smoke-test only and must not be used for final evaluation.")
    parser.add_argument("--run-version", default="clip_lora_patch_pointer_core_v2_v1")
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else value if value != "auto" else "cpu")


def read_examples(root: Path, max_train: int, max_validation: int) -> tuple[list[dict], dict]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    scenes = [json.loads(line) for line in (root / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    selected: list[dict] = []
    for split, limit in (("train", max_train), ("validation", max_validation)):
        group = [scene for scene in scenes if scene["split"] == split]
        selected.extend(group[:limit] if limit > 0 else group)
    if not selected:
        raise ValueError("no dataset scenes selected")
    examples = [{**scene, "instruction": instruction} for scene in selected for instruction in scene["instruction_variants"]]
    return examples, manifest


def prepare_pixels(processor, root: Path, examples: list[dict], batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    scenes = {example["id"]: example for example in examples}
    scene_rows = list(scenes.values())
    pixels: list[torch.Tensor] = []
    for start in range(0, len(scene_rows), batch_size):
        batch = scene_rows[start : start + batch_size]
        images = []
        for row in batch:
            with Image.open(root / row["image"]) as image:
                images.append(image.convert("RGB"))
        pixels.append(processor(images=images, return_tensors="pt")["pixel_values"].cpu())
    index_by_id = {row["id"]: index for index, row in enumerate(scene_rows)}
    return torch.cat(pixels), torch.tensor([index_by_id[row["id"]] for row in examples], dtype=torch.long)


def encode_text(clip_model, processor, examples: list[dict], batch_size: int, device: torch.device) -> torch.Tensor:
    features: list[torch.Tensor] = []
    clip_model.eval()
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        inputs = processor(text=[row["instruction"] for row in batch], return_tensors="pt", padding=True)
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.no_grad():
            text = clip_model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            if hasattr(text, "pooler_output"):
                text = text.pooler_output
            features.append(torch.nn.functional.normalize(text, dim=-1).cpu())
    return torch.cat(features)


def forward(clip_model, head: PatchPointerHead, pixels: torch.Tensor, text: torch.Tensor, grid_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    patches = clip_model.vision_model(pixel_values=pixels).last_hidden_state[:, 1:, :]
    return head(patches, text, grid_size)


def evaluate(clip_model, head: PatchPointerHead, pixels: torch.Tensor, loader: DataLoader, matrix: torch.Tensor, image_size: int, grid_size: int, device: torch.device) -> dict[str, float]:
    clip_model.eval()
    head.eval()
    correct = 0
    count = 0
    errors: list[torch.Tensor] = []
    with torch.no_grad():
        for image_indices, text, labels, target_uv in loader:
            text = text.to(device)
            labels = labels.to(device)
            target_uv = target_uv.to(device)
            logits, predicted_uv, _ = forward(clip_model, head, pixels[image_indices].to(device), text, grid_size)
            predicted_world = (predicted_uv * float(image_size - 1)) @ matrix[:, :2].T + matrix[:, 2]
            target_world = (target_uv * float(image_size - 1)) @ matrix[:, :2].T + matrix[:, 2]
            errors.append(torch.linalg.vector_norm(predicted_world - target_world, dim=1).cpu())
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            count += len(labels)
    error = torch.cat(errors)
    return {
        "intent_accuracy": correct / count,
        "pointer_mae_m": float(error.mean().item()),
        "pointer_rmse_m": float(torch.sqrt(torch.mean(error.square())).item()),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    examples, manifest = read_examples(args.dataset_root, args.max_train_scenes, args.max_validation_scenes)
    train_mask = np.asarray([row["split"] == "train" for row in examples], dtype=bool)
    validation_mask = ~train_mask
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("selected dataset must include both training and validation scenes")

    clip_model, processor = load_clip(args.clip_model)
    frozen_encoder_params = int(sum(parameter.numel() for parameter in clip_model.parameters()))
    for parameter in clip_model.parameters():
        parameter.requires_grad_(False)
    total_layers = len(clip_model.vision_model.encoder.layers)
    layer_indices = list(range(total_layers - args.lora_layers, total_layers))
    inject_vision_lora(clip_model, layer_indices, args.lora_rank, args.lora_alpha)
    for parameter in lora_parameters(clip_model):
        parameter.requires_grad_(True)
    clip_model = clip_model.to(device)

    all_pixels, image_indices = prepare_pixels(processor, args.dataset_root, examples, args.feature_batch_size)
    text = encode_text(clip_model, processor, examples, args.feature_batch_size, device)
    labels = torch.tensor([int(row["task_label"]) for row in examples], dtype=torch.long)
    target_uv = torch.tensor([row["source_pixel_uv_normalized"] for row in examples], dtype=torch.float32)
    head = PatchPointerHead(768, text.shape[1], args.hidden_size, len(manifest["task_labels"])).to(device)
    trainable = [*lora_parameters(clip_model), *head.parameters()]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    train_data = TensorDataset(image_indices[train_mask], text[train_mask], labels[train_mask], target_uv[train_mask])
    validation_data = TensorDataset(image_indices[validation_mask], text[validation_mask], labels[validation_mask], target_uv[validation_mask])
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=device.type == "cuda")
    train_score_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False)
    matrix = torch.tensor(manifest["calibration"]["matrix"], dtype=torch.float32, device=device)
    grid_size = 7
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    history = []
    best_epoch = 0
    best_rmse = float("inf")
    best_head = None
    best_lora = None
    for epoch in range(1, args.epochs + 1):
        clip_model.train()
        head.train()
        losses = []
        for image_batch, text_batch, label_batch, uv_batch in train_loader:
            image_batch = all_pixels[image_batch].to(device)
            text_batch = text_batch.to(device)
            label_batch = label_batch.to(device)
            uv_batch = uv_batch.to(device)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits, prediction, pointer_logits = forward(clip_model, head, image_batch, text_batch, grid_size)
                patch_u = torch.clamp((uv_batch[:, 0] * grid_size).long(), 0, grid_size - 1)
                patch_v = torch.clamp((uv_batch[:, 1] * grid_size).long(), 0, grid_size - 1)
                pointer_targets = patch_v * grid_size + patch_u
                loss = (
                    torch.nn.functional.cross_entropy(logits, label_batch)
                    + 1.5 * torch.nn.functional.cross_entropy(pointer_logits, pointer_targets)
                    + 20.0 * torch.nn.functional.mse_loss(prediction, uv_batch)
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().item()))
        if epoch == 1 or epoch == args.epochs or epoch % args.log_interval == 0:
            train_metrics = evaluate(clip_model, head, all_pixels, train_score_loader, matrix, manifest["image_size"], grid_size, device)
            validation_metrics = evaluate(clip_model, head, all_pixels, validation_loader, matrix, manifest["image_size"], grid_size, device)
            history.append({"epoch": epoch, "loss": float(np.mean(losses)), "train": train_metrics, "validation": validation_metrics})
            if validation_metrics["pointer_rmse_m"] < best_rmse:
                best_epoch = epoch
                best_rmse = validation_metrics["pointer_rmse_m"]
                best_head = copy.deepcopy(head.state_dict())
                best_lora = lora_state_dict(clip_model)
            print(f"epoch={epoch} loss={np.mean(losses):.5f} val_intent={validation_metrics['intent_accuracy']:.3f} val_pointer_rmse_m={validation_metrics['pointer_rmse_m']:.4f}", flush=True)
    if best_head is None or best_lora is None:
        raise RuntimeError("no validation checkpoint was recorded")
    head.load_state_dict(best_head)
    load_lora_state_dict(clip_model, best_lora)
    final_train = evaluate(clip_model, head, all_pixels, train_score_loader, matrix, manifest["image_size"], grid_size, device)
    final_validation = evaluate(clip_model, head, all_pixels, validation_loader, matrix, manifest["image_size"], grid_size, device)
    metadata = {
        "version": args.run_version,
        "method": "CLIP visual-attention LoRA (last two q_proj/v_proj layers) + language-conditioned 2D patch pointer + structured MuJoCo executor",
        "method_boundary": "Runtime uses top RGB, language, a fixed camera-plane calibration, and model predictions. MuJoCo object truth is used only for offline labels and scoring. This is CLIP visual LoRA with a lightweight action-parameter head, not end-to-end VLA or OpenVLA.",
        "offline_truth_boundary": "MuJoCo object positions supervise source-pixel labels and calculate offline error only; runtime does not query object positions.",
        "dataset_manifest": str(args.dataset_root / "manifest.json"),
        "dataset_content_sha256": manifest["dataset_content_sha256"],
        "clip_model": args.clip_model,
        "frozen_encoder_params": frozen_encoder_params,
        "trainable_lora_params": int(sum(parameter.numel() for parameter in lora_parameters(clip_model))),
        "trainable_head_params": int(sum(parameter.numel() for parameter in head.parameters())),
        "trainable_total_params": int(sum(parameter.numel() for parameter in trainable)),
        "lora_config": {"layer_indices": layer_indices, "rank": args.lora_rank, "alpha": args.lora_alpha, "target_modules": ["q_proj", "v_proj"]},
        "task_labels": manifest["task_labels"],
        "workspace_profile": manifest["workspace_profile"],
        "image_size": manifest["image_size"],
        "camera": manifest["camera"],
        "patch_count": 49,
        "patch_grid_size": grid_size,
        "patch_dim": 768,
        "text_dim": int(text.shape[1]),
        "hidden_size": args.hidden_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "amp": amp_enabled,
        "expanded_samples": len(examples),
        "train_samples": int(train_mask.sum()),
        "validation_samples": int(validation_mask.sum()),
        "train_metrics": final_train,
        "validation_metrics": final_validation,
        "selected_checkpoint": "lowest validation pointer RMSE at logged checkpoints",
        "best_epoch": best_epoch,
        "best_validation_pointer_rmse_m": best_rmse,
        "train_time_seconds": time.time() - started,
        "peak_vram_mb": float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "lora_state_dict": lora_state_dict(clip_model), "calibration_matrix": np.asarray(manifest["calibration"]["matrix"], dtype=np.float32), "metadata": metadata}, args.output)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps({"model": str(args.output), "history": history, "metadata": metadata}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"model": str(args.output), "metrics": str(args.metrics), "validation": final_validation}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
