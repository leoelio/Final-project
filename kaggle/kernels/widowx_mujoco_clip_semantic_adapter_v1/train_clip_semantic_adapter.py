"""Kaggle-only frozen CLIP semantic adapter training for Core V2 WidowX tasks."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import zipfile

import numpy as np
from PIL import Image


SEED = 20260721
TASK_LABELS = (
    "place_blue_cube_blue_pad",
    "place_blue_cube_red_pad",
    "place_red_cube_red_pad",
    "move_leftmost_cube_to_bowl",
)
MODEL_NAME = "openai/clip-vit-base-patch32"
BOTTLENECK_DIM = 16
EPOCHS = 400
LEARNING_RATE = 3e-3
BATCH_SIZE = 16


def ensure_transformers():
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "transformers==4.44.2", "safetensors>=0.4.3"]
        )
        from transformers import CLIPModel, CLIPProcessor
    return CLIPModel, CLIPProcessor


def extract_source() -> Path:
    input_root = Path("/kaggle/input")
    manifests = sorted(input_root.rglob("manifest.json"))
    if len(manifests) == 1 and (manifests[0].parent / "episodes").is_dir():
        return manifests[0].parent
    archives = sorted(input_root.rglob("*.zip"))
    if len(archives) != 1:
        raise FileNotFoundError(f"expected one source archive or extracted manifest, found archives={archives}, manifests={manifests}")
    root = Path("/kaggle/working/rlds_source")
    with zipfile.ZipFile(archives[0]) as archive:
        archive.extractall(root)
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(f"manifest.json missing after extraction: {root}")
    return root


def select_device(torch) -> tuple[object, dict]:
    advertised_cuda = bool(torch.cuda.is_available())
    capability = None
    if advertised_cuda:
        try:
            capability = list(torch.cuda.get_device_capability(0))
            if capability[0] >= 7:
                torch.cuda.reset_peak_memory_stats()
                return torch.device("cuda"), {"cuda_advertised": True, "capability": capability}
        except Exception as error:
            return torch.device("cpu"), {"cuda_advertised": True, "capability": capability, "fallback_reason": repr(error)}
    return torch.device("cpu"), {
        "cuda_advertised": advertised_cuda,
        "capability": capability,
        "fallback_reason": "Kaggle runtime GPU is unavailable or has unsupported compute capability.",
    }


def load_initial_samples(source_root: Path) -> tuple[list[Image.Image], list[str], np.ndarray, list[dict]]:
    manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    records = sorted(manifest["episode_manifest"], key=lambda row: int(row["episode_id"]))
    images: list[Image.Image] = []
    texts: list[str] = []
    labels: list[int] = []
    audit: list[dict] = []
    for record in records:
        task = str(record["task"])
        if task not in TASK_LABELS:
            continue
        image_path = source_root / str(record["image_paths"][0])
        images.append(Image.open(image_path).convert("RGB"))
        texts.append(str(record["instruction"]))
        labels.append(TASK_LABELS.index(task))
        audit.append(
            {
                "episode_id": int(record["episode_id"]),
                "task": task,
                "instruction": str(record["instruction"]),
                "image_path": str(record["image_paths"][0]),
            }
        )
    if len(images) != 79:
        raise ValueError(f"expected 79 successful episode starts, found {len(images)}")
    return images, texts, np.asarray(labels, dtype=np.int64), audit


def pooled_feature(output):
    return output.pooler_output if hasattr(output, "pooler_output") else output


def encode_features(model, processor, images: list[Image.Image], texts: list[str], torch) -> np.ndarray:
    device = next(model.parameters()).device
    features: list[np.ndarray] = []
    for start in range(0, len(images), BATCH_SIZE):
        inputs = processor(
            images=images[start : start + BATCH_SIZE],
            text=texts[start : start + BATCH_SIZE],
            return_tensors="pt",
            padding=True,
        )
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.no_grad():
            image_features = pooled_feature(model.get_image_features(pixel_values=inputs["pixel_values"]))
            text_features = pooled_feature(
                model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            )
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            text_features = torch.nn.functional.normalize(text_features, dim=-1)
            features.append(torch.cat([image_features, text_features], dim=-1).cpu().numpy().astype(np.float32))
    return np.concatenate(features, axis=0)


def stratified_split(labels: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    validation = np.zeros(len(labels), dtype=bool)
    for task_id in range(len(TASK_LABELS)):
        indices = np.flatnonzero(labels == task_id)
        count = max(1, int(round(len(indices) * 0.2)))
        validation[rng.permutation(indices)[:count]] = True
    return ~validation, validation


def main() -> None:
    started = time.time()
    random.seed(SEED)
    np.random.seed(SEED)
    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    device, device_info = select_device(torch)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    CLIPModel, CLIPProcessor = ensure_transformers()
    source_root = extract_source()
    images, texts, labels, audit = load_initial_samples(source_root)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    clip_model.eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad_(False)
    features = encode_features(clip_model, processor, images, texts, torch)
    rng = np.random.default_rng(SEED)
    train_mask, val_mask = stratified_split(labels, rng)
    x_mean = features[train_mask].mean(axis=0)
    x_std = features[train_mask].std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    normalized = (features - x_mean) / x_std

    class BottleneckIntentAdapter(nn.Module):
        def __init__(self, input_dim: int, bottleneck_dim: int, output_dim: int):
            super().__init__()
            self.down = nn.Linear(input_dim, bottleneck_dim)
            self.up = nn.Linear(bottleneck_dim, output_dim)

        def forward(self, value):
            return self.up(torch.relu(self.down(value)))

    adapter = BottleneckIntentAdapter(normalized.shape[1], BOTTLENECK_DIM, len(TASK_LABELS)).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    x_train = torch.from_numpy(normalized[train_mask]).to(device)
    y_train = torch.from_numpy(labels[train_mask]).to(device)
    x_val = torch.from_numpy(normalized[val_mask]).to(device)
    y_val = torch.from_numpy(labels[val_mask]).to(device)
    history = []
    for epoch in range(1, EPOCHS + 1):
        order = torch.randperm(len(x_train), device=device)
        adapter.train()
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            loss = torch.nn.functional.cross_entropy(adapter(x_train[batch]), y_train[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch == 1 or epoch % 50 == 0 or epoch == EPOCHS:
            adapter.eval()
            with torch.no_grad():
                train_accuracy = float((adapter(x_train).argmax(dim=1) == y_train).float().mean().item())
                val_accuracy = float((adapter(x_val).argmax(dim=1) == y_val).float().mean().item())
            record = {"epoch": epoch, "train_accuracy": train_accuracy, "val_accuracy": val_accuracy}
            history.append(record)
            print(json.dumps(record), flush=True)

    adapter.eval()
    with torch.no_grad():
        predictions = adapter(torch.from_numpy(normalized).to(device)).argmax(dim=1).cpu().numpy()
    trainable_params = int(sum(parameter.numel() for parameter in adapter.parameters()))
    metadata = {
        "version": "kaggle_clip_semantic_adapter_core_v2_v1",
        "method": "frozen_clip_bottleneck_semantic_adapter",
        "method_boundary": "Frozen CLIP task-intent adapter plus local structured waypoint executor; not an end-to-end VLA and not OpenVLA LoRA.",
        "clip_model": MODEL_NAME,
        "frozen_encoder_params": int(sum(parameter.numel() for parameter in clip_model.parameters())),
        "trainable_adapter_params": trainable_params,
        "architecture": "relu_bottleneck_classifier",
        "bottleneck_dim": BOTTLENECK_DIM,
        "task_labels": list(TASK_LABELS),
        "episode_samples": int(len(labels)),
        "train_samples": int(train_mask.sum()),
        "validation_samples": int(val_mask.sum()),
        "train_accuracy": float((predictions[train_mask] == labels[train_mask]).mean()),
        "validation_accuracy": float((predictions[val_mask] == labels[val_mask]).mean()),
        "source_version": "widowx_mujoco_rlds_source_v1",
        "source_steps": 2528,
        "device": str(device),
        "cuda_advertised": device_info["cuda_advertised"],
        "gpu_execution": device.type == "cuda",
        "cuda_capability": device_info["capability"],
        "device_fallback_reason": device_info.get("fallback_reason"),
        "peak_vram_mb": float(torch.cuda.max_memory_allocated() / (1024**2)) if device.type == "cuda" else 0.0,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "train_time_seconds": time.time() - started,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    output = Path("/kaggle/working")
    np.savez_compressed(
        output / "kaggle_clip_semantic_adapter_core_v2_v1.npz",
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        down_weight=adapter.down.weight.detach().cpu().numpy().T.astype(np.float32),
        down_bias=adapter.down.bias.detach().cpu().numpy().astype(np.float32),
        up_weight=adapter.up.weight.detach().cpu().numpy().T.astype(np.float32),
        up_bias=adapter.up.bias.detach().cpu().numpy().astype(np.float32),
        metadata=json.dumps(metadata, ensure_ascii=False),
    )
    (output / "kaggle_clip_semantic_adapter_core_v2_v1_metrics.json").write_text(
        json.dumps({"metadata": metadata, "history": history, "samples": audit}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output / "kaggle_clip_semantic_adapter_core_v2_v1_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode_id", "task", "instruction", "split", "prediction", "correct"])
        writer.writeheader()
        for index, sample in enumerate(audit):
            writer.writerow(
                {
                    "episode_id": sample["episode_id"],
                    "task": sample["task"],
                    "instruction": sample["instruction"],
                    "split": "validation" if val_mask[index] else "train",
                    "prediction": TASK_LABELS[int(predictions[index])],
                    "correct": int(predictions[index] == labels[index]),
                }
            )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
