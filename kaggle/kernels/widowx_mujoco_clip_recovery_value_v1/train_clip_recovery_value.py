"""Kaggle-only frozen CLIP recovery-value adapter for real MuJoCo failure states."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
from PIL import Image


SEED = 20260729
MODEL_NAME = "openai/clip-vit-base-patch32"
HIDDEN_SIZE = 8
EPOCHS = 300
BATCH_SIZE = 16
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-3


def ensure_transformers():
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "transformers==4.44.2", "safetensors>=0.4.3"])
        from transformers import CLIPModel, CLIPProcessor
    return CLIPModel, CLIPProcessor


def select_device(torch):
    advertised = bool(torch.cuda.is_available())
    capability = None
    if advertised:
        try:
            capability = list(torch.cuda.get_device_capability(0))
            if capability[0] >= 7:
                torch.cuda.reset_peak_memory_stats()
                return torch.device("cuda"), {"cuda_advertised": True, "capability": capability}
        except Exception as error:
            return torch.device("cpu"), {"cuda_advertised": True, "capability": capability, "fallback_reason": repr(error)}
    return torch.device("cpu"), {"cuda_advertised": advertised, "capability": capability, "fallback_reason": "CUDA unavailable or incompatible"}


def clip_tensor(output):
    return output.pooler_output if hasattr(output, "pooler_output") else output


def source_path() -> Path:
    matches = sorted(Path("/kaggle/input").rglob("clip_recovery_training_v1.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one clip_recovery_training_v1.npz, found {matches}")
    return matches[0]


def encode_images(model, processor, images: np.ndarray, torch, batch_size: int) -> np.ndarray:
    device = next(model.parameters()).device
    output = []
    for start in range(0, len(images), batch_size):
        pil = [Image.fromarray(item.astype(np.uint8), mode="RGB") for item in images[start : start + batch_size]]
        inputs = processor(images=pil, return_tensors="pt")
        with torch.no_grad():
            features = clip_tensor(model.get_image_features(pixel_values=inputs["pixel_values"].to(device)))
            features = torch.nn.functional.normalize(features, dim=-1)
        output.append(features.cpu().numpy().astype(np.float32))
    return np.concatenate(output, axis=0)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1
    per_class = []
    recalls = []
    for index, name in enumerate(class_names):
        support = int(matrix[index].sum())
        predicted = int(matrix[:, index].sum())
        true_positive = int(matrix[index, index])
        recall = true_positive / support if support else None
        if recall is not None:
            recalls.append(recall)
        per_class.append({"class": name, "support": support, "precision": true_positive / predicted if predicted else None, "recall": recall})
    return {"accuracy": float(np.mean(y_true == y_pred)), "balanced_accuracy_present_classes": float(np.mean(recalls)) if recalls else None, "confusion_matrix": matrix.tolist(), "per_class": per_class}


def train_head(features: np.ndarray, labels: np.ndarray, splits: np.ndarray, view: str, torch, nn, device, class_names: list[str]):
    x = features[:, :512] if view == "top" else features
    train_mask, test_mask = splits == 0, splits == 1
    train_x_raw, test_x_raw = x[train_mask], x[test_mask]
    train_y, test_y = labels[train_mask], labels[test_mask]
    counts = np.bincount(train_y, minlength=2).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError(f"training class missing: {counts.tolist()}")
    mean, std = train_x_raw.mean(axis=0), train_x_raw.std(axis=0)
    std[std < 1e-6] = 1.0
    train_x = ((train_x_raw - mean) / std).astype(np.float32)
    test_x = ((test_x_raw - mean) / std).astype(np.float32)

    class Adapter(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.down = nn.Linear(input_dim, HIDDEN_SIZE)
            self.up = nn.Linear(HIDDEN_SIZE, 2)

        def forward(self, value):
            return self.up(torch.relu(self.down(value)))

    model = Adapter(train_x.shape[1]).to(device)
    weights = torch.tensor(len(train_y) / (2.0 * counts), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    x_tensor, y_tensor = torch.from_numpy(train_x).to(device), torch.from_numpy(train_y).to(device)
    history = []
    for epoch in range(1, EPOCHS + 1):
        order = torch.randperm(len(train_y), device=device)
        losses = []
        model.train()
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            loss = torch.nn.functional.cross_entropy(model(x_tensor[batch]), y_tensor[batch], weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        if epoch == 1 or epoch % 50 == 0 or epoch == EPOCHS:
            history.append({"epoch": epoch, "weighted_train_loss": float(np.mean(losses))})
    model.eval()
    with torch.no_grad():
        train_pred = model(torch.from_numpy(train_x).to(device)).argmax(dim=1).cpu().numpy()
        test_pred = model(torch.from_numpy(test_x).to(device)).argmax(dim=1).cpu().numpy()
    artifact = {
        "down_weight": model.down.weight.detach().cpu().numpy().T.astype(np.float32),
        "down_bias": model.down.bias.detach().cpu().numpy().astype(np.float32),
        "up_weight": model.up.weight.detach().cpu().numpy().T.astype(np.float32),
        "up_bias": model.up.bias.detach().cpu().numpy().astype(np.float32),
        "x_mean": mean.astype(np.float32),
        "x_std": std.astype(np.float32),
    }
    result = {
        "view": view,
        "trainable_adapter_params": int(sum(parameter.numel() for parameter in model.parameters())),
        "train_metrics": classification_metrics(train_y, train_pred, class_names),
        "test_metrics": classification_metrics(test_y, test_pred, class_names),
        "class_weights": weights.cpu().numpy().tolist(),
        "history": history,
    }
    return artifact, result


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
    with np.load(source_path(), allow_pickle=False) as data:
        top_images = data["top_images"].astype(np.uint8)
        front_images = data["front_images"].astype(np.uint8)
        labels = data["labels"].astype(np.int64)
        splits = data["splits"].astype(np.int8)
        source_metadata = json.loads(data["metadata"].item())
    class_names = list(source_metadata["class_names"])
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    clip = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    clip.eval()
    for parameter in clip.parameters():
        parameter.requires_grad_(False)
    top = encode_images(clip, processor, top_images, torch, BATCH_SIZE)
    views = {"top": top, "top_front": np.concatenate([top, encode_images(clip, processor, front_images, torch, BATCH_SIZE)], axis=1).astype(np.float32)}
    output = Path("/kaggle/working")
    results = []
    frozen_params = int(sum(parameter.numel() for parameter in clip.parameters()))
    for view, features in views.items():
        artifact, result = train_head(features, labels, splits, view, torch, nn, device, class_names)
        metadata = {
            "version": "kaggle_clip_recovery_value_v1",
            "method": "frozen_clip_visual_encoder + bottleneck retry-value adapter",
            "method_boundary": "The adapter is queried only after RGB terminal failure and visual source re-localization. It does not output actions and is not end-to-end VLA or OpenVLA LoRA.",
            "clip_model": MODEL_NAME,
            "view": view,
            "class_names": class_names,
            "feature_dim": int(features.shape[1]),
            "hidden_size": HIDDEN_SIZE,
            "frozen_encoder_params": frozen_params,
            "trainable_adapter_params": result["trainable_adapter_params"],
            "train_samples": int((splits == 0).sum()),
            "test_samples": int((splits == 1).sum()),
            "class_weights": result["class_weights"],
            "train_metrics": result["train_metrics"],
            "test_metrics": result["test_metrics"],
            "device": str(device),
            "cuda_advertised": device_info["cuda_advertised"],
            "cuda_capability": device_info["capability"],
            "gpu_execution": device.type == "cuda",
            "device_fallback_reason": device_info.get("fallback_reason"),
            "peak_vram_mb": float(torch.cuda.max_memory_allocated() / (1024**2)) if device.type == "cuda" else 0.0,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
            "source_dataset_version": source_metadata["version"],
            "runtime_boundary": source_metadata["runtime_boundary"],
            "train_time_seconds": time.time() - started,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        artifact["metadata"] = json.dumps(metadata, ensure_ascii=False)
        model_path = output / f"kaggle_clip_recovery_value_{view}_v1.npz"
        np.savez_compressed(model_path, **artifact)
        results.append({"view": view, "model": model_path.name, "metadata": metadata, "history": result["history"]})
        print(json.dumps({"view": view, "test": metadata["test_metrics"], "model": model_path.name}, ensure_ascii=False), flush=True)
    (output / "kaggle_clip_recovery_value_v1_metrics.json").write_text(
        json.dumps({"version": "kaggle_clip_recovery_value_v1", "source_samples": int(len(labels)), "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
