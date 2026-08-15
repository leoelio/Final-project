from __future__ import annotations

import argparse
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

from torch_runtime import ensure_torch_path  # noqa: E402
from vlm_runtime import ensure_vlm_path  # noqa: E402

ensure_torch_path()
ensure_vlm_path()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import CLIPModel, CLIPProcessor  # noqa: E402
from widowx_env import TASKS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a frozen CLIP visual recovery-value head from real MuJoCo failure states.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "clip_recovery_value")
    parser.add_argument("--views", default="top,top_front")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--include-text", action="store_true", help="Append frozen CLIP features for each task's canonical instruction.")
    parser.add_argument("--include-proprio", action="store_true", help="Append robot joint/actuator observations supplied by the dataset.")
    parser.add_argument("--metrics", type=Path, default=None)
    return parser.parse_args()


def selected_views(value: str) -> list[str]:
    views = [item.strip() for item in value.split(",") if item.strip()]
    if not views or any(item not in {"top", "top_front"} for item in views):
        raise ValueError(f"views must be top and/or top_front, got {value}")
    return views


def device_info() -> tuple[torch.device, dict]:
    if torch.cuda.is_available():
        try:
            capability = list(torch.cuda.get_device_capability(0))
            if capability[0] >= 7:
                torch.cuda.reset_peak_memory_stats()
                return torch.device("cuda"), {"cuda_advertised": True, "capability": capability}
            return torch.device("cpu"), {"cuda_advertised": True, "capability": capability, "fallback_reason": "CUDA capability below the CLIP runtime requirement."}
        except Exception as error:
            return torch.device("cpu"), {"cuda_advertised": True, "capability": None, "fallback_reason": repr(error)}
    return torch.device("cpu"), {"cuda_advertised": False, "capability": None, "fallback_reason": "CUDA unavailable"}


def clip_tensor(output: object) -> torch.Tensor:
    return output.pooler_output if hasattr(output, "pooler_output") else output  # type: ignore[return-value]


def encode_images(model: CLIPModel, processor: CLIPProcessor, images: np.ndarray, batch_size: int) -> np.ndarray:
    device = next(model.parameters()).device
    parts = []
    for start in range(0, len(images), batch_size):
        pil = [Image.fromarray(item.astype(np.uint8), mode="RGB") for item in images[start : start + batch_size]]
        inputs = processor(images=pil, return_tensors="pt")
        with torch.no_grad():
            features = clip_tensor(model.get_image_features(pixel_values=inputs["pixel_values"].to(device)))
            features = torch.nn.functional.normalize(features, dim=-1)
        parts.append(features.cpu().numpy().astype(np.float32))
    return np.concatenate(parts, axis=0)


def encode_texts(model: CLIPModel, processor: CLIPProcessor, instructions: list[str], batch_size: int) -> np.ndarray:
    device = next(model.parameters()).device
    parts = []
    for start in range(0, len(instructions), batch_size):
        inputs = processor(text=instructions[start : start + batch_size], padding=True, return_tensors="pt")
        with torch.no_grad():
            features = clip_tensor(
                model.get_text_features(
                    input_ids=inputs["input_ids"].to(device),
                    attention_mask=inputs["attention_mask"].to(device),
                )
            )
            features = torch.nn.functional.normalize(features, dim=-1)
        parts.append(features.cpu().numpy().astype(np.float32))
    return np.concatenate(parts, axis=0)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1
    classes = []
    recalls = []
    for index, name in enumerate(class_names):
        support = int(matrix[index].sum())
        predicted = int(matrix[:, index].sum())
        true_positive = int(matrix[index, index])
        recall = true_positive / support if support else None
        if recall is not None:
            recalls.append(recall)
        classes.append({"class": name, "support": support, "precision": true_positive / predicted if predicted else None, "recall": recall})
    return {"accuracy": float(np.mean(y_true == y_pred)), "balanced_accuracy_present_classes": float(np.mean(recalls)) if recalls else None, "confusion_matrix": matrix.tolist(), "per_class": classes}


class RecoveryAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.down = nn.Linear(input_dim, hidden_size)
        self.up = nn.Linear(hidden_size, output_size)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.up(torch.relu(self.down(value)))


def train_one(features: np.ndarray, labels: np.ndarray, splits: np.ndarray, view: str, args: argparse.Namespace, device: torch.device, device_meta: dict, class_names: list[str], frozen_encoder_params: int) -> tuple[dict, dict]:
    x = features[:, :512] if view == "top" and not args.include_text and not args.include_proprio else features
    train_mask = splits == 0
    test_mask = splits == 1
    train_x_raw, test_x_raw = x[train_mask], x[test_mask]
    train_y, test_y = labels[train_mask], labels[test_mask]
    counts = np.bincount(train_y, minlength=len(class_names)).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError(f"missing training class counts={counts.tolist()}")
    mean = train_x_raw.mean(axis=0)
    std = train_x_raw.std(axis=0)
    std[std < 1e-6] = 1.0
    train_x = ((train_x_raw - mean) / std).astype(np.float32)
    test_x = ((test_x_raw - mean) / std).astype(np.float32)
    model = RecoveryAdapter(train_x.shape[1], args.hidden_size, len(class_names)).to(device)
    weight = torch.tensor(len(train_y) / (len(class_names) * counts), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    x_tensor = torch.from_numpy(train_x).to(device)
    y_tensor = torch.from_numpy(train_y).to(device)
    history = []
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(train_y), device=device)
        model.train()
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            loss = torch.nn.functional.cross_entropy(model(x_tensor[batch]), y_tensor[batch], weight=weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            history.append({"epoch": epoch, "weighted_train_loss": float(np.mean(losses))})
    model.eval()
    with torch.no_grad():
        train_pred = model(torch.from_numpy(train_x).to(device)).argmax(dim=1).cpu().numpy()
        test_pred = model(torch.from_numpy(test_x).to(device)).argmax(dim=1).cpu().numpy()
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    metadata = {
        "version": "clip_recovery_value_v1",
        "method": "frozen_clip_visual_language_encoder + bottleneck retry-value adapter" if args.include_text else "frozen_clip_visual_encoder + bottleneck retry-value adapter",
        "method_boundary": f"The adapter predicts one of {class_names} only after an RGB terminal failure and visual source re-localization. It does not output continuous robot actions and is not end-to-end VLA or OpenVLA LoRA.",
        "clip_model": args.clip_model,
        "view": view,
        "class_names": class_names,
        "output_classes": len(class_names),
        "feature_dim": int(train_x.shape[1]),
        "uses_instruction": bool(args.include_text),
        "uses_robot_proprioception": bool(args.include_proprio),
        "input_modalities": (["top_rgb", "front_rgb"] if view == "top_front" else ["top_rgb"]) + (["instruction"] if args.include_text else []) + (["robot_proprioception"] if args.include_proprio else []),
        "hidden_size": args.hidden_size,
        "frozen_encoder_params": frozen_encoder_params,
        "trainable_adapter_params": parameter_count,
        "train_samples": int(len(train_y)),
        "test_samples": int(len(test_y)),
        "class_weights": weight.cpu().numpy().tolist(),
        "train_metrics": metrics(train_y, train_pred, class_names),
        "test_metrics": metrics(test_y, test_pred, class_names),
        "device": str(device),
        "cuda_advertised": device_meta["cuda_advertised"],
        "cuda_capability": device_meta["capability"],
        "gpu_execution": device.type == "cuda",
        "device_fallback_reason": device_meta.get("fallback_reason"),
        "peak_vram_mb": float(torch.cuda.max_memory_allocated() / (1024**2)) if device.type == "cuda" else 0.0,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    artifact = {
        "down_weight": model.down.weight.detach().cpu().numpy().T.astype(np.float32),
        "down_bias": model.down.bias.detach().cpu().numpy().astype(np.float32),
        "up_weight": model.up.weight.detach().cpu().numpy().T.astype(np.float32),
        "up_bias": model.up.bias.detach().cpu().numpy().astype(np.float32),
        "x_mean": mean.astype(np.float32),
        "x_std": std.astype(np.float32),
        "metadata": json.dumps(metadata, ensure_ascii=False),
    }
    return artifact, {"view": view, "metadata": metadata, "history": history}


def main() -> None:
    args = parse_args()
    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device, device_meta = device_info()
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    with np.load(args.dataset, allow_pickle=False) as data:
        top_images = data["top_images"].astype(np.uint8)
        front_images = data["front_images"].astype(np.uint8)
        labels = data["labels"].astype(np.int64)
        splits = data["splits"].astype(np.int8)
        task_names = data["task_names"].astype(str)
        proprio = data["proprio"].astype(np.float32) if args.include_proprio else None
        source_metadata = json.loads(data["metadata"].item())
    if args.include_proprio and proprio is None:
        raise ValueError("--include-proprio requires a dataset with a proprio array")
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    clip = CLIPModel.from_pretrained(args.clip_model).to(device)
    clip.eval()
    for parameter in clip.parameters():
        parameter.requires_grad_(False)
    top_features = encode_images(clip, processor, top_images, args.batch_size)
    front_features = encode_images(clip, processor, front_images, args.batch_size)
    visual_features = {"top": top_features, "top_front": np.concatenate([top_features, front_features], axis=1).astype(np.float32)}
    if args.include_text:
        instructions = [TASKS[name].instruction for name in task_names]
        text_features = encode_texts(clip, processor, instructions, args.batch_size)
        features = {view: np.concatenate([value, text_features], axis=1).astype(np.float32) for view, value in visual_features.items()}
    else:
        features = visual_features
    if args.include_proprio:
        features = {view: np.concatenate([value, proprio], axis=1).astype(np.float32) for view, value in features.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for view in selected_views(args.views):
        artifact, result = train_one(
            features[view],
            labels,
            splits,
            view,
            args,
            device,
            device_meta,
            list(source_metadata["class_names"]),
            int(sum(parameter.numel() for parameter in clip.parameters())),
        )
        result["metadata"]["source_dataset"] = str(args.dataset)
        result["metadata"]["source_dataset_version"] = source_metadata["version"]
        result["metadata"]["runtime_boundary"] = source_metadata["runtime_boundary"]
        result["metadata"]["proprio_spec"] = source_metadata.get("proprio_spec") if args.include_proprio else None
        result["metadata"]["train_time_seconds"] = time.time() - started
        artifact["metadata"] = json.dumps(result["metadata"], ensure_ascii=False)
        model_path = args.output_dir / f"clip_recovery_value_{view}_v1.npz"
        np.savez_compressed(model_path, **artifact)
        result["model"] = str(model_path)
        results.append(result)
        print(json.dumps({"view": view, "model": str(model_path), "test": result["metadata"]["test_metrics"]}, ensure_ascii=False), flush=True)
    metrics_path = args.metrics or args.output_dir / "clip_recovery_value_training_metrics_v1.json"
    metrics_path.write_text(json.dumps({"version": "clip_recovery_value_v1", "dataset": str(args.dataset), "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"metrics_path: {metrics_path}")


if __name__ == "__main__":
    main()
