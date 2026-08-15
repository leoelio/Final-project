from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from widowx_env.multiview_features import feature_indices  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact RGB terminal/recovery value head on seed-disjoint MuJoCo data.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "multiview_value_head")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--views", default="top,top_front", help="Comma-separated ablations: top,top_front")
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--metrics", type=Path, default=None)
    return parser.parse_args()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def forward(model: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.maximum(0.0, x @ model["w0"] + model["b0"])
    return hidden @ model["w1"] + model["b1"], hidden


def init_model(input_dim: int, classes: int, hidden_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "w0": (rng.standard_normal((input_dim, hidden_size)) * np.sqrt(2.0 / input_dim)).astype(np.float32),
        "b0": np.zeros(hidden_size, dtype=np.float32),
        "w1": (rng.standard_normal((hidden_size, classes)) * np.sqrt(2.0 / hidden_size)).astype(np.float32),
        "b1": np.zeros(classes, dtype=np.float32),
    }


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    classes: int,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], list[dict], np.ndarray]:
    counts = np.bincount(y_train, minlength=classes).astype(np.float32)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"training split is missing classes {missing}; collect labels before training")
    class_weights = (len(y_train) / (classes * counts)).astype(np.float32)
    model = init_model(x_train.shape[1], classes, hidden_size, rng)
    moments = {f"m_{name}": np.zeros_like(value) for name, value in model.items()}
    moments.update({f"v_{name}": np.zeros_like(value) for name, value in model.items()})
    history: list[dict] = []
    step = 0
    for epoch in range(1, epochs + 1):
        losses = []
        order = rng.permutation(len(y_train))
        for start in range(0, len(y_train), batch_size):
            batch = order[start : start + batch_size]
            x = x_train[batch]
            y = y_train[batch]
            logits, hidden = forward(model, x)
            probabilities = softmax(logits)
            sample_weights = class_weights[y]
            loss = -np.sum(sample_weights * np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-8, 1.0))) / sample_weights.sum()
            loss += 0.5 * weight_decay * (float(np.sum(model["w0"] ** 2)) + float(np.sum(model["w1"] ** 2)))
            losses.append(float(loss))
            grad_logits = probabilities
            grad_logits[np.arange(len(y)), y] -= 1.0
            grad_logits *= (sample_weights / sample_weights.sum())[:, None]
            grads = {
                "w1": hidden.T @ grad_logits + weight_decay * model["w1"],
                "b1": grad_logits.sum(axis=0),
            }
            grad_hidden = grad_logits @ model["w1"].T
            grad_hidden[hidden <= 0.0] = 0.0
            grads["w0"] = x.T @ grad_hidden + weight_decay * model["w0"]
            grads["b0"] = grad_hidden.sum(axis=0)
            step += 1
            for name in model:
                moments[f"m_{name}"] = 0.9 * moments[f"m_{name}"] + 0.1 * grads[name]
                moments[f"v_{name}"] = 0.999 * moments[f"v_{name}"] + 0.001 * (grads[name] ** 2)
                m_hat = moments[f"m_{name}"] / (1.0 - 0.9**step)
                v_hat = moments[f"v_{name}"] / (1.0 - 0.999**step)
                model[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            history.append({"epoch": epoch, "weighted_train_loss": float(np.mean(losses))})
    return model, history, class_weights


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    classes = len(class_names)
    confusion = np.zeros((classes, classes), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        confusion[int(actual), int(predicted)] += 1
    per_class = []
    recalls = []
    for index, name in enumerate(class_names):
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        true_positive = int(confusion[index, index])
        precision = true_positive / predicted if predicted else None
        recall = true_positive / support if support else None
        if recall is not None:
            recalls.append(recall)
        per_class.append({"class": name, "support": support, "precision": precision, "recall": recall})
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy_present_classes": float(np.mean(recalls)) if recalls else None,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def selected_views(value: str) -> list[str]:
    views = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in views if item not in {"top", "top_front"}]
    if not views or unknown:
        raise ValueError(f"views must be top and/or top_front, got {value}")
    return views


def main() -> None:
    args = parse_args()
    with np.load(args.dataset, allow_pickle=False) as data:
        features = data["features"].astype(np.float32)
        labels = data["labels"].astype(np.int64)
        splits = data["splits"].astype(np.int8)
        metadata = json.loads(data["metadata"].item())
    class_names = [str(item) for item in metadata["class_names"]]
    train_mask = splits == 0
    test_mask = splits == 1
    if not train_mask.any() or not test_mask.any():
        raise ValueError("dataset must include explicit train and test samples")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for view in selected_views(args.views):
        indices = feature_indices(metadata["feature_spec"], view)
        x_train_raw = features[train_mask][:, indices]
        x_test_raw = features[test_mask][:, indices]
        x_mean = x_train_raw.mean(axis=0)
        x_std = x_train_raw.std(axis=0)
        x_std[x_std < 1e-6] = 1.0
        x_train = ((x_train_raw - x_mean) / x_std).astype(np.float32)
        x_test = ((x_test_raw - x_mean) / x_std).astype(np.float32)
        model, history, class_weights = train_model(
            x_train,
            labels[train_mask],
            len(class_names),
            args.hidden_size,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.weight_decay,
            np.random.default_rng(args.seed),
        )
        train_pred = forward(model, x_train)[0].argmax(axis=1)
        test_pred = forward(model, x_test)[0].argmax(axis=1)
        train_metrics = classification_metrics(labels[train_mask], train_pred, class_names)
        test_metrics = classification_metrics(labels[test_mask], test_pred, class_names)
        parameter_count = int(sum(value.size for value in model.values()))
        model_metadata = {
            "version": "multiview_value_head_v1",
            "method": "two-layer RGB terminal_or_recovery value head",
            "method_boundary": "The head selects accept/retry/stop from fixed-camera RGB features. It does not output robot actions and is not an end-to-end VLA or OpenVLA LoRA model.",
            "source_dataset": str(args.dataset),
            "source_dataset_version": metadata["version"],
            "runtime_boundary": metadata["runtime_boundary"],
            "class_names": class_names,
            "view": view,
            "feature_indices": indices.tolist(),
            "feature_dim": int(len(indices)),
            "hidden_size": args.hidden_size,
            "trainable_parameters": parameter_count,
            "train_samples": int(train_mask.sum()),
            "test_samples": int(test_mask.sum()),
            "class_weights": class_weights.tolist(),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        model_path = args.output_dir / f"{args.prefix}_{view}_v1.npz"
        np.savez_compressed(
            model_path,
            w0=model["w0"].astype(np.float32),
            b0=model["b0"].astype(np.float32),
            w1=model["w1"].astype(np.float32),
            b1=model["b1"].astype(np.float32),
            x_mean=x_mean.astype(np.float32),
            x_std=x_std.astype(np.float32),
            metadata=json.dumps(model_metadata, ensure_ascii=False),
        )
        result = {"view": view, "model": str(model_path), "train": train_metrics, "test": test_metrics, "history": history, "metadata": model_metadata}
        results.append(result)
        print(json.dumps({"view": view, "model": str(model_path), "test": test_metrics}, ensure_ascii=False), flush=True)
    metrics_path = args.metrics or args.output_dir / f"{args.prefix}_training_metrics_v1.json"
    metrics_path.write_text(json.dumps({"version": "multiview_value_head_v1", "dataset": str(args.dataset), "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"metrics_path: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
