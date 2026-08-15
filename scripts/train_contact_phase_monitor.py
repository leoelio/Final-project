from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from torch_runtime import ensure_torch_path  # noqa: E402
from vlm_runtime import ensure_vlm_path  # noqa: E402

ensure_torch_path()
ensure_vlm_path()

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import CLIPModel, CLIPProcessor  # noqa: E402


VERSION = "frozen_clip_proprio_contact_monitor_v1"
PHASES = ("close_post", "lift_post")


def clip_tensor(output):
    return output.pooler_output if hasattr(output, "pooler_output") else output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a frozen CLIP + robot-proprioception grasp monitor on seed-disjoint contact data.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, default=ROOT / "outputs" / "contact_phase_monitor" / f"{VERSION}.pt")
    parser.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "evaluations" / f"{VERSION}.json")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def load_rows(run_dir: Path) -> list[dict]:
    validation = json.loads((run_dir / "validation.json").read_text(encoding="utf-8"))
    if not validation.get("quota_met"):
        raise RuntimeError("contact data gate is not met")
    rows = [json.loads(line) for line in (run_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if any(row.get("version") != "contact_phase_pairs_v3" for row in rows):
        raise RuntimeError("expected contact_phase_pairs_v3 data")
    return rows


def snapshot_features(run_dir: Path, row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(run_dir / row["state_file"]) as state:
        labels = [str(value) for value in state["snapshot_labels"].tolist()]
        indices = [labels.index(phase) for phase in PHASES]
        frames = state["images"][indices].astype(np.uint8)
        vectors = []
        for index in indices:
            vectors.append(np.concatenate([state["qpos"][index], state["qvel"][index], state["ctrl"][index], state["actuator_force"][index], state["tcp"][index]]))
        close, lift = vectors
        return frames[0], frames[1], np.concatenate([close, lift, lift - close]).astype(np.float32)


def encode_clip(
    close_images: list[np.ndarray], lift_images: list[np.ndarray], instructions: list[str], name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(name).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(name)
    batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for start in range(0, len(instructions), 16):
        stop = min(start + 16, len(instructions))
        images = close_images[start:stop] + lift_images[start:stop]
        image_inputs = processor(images=[Image.fromarray(image, mode="RGB") for image in images], return_tensors="pt")
        text_inputs = processor(text=instructions[start:stop], return_tensors="pt", padding=True)
        image_inputs = {key: value.to(device) for key, value in image_inputs.items()}
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
        with torch.no_grad():
            image_features = clip_tensor(model.get_image_features(pixel_values=image_inputs["pixel_values"]))
            text_features = clip_tensor(model.get_text_features(input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"]))
            image_features = torch.nn.functional.normalize(image_features, dim=-1).cpu().numpy().astype(np.float32)
            text_features = torch.nn.functional.normalize(text_features, dim=-1).cpu().numpy().astype(np.float32)
        size = stop - start
        batches.append((image_features[:size], image_features[size:], text_features))
    close, lift, text = zip(*batches)
    return np.concatenate(close), np.concatenate(lift), np.concatenate(text)


def split_indices(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    validation = np.array([int(hashlib.sha256(row["pair_id"].encode()).hexdigest()[:8], 16) % 5 == 0 for row in rows])
    train = np.flatnonzero(~validation)
    valid = np.flatnonzero(validation)
    if not len(train) or not len(valid):
        raise RuntimeError("invalid group split")
    return train, valid


def metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    predicted = probabilities >= 0.5
    labels = labels.astype(bool)
    tp = int(np.sum(predicted & labels))
    tn = int(np.sum(~predicted & ~labels))
    fp = int(np.sum(predicted & ~labels))
    fn = int(np.sum(~predicted & labels))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    return {
        "samples": int(len(labels)),
        "positives": int(labels.sum()),
        "negatives": int((~labels).sum()),
        "accuracy": float((tp + tn) / max(1, len(labels))),
        "balanced_accuracy": float((recall + specificity) / 2),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / max(1e-8, precision + recall)),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_dir = args.run_dir.resolve()
    rows = load_rows(run_dir)
    close_images, lift_images, proprio = [], [], []
    for row in rows:
        close, lift, vector = snapshot_features(run_dir, row)
        close_images.append(close)
        lift_images.append(lift)
        proprio.append(vector)
    close_features, lift_features, text_features = encode_clip(close_images, lift_images, [row["instruction"] for row in rows], args.clip_model)
    inputs = np.concatenate([close_features, lift_features, text_features, np.stack(proprio)], axis=1).astype(np.float32)
    labels = np.asarray([row["strict_grasp_success"] for row in rows], dtype=np.float32)
    train_index, valid_index = split_indices(rows)
    if len(np.unique(labels[train_index])) != 2 or len(np.unique(labels[valid_index])) != 2:
        raise RuntimeError("seed-disjoint split lacks both grasp classes")
    mean = inputs[train_index].mean(axis=0)
    std = np.maximum(inputs[train_index].std(axis=0), 1e-6)
    normalized = (inputs - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = torch.nn.Sequential(torch.nn.Linear(normalized.shape[1], args.hidden_size), torch.nn.ReLU(), torch.nn.Linear(args.hidden_size, 1)).to(device)
    train_x = torch.from_numpy(normalized[train_index]).to(device)
    train_y = torch.from_numpy(labels[train_index, None]).to(device)
    pos_weight = torch.tensor([(1 - labels[train_index].mean()) / labels[train_index].mean()], device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    net.train()
    for _ in range(args.epochs):
        optimizer.zero_grad()
        loss = criterion(net(train_x), train_y)
        loss.backward()
        optimizer.step()
    net.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(net(torch.from_numpy(normalized).to(device))).cpu().numpy().reshape(-1)
    train_metrics = metrics(probabilities[train_index], labels[train_index])
    valid_metrics = metrics(probabilities[valid_index], labels[valid_index])
    report = {
        "version": VERSION,
        "dataset": str(run_dir),
        "clip_model": args.clip_model,
        "feature_schema": "frozen CLIP close RGB + lift RGB + instruction text + robot-only close/lift proprioception and delta",
        "trainable_parameters": int(sum(parameter.numel() for parameter in net.parameters())),
        "seed_disjoint_split": "sha256(pair_id) modulo 5 == 0 for validation",
        "train": train_metrics,
        "validation": valid_metrics,
        "runtime_candidate_eligible": bool(valid_metrics["balanced_accuracy"] >= 0.70 and valid_metrics["recall"] >= 0.70),
        "runtime_boundary": "The monitor receives only frozen CLIP features and robot-only proprioception. Object truth and contact profile labels are offline labels, not inputs.",
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "x_mean": mean, "x_std": std, "metadata": report}, args.model_output)
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
