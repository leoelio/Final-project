from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from torch_runtime import ensure_torch_path  # noqa: E402
from vlm_runtime import ensure_vlm_path  # noqa: E402

ensure_torch_path()
ensure_vlm_path()

import torch  # noqa: E402
from transformers import CLIPModel, CLIPProcessor  # noqa: E402

from train_clip_recovery_head import encode_images, encode_texts  # noqa: E402
from widowx_env import TASKS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score action-profile head predictions by their counterfactual success, not classification accuracy alone.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "action_profile_bank" / "action_profile_bank_v1.npz")
    parser.add_argument("--records", type=Path, default=ROOT / "data" / "action_profile_bank" / "action_profile_bank_v1.jsonl")
    parser.add_argument("--top-model", type=Path, default=ROOT / "outputs" / "action_profile_value_v1" / "clip_recovery_value_top_v1.npz")
    parser.add_argument("--top-front-model", type=Path, default=ROOT / "outputs" / "action_profile_value_v1" / "clip_recovery_value_top_front_v1.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "evaluations" / "action_profile_selector_offline_v1.json")
    return parser.parse_args()


def load_head(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].astype(np.float32) for name in ("down_weight", "down_bias", "up_weight", "up_bias", "x_mean", "x_std")} | {"metadata": json.loads(data["metadata"].item())}


def predict(head: dict, features: np.ndarray) -> np.ndarray:
    x = (features - head["x_mean"]) / head["x_std"]
    hidden = np.maximum(0.0, x @ head["down_weight"] + head["down_bias"])
    return (hidden @ head["up_weight"] + head["up_bias"]).argmax(axis=1)


def main() -> None:
    args = parse_args()
    with np.load(args.dataset, allow_pickle=False) as data:
        mask = data["splits"].astype(np.int8) == 1
        top_images = data["top_images"][mask].astype(np.uint8)
        front_images = data["front_images"][mask].astype(np.uint8)
        proprio = data["proprio"][mask].astype(np.float32) if "proprio" in data.files else None
        seeds = data["seeds"][mask].astype(int)
        task_names = data["task_names"][mask].astype(str)
    records = {
        (int(row["seed"]), str(row["task"])): row
        for row in (json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line)
        if row["split"] == "test"
    }
    device = torch.device("cpu")
    clip_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(clip_name)
    clip = CLIPModel.from_pretrained(clip_name).to(device)
    clip.eval()
    for parameter in clip.parameters():
        parameter.requires_grad_(False)
    top = encode_images(clip, processor, top_images, 16)
    front = encode_images(clip, processor, front_images, 16)
    text = encode_texts(clip, processor, [TASKS[name].instruction for name in task_names], 16)
    features = {"top": np.concatenate([top, text], axis=1), "top_front": np.concatenate([top, front, text], axis=1)}
    rows = []
    baselines = {"standard": 0, "deep_tight_slow": 0, "oracle": 0}
    for index, key in enumerate(zip(seeds.tolist(), task_names.tolist())):
        outcome = records[key]["outcomes"]
        baselines["standard"] += int(outcome["standard"]["success"])
        baselines["deep_tight_slow"] += int(outcome["deep_tight_slow"]["success"])
        baselines["oracle"] += int(outcome["standard"]["success"] or outcome["deep_tight_slow"]["success"])
        rows.append({"seed": key[0], "task": key[1], "standard_success": bool(outcome["standard"]["success"]), "deep_success": bool(outcome["deep_tight_slow"]["success"])})
    result = {"version": "action_profile_selector_offline_v1", "candidate_states": len(rows), "fixed_profiles": baselines, "selectors": {}}
    for view, path in (("top", args.top_model), ("top_front", args.top_front_model)):
        head = load_head(path)
        labels = list(head["metadata"]["class_names"])
        inputs = features[view]
        if head["metadata"].get("uses_robot_proprioception", False):
            if proprio is None:
                raise ValueError("head requires robot proprioception but dataset does not provide it")
            inputs = np.concatenate([inputs, proprio], axis=1)
        predictions = predict(head, inputs)
        success = 0
        for row, prediction in zip(rows, predictions):
            label = labels[int(prediction)]
            row[f"{view}_prediction"] = label
            success += int(label != "stop" and row["deep_success"] if label == "deep_tight_slow" else label == "standard" and row["standard_success"])
        result["selectors"][view] = {"successes": success, "predicted_action_counts": {label: int(sum(labels[int(value)] == label for value in predictions)) for label in labels}}
    result["rows"] = rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
