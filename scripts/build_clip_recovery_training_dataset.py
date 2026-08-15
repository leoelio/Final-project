from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge real recovery RGB banks while preserving the independent primary test split.")
    parser.add_argument("--primary", type=Path, default=ROOT / "data" / "clip_recovery_bank" / "clip_recovery_bank_v1.npz")
    parser.add_argument("--supplement", type=Path, default=ROOT / "data" / "clip_recovery_bank" / "clip_recovery_bank_supplement_v1.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "clip_recovery_bank" / "clip_recovery_training_v1.npz")
    parser.add_argument("--summary", type=Path, default=ROOT / "data" / "clip_recovery_bank" / "clip_recovery_training_v1_summary.json")
    return parser.parse_args()


def arrays(data: np.lib.npyio.NpzFile, mask: np.ndarray) -> dict[str, np.ndarray]:
    return {name: data[name][mask] for name in ("features", "top_images", "front_images", "labels", "splits", "seeds", "task_names", "domain_names")}


def main() -> None:
    args = parse_args()
    with np.load(args.primary, allow_pickle=False) as primary, np.load(args.supplement, allow_pickle=False) as supplement:
        primary_meta = json.loads(primary["metadata"].item())
        supplement_meta = json.loads(supplement["metadata"].item())
        if primary_meta["feature_spec"] != supplement_meta["feature_spec"] or primary["features"].shape[1] != supplement["features"].shape[1]:
            raise ValueError("banks must share the exact visual feature specification")
        parts = [arrays(primary, np.ones(len(primary["labels"]), dtype=bool)), arrays(supplement, supplement["splits"].astype(np.int8) == 0)]
    merged = {name: np.concatenate([part[name] for part in parts], axis=0) for name in parts[0]}
    metadata = {
        "version": "clip_recovery_training_v1",
        "method": "actual MuJoCo failure/retry RGB bank with seed-disjoint primary test split",
        "class_names": primary_meta["class_names"],
        "feature_spec": primary_meta["feature_spec"],
        "feature_dim": int(merged["features"].shape[1]),
        "source_banks": [str(args.primary), str(args.supplement)],
        "runtime_boundary": primary_meta["runtime_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **merged, metadata=json.dumps(metadata, ensure_ascii=False))
    summary = {
        "version": metadata["version"],
        "samples": int(len(merged["labels"])),
        "image_shape": list(merged["top_images"].shape[1:]),
        "split_class_counts": {
            split: {name: int(np.sum((merged["splits"] == split_id) & (merged["labels"] == class_id))) for class_id, name in enumerate(metadata["class_names"])}
            for split, split_id in (("train", 0), ("test", 1))
        },
        "train_seed_ranges": "760-799 and 820-869",
        "test_seed_range": "800-819",
        "note": "Supplement rows are training only. The primary bank's test rows remain unchanged and are the sole validation split for this training dataset.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
