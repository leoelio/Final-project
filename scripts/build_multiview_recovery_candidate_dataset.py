from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract only visually re-localizable failed attempts for the recovery-value binary head.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "multiview_recovery" / "spatial_recovery_v2.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "multiview_recovery" / "recovery_candidate_v1.npz")
    parser.add_argument("--summary", type=Path, default=ROOT / "data" / "multiview_recovery" / "recovery_candidate_v1_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.dataset, allow_pickle=False) as data:
        metadata = json.loads(data["metadata"].item())
        candidate = (data["first_success_labels"].astype(bool) == 0) & data["source_relocalizable"].astype(bool)
        features = data["features"][candidate].astype(np.float32)
        labels = data["retry_success_labels"][candidate].astype(np.int64)
        splits = data["splits"][candidate].astype(np.int8)
        seeds = data["seeds"][candidate].astype(np.int32)
        tasks = data["task_names"][candidate]
        domains = data["domain_names"][candidate]
    output_metadata = {
        "version": "widowx_multiview_recovery_candidate_v1",
        "method": "retry-success value supervision conditioned on a visually re-localizable failed first attempt",
        "class_names": ["stop", "retry"],
        "feature_spec": metadata["feature_spec"],
        "feature_dim": int(features.shape[1]),
        "source_dataset": str(args.dataset),
        "runtime_boundary": "At runtime this head is queried only after terminal RGB says incomplete and RGB re-localization has found the source object. MuJoCo state remains offline supervision only.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features,
        labels=labels,
        splits=splits,
        seeds=seeds,
        task_names=tasks,
        domain_names=domains,
        metadata=json.dumps(output_metadata, ensure_ascii=False),
    )
    summary = {
        "version": output_metadata["version"],
        "samples": int(len(labels)),
        "split_class_counts": {
            split: {name: int(np.sum((splits == split_id) & (labels == class_id))) for class_id, name in enumerate(output_metadata["class_names"])}
            for split, split_id in (("train", 0), ("test", 1))
        },
        "warning": "The test split may lack a stop example; report class coverage rather than treating missing-class accuracy as evidence.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
