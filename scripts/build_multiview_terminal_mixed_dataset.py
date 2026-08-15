from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge controlled same-color terminal scenes with real rollout terminal observations.")
    parser.add_argument("--terminal-dataset", type=Path, default=ROOT / "data" / "multiview_recovery" / "terminal_verification_v1.npz")
    parser.add_argument("--recovery-dataset", type=Path, default=ROOT / "data" / "multiview_recovery" / "spatial_recovery_v2.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "multiview_recovery" / "terminal_mixed_v1.npz")
    parser.add_argument("--summary", type=Path, default=ROOT / "data" / "multiview_recovery" / "terminal_mixed_v1_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.terminal_dataset, allow_pickle=False) as terminal, np.load(args.recovery_dataset, allow_pickle=False) as recovery:
        terminal_meta = json.loads(terminal["metadata"].item())
        recovery_meta = json.loads(recovery["metadata"].item())
        terminal_x = terminal["features"].astype(np.float32)
        recovery_x = recovery["features"].astype(np.float32)
        if terminal_x.shape[1] != recovery_x.shape[1] or terminal_meta["feature_spec"] != recovery_meta["feature_spec"]:
            raise ValueError("terminal and recovery datasets must share the exact RGB feature specification")
        x = np.concatenate([terminal_x, recovery_x], axis=0)
        y = np.concatenate([terminal["labels"].astype(np.int64), recovery["first_success_labels"].astype(np.int64)], axis=0)
        splits = np.concatenate([terminal["splits"].astype(np.int8), recovery["splits"].astype(np.int8)], axis=0)
        sources = np.asarray(["controlled_terminal"] * len(terminal_x) + ["actual_rollout"] * len(recovery_x))
    metadata = {
        "version": "widowx_multiview_terminal_mixed_v1",
        "method": "controlled same-color terminal scenes plus held-out MuJoCo action-rollout terminal observations",
        "class_names": ["not_complete", "complete"],
        "feature_spec": terminal_meta["feature_spec"],
        "feature_dim": int(x.shape[1]),
        "source_datasets": [str(args.terminal_dataset), str(args.recovery_dataset)],
        "runtime_boundary": "Runtime consumes only top_rgb, front_rgb, and static task configuration. MuJoCo state is offline supervision only.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=x, labels=y, splits=splits, sources=sources, metadata=json.dumps(metadata, ensure_ascii=False))
    summary = {
        "version": metadata["version"],
        "samples": int(len(x)),
        "feature_dim": int(x.shape[1]),
        "split_class_counts": {
            split: {name: int(np.sum((splits == split_id) & (y == class_id))) for class_id, name in enumerate(metadata["class_names"])}
            for split, split_id in (("train", 0), ("test", 1))
        },
        "source_split_counts": {
            source: {split: int(np.sum((sources == source) & (splits == split_id))) for split, split_id in (("train", 0), ("test", 1))}
            for source in sorted(set(sources.tolist()))
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
