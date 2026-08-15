from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("features", "top_images", "front_images", "labels", "splits", "seeds", "task_names", "domain_names")


def load_part(path: Path, split: int) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as data:
        mask = data["splits"].astype(np.int8) == split
        if not np.any(mask):
            raise ValueError(f"no split={split} rows in {path}")
        part = {name: data[name][mask] for name in FIELDS}
        metadata = json.loads(data["metadata"].item())
    return part, metadata


def task_counts(task_names: np.ndarray, labels: np.ndarray, class_names: list[str]) -> dict[str, dict[str, int]]:
    return {
        str(task): {name: int(np.sum((task_names == task) & (labels == index))) for index, name in enumerate(class_names)}
        for task in sorted(set(task_names.tolist()))
    }


def main() -> None:
    old_path = ROOT / "data" / "clip_recovery_bank" / "clip_recovery_training_v1.npz"
    multitask_path = ROOT / "data" / "clip_recovery_bank" / "clip_recovery_bank_multitask_v2.npz"
    heldout_path = ROOT / "data" / "clip_recovery_bank" / "clip_recovery_bank_multitask_v2_heldout.npz"
    output = ROOT / "data" / "clip_recovery_bank" / "clip_recovery_multitask_training_v2.npz"
    summary_path = ROOT / "data" / "clip_recovery_bank" / "clip_recovery_multitask_training_v2_summary.json"

    old_train, old_meta = load_part(old_path, 0)
    multitask_train, multitask_meta = load_part(multitask_path, 0)
    multitask_test, _ = load_part(multitask_path, 1)
    heldout_test, heldout_meta = load_part(heldout_path, 1)
    if old_meta["feature_spec"] != multitask_meta["feature_spec"] or old_meta["feature_spec"] != heldout_meta["feature_spec"]:
        raise ValueError("all recovery banks must use the same feature specification")

    train = {name: np.concatenate([old_train[name], multitask_train[name]], axis=0) for name in FIELDS}
    test = {name: np.concatenate([multitask_test[name], heldout_test[name]], axis=0) for name in FIELDS}
    overlap = set(train["seeds"].tolist()) & set(test["seeds"].tolist())
    if overlap:
        raise ValueError(f"train/test seed overlap: {sorted(overlap)}")
    merged = {name: np.concatenate([train[name], test[name]], axis=0) for name in FIELDS}
    merged["splits"] = np.concatenate(
        [np.zeros(len(train["labels"]), dtype=np.int8), np.ones(len(test["labels"]), dtype=np.int8)]
    )

    class_names = old_meta["class_names"]
    metadata = {
        "version": "clip_recovery_multitask_training_v2",
        "method": "actual MuJoCo failure/retry RGB states with seed-disjoint multitask holdout",
        "class_names": class_names,
        "feature_spec": old_meta["feature_spec"],
        "feature_dim": int(merged["features"].shape[1]),
        "source_banks": [str(old_path), str(multitask_path), str(heldout_path)],
        "runtime_boundary": old_meta["runtime_boundary"],
        "note": "The v1 test split is intentionally excluded. Only prior train rows and new multitask train rows are used for fitting.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **merged, metadata=json.dumps(metadata, ensure_ascii=False))
    summary = {
        "version": metadata["version"],
        "samples": int(len(merged["labels"])),
        "split_class_counts": {
            "train": {name: int(np.sum(train["labels"] == index)) for index, name in enumerate(class_names)},
            "test": {name: int(np.sum(test["labels"] == index)) for index, name in enumerate(class_names)},
        },
        "split_task_class_counts": {
            "train": task_counts(train["task_names"], train["labels"], class_names),
            "test": task_counts(test["task_names"], test["labels"], class_names),
        },
        "train_seed_ranges": "760-799,820-869,1000-1019",
        "test_seed_ranges": "1100-1109,1120-1159",
        "excluded_prior_test_seed_range": "800-819",
        "dataset": str(output),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
