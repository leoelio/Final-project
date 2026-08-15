from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np


REQUIRED_LABELS = (
    "approach_pre",
    "approach_post",
    "descend_post",
    "close_pre",
    "close_post",
    "lift_post",
    "transfer_post",
    "place_descend_post",
    "release_post",
)
REQUIRED_ARRAYS = ("images", "snapshot_labels", "qpos", "qvel", "ctrl", "actuator_force", "tcp", "times")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate paired contact-stage state data and its collection gate.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-success", type=int, default=60)
    parser.add_argument("--min-failure", type=int, default=60)
    return parser.parse_args()


def load_rows(metadata_path: Path) -> list[dict]:
    rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"no rows in {metadata_path}")
    return rows


def validate_state(run_dir: Path, row: dict) -> None:
    state_path = run_dir / row["state_file"]
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    with np.load(state_path) as state:
        if set(state.files) != set(REQUIRED_ARRAYS):
            raise RuntimeError(f"unexpected arrays in {state_path}: {state.files}")
        labels = tuple(str(value) for value in state["snapshot_labels"].tolist())
        if labels != REQUIRED_LABELS or labels != tuple(row["snapshot_labels"]):
            raise RuntimeError(f"invalid phase labels in {state_path}")
        count = len(REQUIRED_LABELS)
        if state["images"].shape[0] != count or state["images"].ndim != 4 or state["images"].shape[-1] != 3:
            raise RuntimeError(f"invalid RGB tensor in {state_path}: {state['images'].shape}")
        expected_shapes = {"qpos": 8, "qvel": 7, "ctrl": 7, "actuator_force": 7, "tcp": 3, "times": None}
        for name in REQUIRED_ARRAYS[2:]:
            if state[name].shape[0] != count or not np.isfinite(state[name]).all():
                raise RuntimeError(f"invalid {name} in {state_path}")
            expected_width = expected_shapes[name]
            if expected_width is not None and state[name].shape[1:] != (expected_width,):
                raise RuntimeError(f"invalid {name} shape in {state_path}: {state[name].shape}")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    rows = load_rows(run_dir / "metadata.jsonl")
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("version") != "contact_phase_pairs_v3":
            raise RuntimeError(f"unexpected dataset version: {row.get('version')}")
        if row.get("profile") not in {"nominal", "stress"}:
            raise RuntimeError(f"unexpected profile: {row.get('profile')}")
        validate_state(run_dir, row)
        by_pair[row["pair_id"]].append(row)

    transitions = Counter()
    for pair_id, pair_rows in by_pair.items():
        if len(pair_rows) != 2 or {row["profile"] for row in pair_rows} != {"nominal", "stress"}:
            raise RuntimeError(f"invalid pair: {pair_id}")
        nominal = next(row for row in pair_rows if row["profile"] == "nominal")
        stress = next(row for row in pair_rows if row["profile"] == "stress")
        if nominal["task"] != stress["task"] or nominal["seed"] != stress["seed"]:
            raise RuntimeError(f"mismatched scene in pair: {pair_id}")
        transitions[f"nominal_{int(nominal['task_success'])}_stress_{int(stress['task_success'])}"] += 1

    total_success = sum(bool(row["task_success"]) for row in rows)
    total_failure = len(rows) - total_success
    by_task = {
        task: {
            "trials": len(task_rows),
            "successes": sum(bool(row["task_success"]) for row in task_rows),
            "failures": sum(not bool(row["task_success"]) for row in task_rows),
        }
        for task, task_rows in sorted(
            ((task, [row for row in rows if row["task"] == task]) for task in {row["task"] for row in rows}),
            key=lambda item: item[0],
        )
    }
    by_profile = {
        profile: {
            "trials": sum(row["profile"] == profile for row in rows),
            "successes": sum(row["profile"] == profile and row["task_success"] for row in rows),
            "failures": sum(row["profile"] == profile and not row["task_success"] for row in rows),
        }
        for profile in ("nominal", "stress")
    }
    report = {
        "version": "contact_phase_pairs_v3_validation",
        "run_dir": str(run_dir),
        "trials": len(rows),
        "pairs": len(by_pair),
        "successes": total_success,
        "failures": total_failure,
        "minimum_successes": args.min_success,
        "minimum_failures": args.min_failure,
        "quota_met": total_success >= args.min_success and total_failure >= args.min_failure,
        "by_task": by_task,
        "by_profile": by_profile,
        "paired_outcome_transitions": dict(sorted(transitions.items())),
        "policy_input_boundary": "Only RGB, robot proprioception and action history are policy inputs. MuJoCo object truth is retained solely as an offline label.",
    }
    output = args.output or run_dir / "validation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["quota_met"]:
        raise RuntimeError("contact-stage data gate is not met")


if __name__ == "__main__":
    main()
