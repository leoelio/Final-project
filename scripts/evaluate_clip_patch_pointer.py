from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_clip_patch_pointer import load_clip_for_policy, load_policy, rollout_episode  # noqa: E402


SPECS = (
    ("place_blue_cube_blue_pad", "medium", 20),
    ("place_blue_cube_red_pad", "medium", 120),
    ("place_red_cube_red_pad", "medium", 220),
    ("move_leftmost_cube_to_bowl", "language", 420),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the frozen-CLIP patch-pointer action head on fixed Core V2 holdout seeds.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--workspace-profile", choices=("legacy", "core_v2"), default="core_v2")
    parser.add_argument("--arm-kp", type=float, default=150.0)
    parser.add_argument("--arm-force", type=float, default=100.0)
    parser.add_argument("--gripper-kp", type=float, default=1200.0)
    parser.add_argument("--gripper-force", type=float, default=200.0)
    parser.add_argument("--friction", type=float, default=5.0)
    parser.add_argument("--place-tcp-z", type=float, default=0.041)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def summarize(rows: list[dict]) -> dict:
    return {
        "episodes": len(rows),
        "semantic_correct": sum(int(row["semantic_correct"]) for row in rows),
        "strict_grasp_success": sum(int(row["strict_grasp_success"]) for row in rows),
        "task_success": sum(int(row["task_success"]) for row in rows),
        "mean_source_error_m": float(np.mean([row["offline_source_error_m"] for row in rows])),
        "mean_target_distance_m": float(np.mean([row["target_distance_m"] for row in rows])),
    }


def main() -> None:
    args = parse_args()
    policy, payload = load_policy(args.model)
    clip_model, processor = load_clip_for_policy(payload)
    rows = []
    for task, complexity, seed in SPECS:
        for offset in range(args.episodes):
            rollout_args = argparse.Namespace(
                **vars(args),
                task=task,
                complexity=complexity,
                seed=seed + offset,
                instruction=None,
                viewer=False,
                duration=0.0,
                speed=0.0,
                video_path=None,
                width=640,
                height=480,
                fps=24,
                frame_stride=12,
            )
            row = rollout_episode(rollout_args, policy, payload, clip_model, processor, seed + offset)
            rows.append(row)
            print(f"task={task} seed={seed + offset} intent={row['semantic_correct']} source_error={row['offline_source_error_m']:.4f} success={row['task_success']}", flush=True)
    by_task = {task: summarize([row for row in rows if row["task"] == task]) for task, _, _ in SPECS}
    summary = {
        "version": "clip_patch_pointer_core_v2_holdout_v1",
        "method": payload["metadata"]["method"],
        "method_boundary": payload["metadata"]["method_boundary"],
        "offline_truth_boundary": payload["metadata"].get(
            "offline_truth_boundary",
            "MuJoCo object truth is used only for offline labels and scoring, never runtime inference.",
        ),
        "model": str(args.model),
        "protocol": {"workspace_profile": args.workspace_profile, "episodes_per_task": args.episodes, "seed_disjoint_from_training": True},
        "overall": summarize(rows),
        "by_task": by_task,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = ["task", "seed", "predicted_intent", "semantic_correct", "predicted_pixel_uv", "predicted_source_xy_m", "offline_source_error_m", "strict_grasp_success", "task_success", "target_distance_m", "out_of_table"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({name: row[name] for name in fieldnames} for row in rows)
    print(json.dumps({"overall": summary["overall"], "by_task": by_task}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
