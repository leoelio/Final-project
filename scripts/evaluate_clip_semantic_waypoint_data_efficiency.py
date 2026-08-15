from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from torch_runtime import ensure_torch_path  # noqa: E402

ensure_torch_path()

from run_clip_action_head import load_clip  # noqa: E402
from run_clip_semantic_waypoint import load_policy, rollout_episode  # noqa: E402


TASKS = (
    ("blue_to_blue", "place_blue_cube_blue_pad", "medium", 20),
    ("blue_to_red", "place_blue_cube_red_pad", "medium", 120),
    ("red_to_red", "place_red_cube_red_pad", "medium", 220),
    ("leftmost_cube", "move_leftmost_cube_to_bowl", "language", 420),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CLIP semantic-waypoint data efficiency on fixed Core V2 held-out seeds.")
    parser.add_argument("--model", action="append", required=True, help="Use BUDGET=PATH, once per data budget.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--workspace-profile", default="core_v2")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--camera", default="top_rgb")
    parser.add_argument("--arm-kp", type=float, default=150.0)
    parser.add_argument("--arm-force", type=float, default=100.0)
    parser.add_argument("--gripper-kp", type=float, default=1200.0)
    parser.add_argument("--gripper-force", type=float, default=200.0)
    parser.add_argument("--friction", type=float, default=5.0)
    parser.add_argument("--place-tcp-z", type=float, default=0.041)
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "core_v2_clip_semantic_data_efficiency.csv")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / "core_v2_clip_semantic_data_efficiency_v1.json")
    return parser.parse_args()


def parse_models(values: list[str]) -> list[tuple[int, Path]]:
    models = []
    for value in values:
        budget, path = value.split("=", 1)
        models.append((int(budget), Path(path)))
    return sorted(models)


def main() -> None:
    args = parse_args()
    models = parse_models(args.model)
    policies = [(budget, path, load_policy(path)) for budget, path in models]
    clip_model, processor = load_clip(str(policies[0][2]["metadata"]["clip_model"]))
    rows = []
    details = {}
    for budget, path, policy in policies:
        for task_key, task, complexity, seed in TASKS:
            args.task, args.complexity = task, complexity
            episodes = [rollout_episode(args, policy, clip_model, processor, seed + offset) for offset in range(args.episodes)]
            successes = sum(int(item["success"]) for item in episodes)
            semantics = sum(int(item["semantic_correct"]) for item in episodes)
            strict_grasps = sum(int(item["strict_grasp_success"]) for item in episodes)
            row = {
                "demo_budget_per_task": budget,
                "stored_samples": int(policy["metadata"]["samples"]),
                "task_key": task_key,
                "success": f"{successes}/{len(episodes)}",
                "success_rate": successes / len(episodes),
                "semantic_correct": f"{semantics}/{len(episodes)}",
                "semantic_accuracy": semantics / len(episodes),
                "strict_grasp_success": f"{strict_grasps}/{len(episodes)}",
                "strict_grasp_rate": strict_grasps / len(episodes),
                "mean_target_distance": float(np.mean([item["target_distance"] for item in episodes])),
                "model": str(path),
            }
            rows.append(row)
            details[f"{budget}:{task_key}"] = episodes
            print(f"budget={budget} task={task_key} success={row['success']} semantic={row['semantic_correct']}", flush=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"version": "core_v2_clip_semantic_data_efficiency_v1", "rows": rows, "episodes": details}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
