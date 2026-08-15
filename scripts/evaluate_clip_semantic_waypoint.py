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
from widowx_env import TASKS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Core V2 frozen CLIP semantic-waypoint candidate.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--version", default="clip_semantic_waypoint_core_v2_v1")
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--complexity", choices=("easy", "medium", "hard", "language"), required=True)
    parser.add_argument("--instruction-normalization", choices=("none", "desktop_alias_v1"), default="none")
    parser.add_argument("--executor", choices=("standard", "contact_fusion"), default="standard")
    parser.add_argument("--workspace-profile", choices=("legacy", "core_v2"), default="core_v2")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--camera", default="top_rgb")
    parser.add_argument("--arm-kp", type=float, default=150.0)
    parser.add_argument("--arm-force", type=float, default=100.0)
    parser.add_argument("--gripper-kp", type=float, default=1200.0)
    parser.add_argument("--gripper-force", type=float, default=200.0)
    parser.add_argument("--friction", type=float, default=5.0)
    parser.add_argument("--place-tcp-z", type=float, default=0.041)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = load_policy(args.model)
    clip_model, processor = load_clip(str(policy["metadata"]["clip_model"]))
    rows = []
    for offset in range(args.episodes):
        result = rollout_episode(args, policy, clip_model, processor, args.seed + offset)
        result["version"] = args.version
        rows.append(result)
        print(
            f"seed={result['seed']} semantic_correct={result['semantic_correct']} "
            f"intent={result['predicted_intent']} success={result['success']} "
            f"strict_grasp={result['strict_grasp_success']} distance={result['target_distance']:.4f}",
            flush=True,
        )
    successes = sum(int(row["success"]) for row in rows)
    semantic_correct = sum(int(row["semantic_correct"]) for row in rows)
    strict_grasps = sum(int(row["strict_grasp_success"]) for row in rows)
    summary = {
        "version": args.version,
        "method_key": "clip_semantic_contact_fusion" if args.executor == "contact_fusion" else "clip_semantic_waypoint",
        "stage": "frozen_pretrained_vlm_semantic_contact_feedback_policy" if args.executor == "contact_fusion" else "frozen_pretrained_vlm_semantic_hierarchical_policy",
        "model": str(args.model),
        "task": args.task,
        "complexity": args.complexity,
        "protocol": {
            "workspace_profile": args.workspace_profile,
            "arm_kp": args.arm_kp,
            "arm_force": args.arm_force,
            "gripper_kp": args.gripper_kp,
            "gripper_force": args.gripper_force,
            "friction": args.friction,
            "place_tcp_z": args.place_tcp_z,
            "instruction_normalization": args.instruction_normalization,
            "executor": args.executor,
        },
        "success": f"{successes}/{len(rows)}",
        "success_rate": successes / len(rows),
        "semantic_correct": f"{semantic_correct}/{len(rows)}",
        "semantic_accuracy": semantic_correct / len(rows),
        "strict_grasp_success": f"{strict_grasps}/{len(rows)}",
        "strict_grasp_rate": strict_grasps / len(rows),
        "mean_target_distance": float(np.mean([row["target_distance"] for row in rows])),
        "rows": rows,
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = [
            "version", "seed", "task", "complexity", "instruction", "normalized_instruction", "predicted_intent", "semantic_correct",
            "selected_object", "target_geom", "executor", "success", "placed", "strict_grasp_success", "contact_regrasp_attempts",
            "transport_hold_confirmed", "contact_recovery_reason", "max_object_z", "lifted_steps_near_tcp", "target_distance",
            "grasp_success", "out_of_table",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"success: {summary['success']}", flush=True)
    print(f"semantic_correct: {summary['semantic_correct']}", flush=True)
    print(f"strict_grasp_success: {summary['strict_grasp_success']}", flush=True)


if __name__ == "__main__":
    main()
