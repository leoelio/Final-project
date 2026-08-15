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


VERSION = "core_v2_clip_semantic_independent_syntax_v1"
SPECS = (
    ("blue_to_blue", "place_blue_cube_blue_pad", "medium", (
        "could you drop the azure cuboid onto the blue platform",
        "guide the cobalt object to the blue disk",
    )),
    ("blue_to_red", "place_blue_cube_red_pad", "medium", (
        "guide the azure cuboid toward the crimson platform",
        "send the cobalt object to the red circle",
    )),
    ("red_to_red", "place_red_cube_red_pad", "medium", (
        "deliver the ruby block to the scarlet disk",
        "move the vermilion cuboid onto the red platform",
    )),
    ("leftmost_cube", "move_leftmost_cube_to_bowl", "language", (
        "send the westernmost cuboid to the vessel",
        "guide the far-left object into the container",
    )),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate new syntax under the fixed tabletop alias vocabulary.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--workspace-profile", choices=("legacy", "core_v2"), default="core_v2")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--camera", default="top_rgb")
    parser.add_argument("--arm-kp", type=float, default=150.0)
    parser.add_argument("--arm-force", type=float, default=100.0)
    parser.add_argument("--gripper-kp", type=float, default=1200.0)
    parser.add_argument("--gripper-force", type=float, default=200.0)
    parser.add_argument("--friction", type=float, default=5.0)
    parser.add_argument("--place-tcp-z", type=float, default=0.041)
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / f"{VERSION}.csv")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / f"{VERSION}.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / f"{VERSION}.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.instruction_normalization = "desktop_alias_v1"
    policy = load_policy(args.model)
    clip_model, processor = load_clip(str(policy["metadata"]["clip_model"]))
    rows: list[dict[str, object]] = []
    for task_index, (task_key, task, complexity, instructions) in enumerate(SPECS):
        for phrase_index, instruction in enumerate(instructions, start=1):
            for offset in range(args.episodes):
                args.task = task
                args.complexity = complexity
                args.instruction = instruction
                seed = 2000 + task_index * 100 + (phrase_index - 1) * args.episodes + offset
                result = rollout_episode(args, policy, clip_model, processor, seed)
                result.update({"task_key": task_key, "phrase_index": phrase_index, "seed": seed})
                rows.append(result)
                print(f"task={task_key} phrase={phrase_index} seed={seed} success={result['task_success']}", flush=True)

    successes = sum(int(row["task_success"]) for row in rows)
    semantic = sum(int(row["semantic_correct"]) for row in rows)
    strict_grasps = sum(int(row["strict_grasp_success"]) for row in rows)
    summary = {
        "version": VERSION,
        "model": str(args.model),
        "protocol": "Independent full sentences; no sentence is copied from the original OOD or augmentation suites. The fixed desktop_alias_v1 vocabulary is intentionally used.",
        "episodes": len(rows),
        "task_success": f"{successes}/{len(rows)}",
        "semantic_correct": f"{semantic}/{len(rows)}",
        "strict_grasp_success": f"{strict_grasps}/{len(rows)}",
        "mean_target_distance": float(np.mean([float(row["target_distance"]) for row in rows])),
        "rows": rows,
    }
    fields = ("task_key", "phrase_index", "seed", "task", "instruction", "normalized_instruction", "predicted_intent", "semantic_correct", "task_success", "strict_grasp_success", "target_distance")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# CLIP 语义词表规范化独立句法验证",
        "",
        f"版本：`{VERSION}`",
        "",
        "- 每个四类任务使用 2 条未出现在原 OOD 或训练改写集中的完整英文句子，每条 5 个新 seed，共 40 episode。",
        "- 固定使用 `desktop_alias_v1` 闭词表规范化；因此该实验只验证新句法在该词表内的泛化，不验证开放词汇泛化。",
        f"- 任务成功：`{summary['task_success']}`；语义正确：`{summary['semantic_correct']}`；严格抓取：`{summary['strict_grasp_success']}`；平均目标距离：`{summary['mean_target_distance']:.4f} m`。",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"task_success: {summary['task_success']}", flush=True)
    print(f"output_json: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
