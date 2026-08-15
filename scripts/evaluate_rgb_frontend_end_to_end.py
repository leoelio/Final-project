from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_rgb_recovery_profiles import DOMAINS, rollout_args, selected_names  # noqa: E402
from run_clip_action_head import load_clip  # noqa: E402
from run_clip_semantic_rgb_feedback import configure_env, rollout  # noqa: E402
from run_clip_semantic_waypoint import load_policy  # noqa: E402
from widowx_env.vision_grounding import load_calibration  # noqa: E402


TASKS = {
    "place_blue_cube_blue_pad": "medium",
    "place_blue_cube_red_pad": "medium",
    "place_red_cube_red_pad": "medium",
    "move_leftmost_cube_to_bowl": "language",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end audit of the revised RGB source-grounding frontend.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=ROOT / "outputs" / "rgb_grounding" / "top_rgb_core_v2_calibration_v1.json")
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--episodes", type=int, default=24, help="Scene seeds per task and contact domain.")
    parser.add_argument("--domains", default="mild_contact_shift,low_contact_shift,severe_contact_shift")
    parser.add_argument("--tasks", default="place_blue_cube_red_pad,move_leftmost_cube_to_bowl")
    parser.add_argument("--version", default="rgb_frontend_end_to_end_preregistered_v1")
    parser.add_argument("--recovery-search", choices=("source", "table"), default="source")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / "rgb_frontend_end_to_end_preregistered_v1.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "rgb_frontend_end_to_end_preregistered_v1.md")
    parser.add_argument("--log-every", type=int, default=12)
    return parser.parse_args()


def counts(rows: list[dict]) -> dict:
    return {
        "episodes": len(rows),
        "initial_grounding_executable": sum(row["initial_grounding_executable"] for row in rows),
        "semantic_correct": sum(row["semantic_correct"] for row in rows),
        "visual_selection_correct": sum(row["visual_selection_correct"] for row in rows),
        "first_attempt_success": sum(row["first_attempt_success"] for row in rows),
        "recovery_triggered": sum(row["recovery_triggered"] for row in rows),
        "task_success": sum(row["task_success"] for row in rows),
    }


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    domain_names = selected_names(args.domains, DOMAINS, "domains")
    task_names = selected_names(args.tasks, TASKS, "tasks")
    policy = load_policy(args.model)
    calibration = load_calibration(args.calibration)
    clip_model, processor = load_clip(str(policy["metadata"]["clip_model"]))
    rows: list[dict] = []
    for domain_name in domain_names:
        for task in task_names:
            for offset in range(args.episodes):
                seed = args.seed + offset
                config = rollout_args(task, TASKS[task], "standard", DOMAINS[domain_name])
                config.recovery_search = args.recovery_search
                record = {"domain": domain_name, "task": task, "seed": seed, "initial_grounding_executable": False, "semantic_correct": False, "visual_selection_correct": False, "first_attempt_success": False, "recovery_triggered": False, "task_success": False}
                try:
                    env, obs = configure_env(config, seed)
                    result = rollout(config, policy, clip_model, processor, calibration, seed, env=env, obs=obs)
                    first = result["attempt_logs"][0]
                    record.update(
                        {
                            "initial_grounding_executable": True,
                            "semantic_correct": bool(result["semantic_correct"]),
                            "visual_selection_correct": bool(result["visual_selection_correct"]),
                            "initial_source_position_error_m": float(result["initial_source_position_error_m"]),
                            "first_attempt_success": bool(first["evaluation_strict_grasp_success"] and first["evaluation_target_distance_m"] < 0.065),
                            "recovery_triggered": bool(result["recovery_triggered"]),
                            "task_success": bool(result["task_success"]),
                            "target_distance_m": float(result["target_distance"]),
                            "recovery_reason": result["recovery_reason"],
                        }
                    )
                except (RuntimeError, LookupError, ValueError) as error:
                    record["error"] = str(error)
                rows.append(record)
                if args.log_every and len(rows) % args.log_every == 0:
                    print(json.dumps(record, ensure_ascii=False), flush=True)
    summary = {
        "version": args.version,
        "method": f"frozen_clip_intent + RGB source grounding with static-pad exclusion + structured standard pick/place + one RGB-triggered {args.recovery_search} search retry",
        "seed_range": f"{args.seed}-{args.seed + args.episodes - 1}",
        "domains": {name: DOMAINS[name] for name in domain_names},
        "tasks": task_names,
        "overall": counts(rows),
        "by_domain": {name: counts([row for row in rows if row["domain"] == name]) for name in domain_names},
        "by_task": {name: counts([row for row in rows if row["task"] == name]) for name in task_names},
        "rows": rows,
        "runtime_boundary": "The policy uses frozen CLIP intent, RGB source grounding, RGB target verification, static task configuration, and fixed standard trajectories. MuJoCo object truth is used only by offline evaluation fields.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 修订 RGB 前端的端到端留出评测",
        "",
        f"固定 seed `{summary['seed_range']}`，默认 `standard` 恢复，任务 `{', '.join(task_names)}`，接触域 `{', '.join(domain_names)}`。",
        "",
        "| 指标 | 结果 |",
        "| --- | ---: |",
        *[f"| {name} | {value}/{summary['overall']['episodes']} |" for name, value in summary["overall"].items() if name != "episodes"],
        "",
        "| 接触域 | 初始定位可执行 | 首轮成功 | 触发恢复 | 最终成功 |",
        "| --- | ---: | ---: | ---: | ---: |",
        *[
            f"| `{name}` | {item['initial_grounding_executable']}/{item['episodes']} | {item['first_attempt_success']}/{item['episodes']} | {item['recovery_triggered']}/{item['episodes']} | {item['task_success']}/{item['episodes']} |"
            for name, item in summary["by_domain"].items()
        ],
        "",
        "该评测只验证修订后的默认实现，不与旧规则的端到端性能作未配对比较。",
    ]
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
