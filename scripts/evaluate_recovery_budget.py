from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_rgb_recovery_profiles import DOMAINS, TASKS, rollout_args  # noqa: E402
from run_clip_action_head import load_clip  # noqa: E402
from run_clip_semantic_rgb_feedback import configure_env, rollout  # noqa: E402
from run_clip_semantic_waypoint import load_policy  # noqa: E402
from widowx_env.vision_grounding import load_calibration  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired severe-contact comparison of one versus two standard RGB retries.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=ROOT / "outputs" / "rgb_grounding" / "top_rgb_core_v2_calibration_v1.json")
    parser.add_argument("--seed", type=int, default=3600)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--task", choices=("move_leftmost_cube_to_bowl",), default="move_leftmost_cube_to_bowl")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / "recovery_budget_preregistered_v1.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "recovery_budget_preregistered_v1.md")
    parser.add_argument("--log-every", type=int, default=8)
    return parser.parse_args()


def exact_two_sided(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if not discordant:
        return 1.0
    tail = sum(comb(discordant, value) for value in range(min(improved, regressed) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def main() -> None:
    args = parse_args()
    policy = load_policy(args.model)
    calibration = load_calibration(args.calibration)
    clip_model, processor = load_clip(str(policy["metadata"]["clip_model"]))
    rows: list[dict] = []
    for offset in range(args.episodes):
        seed = args.seed + offset
        paired_rows = []
        try:
            for retries in (1, 2):
                config = rollout_args(args.task, TASKS[args.task], "standard", DOMAINS["severe_contact_shift"])
                config.feedback_attempts = retries
                env, obs = configure_env(config, seed)
                result = rollout(config, policy, clip_model, processor, calibration, seed, env=env, obs=obs)
                paired_rows.append(
                    {
                        "seed": seed,
                        "retry_budget": retries,
                        "success": bool(result["task_success"]),
                        "attempt_count": int(result["attempt_count"]),
                        "recovery_triggered": bool(result["recovery_triggered"]),
                        "recovery_reason": result["recovery_reason"],
                        "target_distance_m": float(result["target_distance"]),
                    }
                )
        except (RuntimeError, LookupError, ValueError) as error:
            rows.extend({"seed": seed, "retry_budget": retries, "success": False, "error": str(error)} for retries in (1, 2))
            continue
        rows.extend(paired_rows)
        if args.log_every and (offset + 1) % args.log_every == 0:
            print(json.dumps(paired_rows, ensure_ascii=False), flush=True)
    one = {row["seed"]: row for row in rows if row["retry_budget"] == 1}
    two = {row["seed"]: row for row in rows if row["retry_budget"] == 2}
    improved = sum(not one[seed]["success"] and two[seed]["success"] for seed in one)
    regressed = sum(one[seed]["success"] and not two[seed]["success"] for seed in one)
    summary = {
        "version": "recovery_budget_preregistered_v1",
        "method": "frozen_clip_intent + revised RGB grounding + structured standard trajectory + one versus two RGB retries",
        "domain": {"name": "severe_contact_shift", **DOMAINS["severe_contact_shift"]},
        "task": args.task,
        "seed_range": f"{args.seed}-{args.seed + args.episodes - 1}",
        "episodes_per_budget": len(one),
        "successes": {"one_retry": sum(row["success"] for row in one.values()), "two_retries": sum(row["success"] for row in two.values())},
        "paired": {"improved": improved, "regressed": regressed, "discordant": improved + regressed, "exact_two_sided_p": exact_two_sided(improved, regressed)},
        "rows": rows,
        "runtime_boundary": "Both variants use frozen CLIP, RGB source grounding, RGB target confirmation, and the same standard trajectory. MuJoCo truth is offline scoring only.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# RGB 恢复预算预注册对照",
        "",
        f"严重接触域，任务 `{args.task}`，seed `{summary['seed_range']}`。两种方案仅差一次额外的 RGB 重定位标准重试。",
        "",
        "| 恢复预算 | 成功 |",
        "| --- | ---: |",
        f"| 1 次重试 | {summary['successes']['one_retry']}/{summary['episodes_per_budget']} |",
        f"| 2 次重试 | {summary['successes']['two_retries']}/{summary['episodes_per_budget']} |",
        "",
        f"配对变化：改进 {improved}、回退 {regressed}、不一致 {improved + regressed}、精确双侧 p={summary['paired']['exact_two_sided_p']:.4f}。",
    ]
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
