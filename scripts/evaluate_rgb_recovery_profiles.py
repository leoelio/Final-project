from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_clip_action_head import load_clip  # noqa: E402
from run_clip_semantic_rgb_feedback import configure_env, rollout  # noqa: E402
from run_clip_semantic_waypoint import load_policy  # noqa: E402
from widowx_env.vision_grounding import load_calibration  # noqa: E402


TASKS = {
    "place_blue_cube_red_pad": "medium",
    "move_leftmost_cube_to_bowl": "language",
}
DOMAINS = {
    "mild_contact_shift": {"arm_kp": 135.0, "arm_force": 90.0, "gripper_kp": 950.0, "gripper_force": 150.0, "friction": 2.5},
    "low_contact_shift": {"arm_kp": 120.0, "arm_force": 80.0, "gripper_kp": 750.0, "gripper_force": 110.0, "friction": 1.5},
    "severe_contact_shift": {"arm_kp": 105.0, "arm_force": 70.0, "gripper_kp": 550.0, "gripper_force": 75.0, "friction": 0.8},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired held-out comparison of recovery profiles under RGB terminal verification.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=ROOT / "outputs" / "rgb_grounding" / "top_rgb_core_v2_calibration_v1.json")
    parser.add_argument("--seed", type=int, default=1300)
    parser.add_argument("--episodes", type=int, default=10, help="Paired seeds per task.")
    parser.add_argument("--domains", default="severe_contact_shift", help="Comma-separated fixed contact domains.")
    parser.add_argument("--tasks", default=",".join(TASKS), help="Comma-separated RGB-verifiable tasks.")
    parser.add_argument("--log-every", type=int, default=12, help="Print each completed paired scene at this interval; 0 disables progress rows.")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / "rgb_recovery_profile_heldout_v1.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "rgb_recovery_profile_heldout_v1.md")
    return parser.parse_args()


def selected_names(value: str, known: dict[str, object], label: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in known]
    if not names or unknown:
        raise KeyError(f"unknown {label}: {unknown}")
    return names


def rollout_args(task: str, complexity: str, profile: str, domain: dict) -> SimpleNamespace:
    return SimpleNamespace(
        task=task,
        complexity=complexity,
        workspace_profile="core_v2",
        image_size=224,
        camera="top_rgb",
        instruction=None,
        instruction_normalization="none",
        feedback_attempts=1,
        recovery_profile=profile,
        place_tcp_z=0.041,
        speed=0.0,
        **domain,
    )


def exact_two_sided(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if not discordant:
        return 1.0
    tail = sum(comb(discordant, value) for value in range(min(improved, regressed) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def main() -> None:
    args = parse_args()
    domain_names = selected_names(args.domains, DOMAINS, "domains")
    task_names = selected_names(args.tasks, TASKS, "tasks")
    policy = load_policy(args.model)
    calibration = load_calibration(args.calibration)
    clip_model, processor = load_clip(str(policy["metadata"]["clip_model"]))
    rows: list[dict] = []
    skipped_resets: list[dict] = []
    for domain_name in domain_names:
        for task in task_names:
            complexity = TASKS[task]
            for offset in range(args.episodes):
                seed = args.seed + offset
                paired_rows = []
                try:
                    for profile in ("standard", "deep_tight_slow"):
                        config = rollout_args(task, complexity, profile, DOMAINS[domain_name])
                        env, obs = configure_env(config, seed)
                        result = rollout(config, policy, clip_model, processor, calibration, seed, env=env, obs=obs)
                        first = result["attempt_logs"][0]
                        first_success = bool(first["evaluation_strict_grasp_success"] and first["evaluation_target_distance_m"] < 0.065)
                        paired_rows.append(
                            {
                                "profile": profile,
                                "domain": domain_name,
                                "task": task,
                                "seed": seed,
                                "success": bool(result["task_success"]),
                                "first_success": first_success,
                                "retry_executed": bool(result["recovery_triggered"]),
                                "recovery_reason": result["recovery_reason"],
                                "target_distance_m": float(result["target_distance"]),
                            }
                        )
                except (RuntimeError, LookupError, ValueError) as error:
                    skipped_resets.append({"domain": domain_name, "task": task, "seed": seed, "reason": str(error)})
                    continue
                rows.extend(paired_rows)
                completed_pairs = len(rows) // 2
                if args.log_every and completed_pairs % args.log_every == 0:
                    print(json.dumps(paired_rows, ensure_ascii=False), flush=True)
    standard = {(row["domain"], row["task"], row["seed"]): row for row in rows if row["profile"] == "standard"}
    deep = {(row["domain"], row["task"], row["seed"]): row for row in rows if row["profile"] == "deep_tight_slow"}
    improved = sum(int(not standard[key]["success"] and deep[key]["success"]) for key in standard)
    regressed = sum(int(standard[key]["success"] and not deep[key]["success"]) for key in standard)
    by_domain = {}
    for domain_name in domain_names:
        keys = [key for key in standard if key[0] == domain_name]
        domain_improved = sum(int(not standard[key]["success"] and deep[key]["success"]) for key in keys)
        domain_regressed = sum(int(standard[key]["success"] and not deep[key]["success"]) for key in keys)
        by_domain[domain_name] = {
            "episodes_per_profile": len(keys),
            "profile_successes": {
                "standard": int(sum(standard[key]["success"] for key in keys)),
                "deep_tight_slow": int(sum(deep[key]["success"] for key in keys)),
            },
            "paired": {"improved": domain_improved, "regressed": domain_regressed, "discordant": domain_improved + domain_regressed, "exact_two_sided_p": exact_two_sided(domain_improved, domain_regressed)},
        }
    summary = {
        "version": "rgb_recovery_profile_multidomain_v1",
        "method": "frozen_clip_intent + RGB terminal verification + RGB re-localization + profile-specific structured retry",
        "domains": {name: DOMAINS[name] for name in domain_names},
        "seed_range": f"{args.seed}-{args.seed + args.episodes - 1}",
        "tasks": task_names,
        "episodes_per_profile": len(rows) // 2,
        "skipped_resets": skipped_resets,
        "profile_successes": {profile: int(sum(row["success"] for row in rows if row["profile"] == profile)) for profile in ("standard", "deep_tight_slow")},
        "profile_retries": {profile: int(sum(row["retry_executed"] for row in rows if row["profile"] == profile)) for profile in ("standard", "deep_tight_slow")},
        "paired": {"improved": improved, "regressed": regressed, "discordant": improved + regressed, "exact_two_sided_p": exact_two_sided(improved, regressed)},
        "by_domain": by_domain,
        "rows": rows,
        "runtime_boundary": "Retry triggering and trajectory planning use only frozen CLIP intent, RGB target verification, RGB source localization, static task configuration, and the fixed profile. MuJoCo state is final evaluation only.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# RGB 终局确认下的恢复轨迹 Profile 留出评测",
        "",
        f"固定接触域 `{', '.join(domain_names)}`，任务 `{', '.join(task_names)}`，seed `{summary['seed_range']}`。同色目标任务因顶视 RGB 无法可靠区分方块与同色盘，不纳入自动重试比较。",
        "",
        "| Profile | 成功 | RGB 触发重试 |",
        "| --- | ---: | ---: |",
        f"| `standard` | {summary['profile_successes']['standard']}/{summary['episodes_per_profile']} | {summary['profile_retries']['standard']} |",
        f"| `deep_tight_slow` | {summary['profile_successes']['deep_tight_slow']}/{summary['episodes_per_profile']} | {summary['profile_retries']['deep_tight_slow']} |",
        "",
        f"配对变化：改进 {improved}，回退 {regressed}，不一致 {improved + regressed}，精确双侧 p={summary['paired']['exact_two_sided_p']:.4f}。样本有限，不能据此做统计显著性声明。",
        "",
        "| 接触域 | 标准 | 深抓取 | 改进 | 回退 | 精确双侧 p |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| `{name}` | {item['profile_successes']['standard']}/{item['episodes_per_profile']} | {item['profile_successes']['deep_tight_slow']}/{item['episodes_per_profile']} | {item['paired']['improved']} | {item['paired']['regressed']} | {item['paired']['exact_two_sided_p']:.4f} |"
            for name, item in by_domain.items()
        ],
        "",
        "该比较评估的是已固定的结构化恢复动作，不是学习出的连续动作策略或端到端 VLA。",
    ]
    if skipped_resets:
        lines.extend(["", f"环境初始化无法生成无重叠布局的 seed 已成对跳过：{len(skipped_resets)} 个。"])
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"output_json: {args.output_json}")


if __name__ == "__main__":
    main()
