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
from run_clip_semantic_multiview_gate import configure_env, load_head, rollout  # noqa: E402
from run_clip_semantic_waypoint import load_policy  # noqa: E402
from widowx_env.vision_grounding import load_calibration  # noqa: E402


TASKS = (
    ("place_blue_cube_blue_pad", "medium"),
    ("place_blue_cube_red_pad", "medium"),
    ("place_red_cube_red_pad", "medium"),
    ("move_leftmost_cube_to_bowl", "language"),
)
SEVERE = {"arm_kp": 105.0, "arm_force": 70.0, "gripper_kp": 550.0, "gripper_force": 75.0, "friction": 0.8}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired held-out comparison of standard and deep-tight-slow visual recovery trajectories.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--terminal-head", type=Path, required=True)
    parser.add_argument("--recovery-head", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=ROOT / "outputs" / "rgb_grounding" / "top_rgb_core_v2_calibration_v1.json")
    parser.add_argument("--seed", type=int, default=1300)
    parser.add_argument("--episodes", type=int, default=5, help="Paired seeds per task.")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / "recovery_profile_heldout_v1.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "recovery_profile_heldout_v1.md")
    return parser.parse_args()


def rollout_args(task: str, complexity: str, profile: str) -> SimpleNamespace:
    return SimpleNamespace(
        task=task,
        complexity=complexity,
        workspace_profile="core_v2",
        image_size=224,
        place_tcp_z=0.041,
        recovery_mode="rule",
        recovery_profile=profile,
        speed=0.0,
        **SEVERE,
    )


def exact_two_sided(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if not discordant:
        return 1.0
    lower = min(improved, regressed)
    tail = sum(comb(discordant, index) for index in range(lower + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def main() -> None:
    args = parse_args()
    policy = load_policy(args.model)
    calibration = load_calibration(args.calibration)
    clip_model, processor = load_clip(str(policy["metadata"]["clip_model"]))
    terminal_head = load_head(args.terminal_head)
    recovery_head = load_head(args.recovery_head)
    rows: list[dict] = []
    for task, complexity in TASKS:
        for offset in range(args.episodes):
            seed = args.seed + offset
            for profile in ("standard", "deep_tight_slow"):
                config = rollout_args(task, complexity, profile)
                env, obs = configure_env(config, seed)
                result = rollout(config, policy, clip_model, processor, calibration, terminal_head, recovery_head, seed, env=env, obs=obs)
                row = {
                    "profile": profile,
                    "task": task,
                    "seed": seed,
                    "success": bool(result["task_success"]),
                    "first_success": bool(result["first_execution_evaluation_success"]),
                    "retry_executed": bool(result["retry_executed"]),
                    "retry_success": result["retry_execution_evaluation_success"],
                    "gate_decision": result["gate_decision"],
                    "target_distance_m": float(result["target_distance_m"]),
                }
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    standard = {(row["task"], row["seed"]): row for row in rows if row["profile"] == "standard"}
    deep = {(row["task"], row["seed"]): row for row in rows if row["profile"] == "deep_tight_slow"}
    improved = sum(int(not standard[key]["success"] and deep[key]["success"]) for key in standard)
    regressed = sum(int(standard[key]["success"] and not deep[key]["success"]) for key in standard)
    summary = {
        "version": "recovery_profile_heldout_v1",
        "method": "frozen_clip_intent + RGB grounding + learned terminal gate + profile-specific structured visual retry",
        "domain": {"name": "severe_contact_shift", **SEVERE},
        "seed_range": f"{args.seed}-{args.seed + args.episodes - 1}",
        "episodes_per_profile": len(rows) // 2,
        "profile_successes": {profile: int(sum(row["success"] for row in rows if row["profile"] == profile)) for profile in ("standard", "deep_tight_slow")},
        "profile_retries": {profile: int(sum(row["retry_executed"] for row in rows if row["profile"] == profile)) for profile in ("standard", "deep_tight_slow")},
        "paired": {"improved": improved, "regressed": regressed, "discordant": improved + regressed, "exact_two_sided_p": exact_two_sided(improved, regressed)},
        "rows": rows,
        "runtime_boundary": "Only frozen CLIP intent, RGB localization, static task configuration, and the fixed trajectory profile are runtime inputs. MuJoCo state is only used internally by IK/physics and final scoring.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 视觉恢复轨迹 Profile 留出评测",
        "",
        f"极端接触扰动，seed `{summary['seed_range']}`，每个 profile {summary['episodes_per_profile']} 个配对 episode。",
        "",
        "| Profile | 成功 | 执行重试 |",
        "| --- | ---: | ---: |",
        f"| `standard` | {summary['profile_successes']['standard']}/{summary['episodes_per_profile']} | {summary['profile_retries']['standard']} |",
        f"| `deep_tight_slow` | {summary['profile_successes']['deep_tight_slow']}/{summary['episodes_per_profile']} | {summary['profile_retries']['deep_tight_slow']} |",
        "",
        f"配对变化：改进 {improved}，回退 {regressed}，不一致 {improved + regressed}，精确双侧 p={summary['paired']['exact_two_sided_p']:.4f}。样本有限，p 值不能作为显著性声明。",
        "",
        "`deep_tight_slow` 只在第一条标准轨迹失败、终局门判断未完成且 RGB 可重新定位物体时生效；它并非端到端 VLA 或学习出的连续动作策略。",
    ]
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"output_json: {args.output_json}")


if __name__ == "__main__":
    main()
