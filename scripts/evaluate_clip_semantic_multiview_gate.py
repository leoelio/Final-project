from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_clip_action_head import load_clip  # noqa: E402
from run_clip_semantic_multiview_gate import configure_env as configure_gate_env  # noqa: E402
from run_clip_semantic_multiview_gate import load_clip_recovery_head, load_head, rollout as gate_rollout  # noqa: E402
from run_clip_semantic_rgb_feedback import configure_env as configure_rule_env  # noqa: E402
from run_clip_semantic_rgb_feedback import rollout as rule_rollout  # noqa: E402
from run_clip_semantic_waypoint import load_policy  # noqa: E402
from widowx_env.vision_grounding import load_calibration  # noqa: E402


TASKS = (
    ("place_blue_cube_blue_pad", "medium"),
    ("place_blue_cube_red_pad", "medium"),
    ("place_red_cube_red_pad", "medium"),
    ("move_leftmost_cube_to_bowl", "language"),
)
SPATIAL_TASK = ("move_leftmost_cube_to_bowl", "language")
DOMAINS = {
    "nominal": {"arm_kp": 150.0, "arm_force": 100.0, "gripper_kp": 1200.0, "gripper_force": 200.0, "friction": 5.0},
    "mild_contact_shift": {"arm_kp": 135.0, "arm_force": 90.0, "gripper_kp": 950.0, "gripper_force": 150.0, "friction": 2.5},
    "low_contact_shift": {"arm_kp": 120.0, "arm_force": 80.0, "gripper_kp": 750.0, "gripper_force": 110.0, "friction": 1.5},
    "severe_contact_shift": {"arm_kp": 105.0, "arm_force": 70.0, "gripper_kp": 550.0, "gripper_force": 75.0, "friction": 0.8},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rule-based RGB retry versus learned top-only and top+front visual gates.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--terminal-top", type=Path, required=True)
    parser.add_argument("--recovery-top", type=Path, required=True)
    parser.add_argument("--terminal-top-front", type=Path, required=True)
    parser.add_argument("--recovery-top-front", type=Path, required=True)
    parser.add_argument("--clip-recovery-top", type=Path, default=None)
    parser.add_argument("--clip-recovery-top-front", type=Path, default=None)
    parser.add_argument("--calibration", type=Path, default=ROOT / "outputs" / "rgb_grounding" / "top_rgb_core_v2_calibration_v1.json")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / "clip_semantic_multiview_gate_v1.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "clip_semantic_multiview_gate_v1.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "clip_semantic_multiview_gate_v1.md")
    parser.add_argument("--seed", type=int, default=750, help="Held-out base seed; keep it outside all head-training splits.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--domains", default=",".join(DOMAINS))
    parser.add_argument("--recovery-mode", choices=("rule", "head", "clip_head"), default="rule")
    parser.add_argument("--recovery-train-seeds", default="not_recorded", help="Recovery-head training seed ranges for result metadata.")
    parser.add_argument("--recovery-test-seeds", default="not_recorded", help="Held-out recovery-bank seed ranges for result metadata.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--place-tcp-z", type=float, default=0.041)
    parser.add_argument("--log-every", type=int, default=0)
    return parser.parse_args()


def selected_domains(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in names if item not in DOMAINS]
    if unknown:
        raise KeyError(f"unknown domains: {unknown}")
    return names


def base_args(args: argparse.Namespace, task: str, complexity: str, domain: dict) -> dict:
    return {
        "task": task,
        "complexity": complexity,
        "workspace_profile": "core_v2",
        "image_size": args.image_size,
        "arm_kp": domain["arm_kp"],
        "arm_force": domain["arm_force"],
        "gripper_kp": domain["gripper_kp"],
        "gripper_force": domain["gripper_force"],
        "friction": domain["friction"],
        "place_tcp_z": args.place_tcp_z,
        "speed": 0.0,
    }


def rule_args(args: argparse.Namespace, task: str, complexity: str, domain: dict) -> SimpleNamespace:
    return SimpleNamespace(
        **base_args(args, task, complexity, domain),
        instruction=None,
        instruction_normalization="none",
        camera="top_rgb",
        feedback_attempts=1,
    )


def gate_args(args: argparse.Namespace, task: str, complexity: str, domain: dict) -> SimpleNamespace:
    return SimpleNamespace(**base_args(args, task, complexity, domain), recovery_mode=args.recovery_mode)


def flatten_rule(result: dict, domain: str) -> dict:
    return {
        "mode": "rule_rgb_retry",
        "domain": domain,
        "task": result["task"],
        "seed": result["seed"],
        "success": bool(result["task_success"]),
        "strict_grasp_success": bool(result["strict_grasp_success"]),
        "attempt_count": int(result["attempt_count"]),
        "retry_executed": bool(result["recovery_triggered"]),
        "gate_decision": str(result["recovery_reason"]),
        "terminal_label": None,
        "recovery_label": None,
        "target_distance_m": float(result["target_distance"]),
    }


def flatten_gate(result: dict, mode: str, domain: str) -> dict:
    return {
        "mode": mode,
        "domain": domain,
        "task": result["task"],
        "seed": result["seed"],
        "success": bool(result["task_success"]),
        "strict_grasp_success": bool(result["strict_grasp_success"]),
        "attempt_count": 1 + int(result["retry_executed"]),
        "retry_executed": bool(result["retry_executed"]),
        "gate_decision": result["gate_decision"],
        "terminal_label": result["terminal_prediction"]["label"],
        "recovery_label": result["recovery_prediction"]["label"],
        "target_distance_m": float(result["target_distance_m"]),
    }


def summaries(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["mode"], row["domain"], row["task"]), []).append(row)
    return [
        {
            "mode": key[0],
            "domain": key[1],
            "task": key[2],
            "episodes": len(group),
            "successes": sum(int(item["success"]) for item in group),
            "retry_executed": sum(int(item["retry_executed"]) for item in group),
            "mean_attempts": sum(float(item["attempt_count"]) for item in group) / len(group),
            "mean_target_distance_m": sum(float(item["target_distance_m"]) for item in group) / len(group),
        }
        for key, group in sorted(grouped.items())
    ]


def paired(rows: list[dict]) -> list[dict]:
    lookup: dict[tuple[str, str, int], dict[str, dict]] = {}
    for row in rows:
        lookup.setdefault((row["domain"], row["task"], int(row["seed"])), {})[row["mode"]] = row
    output = []
    for learned in sorted({row["mode"] for row in rows if row["mode"] != "rule_rgb_retry"}):
        groups: dict[tuple[str, str], dict[str, int]] = {}
        for (domain, task, _seed), item in lookup.items():
            if "rule_rgb_retry" not in item or learned not in item:
                continue
            group = groups.setdefault((domain, task), {"episodes": 0, "rule_successes": 0, "learned_successes": 0, "improved": 0, "regressed": 0})
            rule_success = bool(item["rule_rgb_retry"]["success"])
            learned_success = bool(item[learned]["success"])
            group["episodes"] += 1
            group["rule_successes"] += int(rule_success)
            group["learned_successes"] += int(learned_success)
            group["improved"] += int(not rule_success and learned_success)
            group["regressed"] += int(rule_success and not learned_success)
        for (domain, task), group in sorted(groups.items()):
            discordant = group["improved"] + group["regressed"]
            group["discordant"] = discordant
            group["exact_two_sided_p"] = min(1.0, 2.0 * (0.5**discordant)) if discordant else 1.0
            output.append({"learned_mode": learned, "domain": domain, "task": task, **group})
    return output


def report(rows: list[dict], grouped: list[dict], comparisons: list[dict], args: argparse.Namespace) -> str:
    lines = [
        "# 轻量多视角终局与恢复门禁评测",
        "",
        "版本：`clip_semantic_multiview_gate_v1`",
        "",
        "## 协议",
        "",
        f"- 留出 seed：`{args.seed}` 至 `{args.seed + args.episodes - 1}`，不与恢复头训练 `720–739` 或其验证 `740–749` 重叠。",
        f"- `rule_rgb_retry`：原始 RGB 规则重试；学习门禁的恢复模式为 `{args.recovery_mode}`，并比较顶视图与顶视图加前视图。",
        "- 运行时只读取 RGB 图像、冻结语义意图与静态任务配置。MuJoCo 物体状态只用于事后评分。",
        "- 终局头和恢复头只做 accept/retry/stop 决策，结构化 waypoint 执行器保持不变；这不是端到端 VLA 或 OpenVLA LoRA。",
        "",
        "## 分组结果",
        "",
        "| 方法 | 域 | 任务 | 成功 | 重试 | 平均尝试 | 平均目标距离 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in grouped:
        lines.append(f"| `{row['mode']}` | `{row['domain']}` | `{row['task']}` | {row['successes']}/{row['episodes']} | {row['retry_executed']}/{row['episodes']} | {row['mean_attempts']:.2f} | {row['mean_target_distance_m']:.4f} m |")
    lines.extend(["", "## 同 seed 配对", "", "| 学习门禁 | 域 | 任务 | 规则 | 学习门禁 | 改善 | 回退 | p 值 |", "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in comparisons:
        lines.append(f"| `{row['learned_mode']}` | `{row['domain']}` | `{row['task']}` | {row['rule_successes']}/{row['episodes']} | {row['learned_successes']}/{row['episodes']} | {row['improved']} | {row['regressed']} | {row['exact_two_sided_p']:.4f} |")
    lines.extend([
        "",
        "## 解释边界",
        "",
        "双视角被保留为消融项。若它没有优于顶视图，结论应是当前固定顶视相机已经包含足够判别信息，额外前视图引入有限样本下的噪声，而不是宣称多视角必然更好。",
        "测试集中若某个类别没有出现，应只报告该类别训练覆盖，不能将其写成该类别的独立泛化结果。小样本精确检验 p 值仅防止过度解读，不能作为统计显著性声明。",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    policy = load_policy(args.model)
    calibration = load_calibration(args.calibration)
    clip_model, processor = load_clip(str(policy["metadata"]["clip_model"]))
    if args.recovery_mode == "clip_head" and (args.clip_recovery_top is None or args.clip_recovery_top_front is None):
        raise ValueError("--clip-recovery-top and --clip-recovery-top-front are required for clip_head evaluation")
    suffix = {"rule": "rule_recovery", "head": "head_recovery", "clip_head": "clip_recovery"}[args.recovery_mode]
    heads = {
        f"learned_top_{suffix}": (
            load_head(args.terminal_top),
            load_head(args.recovery_top),
            load_clip_recovery_head(args.clip_recovery_top) if args.recovery_mode == "clip_head" else None,
        ),
        f"learned_top_front_{suffix}": (
            load_head(args.terminal_top_front),
            load_head(args.recovery_top_front),
            load_clip_recovery_head(args.clip_recovery_top_front) if args.recovery_mode == "clip_head" else None,
        ),
    }
    rows: list[dict] = []
    raw_results: list[dict] = []
    for domain_name in selected_domains(args.domains):
        task_specs = TASKS if domain_name == "nominal" else (SPATIAL_TASK,)
        for task, complexity in task_specs:
            for offset in range(args.episodes):
                seed = args.seed + offset
                rule_config = rule_args(args, task, complexity, DOMAINS[domain_name])
                env, obs = configure_rule_env(rule_config, seed)
                rule = rule_rollout(rule_config, policy, clip_model, processor, calibration, seed, env=env, obs=obs)
                rows.append(flatten_rule(rule, domain_name))
                raw_results.append({"mode": "rule_rgb_retry", "domain": domain_name, "result": rule})
                for mode, (terminal_head, recovery_head, clip_recovery_head) in heads.items():
                    config = gate_args(args, task, complexity, DOMAINS[domain_name])
                    env, obs = configure_gate_env(config, seed)
                    result = gate_rollout(
                        config,
                        policy,
                        clip_model,
                        processor,
                        calibration,
                        terminal_head,
                        recovery_head,
                        seed,
                        env=env,
                        obs=obs,
                        clip_recovery_head=clip_recovery_head,
                    )
                    rows.append(flatten_gate(result, mode, domain_name))
                    raw_results.append({"mode": mode, "domain": domain_name, "result": result})
                    if args.log_every and len(rows) % args.log_every == 0:
                        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    grouped = summaries(rows)
    comparisons = paired(rows)
    payload = {
        "version": "clip_semantic_multiview_gate_v1",
        "method": "frozen_clip_intent + rgb_grounding + learned terminal/recovery value heads + structured executor",
        "seed_protocol": {
            "evaluation_base": args.seed,
            "episodes": args.episodes,
            "recovery_train": args.recovery_train_seeds,
            "recovery_test": args.recovery_test_seeds,
        },
        "rows": rows,
        "raw_results": raw_results,
        "summary": grouped,
        "paired_comparisons": comparisons,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report(rows, grouped, comparisons, args), encoding="utf-8")
    for mode in sorted({row["mode"] for row in rows}):
        group = [row for row in rows if row["mode"] == mode]
        print(f"{mode}_success: {sum(int(row['success']) for row in group)}/{len(group)}", flush=True)
    print(f"output_json: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
