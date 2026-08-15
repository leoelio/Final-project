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

from torch_runtime import ensure_torch_path  # noqa: E402
from vlm_runtime import ensure_vlm_path  # noqa: E402

ensure_torch_path()
ensure_vlm_path()

from run_clip_action_head import load_clip  # noqa: E402
from run_clip_semantic_rgb_feedback import configure_env, load_calibration, rollout  # noqa: E402
from run_clip_semantic_waypoint import load_policy  # noqa: E402


TASKS = (
    ("place_blue_cube_blue_pad", "medium"),
    ("place_blue_cube_red_pad", "medium"),
    ("place_red_cube_red_pad", "medium"),
    ("move_leftmost_cube_to_bowl", "language"),
)
TASK_SEED_OFFSETS = {task: index * 100 for index, (task, _) in enumerate(TASKS)}
SPATIAL_TASK = "move_leftmost_cube_to_bowl"
DOMAINS = {
    "nominal": {
        "arm_kp": 150.0,
        "arm_force": 100.0,
        "gripper_kp": 1200.0,
        "gripper_force": 200.0,
        "friction": 5.0,
        "description": "Core V2 标准接触参数",
    },
    "mild_contact_shift": {
        "arm_kp": 135.0,
        "arm_force": 90.0,
        "gripper_kp": 950.0,
        "gripper_force": 150.0,
        "friction": 2.5,
        "description": "中等摩擦和夹爪力下降，用于检验闭环恢复敏感性",
    },
    "low_contact_shift": {
        "arm_kp": 120.0,
        "arm_force": 80.0,
        "gripper_kp": 750.0,
        "gripper_force": 110.0,
        "friction": 1.5,
        "description": "较强接触扰动，不等同于真实 sim-to-real 验证",
    },
}
MODES = {"rgb_open_loop": 0, "rgb_visual_retry": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RGB open-loop versus RGB feedback retry across Core V2 tasks and contact shifts.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=ROOT / "outputs" / "rgb_grounding" / "top_rgb_core_v2_calibration_v1.json")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / "clip_semantic_rgb_feedback_closed_loop_v1.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "clip_semantic_rgb_feedback_closed_loop_v1.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "clip_semantic_rgb_feedback_closed_loop_v1.md")
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--task-seed-offsets",
        default="0,100,200,300",
        help="Comma-separated seed offsets in TASKS order; default preserves the original protocol.",
    )
    parser.add_argument("--domains", default="nominal,mild_contact_shift,low_contact_shift")
    parser.add_argument("--modes", default="rgb_open_loop,rgb_visual_retry")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--camera", default="top_rgb")
    parser.add_argument("--place-tcp-z", type=float, default=0.041)
    parser.add_argument("--log-every", type=int, default=0)
    return parser.parse_args()


def selected(value: str, available: dict, label: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in names if item not in available]
    if unknown:
        raise KeyError(f"unknown {label}: {unknown}")
    return names


def task_seed_offsets(value: str) -> dict[str, int]:
    offsets = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(offsets) != len(TASKS):
        raise ValueError(f"expected {len(TASKS)} task seed offsets, got {len(offsets)}")
    return {task: offset for (task, _), offset in zip(TASKS, offsets, strict=True)}


def rollout_args(args: argparse.Namespace, task: str, complexity: str, domain: dict, feedback_attempts: int) -> SimpleNamespace:
    return SimpleNamespace(
        task=task,
        complexity=complexity,
        workspace_profile="core_v2",
        instruction=None,
        instruction_normalization="none",
        image_size=args.image_size,
        camera=args.camera,
        arm_kp=domain["arm_kp"],
        arm_force=domain["arm_force"],
        gripper_kp=domain["gripper_kp"],
        gripper_force=domain["gripper_force"],
        friction=domain["friction"],
        place_tcp_z=args.place_tcp_z,
        feedback_attempts=feedback_attempts,
        recovery_profile="standard",
        speed=0.0,
    )


def flat_row(result: dict, mode: str, domain_name: str, domain: dict) -> dict:
    return {
        "version": "clip_semantic_rgb_feedback_closed_loop_v1",
        "mode": mode,
        "domain": domain_name,
        "task": result["task"],
        "seed": result["seed"],
        "success": result["task_success"],
        "semantic_correct": result["semantic_correct"],
        "visual_selection_correct": result["visual_selection_correct"],
        "strict_grasp_success": result["strict_grasp_success"],
        "attempt_count": result["attempt_count"],
        "recovery_triggered": result["recovery_triggered"],
        "recovery_reason": result["recovery_reason"],
        "initial_source_position_error_m": result["initial_source_position_error_m"],
        "final_source_position_error_m": result["final_source_position_error_m"],
        "target_distance_m": result["target_distance"],
        "out_of_table": result["out_of_table"],
        "arm_kp": domain["arm_kp"],
        "arm_force": domain["arm_force"],
        "gripper_kp": domain["gripper_kp"],
        "gripper_force": domain["gripper_force"],
        "friction": domain["friction"],
    }


def summary_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["mode"], row["domain"], row["task"]), []).append(row)
    summaries = []
    for key, group in sorted(grouped.items()):
        count = len(group)
        summaries.append(
            {
                "mode": key[0],
                "domain": key[1],
                "task": key[2],
                "episodes": count,
                "successes": sum(int(row["success"]) for row in group),
                "semantic_correct": sum(int(row["semantic_correct"]) for row in group),
                "visual_selection_correct": sum(int(row["visual_selection_correct"]) for row in group),
                "strict_grasp_success": sum(int(row["strict_grasp_success"]) for row in group),
                "recovery_triggered": sum(int(row["recovery_triggered"]) for row in group),
                "mean_attempt_count": sum(float(row["attempt_count"]) for row in group) / count,
                "mean_target_distance_m": sum(float(row["target_distance_m"]) for row in group) / count,
            }
        )
    return summaries


def paired_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int], dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault((row["domain"], row["task"], int(row["seed"])), {})[row["mode"]] = row
    comparisons: dict[tuple[str, str], dict[str, int]] = {}
    for (domain, task, _seed), pair in grouped.items():
        if set(pair) != {"rgb_open_loop", "rgb_visual_retry"}:
            continue
        item = comparisons.setdefault(
            (domain, task),
            {"episodes": 0, "open_successes": 0, "retry_successes": 0, "improved": 0, "regressed": 0},
        )
        open_success = bool(pair["rgb_open_loop"]["success"])
        retry_success = bool(pair["rgb_visual_retry"]["success"])
        item["episodes"] += 1
        item["open_successes"] += int(open_success)
        item["retry_successes"] += int(retry_success)
        item["improved"] += int(not open_success and retry_success)
        item["regressed"] += int(open_success and not retry_success)
    output = []
    for (domain, task), item in sorted(comparisons.items()):
        discordant = item["improved"] + item["regressed"]
        exact_two_sided_p = min(1.0, 2.0 * (0.5 ** discordant)) if discordant else 1.0
        output.append({"domain": domain, "task": task, **item, "discordant": discordant, "exact_two_sided_p": exact_two_sided_p})
    return output


def markdown_report(args: argparse.Namespace, rows: list[dict], summaries: list[dict], paired: list[dict], domain_names: list[str]) -> str:
    lines = [
        "# RGB 视觉反馈闭环恢复评测",
        "",
        "版本：`clip_semantic_rgb_feedback_closed_loop_v1`",
        "",
        "## 研究问题",
        "",
        "在冻结 CLIP 语义选择和结构化抓放不变时，RGB 终局观测触发的一次重定位能否降低初始视觉定位在接触阶段造成的失败？",
        "",
        "## 方法边界",
        "",
        "- `rgb_open_loop`：只用初始 top RGB 定位一次源物体。",
        "- `rgb_visual_retry`：首次执行后再次从 RGB 观察源物体；仅当物体仍被视觉检测到位于源工作区时，才允许一次重新定位和重试。",
        "- 轨迹规划和重试决策不读取 MuJoCo 动态物体/目标坐标；真值仅用于评分。",
        "- 同色物体放入同色盘时，纯颜色分割不能可靠视觉验证终局，因此会如实记录为 `same_color_object_and_pad`，不伪造完成信号。",
        "- 接触扰动只是在 MuJoCo 中改变摩擦、执行器和夹爪参数，不代表 Isaac 或真实机械臂迁移。",
        "",
        "## 扰动域",
        "",
        "| 域 | arm kp/force | gripper kp/force | 摩擦 | 说明 |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for name in domain_names:
        domain = DOMAINS[name]
        lines.append(f"| `{name}` | {domain['arm_kp']} / {domain['arm_force']} | {domain['gripper_kp']} / {domain['gripper_force']} | {domain['friction']} | {domain['description']} |")
    lines.extend(
        [
            "",
            "## 汇总结果",
            "",
            "| 策略 | 域 | 任务 | 成功 | 语义 | 视觉对象 | 严格抓取 | 重试触发 | 平均尝试次数 | 平均目标距离 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summaries:
        lines.append(
            f"| `{row['mode']}` | `{row['domain']}` | `{row['task']}` | {row['successes']}/{row['episodes']} | "
            f"{row['semantic_correct']}/{row['episodes']} | {row['visual_selection_correct']}/{row['episodes']} | "
            f"{row['strict_grasp_success']}/{row['episodes']} | {row['recovery_triggered']}/{row['episodes']} | "
            f"{row['mean_attempt_count']:.2f} | {row['mean_target_distance_m']:.4f} m |"
        )
    lines.extend(
        [
            "",
            "## 同 seed 配对差异",
            "",
            "| 域 | 任务 | 单次定位 | 视觉重试 | 改善 | 回退 | 不同结果对 | 精确双侧 p 值 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in paired:
        lines.append(
            f"| `{row['domain']}` | `{row['task']}` | {row['open_successes']}/{row['episodes']} | "
            f"{row['retry_successes']}/{row['episodes']} | {row['improved']} | {row['regressed']} | "
            f"{row['discordant']} | {row['exact_two_sided_p']:.4f} |"
        )
    lines.extend(
        [
            "",
            "样本量为每个配对 5 个 seed；精确检验只用于避免过度解读，不将 p 值大于 0.05 的小样本改善写成统计显著。",
        ]
    )
    failures = [row for row in rows if not row["success"]]
    lines.extend(["", "## 失败样本索引", ""])
    if failures:
        for row in failures:
            lines.append(
                f"- `{row['mode']}` / `{row['domain']}` / `{row['task']}` / seed `{row['seed']}`："
                f"strict_grasp={row['strict_grasp_success']}，attempts={row['attempt_count']}，"
                f"recovery=`{row['recovery_reason']}`，target_distance={float(row['target_distance_m']):.4f} m。"
            )
    else:
        lines.append("- 本次矩阵中没有失败样本。")
    lines.extend(
        [
            "",
            "## 复现与视频",
            "",
            "- 逐 episode 数据：`outputs/evaluations/clip_semantic_rgb_feedback_closed_loop_v1.json`。",
            "- 汇总 CSV：`docs/clip_semantic_rgb_feedback_closed_loop_v1.csv`。",
            "- 代表性视频仅保存到 `presentation_videos/rgb_feedback_loop_v1/`，按失败、视觉恢复和扰动诊断分别保留，避免重复案例。",
            "",
            "## 论文表述边界",
            "",
            "可以写成固定桌面、固定相机和闭集颜色条件下的 RGB 视觉反馈闭环消融；不能写成端到端 VLA、OpenVLA LoRA、通用视觉抓取或真实机器人结果。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    domain_names = selected(args.domains, DOMAINS, "domains")
    mode_names = selected(args.modes, MODES, "modes")
    seed_offsets = task_seed_offsets(args.task_seed_offsets)
    policy = load_policy(args.model)
    calibration = load_calibration(args.calibration)
    clip_model, processor = load_clip(str(policy["metadata"]["clip_model"]))
    rows: list[dict] = []
    raw_results: list[dict] = []
    for domain_name in domain_names:
        task_specs = TASKS if domain_name == "nominal" else tuple(item for item in TASKS if item[0] == SPATIAL_TASK)
        for mode_index, mode_name in enumerate(mode_names):
            for task, complexity in task_specs:
                run_args = rollout_args(args, task, complexity, DOMAINS[domain_name], MODES[mode_name])
                for offset in range(args.episodes):
                    seed = args.seed + seed_offsets[task] + offset
                    env, obs = configure_env(run_args, seed)
                    result = rollout(run_args, policy, clip_model, processor, calibration, seed, env=env, obs=obs)
                    raw_results.append({"mode": mode_name, "domain": domain_name, "result": result})
                    rows.append(flat_row(result, mode_name, domain_name, DOMAINS[domain_name]))
                    if args.log_every and len(rows) % args.log_every == 0:
                        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    summaries = summary_rows(rows)
    paired = paired_rows(rows)
    payload = {
        "version": "clip_semantic_rgb_feedback_closed_loop_v1",
        "method": "frozen_clip_intent + rgb_grounding + one_visual_retry",
        "model": str(args.model),
        "calibration": str(args.calibration),
        "domains": {name: DOMAINS[name] for name in domain_names},
        "modes": MODES,
        "task_seed_offsets": seed_offsets,
        "rows": rows,
        "raw_results": raw_results,
        "summary": summaries,
        "paired_comparisons": paired,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown_report(args, rows, summaries, paired, domain_names), encoding="utf-8")
    total = len(rows)
    print(f"summary_path: {args.output_json}")
    print(f"episodes: {total}")
    for mode_name in mode_names:
        mode_rows = [row for row in rows if row["mode"] == mode_name]
        print(f"{mode_name}_success: {sum(int(row['success']) for row in mode_rows)}/{len(mode_rows)}")


if __name__ == "__main__":
    main()
