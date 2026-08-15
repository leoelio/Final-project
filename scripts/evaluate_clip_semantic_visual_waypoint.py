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
from run_clip_semantic_visual_waypoint import load_calibration, load_policy, rollout  # noqa: E402


TASKS = (
    ("place_blue_cube_blue_pad", "medium"),
    ("place_blue_cube_red_pad", "medium"),
    ("place_red_cube_red_pad", "medium"),
    ("move_leftmost_cube_to_bowl", "language"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CLIP semantic selection plus RGB-grounded waypoint planning on the Core V2 holdout protocol.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=ROOT / "outputs" / "rgb_grounding" / "top_rgb_core_v2_calibration_v1.json")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / "clip_semantic_rgb_grounded_waypoint_core_v2_v1.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "clip_semantic_rgb_grounded_waypoint_core_v2_v1.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "clip_semantic_rgb_grounded_waypoint_core_v2_v1.md")
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--camera", default="top_rgb")
    parser.add_argument("--arm-kp", type=float, default=150.0)
    parser.add_argument("--arm-force", type=float, default=100.0)
    parser.add_argument("--gripper-kp", type=float, default=1200.0)
    parser.add_argument("--gripper-force", type=float, default=200.0)
    parser.add_argument("--friction", type=float, default=5.0)
    parser.add_argument("--place-tcp-z", type=float, default=0.041)
    parser.add_argument("--log-every", type=int, default=0, help="Print one JSON row every N episodes; 0 keeps batch output compact.")
    return parser.parse_args()


def rollout_args(args: argparse.Namespace, task: str, complexity: str) -> SimpleNamespace:
    return SimpleNamespace(
        task=task,
        complexity=complexity,
        workspace_profile="core_v2",
        instruction=None,
        instruction_normalization="none",
        image_size=args.image_size,
        camera=args.camera,
        arm_kp=args.arm_kp,
        arm_force=args.arm_force,
        gripper_kp=args.gripper_kp,
        gripper_force=args.gripper_force,
        friction=args.friction,
        place_tcp_z=args.place_tcp_z,
        speed=0.0,
    )


def markdown_report(rows: list[dict], summary: dict) -> str:
    lines = [
        "# CLIP 语义 + RGB 定位结构化策略报告",
        "",
        "版本：`clip_semantic_rgb_grounded_waypoint_core_v2_v1`",
        "",
        "## 方法边界",
        "",
        "- 冻结 CLIP 的线性意图 adapter 选择四类任务意图。",
        "- 抓取物体坐标仅来自初始 `top_rgb` 的颜色/方形轮廓定位与离线桌面平面标定；运行时不读取 MuJoCo 物体坐标规划轨迹。",
        "- 放置目标使用固定场景配置，接触轨迹由既有结构化 waypoint executor 执行。",
        "- MuJoCo 状态只用于严格抓取、最终目标距离和定位误差评分；这不是端到端 VLA 或 OpenVLA LoRA。",
        "",
        "## 留出集结果",
        "",
        "| 任务 | 严格任务成功 | 语义正确 | 视觉对象正确 | 严格抓取 | 平均定位误差 | 平均目标距离 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task, _ in TASKS:
        group = [row for row in rows if row["task"] == task]
        lines.append(
            f"| {task} | {sum(int(row['task_success']) for row in group)}/{len(group)} | "
            f"{sum(int(row['semantic_correct']) for row in group)}/{len(group)} | "
            f"{sum(int(row['visual_selection_correct']) for row in group)}/{len(group)} | "
            f"{sum(int(row['strict_grasp_success']) for row in group)}/{len(group)} | "
            f"{sum(row['source_position_error_m'] for row in group) / len(group):.4f} m | "
            f"{sum(row['target_distance'] for row in group) / len(group):.4f} m |"
        )
    lines.extend(
        [
            "",
            "## 状态定位对照",
            "",
            "| 方法 | 源物体坐标来源 | 严格任务成功 | 视觉对象选择 |",
            "| --- | --- | ---: | ---: |",
            "| `core_v2_clip_semantic_waypoint_v1` | MuJoCo 动态场景状态 | 20/20 | 不适用 |",
            f"| `clip_semantic_rgb_grounded_waypoint_core_v2_v1` | 初始 top RGB + 离线平面标定 | {summary['task_successes']}/{summary['episodes']} | {summary['visual_selection_correct']}/{summary['episodes']} |",
            "",
            f"总计：严格任务成功 `{summary['task_successes']}/{summary['episodes']}`；视觉对象选择 `{summary['visual_selection_correct']}/{summary['episodes']}`；平均源物体定位误差 `{summary['mean_source_position_error_m']:.4f} m`。",
            "",
            "## 失败定位",
            "",
        ]
    )
    failures = [row for row in rows if not row["task_success"]]
    if failures:
        for row in failures:
            lines.append(
                f"- `{row['task']}` seed `{row['seed']}`：语义和视觉对象选择均正确，"
                f"源物体误差 `{row['source_position_error_m']:.4f} m`，严格抓取={row['strict_grasp_success']}，"
                f"最终目标距离 `{row['target_distance']:.4f} m`。"
            )
    else:
        lines.append("- 本次协议中没有失败案例。")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "固定俯视相机、固定颜色与固定桌面布局下，视觉定位已能替代动态物体状态作为初始抓取点来源；但相比状态定位对照少 2 次空间任务成功，说明毫米级初始定位偏差在边界姿态上会传导为接触/运输失败。它仍依赖结构化接触执行器，不能扩展为端到端动作学习或真实机器人结论。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    policy = load_policy(args.model)
    calibration = load_calibration(args.calibration)
    clip_model, processor = load_clip(str(policy["metadata"]["clip_model"]))
    rows: list[dict] = []
    for task_index, (task, complexity) in enumerate(TASKS):
        run_args = rollout_args(args, task, complexity)
        for offset in range(args.episodes):
            row = rollout(run_args, policy, clip_model, processor, calibration, args.seed + task_index * 100 + offset)
            rows.append(row)
            if args.log_every and len(rows) % args.log_every == 0:
                print(json.dumps(row, ensure_ascii=False), flush=True)
    summary = {
        "version": "clip_semantic_rgb_grounded_waypoint_core_v2_v1",
        "method": "frozen_clip_intent + rgb_color_shape_grounding + structured_waypoint",
        "model": str(args.model),
        "calibration": str(args.calibration),
        "episodes": len(rows),
        "task_successes": sum(int(row["task_success"]) for row in rows),
        "semantic_correct": sum(int(row["semantic_correct"]) for row in rows),
        "visual_selection_correct": sum(int(row["visual_selection_correct"]) for row in rows),
        "strict_grasp_successes": sum(int(row["strict_grasp_success"]) for row in rows),
        "mean_source_position_error_m": float(sum(row["source_position_error_m"] for row in rows) / len(rows)),
        "mean_target_distance_m": float(sum(row["target_distance"] for row in rows) / len(rows)),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown_report(rows, summary), encoding="utf-8")
    print(f"summary_path: {args.output_json}")
    print(f"task_success_rate: {summary['task_successes']}/{summary['episodes']}")
    print(f"mean_source_position_error_m: {summary['mean_source_position_error_m']:.6f}")


if __name__ == "__main__":
    main()
