from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fixed_counts(data: dict) -> dict:
    rows = [row for row in data["rows"] if row["mode"] == "rgb_open_loop" and row["domain"] == "nominal"]
    return {
        "episodes": len(rows),
        "grounded": sum(bool(row["visual_selection_correct"]) for row in rows),
        "success": sum(bool(row["success"]) for row in rows),
        "red_success": sum(bool(row["success"]) for row in rows if row["task"] == "place_red_cube_red_pad"),
        "red_distance": sum(float(row["target_distance_m"]) for row in rows if row["task"] == "place_red_cube_red_pad") / 5,
    }


def format_ratio(value: int, total: int) -> str:
    return f"{value}/{total} ({100.0 * value / total:.1f}%)"


def wilson_95(value: int, total: int) -> str:
    z = 1.959963984540054
    rate = value / total
    denominator = 1.0 + z**2 / total
    center = (rate + z**2 / (2.0 * total)) / denominator
    radius = z * ((rate * (1.0 - rate) / total + z**2 / (4.0 * total**2)) ** 0.5) / denominator
    return f"[{100.0 * (center - radius):.1f}%, {100.0 * (center + radius):.1f}%]"

def main() -> None:
    before = fixed_counts(load(ROOT / "outputs" / "evaluations" / "clip_semantic_rgb_feedback_patch_pointer_holdout_v1.json"))
    after = fixed_counts(load(ROOT / "outputs" / "evaluations" / "rgb_grounding_refinement_v2_fixed20.json"))
    extended = load(ROOT / "outputs" / "evaluations" / "rgb_grounding_refinement_v2_extended_v1.json")
    overall = extended["overall"]
    failures = [row for row in extended["rows"] if not row["task_success"]]
    failure_groups = Counter((row["task"], str(row.get("recovery_reason", "unknown"))) for row in failures)

    lines = [
        "# RGB 同色目标剔除阶段报告",
        "",
        "版本：`rgb_grounding_refinement_v2_extended_v1`",
        "",
        "## 结论",
        "",
        "在固定顶视相机、封闭颜色集合和 MuJoCo Core V2 桌面环境中，静态同色目标盘会与邻近源方块在 RGB 颜色掩码中粘连。基于离线平面标定剔除静态盘半径 6.5 cm 内的同色像素后，固定 20 条同 seed 留出从 19/20 严格成功恢复到 20/20；扩展 144 条跨任务、跨接触域评测中没有初始 RGB 定位失败，最终严格成功为 132/144。",
        "",
        "这是一项运行时 RGB 几何前端修复，不是端到端 VLA、OpenVLA 微调、LoRA 成功结果或真实机械臂结论。MuJoCo 物体位姿、分割图和接触真值只用于离线评分，未参与定位、重试决策或轨迹规划。",
        "",
        "## 同 seed 前后对照",
        "",
        "固定任务 seed 为蓝到蓝 `20-24`、蓝到红 `120-124`、红到红 `220-224`、最左方块到碗 `420-424`。以下仅比较 `rgb_open_loop / nominal`，因此差异可归因于 RGB 源定位规则，而不是接触参数或策略模型变化。",
        "",
        "| 指标 | 静态盘剔除前 | 静态盘剔除后 |",
        "| --- | ---: | ---: |",
        f"| 初始对象选择正确 | {format_ratio(before['grounded'], before['episodes'])} | {format_ratio(after['grounded'], after['episodes'])} |",
        f"| 严格任务成功 | {format_ratio(before['success'], before['episodes'])} | {format_ratio(after['success'], after['episodes'])} |",
        f"| 红方块到红盘 | {before['red_success']}/5 | {after['red_success']}/5 |",
        f"| 红方块到红盘平均目标距离 | {before['red_distance'] * 100:.2f} cm | {after['red_distance'] * 100:.2f} cm |",
        "",
        "修复前唯一失败是 `place_red_cube_red_pad / seed 222`：初始红色候选完全被拒绝。第一次仅放宽填充率后，候选可被检测但位置误差仍约 9.7 mm，抓取失败；最终版本先剔除静态红盘，再保持保守的方形阈值，离线定位误差为 2.5 mm，并在 viewer 中严格完成放置。",
        "",
        "## 扩展 144 条结果",
        "",
        "协议固定为 4 项任务 × 3 档接触域 × 12 个 seed（`3400-3411`），详情见 `docs/rgb_grounding_refinement_v2_preregistered.md`。它与历史 144 条协议的任务组成不同，不能用于历史前端的配对因果比较。",
        "",
        "| 指标 | 结果 | Wilson 95% CI |",
        "| --- | ---: | ---: |",
        f"| 初始 RGB 定位可执行 | {format_ratio(overall['initial_grounding_executable'], overall['episodes'])} | {wilson_95(overall['initial_grounding_executable'], overall['episodes'])} |",
        f"| 冻结 CLIP 语义正确 | {format_ratio(overall['semantic_correct'], overall['episodes'])} | {wilson_95(overall['semantic_correct'], overall['episodes'])} |",
        f"| 视觉对象选择正确 | {format_ratio(overall['visual_selection_correct'], overall['episodes'])} | {wilson_95(overall['visual_selection_correct'], overall['episodes'])} |",
        f"| 首轮严格成功 | {format_ratio(overall['first_attempt_success'], overall['episodes'])} | {wilson_95(overall['first_attempt_success'], overall['episodes'])} |",
        f"| 触发 RGB 重试 | {overall['recovery_triggered']}/{overall['episodes']} | - |",
        f"| 最终严格任务成功 | {format_ratio(overall['task_success'], overall['episodes'])} | {wilson_95(overall['task_success'], overall['episodes'])} |",
        "",
        "| 任务 | 初始定位 | 对象选择 | 最终成功 | Wilson 95% CI |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for task, item in extended["by_task"].items():
        lines.append(
            f"| `{task}` | {item['initial_grounding_executable']}/{item['episodes']} | "
            f"{item['visual_selection_correct']}/{item['episodes']} | {format_ratio(item['task_success'], item['episodes'])} | "
            f"{wilson_95(item['task_success'], item['episodes'])} |"
        )
    lines.extend(["", "| 接触域 | 首轮成功 | 最终成功 | Wilson 95% CI |", "| --- | ---: | ---: | ---: |"])
    for domain, item in extended["by_domain"].items():
        lines.append(
            f"| `{domain}` | {format_ratio(item['first_attempt_success'], item['episodes'])} | "
            f"{format_ratio(item['task_success'], item['episodes'])} | {wilson_95(item['task_success'], item['episodes'])} |"
        )
    lines.extend(["", "## 失败分层", "", "| 任务 | 终止原因 | episode 数 |"])
    lines.append("| --- | --- | ---: |")
    for (task, reason), count in sorted(failure_groups.items()):
        lines.append(f"| `{task}` | `{reason}` | {count} |")
    lines.extend(
        [
            "",
            "12 次失败均发生在接触执行或最左关系任务，源定位、语义和对象选择均为正确。4 次 RGB 重试均未将首轮失败转为严格成功，因此不能把该重试模块描述为本协议中的性能增益；它只保留为可观测的故障恢复机制。",
            "",
            "## 视频证据",
            "",
            "- 蓝方块到蓝盘 viewer 成功案例：`videos/rgb_grounding_refinement_v2/seed3400_blue_to_blue_success.mp4`。该视频的离线定位误差为 0.64 mm、最终目标距离为 1.04 cm。",
            "- 同色定位修复后的 viewer 成功案例：`videos/rgb_grounding_refinement_v1/seed222_target_mask_success.mp4`。该视频的离线定位误差为 2.5 mm、最终目标距离为 2.8 mm；它用于复核，不替代上表的统计结果。",
            "- 已保留的最左任务严重接触失败诊断：`videos/leftmost_margin_v1/seed3403_severe_retry_failure.mp4`。失败视频只保留一条代表性案例，避免用重复失败素材替代失败分层统计。",
            "",
            "## 完整复现",
            "",
            "```powershell",
            "$env:VLA_TORCH_PACKAGE_DIR='D:\\vla_torch_cuda_pkgs'",
            "& .\\.venv\\Scripts\\python.exe .\\scripts\\evaluate_clip_semantic_rgb_feedback.py `",
            "  --model .\\outputs\\clip_semantic_waypoint\\clip_semantic_waypoint_core_v2_v1_20260721_110325.npz `",
            "  --seed 20 --task-seed-offsets 0,100,200,400 --episodes 5 --domains nominal `",
            "  --modes rgb_open_loop,rgb_visual_retry `",
            "  --output-json .\\outputs\\evaluations\\rgb_grounding_refinement_v2_fixed20.json `",
            "  --output-csv .\\docs\\rgb_grounding_refinement_v2_fixed20.csv `",
            "  --output-md .\\docs\\rgb_grounding_refinement_v2_fixed20.md",
            "",
            "& .\\.venv\\Scripts\\python.exe .\\scripts\\evaluate_rgb_frontend_end_to_end.py `",
            "  --model .\\outputs\\clip_semantic_waypoint\\clip_semantic_waypoint_core_v2_v1_20260721_110325.npz `",
            "  --seed 3400 --episodes 12 `",
            "  --tasks place_blue_cube_blue_pad,place_blue_cube_red_pad,place_red_cube_red_pad,move_leftmost_cube_to_bowl `",
            "  --version rgb_grounding_refinement_v2_extended_v1 `",
            "  --output-json .\\outputs\\evaluations\\rgb_grounding_refinement_v2_extended_v1.json `",
            "  --output-md .\\docs\\rgb_grounding_refinement_v2_extended_v1_raw.md",
            "",
            "& .\\.venv\\Scripts\\python.exe .\\scripts\\build_rgb_grounding_refinement_v2_report.py",
            "```",
        ]
    )
    output = ROOT / "docs" / "rgb_grounding_refinement_v2_stage.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report_path: {output}")


if __name__ == "__main__":
    main()
