from __future__ import annotations

import json
from collections import Counter
from math import comb
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V2_PATH = ROOT / "outputs" / "evaluations" / "rgb_object_identity_v2_extended_v1.json"
V3_PATH = ROOT / "outputs" / "evaluations" / "rgb_occlusion_recovery_v3_extended_v1.json"
OUTPUT = ROOT / "docs" / "rgb_occlusion_recovery_v3_stage.md"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ratio(value: int, total: int) -> str:
    return f"{value}/{total} ({100.0 * value / total:.1f}%)"


def wilson_95(value: int, total: int) -> str:
    z = 1.959963984540054
    rate = value / total
    denominator = 1.0 + z**2 / total
    center = (rate + z**2 / (2.0 * total)) / denominator
    radius = z * ((rate * (1.0 - rate) / total + z**2 / (4.0 * total**2)) ** 0.5) / denominator
    return f"[{100.0 * (center - radius):.1f}%, {100.0 * (center + radius):.1f}%]"


def exact_two_sided(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if not discordant:
        return 1.0
    tail = sum(comb(discordant, value) for value in range(min(improved, regressed) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def main() -> None:
    v2 = load(V2_PATH)
    v3 = load(V3_PATH)
    overall = v3["overall"]
    total = overall["episodes"]
    recovered = [row for row in v3["rows"] if not row["first_attempt_success"] and row["task_success"]]
    regressions = [row for row in v3["rows"] if row["first_attempt_success"] and not row["task_success"]]
    failures = [row for row in v3["rows"] if not row["task_success"]]
    failure_groups = Counter((row["task"], str(row.get("recovery_reason", "unknown"))) for row in failures)
    v2_overall = v2["overall"]

    lines = [
        "# RGB 遮挡恢复 V3 阶段报告",
        "",
        "版本：`rgb_occlusion_recovery_v3_extended_v1`",
        "",
        "## 结论",
        "",
        "在固定 MuJoCo Core V2 桌面场景、冻结 CLIP 意图、RGB 几何定位和结构化抓放轨迹不变的前提下，V3 将 RGB 重定位的填充率阈值仅在首轮失败后由 `0.70` 放宽为 `0.50`，用于应对机械臂遮挡造成的方块轮廓缺失。最终结论只来自未参与调参的新 seed `3800-3811`。",
        "",
        "该结果描述的是分层 RGB 视觉-语言-动作系统，不是端到端 VLA、OpenVLA 微调、LoRA 的正向性能结果，也不外推到真实机械臂。",
        "",
        "## 协议与运行时边界",
        "",
        "- 4 项任务 × 3 档 MuJoCo 接触域 × 12 个新 seed，共 144 个 episode；完整协议见 `docs/rgb_occlusion_recovery_v3_preregistered.md`。",
        "- 初始源物体识别保持严格的面积 `160-650 px`、填充率 `>=0.70` 和同色静态目标盘排除，防止把同色圆柱或球当成方块。",
        "- 仅在首轮未完成时允许一次 RGB 重定位；恢复阶段仍要求面积、源工作区和最近历史位置，只将填充率下限设为 `0.50`。",
        "- 运行时不读取 MuJoCo 物体真值决定重试或规划；真值仅用于离线成功率和对象选择误差统计。",
        "",
        "## V3 独立结果",
        "",
        "| 指标 | 结果 | Wilson 95% CI |",
        "| --- | ---: | ---: |",
    ]
    for key, label in (
        ("initial_grounding_executable", "初始 RGB 定位可执行"),
        ("semantic_correct", "冻结 CLIP 语义正确"),
        ("visual_selection_correct", "离线 RGB 对象选择正确"),
        ("first_attempt_success", "首轮严格成功"),
        ("task_success", "最终严格任务成功"),
    ):
        lines.append(f"| {label} | {ratio(overall[key], total)} | {wilson_95(overall[key], total)} |")
    lines.append(f"| 触发 RGB 重试 | {overall['recovery_triggered']}/{total} | - |")
    lines.append(f"| 首轮失败后被重试挽回 | {len(recovered)}/{total} | - |")

    lines.extend(
        [
            "",
            "## 同 episode 配对对照：一次开环动作 vs. 允许一次 RGB 重试",
            "",
            "每个 episode 的首轮动作在是否重试之前完全相同，因此首轮严格成功可作为该 episode 的一次开环对照。允许重试后，成功数由 "
            f"`{overall['first_attempt_success']}/{total}` 变为 `{overall['task_success']}/{total}`：挽回 {len(recovered)} 条，回退 {len(regressions)} 条，"
            f"不一致配对 {len(recovered) + len(regressions)} 条，精确双侧检验 `p={exact_two_sided(len(recovered), len(regressions)):.4f}`。",
            "",
            "这表明一次 RGB 重定位在本协议中有正向挽回迹象，但因有效不一致配对仅为少量样本，不能称为统计显著的性能提升。",
        ]
    )

    lines.extend(["", "## 分任务结果", "", "| 任务 | 初始定位 | 对象选择 | 首轮成功 | 触发重试 | 最终成功 | Wilson 95% CI |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for task, item in v3["by_task"].items():
        episodes = item["episodes"]
        lines.append(
            f"| `{task}` | {item['initial_grounding_executable']}/{episodes} | {item['visual_selection_correct']}/{episodes} | "
            f"{item['first_attempt_success']}/{episodes} | {item['recovery_triggered']}/{episodes} | "
            f"{ratio(item['task_success'], episodes)} | {wilson_95(item['task_success'], episodes)} |"
        )

    lines.extend(["", "## 分接触域结果", "", "| 接触域 | 首轮成功 | 触发重试 | 最终成功 | Wilson 95% CI |", "| --- | ---: | ---: | ---: | ---: |"])
    for domain, item in v3["by_domain"].items():
        episodes = item["episodes"]
        lines.append(
            f"| `{domain}` | {ratio(item['first_attempt_success'], episodes)} | {item['recovery_triggered']}/{episodes} | "
            f"{ratio(item['task_success'], episodes)} | {wilson_95(item['task_success'], episodes)} |"
        )

    lines.extend(["", "## 前序诊断与独立性", "", "V2 的 seed `3600-3611` 在身份规则冻结后得到 `" + ratio(v2_overall["task_success"], v2_overall["episodes"]) + "` 最终成功，其中严重接触域的最左方块任务残留两条失败。对这两条失败进行 RGB 画面诊断后，发现正确方块仍可见，但填充率被遮挡压低到严格初始阈值以下。随后才引入 V3 的仅恢复阶段阈值放宽。",
        "",
        "因此，V2 结果只作为问题诊断，不与 V3 做未配对的性能增益声明；V3 的新 seed `3800-3811` 才是该恢复规则的独立端到端证据。",
        "",
        "## 失败分层",
        "",
        "| 任务 | 终止原因 | episode 数 |",
        "| --- | --- | ---: |",
    ])
    if failure_groups:
        for (task, reason), count in sorted(failure_groups.items()):
            lines.append(f"| `{task}` | `{reason}` | {count} |")
    else:
        lines.append("| - | 无最终失败 | 0 |")

    lines.extend([
        "",
        "## 视频证据",
        "",
        "- 蓝方块到蓝盘的 viewer 成功案例：`videos/rgb_grounding_refinement_v2/seed3400_blue_to_blue_success.mp4`。",
        "- 同色红方块到红盘的 viewer 成功案例：`videos/rgb_grounding_refinement_v1/seed222_target_mask_success.mp4`。",
        "- 同色异形干扰修复后的蓝方块案例：`videos/rgb_terminal_identity_v1/seed3410_after.mp4`。",
        "- 首轮失败后由 RGB 重定位挽回的最左方块案例：`videos/rgb_object_identity_v2/seed3609_leftmost_recovery_success.mp4`。",
        "- V3 将补录一条新 seed 的 viewer 案例，作为本协议的可视化证据；视频只用于复核，不替代上述 144 条统计。",
        "",
        "## 复现",
        "",
        "```powershell",
        "$env:VLA_TORCH_PACKAGE_DIR='D:\\vla_torch_cuda_pkgs'",
        "& .\\.venv\\Scripts\\python.exe .\\scripts\\evaluate_rgb_frontend_end_to_end.py `",
        "  --model .\\outputs\\clip_semantic_waypoint\\clip_semantic_waypoint_core_v2_v1_20260721_110325.npz `",
        "  --seed 3800 --episodes 12 `",
        "  --tasks place_blue_cube_blue_pad,place_blue_cube_red_pad,place_red_cube_red_pad,move_leftmost_cube_to_bowl `",
        "  --version rgb_occlusion_recovery_v3_extended_v1 `",
        "  --output-json .\\outputs\\evaluations\\rgb_occlusion_recovery_v3_extended_v1.json `",
        "  --output-md .\\docs\\rgb_occlusion_recovery_v3_extended_v1_raw.md",
        "",
        "& .\\.venv\\Scripts\\python.exe .\\scripts\\build_rgb_occlusion_recovery_v3_report.py",
        "```",
    ])
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report_path: {OUTPUT}")


if __name__ == "__main__":
    main()
