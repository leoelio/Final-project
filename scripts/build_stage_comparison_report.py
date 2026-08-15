from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


GROUP_ORDER = [
    "环境与示范",
    "数据验证",
    "结构化控制强基线",
    "普通模仿学习",
    "动作块 / 轨迹条件",
    "ACT-style",
    "Diffusion Policy",
    "VLA/action-head 代理",
    "VLM 表征代理",
    "轻量后训练代理",
    "参数高效后训练代理",
    "多任务 action-head 代理",
]


GROUP_TEXT = {
    "环境与示范": {
        "purpose": "验证桌面任务本身可执行，并生成可复现示范数据。",
        "finding": "规则 expert 仍是当前上界，适合解释任务可行性和示范来源。",
        "talk": "先播放 expert 视频，说明环境、物体、成功指标和数据采集链路。",
    },
    "数据验证": {
        "purpose": "验证保存的动作轨迹能够被 MuJoCo 重新回放。",
        "finding": "replay 只证明数据可复现，不代表策略学习能力。",
        "talk": "用于说明数据不是只存在于运行时，而是可以离线保存、复现和作为训练集。",
    },
    "结构化控制强基线": {
        "purpose": "提供可解释的阶段化控制上界，用来确认任务不是物理上不可完成。",
        "finding": "结构化 waypoint 在主任务和语言/空间任务上明显强于多数 learned baseline。",
        "talk": "强调它使用显式状态和规则，不是 VLA，但能揭示抓取任务需要阶段恢复和接触控制。",
    },
    "普通模仿学习": {
        "purpose": "建立 Linear BC、kNN BC、MLP BC 等普通 imitation learning 对照组。",
        "finding": "单步回归闭环不稳定；kNN 训练范围强但留出和语言泛化弱。",
        "talk": "这一阶段回答“简单 BC 是否足够”，结论是否定的。",
    },
    "动作块 / 轨迹条件": {
        "purpose": "测试历史窗口和动作块是否能补足单步 BC 的短视问题。",
        "finding": "trajectory-kNN 能记住训练轨迹，但 held-out 失败；动作块 MLP 更平滑但仍不稳定抓取。",
        "talk": "适合作为 ACT 前的过渡 baseline，不能写成完整 ACT。",
    },
    "ACT-style": {
        "purpose": "用 Transformer action chunk、CVAE latent 和轻量视觉代理特征建立 ACT-style baseline。",
        "finding": "结构更接近 ACT，但 state-only 或 pooled RGB 代理版本仍不能稳定解决接触、夹紧和抬升。",
        "talk": "这里的结论应写成 state-only/ACT-lite baseline 不足，而不是完整视觉 ACT 失败。",
    },
    "Diffusion Policy": {
        "purpose": "建立动作块扩散策略对照，区分 NumPy-lite 和 PyTorch state-only diffusion。",
        "finding": "PyTorch diffusion 能记录资源和训练指标，但闭环仍未稳定成功。",
        "talk": "用于说明失败不只是 NumPy 实现过弱，state-only 小数据 diffusion 也不足。",
    },
    "VLA/action-head 代理": {
        "purpose": "在本地可复现条件下模拟 VLA action-head-only 路线。",
        "finding": "对象/语言/视觉统计特征 action head 已搭好，但当前闭环成功率仍不足。",
        "talk": "强调这是 VLA 路线的本地 proxy，不是 OpenVLA 或 RT-2 后训练。",
    },
    "VLM 表征代理": {
        "purpose": "测试冻结通用 VLM 表征加轻量 action head 的可行性。",
        "finding": "冻结 CLIP 表征不能直接补足机器人接触控制和阶段执行。",
        "talk": "可以作为“已有 VLM 表征 + action head”的弱代理，但不能写成机器人 VLA。",
    },
    "轻量后训练代理": {
        "purpose": "测试 reward-weighted BC 是否能作为轻量后训练代理。",
        "finding": "加权后训练代理未带来稳定闭环提升，说明偏好/奖励权重需要更强数据或在线修正。",
        "talk": "不能写成在线 RL，只能写成 reward-weighted BC proxy。",
    },
    "参数高效后训练代理": {
        "purpose": "对比 action head only、Adapter、LoRA-style 等可训练参数规模。",
        "finding": "Adapter/LoRA-style 可训练参数很少，但当前 proxy 效果仍不稳定。",
        "talk": "这部分适合回答“省算力/省参数”，但不能宣称完成 pretrained VLA LoRA。",
    },
    "多任务 action-head 代理": {
        "purpose": "测试 naive 多任务混合数据是否自然带来语言/空间泛化。",
        "finding": "多任务 action head 离线误差正常但闭环失败，说明简单混合不等于 VLA 泛化。",
        "talk": "用于引出后续需要真实机器人预训练表征和更明确任务条件。",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a stage-level Chinese comparison report from the method audit table.")
    parser.add_argument("--audit-csv", type=Path, default=ROOT / "docs" / "method_stage_audit.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "stage_comparison_report.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "stage_comparison_report.csv")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def success_value(text: str) -> tuple[int, int] | None:
    match = re.search(r"(\d+)\s*/\s*(\d+)", text or "")
    if not match:
        return None
    total = int(match.group(2))
    if total <= 0:
        return None
    return int(match.group(1)), total


def best_success(rows: list[dict[str, str]], column: str) -> str:
    best_row = None
    best_rate = -1.0
    best_pair = None
    for row in rows:
        pair = success_value(row.get(column, ""))
        if not pair:
            continue
        rate = pair[0] / pair[1]
        if rate > best_rate:
            best_rate = rate
            best_row = row
            best_pair = pair
    if not best_row or not best_pair:
        return "未评测"
    return f"{best_row['版本']}：{best_pair[0]}/{best_pair[1]} ({best_rate:.0%})"


def parse_param(text: str) -> int | None:
    cleaned = (text or "").replace(",", "").strip()
    if not cleaned or not cleaned.isdigit():
        return None
    return int(cleaned)


def param_range(rows: list[dict[str, str]]) -> str:
    values = [value for value in (parse_param(row.get("可训练参数", "")) for row in rows) if value is not None]
    if not values:
        return "未记录"
    if min(values) == max(values):
        return f"{min(values):,}"
    return f"{min(values):,} - {max(values):,}"


def group_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["阶段分组"], []).append(row)
    return groups


def ordered_groups(groups: dict[str, list[dict[str, str]]]) -> list[str]:
    ordered = [group for group in GROUP_ORDER if group in groups]
    ordered.extend(group for group in groups if group not in ordered)
    return ordered


def md_escape(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(md_escape(value) for value in values) + " |"


def stage_summary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups = group_rows(rows)
    summaries = []
    for group in ordered_groups(groups):
        group_items = groups[group]
        text = GROUP_TEXT.get(group, {})
        summaries.append(
            {
                "阶段分组": group,
                "方法数": str(len(group_items)),
                "代表版本": "、".join(row["版本"] for row in group_items[:3]),
                "最好训练范围": best_success(group_items, "主任务训练范围"),
                "最好留出范围": best_success(group_items, "主任务留出范围"),
                "最好语言泛化": best_success(group_items, "语言泛化"),
                "可训练参数范围": param_range(group_items),
                "阶段目的": text.get("purpose", "未记录"),
                "阶段结论": text.get("finding", "未记录"),
                "讲解建议": text.get("talk", "未记录"),
                "视频证据": "、".join(row["固定视频"] for row in group_items[:3]),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, audit_rows: list[dict[str, str]], summaries: list[dict[str, str]]) -> None:
    groups = group_rows(audit_rows)
    lines = [
        "# 阶段对比报告",
        "",
        "版本：`stage_comparison_report_v1`",
        "",
        "用途：按毕业设计研究流程，把当前正式实验版本从“单方法表”整理成“阶段说明 + 评测比较 + 视频证据”的中文报告。该报告面向论文结果章节、答辩讲解和后续实验续写。",
        "",
        "数据来源：`docs/method_stage_audit.csv`、`docs/evaluation_summary.csv`、`docs/language_generalization_summary.csv`、`docs/model_resource_summary.csv` 和 `outputs/videos`。",
        "",
        "## 1. 阶段总览",
        "",
        md_row(["阶段", "方法数", "最好训练范围", "最好留出范围", "最好语言泛化", "参数范围", "阶段结论"]),
        md_row(["---", "---:", "---", "---", "---", "---:", "---"]),
    ]
    for item in summaries:
        lines.append(
            md_row(
                [
                    item["阶段分组"],
                    item["方法数"],
                    item["最好训练范围"],
                    item["最好留出范围"],
                    item["最好语言泛化"],
                    item["可训练参数范围"],
                    item["阶段结论"],
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 2. 阶段详表",
            "",
        ]
    )
    summary_by_group = {item["阶段分组"]: item for item in summaries}
    detail_header = ["版本", "方法", "主任务训练范围", "主任务留出范围", "语言泛化", "可训练参数", "固定视频", "论文可写", "论文红线"]
    for group in ordered_groups(groups):
        summary = summary_by_group[group]
        lines.extend(
            [
                f"### {group}",
                "",
                f"阶段目的：{summary['阶段目的']}",
                "",
                f"阶段结论：{summary['阶段结论']}",
                "",
                f"讲解建议：{summary['讲解建议']}",
                "",
                md_row(detail_header),
                md_row(["---", "---", "---:", "---:", "---:", "---:", "---", "---", "---"]),
            ]
        )
        for row in groups[group]:
            lines.append(
                md_row(
                    [
                        f"`{row['版本']}`",
                        row["方法"],
                        row["主任务训练范围"],
                        row["主任务留出范围"],
                        row["语言泛化"],
                        row["可训练参数"],
                        f"`{row['固定视频']}`",
                        row["论文可写"],
                        row["论文红线"],
                    ]
                )
            )
        lines.append("")

    lines.extend(
        [
            "## 3. 可写结论",
            "",
            "1. 环境、示范、回放、固定视频和 dashboard 已经形成可复现实验闭环。",
            "2. 普通 BC、动作块、ACT-style、Diffusion-style 和 action-head proxy 都已经有正式版本名、artifact、评测表和视频证据。",
            "3. 当前最强的可解释对照仍是结构化 waypoint；学习型 proxy 的失败主要体现在接触、夹紧、抬升和跨 seed 泛化。",
            "4. LoRA/Adapter-style proxy 的参数规模优势可以写，但不能写成真实 pretrained VLA 的 LoRA/Adapter 后训练结果。",
            "5. 后续若加入 OpenVLA、Isaac 或真实机械臂，需要继续沿用本报告的字段：版本名、阶段、artifact、主任务评测、语言泛化、资源记录、固定视频和论文红线。",
            "",
            "## 4. 视频证据入口",
            "",
            "```text",
            "docs/defense_deck.html",
            "docs/presentation_video_pack.md",
            "outputs/presentation_clips/00_defense_video_reel.mp4",
            "outputs/showcase/all_registered_methods_grid.mp4",
            "outputs/showcase/language_generalization_grid.mp4",
            "outputs/videos/*.mp4",
            "```",
            "",
            "## 5. 重建命令",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_stage_comparison_report.py"}"',
            "```",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    audit_rows = read_rows(args.audit_csv)
    summaries = stage_summary_rows(audit_rows)
    write_csv(args.output_csv, summaries)
    write_md(args.output_md, audit_rows, summaries)
    print(f"stage_comparison_md: {args.output_md}", flush=True)
    print(f"stage_comparison_csv: {args.output_csv}", flush=True)
    print(f"stage_groups: {len(summaries)}", flush=True)


if __name__ == "__main__":
    main()
