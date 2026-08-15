from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "trajectory_act_conclusion_brief_v1"

FIELDNAMES = [
    "分组",
    "版本",
    "方法",
    "结构定位",
    "训练范围",
    "留出范围",
    "语言/空间泛化",
    "固定视频",
    "固定视频结果",
    "抓取标志",
    "物体高度",
    "可写结论",
    "论文红线",
    "推荐展示命令",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese paper/defense conclusion brief for trajectory/ACT baselines.")
    parser.add_argument("--record", type=Path, default=ROOT / "docs" / "trajectory_act_experiment_record.csv")
    parser.add_argument("--stage-report", type=Path, default=ROOT / "docs" / "trajectory_act_stage_report.csv")
    parser.add_argument("--diagnosis", type=Path, default=ROOT / "docs" / "trajectory_act_failure_diagnosis.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "trajectory_act_conclusion_brief.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "trajectory_act_conclusion_brief.csv")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def ratio(value: str) -> tuple[int, int] | None:
    if "/" not in value:
        return None
    left, right = value.split("/", 1)
    try:
        return int(left), int(right)
    except ValueError:
        return None


def ratio_sum(rows: list[dict[str, str]], field: str) -> tuple[int, int]:
    success = 0
    total = 0
    for row in rows:
        parsed = ratio(row.get(field, ""))
        if parsed is None:
            continue
        success += parsed[0]
        total += parsed[1]
    return success, total


def percent(success: int, total: int) -> str:
    if total == 0:
        return "未登记"
    return f"{success}/{total} ({success / total:.0%})"


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(value.replace("|", "\\|").replace("\n", "<br>") for value in values) + " |"


def build_rows(
    record_rows: list[dict[str, str]],
    stage_rows: list[dict[str, str]],
    diagnosis_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    core_versions = {row["版本"] for row in record_rows}
    record_by_version = {row["版本"]: row for row in record_rows}
    diagnosis_by_version = {row["版本"]: row for row in diagnosis_rows}
    rows: list[dict[str, str]] = []
    for stage in stage_rows:
        version = stage["版本"]
        record = record_by_version.get(version, {})
        diagnosis = diagnosis_by_version.get(version, {})
        is_core = version in core_versions
        viewer_command = record.get("主任务Viewer命令") or stage.get("主任务viewer命令", "")
        rows.append(
            {
                "分组": "trajectory-conditioned BC / ACT 正式对照" if is_core else "Diffusion 相邻对照",
                "版本": version,
                "方法": stage["方法"],
                "结构定位": stage["结构定位"],
                "训练范围": stage["主任务训练范围"],
                "留出范围": stage["主任务留出范围"],
                "语言/空间泛化": stage["语言/空间泛化"],
                "固定视频": stage["主任务视频"],
                "固定视频结果": diagnosis.get("success", record.get("固定视频结果", "")),
                "抓取标志": diagnosis.get("grasp_success", record.get("抓取标志", "")),
                "物体高度": diagnosis.get("object_z", record.get("物体高度", "")),
                "可写结论": stage["论文结论"],
                "论文红线": stage["论文红线"],
                "推荐展示命令": viewer_command,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, str]], record_rows: list[dict[str, str]]) -> None:
    core_rows = [row for row in rows if row["分组"] == "trajectory-conditioned BC / ACT 正式对照"]
    adjacent_rows = [row for row in rows if row["分组"] == "Diffusion 相邻对照"]
    train_success, train_total = ratio_sum(record_rows, "训练范围成功率")
    heldout_success, heldout_total = ratio_sum(record_rows, "留出范围成功率")
    language_success, language_total = ratio_sum(record_rows, "语言/空间泛化")
    fixed_video_successes = sum(1 for row in core_rows if row["固定视频结果"] in {"True", "success=True"})
    grasp_successes = sum(1 for row in core_rows if row["抓取标志"] in {"True", "success=True"})
    lines = [
        "# Trajectory / ACT 论文结论摘要",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：把 trajectory-conditioned BC / ACT-style baseline 的版本名称、评测结果、视频证据、可写结论和论文红线压缩成一页式中文摘要，服务论文结果章节和答辩讲解。该文档由现有 CSV 自动生成，不新增实验结论。",
        "",
        "## 1. 一句话结论",
        "",
        "- 当前已经打通 trajectory-conditioned BC / ACT-style baseline 的训练、闭环运行、固定视频和 viewer 复现链路。",
        "- `trajectory_knn_chunk_bc_v1` 在训练范围成功率高，但 held-out 和语言/空间泛化失败，适合写成轨迹记忆型 baseline，不适合写成泛化策略。",
        "- state-only ACT、ACT-CVAE 和视觉 ACT-lite 的结构升级没有自动解决接触、夹紧和抬升，当前结果只能说明本地轻量代理不足，不能证明完整官方 ACT 无效。",
        "- 固定视频中存在目标距离成功样例，但严格抓取口径仍显示抓取/抬升不足，因此论文必须同时报告 `success`、`grasp_success` 和 `object_z`。",
        "",
        "## 2. 覆盖统计",
        "",
        md_row(["项目", "数量/结果"]),
        md_row(["---", "---:"]),
        md_row(["trajectory-conditioned BC / ACT 正式对照", str(len(core_rows))]),
        md_row(["Diffusion 相邻对照", str(len(adjacent_rows))]),
        md_row(["核心对照训练范围总成功", percent(train_success, train_total)]),
        md_row(["核心对照留出范围总成功", percent(heldout_success, heldout_total)]),
        md_row(["核心对照语言/空间泛化总成功", percent(language_success, language_total)]),
        md_row(["核心固定视频 success=True", str(fixed_video_successes)]),
        md_row(["核心固定视频 grasp_success=True", str(grasp_successes)]),
        "",
        "## 3. 论文可写口径",
        "",
        md_row(["版本", "结构定位", "训练范围", "留出范围", "语言/空间泛化", "固定视频", "抓取标志", "可写结论", "论文红线"]),
        md_row(["---", "---", "---:", "---:", "---:", "---", "---", "---", "---"]),
    ]
    for row in rows:
        lines.append(
            md_row(
                [
                    f"`{row['版本']}`",
                    row["结构定位"],
                    row["训练范围"],
                    row["留出范围"],
                    row["语言/空间泛化"],
                    f"`{row['固定视频']}`",
                    row["抓取标志"],
                    row["可写结论"],
                    row["论文红线"],
                ]
            )
        )
    lines.extend(
        [
            "",
            "## 4. 推荐展示顺序",
            "",
            "1. 先播放 `outputs/presentation_clips/03_trajectory_act_diffusion.mp4`，说明本阶段不是单一模型，而是一组动作块、trajectory、ACT-style 和 Diffusion 对照。",
            "2. 再打开 `trajectory_knn_chunk_bc_v1` 的固定视频或 viewer，说明训练范围成功与泛化失败之间的差异。",
            "3. 接着展示 `torch_act_state_chunk_v1`、`torch_act_cvae_state_chunk_v1` 和 `visual_act_cnn_cvae_v1`，说明结构增强没有自动转化为稳定抓取。",
            "4. 最后引用 `docs/trajectory_act_failure_diagnosis.md` 和 `docs/strict_grasp_success_audit.md`，强调成功率、抓取标志和物体高度必须同时报告。",
            "",
            "## 5. 关键打开命令",
            "",
            "打开本摘要：",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "showcase_launcher.py"}" --target trajectory-act-brief',
            "```",
            "",
            "打开 trajectory/ACT 中文实验台账：",
            "",
            "```powershell",
            f'Start-Process notepad.exe "{ROOT / "docs" / "trajectory_act_experiment_record.md"}"',
            "```",
            "",
            "打开 trajectory/ACT 阶段报告：",
            "",
            "```powershell",
            f'Start-Process notepad.exe "{ROOT / "docs" / "trajectory_act_stage_report.md"}"',
            "```",
            "",
            "重建本摘要：",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_trajectory_act_conclusion_brief.py"}"',
            "```",
            "",
            "完整验证：",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "verify_experiment_artifacts.py"}"',
            "```",
            "",
            f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    record_rows = read_csv(args.record)
    stage_rows = read_csv(args.stage_report)
    diagnosis_rows = read_csv(args.diagnosis)
    rows = build_rows(record_rows, stage_rows, diagnosis_rows)
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows, record_rows)
    print(f"trajectory_act_conclusion_brief_md: {args.output_md}", flush=True)
    print(f"trajectory_act_conclusion_brief_csv: {args.output_csv}", flush=True)
    print(f"trajectory_act_conclusion_brief_rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
