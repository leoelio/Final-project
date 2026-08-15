from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "trajectory_act_failure_diagnosis_v1"

sys.path.insert(0, str(ROOT / "scripts"))
from build_trajectory_act_stage_report import STAGE_VERSIONS, STRUCTURE_NOTES  # noqa: E402


FIELDNAMES = [
    "版本",
    "方法",
    "阶段",
    "结构定位",
    "主任务训练范围",
    "主任务留出范围",
    "语言/空间泛化",
    "固定视频",
    "success",
    "target_distance",
    "ee_object_distance",
    "object_z",
    "grasp_success",
    "contact_count",
    "mean_action_norm",
    "max_action_norm",
    "接触诊断",
    "夹紧/抬升诊断",
    "泛化诊断",
    "动作速度诊断",
    "可写结论",
    "论文红线",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a trajectory/ACT failure diagnosis table from fixed video metadata.")
    parser.add_argument("--trajectory-stage", type=Path, default=ROOT / "docs" / "trajectory_act_stage_report.csv")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "trajectory_act_failure_diagnosis.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "trajectory_act_failure_diagnosis.md")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", "<br>") for value in values) + " |"


def score(text: str) -> tuple[int, int]:
    if "/" not in text:
        return 0, 0
    left, right = text.split("/", 1)
    try:
        return int(float(left)), int(float(right))
    except ValueError:
        return 0, 0


def yes(value: object) -> bool:
    return str(value).lower() == "true"


def fmt(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def contact_diagnosis(metrics: dict[str, object]) -> str:
    contact_count = float(metrics.get("contact_count", 0.0))
    ee_distance = float(metrics.get("ee_object_distance", 999.0))
    success = yes(metrics.get("success", False))
    if contact_count <= 0 and ee_distance > 0.15:
        return "末端没有稳定进入接触区。"
    if success:
        return "固定 seed 已到达目标附近，但仍需看批量成功率。"
    if contact_count > 0:
        return "有接触，但接触没有转化为稳定夹紧和移动。"
    return "接近阶段不稳定，未形成有效抓取。"


def lift_diagnosis(metrics: dict[str, object]) -> str:
    grasp = yes(metrics.get("grasp_success", False))
    object_z = float(metrics.get("object_z", 0.0))
    if grasp and object_z > 0.05:
        return "出现夹紧和抬升迹象。"
    if grasp:
        return "夹紧成立但抬升不足。"
    if object_z <= 0.04:
        return "未稳定夹住，物体基本停留在桌面高度。"
    return "物体有扰动，但没有稳定抬升。"


def generalization_diagnosis(stage_row: dict[str, str]) -> str:
    train_ok, train_total = score(stage_row["主任务训练范围"])
    heldout_ok, heldout_total = score(stage_row["主任务留出范围"])
    language_ok, language_total = score(stage_row["语言/空间泛化"])
    if train_ok > 0 and heldout_ok == 0 and language_ok == 0:
        return "训练范围有成功但留出和语言任务失败，偏轨迹记忆。"
    if heldout_ok > 0 and language_ok == 0:
        return "留出范围有少量成功，但语言/空间泛化仍缺失。"
    if train_total and train_ok == 0 and heldout_ok == 0:
        return "训练范围和留出范围都不稳定，不能作为可靠闭环策略。"
    if language_total and language_ok == 0:
        return "语言/空间泛化失败。"
    return "泛化证据不足，需看批量评测。"


def speed_diagnosis(summary: dict[str, object], stage_row: dict[str, str]) -> str:
    max_norm = float(summary.get("max_action_norm", 0.0))
    mean_norm = float(summary.get("mean_action_norm", 0.0))
    command = stage_row["主任务viewer命令"]
    has_safety = "--speed 0.05" in command and "--max-arm-delta" in command and "--stop-on-unsafe" in command
    if has_safety and max_norm > 1.0:
        return "viewer 已慢速限幅；模型原始动作波动仍偏大，失败不能只归因于播放速度。"
    if has_safety:
        return "viewer 已慢速限幅，主要问题应看接触、夹紧和泛化。"
    if mean_norm > 0.8 or max_norm > 1.2:
        return "动作幅度偏大，后续可做控制限幅扫表。"
    return "动作幅度不是主要异常。"


def paper_claim(stage_row: dict[str, str], metrics: dict[str, object]) -> str:
    if yes(metrics.get("success", False)) and not yes(metrics.get("grasp_success", False)):
        return "可写固定 seed 到达目标附近，但不能写成稳定抓取成功。"
    if stage_row["版本"] == "trajectory_knn_chunk_bc_v1":
        return "可写训练分布内轨迹记忆有效，但不能写成泛化策略。"
    return "可写本地 trajectory/ACT 对照组仍主要失败在接触、夹紧、抬升或泛化。"


def build_rows(stage_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_version = {row["版本"]: row for row in stage_rows}
    rows: list[dict[str, str]] = []
    for version in STAGE_VERSIONS:
        stage_row = by_version[version]
        video = stage_row["主任务视频"]
        metadata_path = ROOT / video.replace("/", "\\")
        metadata = read_json(metadata_path.with_suffix(".json"))
        summary = metadata.get("summary", {})
        metrics = summary.get("metrics", {})
        if not isinstance(summary, dict) or not isinstance(metrics, dict):
            raise RuntimeError(f"video metadata lacks summary metrics: {metadata_path}")
        rows.append(
            {
                "版本": version,
                "方法": stage_row["方法"],
                "阶段": stage_row["阶段"],
                "结构定位": STRUCTURE_NOTES[version],
                "主任务训练范围": stage_row["主任务训练范围"],
                "主任务留出范围": stage_row["主任务留出范围"],
                "语言/空间泛化": stage_row["语言/空间泛化"],
                "固定视频": video,
                "success": str(summary.get("success", metrics.get("success", ""))),
                "target_distance": fmt(summary.get("target_distance", metrics.get("target_distance"))),
                "ee_object_distance": fmt(metrics.get("ee_object_distance")),
                "object_z": fmt(summary.get("object_z", metrics.get("object_z"))),
                "grasp_success": str(summary.get("grasp_success", metrics.get("grasp_success", ""))),
                "contact_count": fmt(metrics.get("contact_count"), digits=1),
                "mean_action_norm": fmt(summary.get("mean_action_norm")),
                "max_action_norm": fmt(summary.get("max_action_norm")),
                "接触诊断": contact_diagnosis(metrics),
                "夹紧/抬升诊断": lift_diagnosis(metrics),
                "泛化诊断": generalization_diagnosis(stage_row),
                "动作速度诊断": speed_diagnosis(summary, stage_row),
                "可写结论": paper_claim(stage_row, metrics),
                "论文红线": stage_row["论文红线"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, str]]) -> None:
    success_rows = [row for row in rows if row["success"] == "True"]
    grasp_rows = [row for row in rows if row["grasp_success"] == "True"]
    contact_rows = [row for row in rows if float(row["contact_count"] or 0.0) > 0.0]
    high_action_rows = [row for row in rows if float(row["max_action_norm"] or 0.0) > 1.0]

    lines = [
        "# Trajectory / ACT 失败诊断矩阵",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：基于固定视频的 JSON 元数据，细分 trajectory-conditioned BC / ACT / Diffusion 对照组的失败原因。该诊断不替代成功率表，只解释为什么这些方法还不能作为可靠策略。",
        "",
        "关键边界：当前 viewer 命令已经统一使用 `--duration 60 --speed 0.05` 和动作限幅；因此本阶段主要问题不能简单写成“播放速度太快”，而应定位到闭环接触、夹紧、抬升和泛化。",
        "",
        "## 1. 诊断总览",
        "",
        md_row(["项目", "数量"]),
        md_row(["---", "---:"]),
        md_row(["纳入版本", str(len(rows))]),
        md_row(["固定视频 success=True", str(len(success_rows))]),
        md_row(["grasp_success=True", str(len(grasp_rows))]),
        md_row(["contact_count > 0", str(len(contact_rows))]),
        md_row(["max_action_norm > 1.0", str(len(high_action_rows))]),
        "",
        "## 2. 方法级诊断",
        "",
        md_row(["版本", "方法", "成功", "目标距离", "末端-物体距离", "接触数", "夹住", "物体高度", "动作速度诊断", "接触诊断", "夹紧/抬升诊断", "泛化诊断"]),
        md_row(["---", "---", "---", "---:", "---:", "---:", "---", "---:", "---", "---", "---", "---"]),
    ]
    for row in rows:
        lines.append(
            md_row(
                [
                    f"`{row['版本']}`",
                    row["方法"],
                    row["success"],
                    row["target_distance"],
                    row["ee_object_distance"],
                    row["contact_count"],
                    row["grasp_success"],
                    row["object_z"],
                    row["动作速度诊断"],
                    row["接触诊断"],
                    row["夹紧/抬升诊断"],
                    row["泛化诊断"],
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 3. 论文可写结论与红线",
            "",
            md_row(["版本", "可写结论", "论文红线", "固定视频"]),
            md_row(["---", "---", "---", "---"]),
        ]
    )
    for row in rows:
        lines.append(md_row([f"`{row['版本']}`", row["可写结论"], row["论文红线"], f"`{row['固定视频']}`"]))

    lines.extend(
        [
            "",
            "## 4. 使用说明",
            "",
            "1. 本表用于解释 trajectory/ACT 对照组，不用于替代 `docs/trajectory_act_stage_report.csv` 的批量成功率。",
            "2. 如果要展示“速度是否太快”，优先引用本表的动作速度诊断和 `docs/reproducible_command_index.md` 中的慢速 viewer 命令。",
            "3. 如果后续重新训练 ACT 或修改控制限幅，需要重新生成本表并保存新视频，不能只覆盖旧结论。",
            "",
            "## 5. 重建命令",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_trajectory_act_failure_diagnosis.py"}"',
            "```",
            "",
            "## 6. 代表视频",
            "",
            "```powershell",
            f'Start-Process "{ROOT / "outputs" / "presentation_clips" / "03_trajectory_act_diffusion.mp4"}"',
            f'Start-Process "{ROOT / "outputs" / "videos" / "trajectory_knn_chunk_bc_v1_seed0.mp4"}"',
            f'Start-Process "{ROOT / "outputs" / "videos" / "torch_act_state_chunk_v1_seed0.mp4"}"',
            f'Start-Process "{ROOT / "outputs" / "videos" / "visual_act_cnn_cvae_v1_seed0.mp4"}"',
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    stage_rows = read_csv(args.trajectory_stage)
    rows = build_rows(stage_rows)
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows)
    print(f"trajectory_act_failure_diagnosis_md: {args.output_md}", flush=True)
    print(f"trajectory_act_failure_diagnosis_csv: {args.output_csv}", flush=True)
    print(f"trajectory_act_failure_diagnosis_rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
