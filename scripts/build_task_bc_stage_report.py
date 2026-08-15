from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_experiment_command_index import MAIN_RUN_DIR, ps_command, training_command, viewer_command  # noqa: E402


STAGE_VERSIONS = [
    "expert_scripted_v1",
    "structured_waypoint_policy_v1",
    "replay_demo_v1",
    "linear_bc_v1",
    "knn_bc_v1",
    "mlp_bc_v1",
]


STRUCTURE_NOTES = {
    "expert_scripted_v1": "脚本专家 / oracle / 示范数据生成器",
    "structured_waypoint_policy_v1": "显式阶段控制 + 目标物/目标区域状态访问的结构化强对照",
    "replay_demo_v1": "保存的示范轨迹回放，用于验证数据可复现",
    "linear_bc_v1": "单步线性行为克隆弱 baseline",
    "knn_bc_v1": "非神经网络 kNN 行为克隆 / 轨迹记忆 baseline",
    "mlp_bc_v1": "单步 MLP 行为克隆神经网络 baseline",
}


PAPER_CONCLUSIONS = {
    "expert_scripted_v1": "任务在 MuJoCo 中可解，脚本专家可用于采集示范数据；不能写成 learned policy。",
    "structured_waypoint_policy_v1": "显式阶段结构和状态访问可以稳定完成主任务，并能作为 learned 方法失败时的任务可解性强对照。",
    "replay_demo_v1": "保存的 `.npz` 示范轨迹可以重新回放，证明采集数据可复现；它不是策略学习方法。",
    "linear_bc_v1": "线性单步回归在闭环中失败，说明离线 MSE 或单步动作拟合不能替代闭环成功率。",
    "knn_bc_v1": "训练范围成功但 held-out 和语言泛化弱，说明它更像轨迹记忆而非泛化策略。",
    "mlp_bc_v1": "MLP 单步 BC 仍不稳定，说明仅提高函数逼近能力不足以解决长时序接触控制。",
}


FIELDNAMES = [
    "版本",
    "阶段",
    "方法",
    "结构定位",
    "artifact",
    "主任务训练范围",
    "主任务留出范围",
    "语言/空间泛化",
    "可训练参数",
    "模型大小MB",
    "feature_dim",
    "stored_samples",
    "主任务视频",
    "语言视频",
    "失败模式",
    "论文结论",
    "论文红线",
    "主任务viewer命令",
    "语言viewer命令",
    "训练/采集命令",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese stage report for task/data/oracle/basic BC baselines.")
    parser.add_argument("--evaluation", type=Path, default=ROOT / "docs" / "evaluation_summary.csv")
    parser.add_argument("--language-summary", type=Path, default=ROOT / "docs" / "language_generalization_summary.csv")
    parser.add_argument("--resource-summary", type=Path, default=ROOT / "docs" / "model_resource_summary.csv")
    parser.add_argument("--video-evidence", type=Path, default=ROOT / "docs" / "video_evidence_index.csv")
    parser.add_argument("--failure-taxonomy", type=Path, default=ROOT / "docs" / "failure_mode_taxonomy.csv")
    parser.add_argument("--data-efficiency", type=Path, default=ROOT / "docs" / "data_efficiency_summary.csv")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "task_bc_stage_report.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "task_bc_stage_report.md")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def by_version(rows: list[dict[str, str]], key: str = "version") -> dict[str, dict[str, str]]:
    return {row[key]: row for row in rows}


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", "<br>") for value in values) + " |"


def success_rate(text: str) -> float:
    if "/" not in text:
        return 0.0
    left, right = text.split("/", 1)
    try:
        return float(left) / max(1.0, float(right))
    except ValueError:
        return 0.0


def aggregate_failure_modes(rows: list[dict[str, str]]) -> dict[str, str]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[row["版本"]][row["失败模式"]] += 1
    return {
        version: "；".join(f"{mode} x{count}" for mode, count in counter.most_common())
        for version, counter in grouped.items()
    }


def first_video(rows: list[dict[str, str]], version: str, video_type: str) -> str:
    for row in rows:
        if row.get("版本") == version and row.get("视频类型") == video_type:
            return row.get("视频文件", "")
    if version == "expert_scripted_v1" and video_type == "语言/空间泛化片段":
        for row in rows:
            if row.get("版本") == "expert_scripted_language_v1":
                return row.get("视频文件", "")
    return ""


def collect_demos_command() -> str:
    return ps_command(
        "scripts/collect_demos.py",
        [
            "--task",
            "place_blue_cube_blue_pad",
            "--complexity",
            "medium",
            "--seed",
            "0",
            "--episodes",
            "5",
            "--output",
            ROOT / "data" / "demos",
            "--run-name",
            "place_blue_cube_blue_pad_medium_visual_check",
            "--min-success-rate",
            "0.7",
            "--viewer",
            "--duration",
            "60",
            "--speed",
            "0.05",
        ],
    )


def build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    evaluations = by_version(read_csv(args.evaluation), "version")
    language = by_version(read_csv(args.language_summary), "version")
    resources = by_version(read_csv(args.resource_summary), "version")
    video_rows = read_csv(args.video_evidence)
    failure_modes = aggregate_failure_modes(read_csv(args.failure_taxonomy))

    rows: list[dict[str, str]] = []
    for version in STAGE_VERSIONS:
        evaluation = evaluations[version]
        resource = resources.get(version, {})
        language_key = "expert_scripted_language_v1" if version == "expert_scripted_v1" else version
        language_row = language.get(language_key, {})
        method = {
            "version": version,
            "stage": evaluation["stage"],
            "method": evaluation["method"],
            "artifact": evaluation["artifact"],
            "clip": evaluation["clip"],
        }
        train = collect_demos_command() if version == "expert_scripted_v1" else training_command(version)
        rows.append(
            {
                "版本": version,
                "阶段": evaluation["stage"],
                "方法": evaluation["method"],
                "结构定位": STRUCTURE_NOTES[version],
                "artifact": evaluation["artifact"],
                "主任务训练范围": evaluation["train_range_success"],
                "主任务留出范围": evaluation["heldout_success"],
                "语言/空间泛化": language_row.get("success", "不适用"),
                "可训练参数": resource.get("trainable_params", "0"),
                "模型大小MB": resource.get("artifact_size_mb", ""),
                "feature_dim": resource.get("feature_dim", ""),
                "stored_samples": resource.get("stored_samples", ""),
                "主任务视频": evaluation["clip"],
                "语言视频": first_video(video_rows, version, "语言/空间泛化片段"),
                "失败模式": failure_modes.get(version, ""),
                "论文结论": PAPER_CONCLUSIONS[version],
                "论文红线": evaluation["note"],
                "主任务viewer命令": viewer_command(method, task="place_blue_cube_blue_pad", complexity="medium", seed=0),
                "语言viewer命令": viewer_command(method, task="move_leftmost_to_bowl", complexity="language", seed=200) if language_row and version != "replay_demo_v1" else "",
                "训练/采集命令": train,
            }
        )
    return rows


def data_efficiency_lines(rows: list[dict[str, str]]) -> list[str]:
    knn_rows = [row for row in rows if row.get("method_key") == "knn_bc"]
    if not knn_rows:
        return []
    lines = [
        "## 5. kNN 数据效率补充",
        "",
        "`data_efficiency_v2` 显示：`knn_bc` 在 10/25/50/92 条示范下训练范围均能达到 3/3，但 held-out 只有全量 92 条时达到 1/3。这说明 kNN 的数据效率主要来自训练分布内轨迹记忆，不能写成泛化能力。",
        "",
        md_row(["示范预算", "split", "成功率", "平均目标距离"]),
        md_row(["---:", "---", "---:", "---:"]),
    ]
    for row in knn_rows:
        lines.append(md_row([row["demo_budget"], row["split"], row["success"], f"{float(row['mean_target_distance']):.4f}"]))
    return lines


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, str]], data_efficiency: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    learned_rows = [row for row in rows if row["版本"] in {"linear_bc_v1", "knn_bc_v1", "mlp_bc_v1"}]
    learned_train = sum(success_rate(row["主任务训练范围"]) for row in learned_rows)
    learned_heldout = sum(success_rate(row["主任务留出范围"]) for row in learned_rows)
    learned_language = sum(success_rate(row["语言/空间泛化"]) for row in learned_rows)

    lines = [
        "# 任务 / 数据 / 普通 BC 阶段报告",
        "",
        "版本：`task_bc_stage_report_v1`",
        "",
        "用途：把桌面任务定义、脚本专家、示范回放、结构化强对照和普通 BC baseline 集中成一份中文阶段报告。这一阶段对应毕业设计前 3 层：任务可视化、数据采集/回放、普通学习 baseline。",
        "",
        "数据来源：`docs/evaluation_summary.csv`、`docs/model_resource_summary.csv`、`docs/language_generalization_summary.csv`、`docs/video_evidence_index.csv`、`docs/failure_mode_taxonomy.csv`、`docs/data_efficiency_summary.csv`。",
        "",
        "阶段展示视频：`outputs/presentation_clips/01_task_data_oracle.mp4`、`outputs/presentation_clips/02_basic_bc_baselines.mp4`。完整视频浏览页：`docs/video_evidence_gallery.html`。",
        "",
        "论文边界：`expert_scripted_v1` 不能写成 learned policy；`structured_waypoint_policy_v1` 不能写成 learned VLA；`replay_demo_v1` 不是策略学习方法；`linear_bc_v1`、`knn_bc_v1`、`mlp_bc_v1` 是普通 BC baseline，不能写成语言理解或 VLA 泛化。",
        "",
        "## 1. 阶段结论",
        "",
        "- Expert、replay 和 structured waypoint 共同证明：MuJoCo 桌面抓取/放置任务可执行，示范数据可保存和回放，任务不是不可解。",
        f"- 普通 learned BC 三个方法的主任务训练范围成功率总和：{learned_train:.2f} / 3；主要来自 `knn_bc_v1` 的训练范围记忆。",
        f"- 普通 learned BC 三个方法的 held-out 成功率总和：{learned_heldout:.2f} / 3，说明泛化弱。",
        f"- 普通 learned BC 三个方法的语言/空间泛化成功率总和：{learned_language:.2f} / 3，说明单任务普通 BC 不具备语言/空间泛化能力。",
        "- 主要失败模式：linear 和 MLP 是未形成有效抓取/未抬升；replay 是数据回放/可复现；kNN 是训练范围成功但泛化不足。",
        "",
        "## 2. 方法对比表",
        "",
        md_row(["版本", "结构定位", "Train", "Held-out", "Language", "参数", "大小MB", "失败模式"]),
        md_row(["---", "---", "---:", "---:", "---:", "---:", "---:", "---"]),
    ]
    for row in rows:
        lines.append(
            md_row(
                [
                    f"`{row['版本']}`",
                    row["结构定位"],
                    row["主任务训练范围"],
                    row["主任务留出范围"],
                    row["语言/空间泛化"],
                    row["可训练参数"],
                    row["模型大小MB"],
                    row["失败模式"],
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 3. 论文可写结论",
            "",
            md_row(["版本", "可写结论", "论文红线"]),
            md_row(["---", "---", "---"]),
        ]
    )
    for row in rows:
        lines.append(md_row([f"`{row['版本']}`", row["论文结论"], row["论文红线"]]))

    lines.extend(
        [
            "",
            "## 4. 视频证据入口",
            "",
            md_row(["版本", "主任务视频", "语言视频"]),
            md_row(["---", "---", "---"]),
        ]
    )
    for row in rows:
        lines.append(md_row([f"`{row['版本']}`", f"`{row['主任务视频']}`", f"`{row['语言视频']}`" if row["语言视频"] else "未导出/不适用"]))

    lines.extend(["", *data_efficiency_lines(data_efficiency)])

    lines.extend(
        [
            "",
            "## 6. 主任务慢速 Viewer 命令",
            "",
            "以下命令都会打开 MuJoCo viewer，统一使用 `--duration 60 --speed 0.05`。",
            "",
        ]
    )
    for row in rows:
        lines.extend([f"### `{row['版本']}`", "", "```powershell", row["主任务viewer命令"], "```", ""])

    lines.extend(["## 7. 语言/空间泛化慢速 Viewer 命令", ""])
    for row in rows:
        if row["语言viewer命令"]:
            lines.extend([f"### `{row['版本']}`", "", "```powershell", row["语言viewer命令"], "```", ""])

    lines.extend(["## 8. 训练/采集/重建命令", ""])
    for row in rows:
        if row["训练/采集命令"]:
            lines.extend([f"### `{row['版本']}`", "", "```powershell", row["训练/采集命令"], "```", ""])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    data_efficiency = read_csv(args.data_efficiency)
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows, data_efficiency)
    print(f"task_bc_stage_rows: {len(rows)}", flush=True)
    print(f"task_bc_stage_csv: {args.output_csv}", flush=True)
    print(f"task_bc_stage_md: {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
