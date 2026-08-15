from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "preference_post_training_ablation_matrix_v1"

FIELDNAMES = [
    "版本",
    "偏好来源",
    "权重策略",
    "训练范围放置",
    "留出范围放置",
    "训练范围TCP抬升",
    "留出范围TCP抬升",
    "训练范围严格抓取",
    "留出范围严格抓取",
    "固定视频结果",
    "固定视频",
    "固定视频目标距离",
    "固定视频物体高度",
    "是否允许升级formal",
    "主要失败原因",
    "下一轮设计指向",
    "完整viewer命令",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese ablation matrix for preference post-training candidates.")
    parser.add_argument("--upgrade-gate", type=Path, default=ROOT / "docs" / "preference_post_training_upgrade_gate.csv")
    parser.add_argument("--candidate-index", type=Path, default=ROOT / "docs" / "candidate_diagnostic_video_index.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "preference_post_training_ablation_matrix.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "preference_post_training_ablation_matrix.csv")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def ps_command(script: str, args: list[str] | None = None) -> str:
    args = args or []
    return f'& "{PYTHON}" "{ROOT / script}" ' + " ".join(args)


def ratio_success(value: str) -> tuple[int, int]:
    if "/" not in value:
        return 0, 0
    left, right = value.split("/", 1)
    try:
        return int(float(left)), int(float(right))
    except ValueError:
        return 0, 0


def design_implication(row: dict[str, str]) -> str:
    train_strict, _ = ratio_success(row["train_range 严格抓取"])
    heldout_strict, _ = ratio_success(row["heldout 严格抓取"])
    heldout_place, _ = ratio_success(row["heldout 放置成功"])
    train_tcp, _ = ratio_success(row["train_range TCP抬升"])
    heldout_tcp, _ = ratio_success(row["heldout TCP抬升"])

    if train_strict + heldout_strict == 0 and train_tcp + heldout_tcp > 0:
        return "已有局部 TCP 抬升迹象，但严格抓取仍为 0；下一轮应把接触保持、夹爪闭合时序和物体随动作为显式 preference。"
    if heldout_place == 0:
        return "留出范围放置失败；下一轮应先增加跨 seed/位置的数据多样性，再谈 preference 升级。"
    if train_tcp + heldout_tcp == 0:
        return "偏好权重主要改善目标距离，没有改善接触/抬升；下一轮应加入接触阶段成功示范或 grasp-hold 子目标。"
    return "仍不能升级 formal；下一轮需要同时提高 held-out、标准抓取和严格抓取，而不是只优化目标距离。"


def build_rows(gate_rows: list[dict[str, str]], candidate_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    candidate_by_version = {row["版本"]: row for row in candidate_rows}
    rows: list[dict[str, str]] = []
    for gate in gate_rows:
        candidate = candidate_by_version.get(gate["版本"], {})
        row = {
            "版本": gate["版本"],
            "偏好来源": gate["偏好来源"],
            "权重策略": gate["权重策略"],
            "训练范围放置": gate["train_range 放置成功"],
            "留出范围放置": gate["heldout 放置成功"],
            "训练范围TCP抬升": gate["train_range TCP抬升"],
            "留出范围TCP抬升": gate["heldout TCP抬升"],
            "训练范围严格抓取": gate["train_range 严格抓取"],
            "留出范围严格抓取": gate["heldout 严格抓取"],
            "固定视频结果": candidate.get("结果", ""),
            "固定视频": gate["固定视频"],
            "固定视频目标距离": candidate.get("目标距离", ""),
            "固定视频物体高度": candidate.get("物体高度", ""),
            "是否允许升级formal": gate["是否允许升级formal"],
            "主要失败原因": gate["不能升级原因"],
            "下一轮设计指向": "",
            "完整viewer命令": gate["viewer命令"],
        }
        row["下一轮设计指向"] = design_implication(gate)
        rows.append(row)
    return rows


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(value.replace("|", "\\|").replace("\n", "<br>") for value in values) + " |"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, str]]) -> None:
    strict_total = 0
    heldout_place_total = 0
    for row in rows:
        strict_total += ratio_success(row["训练范围严格抓取"])[0] + ratio_success(row["留出范围严格抓取"])[0]
        heldout_place_total += ratio_success(row["留出范围放置"])[0]

    lines = [
        "# Preference 后训练消融矩阵",
        "",
        f"版本：`{VERSION}`",
        "",
        f"用途：把当前 {len(rows)} 个 preference post-training 候选的偏好来源、权重策略、训练范围/留出范围、TCP 抬升、严格抓取、固定视频和不能升级原因统一到一张中文矩阵。它不新增实验结果，用于指导下一轮 preference objective 设计。",
        "",
        "打开本页命令：",
        "",
        "```powershell",
        ps_command("scripts/showcase_launcher.py", ["--target", "preference-ablation"]),
        "```",
        "",
        "## 1. 结论摘要",
        "",
        f"- 当前候选数量：`{len(rows)}`。",
        f"- 当前严格抓取总成功数：`{strict_total}`。",
        f"- 当前留出范围放置成功候选累计：`{heldout_place_total}`。",
        "- 结论：preference weighting / relative geometry / episode ranking 能改善部分目标距离或 TCP 抬升迹象，但还不能把 `preference_trajectory_post_training_v1` 升级为正式后训练方法。",
        "- 下一轮优先方向：不要继续只调目标距离权重；应把接触保持、夹爪闭合时序、物体随动和跨 seed/位置数据多样性写进 preference objective。",
        "",
        "## 2. 消融矩阵",
        "",
        md_row(FIELDNAMES),
        md_row(["---"] * len(FIELDNAMES)),
    ]
    for row in rows:
        lines.append(md_row([row[field] for field in FIELDNAMES]))

    lines.extend(
        [
            "",
            "## 3. 重建命令",
            "",
            "```powershell",
            ps_command("scripts/build_preference_post_training_ablation_matrix.py"),
            "```",
            "",
            "## 4. 总体验证命令",
            "",
            "```powershell",
            ps_command("scripts/verify_experiment_artifacts.py"),
            "```",
            "",
            f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = build_rows(read_csv(args.upgrade_gate), read_csv(args.candidate_index))
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows)
    print(f"preference_post_training_ablation_matrix_md: {args.output_md}", flush=True)
    print(f"preference_post_training_ablation_matrix_csv: {args.output_csv}", flush=True)
    print(f"rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
