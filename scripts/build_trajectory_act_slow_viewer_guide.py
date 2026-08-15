from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "trajectory_act_slow_viewer_guide_v1"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_experiment_command_index import training_command, viewer_command  # noqa: E402


CORE_VERSIONS = [
    {
        "version": "trajectory_conditioned_chunk_bc_v2",
        "定位": "Trajectory-conditioned Action-Chunk BC",
        "观看目的": "观察加入 8 帧历史状态后动作是否更平滑，以及接触、夹紧、抬升阶段为什么仍失败。",
        "论文口径": "只能写成历史状态条件动作块 baseline，不能写成完整 ACT 或稳定抓取成功。",
    },
    {
        "version": "trajectory_knn_chunk_bc_v1",
        "定位": "Trajectory-kNN Action-Chunk BC",
        "观看目的": "观察训练范围成功样例，同时区分轨迹记忆型成功与 held-out / 语言泛化失败。",
        "论文口径": "可以写成轨迹记忆型强 baseline，不能写成泛化策略。",
    },
    {
        "version": "torch_act_state_chunk_v1",
        "定位": "State-only Transformer ACT-style",
        "观看目的": "观察 ACT-style 动作块闭环控制是否比普通 BC 更稳定，以及为什么仍卡在抓取/抬升。",
        "论文口径": "只能写成 state-only ACT-style baseline，不能写成官方完整视觉 ACT。",
    },
    {
        "version": "torch_act_cvae_state_chunk_v1",
        "定位": "State-only ACT-CVAE-lite",
        "观看目的": "观察加入 CVAE latent 后是否改善动作块形态，以及离线误差下降为何没有转成稳定闭环成功。",
        "论文口径": "只能写成 ACT-CVAE-lite 代理，不能写成官方 ACT 复现。",
    },
    {
        "version": "visual_act_cnn_cvae_v1",
        "定位": "Visual ACT-CNN-CVAE-lite",
        "观看目的": "观察小型 CNN 视觉链路是否能补充状态输入，以及当前数据规模下为什么仍不稳定。",
        "论文口径": "只能写成本地视觉 ACT-lite 代理，不能写成真实 VLA 或真实机器人视觉 ACT。",
    },
]


FIELDNAMES = [
    "版本",
    "定位",
    "观看目的",
    "训练范围成功率",
    "留出范围成功率",
    "语言/空间泛化",
    "固定视频",
    "标准慢速viewer命令",
    "超慢学习viewer命令",
    "launcher超慢命令",
    "训练命令",
    "论文口径",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese slow-viewer guide for trajectory-conditioned BC / ACT baselines.")
    parser.add_argument("--versions", type=Path, default=ROOT / "docs" / "experiment_versions.json")
    parser.add_argument("--language-summary", type=Path, default=ROOT / "docs" / "language_generalization_summary.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "trajectory_act_slow_viewer_guide.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "trajectory_act_slow_viewer_guide.csv")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def ps_command(script: str, args: list[str]) -> str:
    rendered = [f'"{PYTHON}"', f'"{ROOT / script}"', *args]
    return "& " + " ".join(rendered)


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(value.replace("|", "\\|").replace("\n", "<br>") for value in values) + " |"


def build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    methods = {row["version"]: row for row in read_json(args.versions)["methods"]}
    language = {row["version"]: row for row in read_csv(args.language_summary)}
    rows: list[dict[str, str]] = []
    for spec in CORE_VERSIONS:
        method = methods[spec["version"]]
        language_row = language.get(spec["version"], {})
        standard_viewer = viewer_command(
            method,
            task="place_blue_cube_blue_pad",
            complexity="medium",
            seed=0,
            duration=60,
            speed=0.05,
        )
        slow_viewer = viewer_command(
            method,
            task="place_blue_cube_blue_pad",
            complexity="medium",
            seed=0,
            duration=90,
            speed=0.02,
        )
        launcher_slow = ps_command(
            "scripts/showcase_launcher.py",
            [
                "--target",
                f"method:{spec['version']}",
                "--action",
                "viewer",
                "--viewer-speed",
                "0.02",
                "--viewer-duration",
                "90",
            ],
        )
        rows.append(
            {
                "版本": spec["version"],
                "定位": spec["定位"],
                "观看目的": spec["观看目的"],
                "训练范围成功率": method["train_range_success"],
                "留出范围成功率": method["heldout_success"],
                "语言/空间泛化": language_row.get("success", "未登记"),
                "固定视频": method["clip"],
                "标准慢速viewer命令": standard_viewer,
                "超慢学习viewer命令": slow_viewer,
                "launcher超慢命令": launcher_slow,
                "训练命令": training_command(spec["version"]) or "",
                "论文口径": spec["论文口径"],
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
    lines = [
        "# Trajectory-conditioned BC / ACT 超慢可视化指南",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：把 trajectory-conditioned BC / ACT 阶段最需要观察的几个版本集中成中文学习入口，提供标准慢速 viewer 和超慢学习 viewer 的完整命令。该文档不新增量化评测，不改变模型权重，不改变论文中的成功率口径。",
        "",
        "关键说明：`--speed` 只改变 viewer 播放等待时间，方便肉眼观察；动作限幅、replan、temporal ensemble 等控制参数仍按各方法的既有安全档位记录。超慢命令适合学习和诊断，不作为新的评测结果。",
        "",
        "打开本页：",
        "",
        "```powershell",
        ps_command("scripts/showcase_launcher.py", ["--target", "trajectory-act-slow"]),
        "```",
        "",
        "## 1. 推荐观看顺序",
        "",
        "1. 先看 `trajectory_knn_chunk_bc_v1`：它在训练范围最容易展示“看起来能完成”的现象，用来理解轨迹记忆型 baseline。",
        "2. 再看 `trajectory_conditioned_chunk_bc_v2`：它说明历史轨迹条件能让动作更平滑，但不足以解决接触和夹紧。",
        "3. 接着看 `torch_act_state_chunk_v1`：它代表 state-only ACT-style baseline，重点观察动作块预测在闭环里如何累积误差。",
        "4. 最后看 `torch_act_cvae_state_chunk_v1` 和 `visual_act_cnn_cvae_v1`：说明结构更接近 ACT/视觉 ACT 后，仍需要更强数据和接触建模。",
        "",
        "## 2. 版本总表",
        "",
        md_row(["版本", "定位", "Train", "Held-out", "Language", "固定视频", "论文口径"]),
        md_row(["---", "---", "---:", "---:", "---:", "---", "---"]),
    ]
    for row in rows:
        lines.append(
            md_row(
                [
                    f"`{row['版本']}`",
                    row["定位"],
                    row["训练范围成功率"],
                    row["留出范围成功率"],
                    row["语言/空间泛化"],
                    f"`{row['固定视频']}`",
                    row["论文口径"],
                ]
            )
        )

    lines.extend(["", "## 3. 完整 viewer 命令", ""])
    for row in rows:
        lines.extend(
            [
                f"### `{row['版本']}`",
                "",
                f"- 定位：{row['定位']}",
                f"- 观看目的：{row['观看目的']}",
                f"- 论文口径：{row['论文口径']}",
                "",
                "超慢学习 viewer（推荐学习时使用）：",
                "",
                "```powershell",
                row["超慢学习viewer命令"],
                "```",
                "",
                "通过 launcher 启动同一个超慢 viewer：",
                "",
                "```powershell",
                row["launcher超慢命令"],
                "```",
                "",
                "标准慢速 viewer（和当前实验台账一致）：",
                "",
                "```powershell",
                row["标准慢速viewer命令"],
                "```",
                "",
            ]
        )
        if row["训练命令"]:
            lines.extend(
                [
                    "训练命令：",
                    "",
                    "```powershell",
                    row["训练命令"],
                    "```",
                    "",
                ]
            )

    lines.extend(
        [
            "## 4. 实验记录边界",
            "",
            "- 中文实验记录优先写在 `docs/trajectory_act_experiment_record.md`、`docs/trajectory_act_stage_report.md`、`docs/trajectory_act_conclusion_brief.md` 和本文档中。",
            "- 量化结果继续以 `docs/evaluation_summary.csv`、`docs/language_generalization_summary.csv`、`docs/video_evidence_index.csv` 和 `outputs/evaluations/*.json` 为准。",
            "- 如果只是调整 `--viewer-speed` 或 `--viewer-duration` 观察过程，不要登记成新方法成功率。",
            "- 如果后续改变动作限幅、夹爪力度、grasp gate、模型结构或训练数据，需要新建版本名并重新评测。",
            "",
            f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows)
    print(f"trajectory_act_slow_viewer_guide_md: {args.output_md}", flush=True)
    print(f"trajectory_act_slow_viewer_guide_csv: {args.output_csv}", flush=True)
    print(f"rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
