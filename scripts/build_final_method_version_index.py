from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "final_method_version_index_v1"


FIELDNAMES = [
    "序号",
    "版本",
    "阶段",
    "方法",
    "artifact",
    "主任务训练范围",
    "主任务留出范围",
    "语言/空间泛化",
    "可训练参数",
    "模型大小MB",
    "固定视频",
    "入包状态",
    "简介",
    "论文边界",
    "主任务viewer命令",
    "语言viewer命令",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a final Chinese index of all registered method versions.")
    parser.add_argument("--versions", type=Path, default=ROOT / "docs" / "experiment_versions.json")
    parser.add_argument("--method-gate", type=Path, default=ROOT / "docs" / "method_evidence_gate.csv")
    parser.add_argument("--resources", type=Path, default=ROOT / "docs" / "model_resource_summary.csv")
    parser.add_argument("--language", type=Path, default=ROOT / "docs" / "language_generalization_summary.csv")
    parser.add_argument("--command-index", type=Path, default=ROOT / "docs" / "reproducible_command_index.md")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "final_method_version_index.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "final_method_version_index.csv")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def by_key(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row[key]: row for row in rows}


def clean_command(lines: list[str]) -> str:
    return "\n".join(line.rstrip() for line in lines if line.strip()).strip()


def extract_viewer_commands(command_index: Path) -> tuple[dict[str, str], dict[str, str]]:
    text = command_index.read_text(encoding="utf-8-sig").splitlines()
    main: dict[str, str] = {}
    language: dict[str, str] = {}
    mode = "other"
    current_version = ""
    in_code = False
    buffer: list[str] = []

    heading_re = re.compile(r"^### (?:\d+\. )?`([^`]+)`")
    for line in text:
        if line.startswith("## 3. 主任务慢速 Viewer 命令"):
            mode = "main"
            continue
        if line.startswith("## 4. 语言/空间泛化慢速 Viewer 命令"):
            mode = "language"
            continue
        if line.startswith("## ") and not line.startswith(("## 3.", "## 4.")):
            mode = "other"

        match = heading_re.match(line)
        if match and mode in {"main", "language"}:
            current_version = match.group(1)
            continue

        if line.strip() == "```powershell" and current_version and mode in {"main", "language"}:
            in_code = True
            buffer = []
            continue
        if in_code and line.strip() == "```":
            command = clean_command(buffer)
            if command:
                target = main if mode == "main" else language
                target.setdefault(current_version, command)
            in_code = False
            buffer = []
            continue
        if in_code:
            buffer.append(line)

    return main, language


def md_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def md_row(values: list[object]) -> str:
    return "| " + " | ".join(md_escape(value) for value in values) + " |"


def build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    versions = read_json(args.versions)["methods"]
    gate = by_key(read_csv(args.method_gate), "版本")
    resources = by_key(read_csv(args.resources), "version")
    language = by_key(read_csv(args.language), "version")
    main_commands, language_commands = extract_viewer_commands(args.command_index)

    rows: list[dict[str, str]] = []
    for index, method in enumerate(versions, start=1):
        version = method["version"]
        gate_row = gate.get(version, {})
        resource_row = resources.get(version, {})
        language_row = language.get(version, {})
        trainable_params = gate_row.get("可训练参数") or resource_row.get("trainable_params", "")
        model_size = gate_row.get("模型大小MB") or resource_row.get("artifact_size_mb", "")
        paper_boundary = gate_row.get("论文红线") or method.get("note", "")
        rows.append(
            {
                "序号": str(index),
                "版本": version,
                "阶段": method["stage"],
                "方法": method["method"],
                "artifact": method["artifact"],
                "主任务训练范围": method["train_range_success"],
                "主任务留出范围": method["heldout_success"],
                "语言/空间泛化": language_row.get("success", gate_row.get("语言/空间泛化", "not_applicable")),
                "可训练参数": trainable_params,
                "模型大小MB": model_size,
                "固定视频": method["clip"],
                "入包状态": gate_row.get("入包状态", "未检查"),
                "简介": method.get("note", ""),
                "论文边界": paper_boundary,
                "主任务viewer命令": main_commands.get(version, ""),
                "语言viewer命令": language_commands.get(version, ""),
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
    passed = sum(1 for row in rows if row["入包状态"] == "通过")
    with_commands = sum(1 for row in rows if row["主任务viewer命令"])
    language_commands = sum(1 for row in rows if row["语言viewer命令"])
    lines = [
        "# 最终方法版本索引",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：把当前正式登记的每个方法版本整理成“最终版”索引，集中保留版本名、阶段、artifact、量化结果、固定视频、简介、论文边界和慢速 MuJoCo viewer 启动命令。这个文件不新增实验结果，只作为论文写作、答辩讲解和逐方法复现入口。",
        "",
        "## 1. 覆盖统计",
        "",
        md_row(["项目", "数量"]),
        md_row(["---", "---:"]),
        md_row(["正式方法版本", len(rows)]),
        md_row(["入包状态通过", passed]),
        md_row(["主任务 viewer 命令", with_commands]),
        md_row(["语言/空间 viewer 命令", language_commands]),
        "",
        "边界：本索引中的 25 个版本是当前 MuJoCo 实验包的正式方法版本；`robot_vla_action_head_handoff_v1`、`openvla_dataset_bridge_v1` 和 `openvla_feasibility_check_v1` 是下一阶段前置门禁，不计入当前方法成功率比较。",
        "",
        "## 2. 最终版本总表",
        "",
        md_row(["序号", "版本", "阶段", "方法", "Train", "Held-out", "Language", "参数", "固定视频", "入包状态"]),
        md_row(["---:", "---", "---", "---", "---:", "---:", "---:", "---:", "---", "---"]),
    ]
    for row in rows:
        lines.append(
            md_row(
                [
                    row["序号"],
                    f"`{row['版本']}`",
                    row["阶段"],
                    row["方法"],
                    row["主任务训练范围"],
                    row["主任务留出范围"],
                    row["语言/空间泛化"],
                    row["可训练参数"],
                    f"`{row['固定视频']}`",
                    row["入包状态"],
                ]
            )
        )

    lines.extend(["", "## 3. 逐方法最终版说明与启动命令", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row['序号']}. `{row['版本']}`",
                "",
                f"- 阶段：{row['阶段']}",
                f"- 方法：{row['方法']}",
                f"- artifact：`{row['artifact']}`",
                f"- 主任务 train-range / held-out：{row['主任务训练范围']} / {row['主任务留出范围']}",
                f"- 语言/空间泛化：{row['语言/空间泛化']}",
                f"- 可训练参数 / 模型大小：{row['可训练参数']} / {row['模型大小MB']} MB",
                f"- 固定视频：`{row['固定视频']}`",
                f"- 简介：{row['简介']}",
                f"- 论文边界：{row['论文边界']}",
                "",
                "主任务慢速 viewer 命令：",
                "",
                "```powershell",
                row["主任务viewer命令"] or "# 未登记主任务 viewer 命令",
                "```",
            ]
        )
        if row["语言viewer命令"]:
            lines.extend(
                [
                    "",
                    "语言/空间慢速 viewer 命令：",
                    "",
                    "```powershell",
                    row["语言viewer命令"],
                    "```",
                ]
            )
        lines.append("")

    lines.extend(
        [
            "## 4. 复现入口",
            "",
            "完整命令索引：`docs/reproducible_command_index.md`",
            "",
            "重建本索引：",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_final_method_version_index.py"}"',
            "```",
            "",
            "总体验证：",
            "",
            "```powershell",
            '$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "verify_experiment_artifacts.py"}"',
            "```",
            "",
            f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    if len(rows) != 25:
        raise RuntimeError(f"expected 25 formal method versions, found {len(rows)}")
    missing_commands = [row["版本"] for row in rows if not row["主任务viewer命令"]]
    if missing_commands:
        raise RuntimeError(f"missing main viewer commands: {missing_commands}")
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows)
    print(f"final_method_version_index_md: {args.output_md}", flush=True)
    print(f"final_method_version_index_csv: {args.output_csv}", flush=True)
    print(f"methods: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
