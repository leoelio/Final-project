from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "method_evidence_gate_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a per-method evidence gate report.")
    parser.add_argument("--versions", type=Path, default=ROOT / "docs" / "experiment_versions.json")
    parser.add_argument("--resource-summary", type=Path, default=ROOT / "docs" / "model_resource_summary.csv")
    parser.add_argument("--task-bc-stage", type=Path, default=ROOT / "docs" / "task_bc_stage_report.csv")
    parser.add_argument("--trajectory-act-stage", type=Path, default=ROOT / "docs" / "trajectory_act_stage_report.csv")
    parser.add_argument("--action-head-stage", type=Path, default=ROOT / "docs" / "action_head_stage_report.csv")
    parser.add_argument("--video-quality-audit", type=Path, default=ROOT / "docs" / "video_quality_audit.csv")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "method_evidence_gate.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "method_evidence_gate.md")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", "<br>") for value in values) + " |"


def collect_stage_rows(*groups: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for group in groups:
        for row in group:
            version = row.get("版本", "")
            if version:
                rows[version] = row
    return rows


def command_for(row: dict[str, str]) -> str:
    return row.get("训练/采集命令") or row.get("训练命令") or "不适用"


def exists_text(path_text: str) -> str:
    path = ROOT / path_text
    return "是" if path.exists() else "否"


def build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    versions = read_json(args.versions)["methods"]
    resources = {row["version"]: row for row in read_csv(args.resource_summary)}
    stage_rows = collect_stage_rows(
        read_csv(args.task_bc_stage),
        read_csv(args.trajectory_act_stage),
        read_csv(args.action_head_stage),
    )
    video_quality = {
        row["版本"]: row
        for row in read_csv(args.video_quality_audit)
        if row.get("视频类型") == "主任务固定片段"
    }

    rows: list[dict[str, str]] = []
    for method in versions:
        version = method["version"]
        stage_row = stage_rows.get(version, {})
        resource_row = resources.get(version, {})
        video_row = video_quality.get(version, {})
        artifact = method["artifact"]
        clip = method["clip"]
        viewer = stage_row.get("主任务viewer命令", "")
        train_command = command_for(stage_row)

        checks = {
            "artifact存在": exists_text(artifact),
            "资源记录": "是" if resource_row else "否",
            "主视频存在": exists_text(clip),
            "视频审计通过": "是" if video_row.get("审计状态") == "通过" else "否",
            "viewer命令存在": "是" if "--viewer" in viewer and "--duration 60" in viewer and "--speed 0.05" in viewer else "否",
            "训练或采集命令存在": "是" if train_command and train_command != "不适用" else ("不适用" if method["stage"] == "data_verification" else "否"),
            "论文红线存在": "是" if stage_row.get("论文红线") else "否",
        }
        failed = [name for name, value in checks.items() if value == "否"]
        status = "通过" if not failed else "需补齐"
        rows.append(
            {
                "版本": version,
                "阶段": method["stage"],
                "方法": method["method"],
                "artifact": artifact,
                "artifact存在": checks["artifact存在"],
                "主任务训练范围": method["train_range_success"],
                "主任务留出范围": method["heldout_success"],
                "语言/空间泛化": stage_row.get("语言/空间泛化", "未登记"),
                "可训练参数": resource_row.get("trainable_params", ""),
                "模型大小MB": resource_row.get("artifact_size_mb", ""),
                "固定视频": clip,
                "视频审计通过": checks["视频审计通过"],
                "viewer命令存在": checks["viewer命令存在"],
                "训练或采集命令存在": checks["训练或采集命令存在"],
                "论文红线": stage_row.get("论文红线", ""),
                "入包状态": status,
                "需补齐": "；".join(failed) if failed else "无",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, str]]) -> None:
    passed = sum(1 for row in rows if row["入包状态"] == "通过")
    train_commands = sum(1 for row in rows if row["训练或采集命令存在"] in {"是", "不适用"})
    lines = [
        "# 方法证据门禁",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：逐个检查 25 个正式方法版本是否具备最终论文和答辩需要的最小证据包：版本名称、artifact、主任务结果、资源记录、固定视频、视频质量审计、慢速 viewer 命令、训练/采集命令和论文红线。",
        "",
        "该报告不新增实验结果，只把现有 `experiment_versions`、阶段报告、资源表和视频质量审计合并成方法级入包检查。展示视频不是额外成功率结论，成功率仍以 CSV 评测表为准。",
        "",
        "## 1. 总览",
        "",
        md_row(["项目", "数量"]),
        md_row(["---", "---:"]),
        md_row(["正式方法版本", str(len(rows))]),
        md_row(["入包状态通过", str(passed)]),
        md_row(["有训练/采集命令或不适用", str(train_commands)]),
        md_row(["视频审计通过", str(sum(1 for row in rows if row["视频审计通过"] == "是"))]),
        md_row(["viewer 命令通过", str(sum(1 for row in rows if row["viewer命令存在"] == "是"))]),
        "",
        "## 2. 方法明细",
        "",
        md_row(["版本", "阶段", "Train", "Held-out", "Language", "参数", "视频审计", "viewer", "入包状态", "需补齐"]),
        md_row(["---", "---", "---:", "---:", "---:", "---:", "---", "---", "---", "---"]),
    ]
    for row in rows:
        lines.append(
            md_row(
                [
                    f"`{row['版本']}`",
                    row["阶段"],
                    row["主任务训练范围"],
                    row["主任务留出范围"],
                    row["语言/空间泛化"],
                    row["可训练参数"],
                    row["视频审计通过"],
                    row["viewer命令存在"],
                    row["入包状态"],
                    row["需补齐"],
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 3. 论文和答辩使用边界",
            "",
            "1. `入包状态=通过` 表示该方法具备可追溯的版本、artifact、评测、视频、viewer 命令和论文红线。",
            "2. `trajectory_conditioned_chunk_bc_v2`、`trajectory_knn_chunk_bc_v1`、`torch_act_state_chunk_v1` 和 `torch_act_cvae_state_chunk_v1` 可作为 trajectory-conditioned BC / ACT 对照组，但不能写成完整官方视觉 ACT。",
            "3. `adapter_action_head_lite_v1`、`lora_action_head_lite_v1`、`clip_action_head_lite_v1` 等只能写成本地 action-head/PEFT/CLIP proxy，不能写成真实 OpenVLA/RT-2 后训练。",
            "4. 当前完整研究路线仍未完成；真实 OpenVLA/机器人 VLA、Isaac domain randomization 和真实 WidowX 验证仍需按下一阶段注册表继续执行。",
            "",
            "## 4. 重建命令",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_method_evidence_gate.py"}"',
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows)
    print(f"method_evidence_gate_md: {args.output_md}", flush=True)
    print(f"method_evidence_gate_csv: {args.output_csv}", flush=True)
    print(f"method_evidence_gate_rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
