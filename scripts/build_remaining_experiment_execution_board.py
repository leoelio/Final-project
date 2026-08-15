from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "remaining_experiment_execution_board_v1"

FIELDNAMES = [
    "优先级",
    "版本",
    "类别",
    "当前状态",
    "执行环境",
    "阻塞条件",
    "下一步动作",
    "必需回填工件",
    "成功/升级门槛",
    "完成后重建命令",
    "完成后验证命令",
    "论文红线",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese execution board for remaining planned experiments.")
    parser.add_argument("--next-registry", type=Path, default=ROOT / "docs" / "next_experiment_registry.csv")
    parser.add_argument("--readiness", type=Path, default=ROOT / "docs" / "external_dependency_readiness_audit.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "remaining_experiment_execution_board.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "remaining_experiment_execution_board.csv")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def ps_command(script: str, args: list[str] | None = None) -> str:
    args = args or []
    return f'& "{PYTHON}" "{ROOT / script}" ' + " ".join(args)


def priority(version: str, readiness_status: str) -> int:
    order = {
        "preference_trajectory_post_training_v1": 1,
        "robot_vla_action_head_lite_v1": 2,
        "robot_vla_adapter_lite_v1": 3,
        "robot_vla_lora_lite_v1": 4,
        "isaac_domain_randomization_v1": 5,
        "real_widowx_validation_v1": 6,
    }
    if readiness_status.startswith("waiting"):
        return order.get(version, 90)
    return order.get(version, 80)


def environment_for(category: str, readiness_status: str) -> str:
    if category == "robot_vla_post_training":
        return "48GB+ GPU 或云端真实 VLA 环境"
    if category == "sim_to_real_proxy":
        return "安装 Isaac / Isaac Sim 的高保真仿真环境"
    if category == "real_robot_validation":
        return "真实 WidowX、相机、桌面物体和安全操作空间"
    if readiness_status == "ready_for_local_redesign":
        return "当前本地 MuJoCo 环境；需要重新设计 preference objective"
    return "按 registry 中的依赖环境执行"


def rebuild_command(version: str, category: str) -> str:
    if version.startswith("robot_vla"):
        return "；".join(
            [
                ps_command("scripts/build_robot_vla_remote_result_intake.py"),
                ps_command("scripts/build_external_dependency_readiness_audit.py"),
                ps_command("scripts/build_remaining_experiment_execution_board.py"),
            ]
        )
    if version == "isaac_domain_randomization_v1":
        return "；".join(
            [
                ps_command("scripts/build_isaac_domain_randomization_handoff.py"),
                ps_command("scripts/build_external_dependency_readiness_audit.py"),
                ps_command("scripts/build_remaining_experiment_execution_board.py"),
            ]
        )
    if version == "real_widowx_validation_v1":
        return "；".join(
            [
                ps_command("scripts/build_real_widowx_validation_handoff.py"),
                ps_command("scripts/build_external_dependency_readiness_audit.py"),
                ps_command("scripts/build_remaining_experiment_execution_board.py"),
            ]
        )
    if category == "post_training":
        return "；".join(
            [
                ps_command("scripts/build_preference_post_training_ablation_matrix.py"),
                ps_command("scripts/build_preference_post_training_upgrade_gate.py"),
                ps_command("scripts/build_remaining_experiment_execution_board.py"),
            ]
        )
    return ps_command("scripts/build_remaining_experiment_execution_board.py")


def should_include(row: dict[str, str]) -> bool:
    if row["formal_method_allowed_now"] == "是":
        return False
    return row["registry_status"] in {"planned", "planned_external_dependency"} or row["readiness_status"] == "ready_for_local_redesign"


def build_rows(registry_rows: list[dict[str, str]], readiness_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    registry_by_version = {row["version"]: row for row in registry_rows}
    rows: list[dict[str, str]] = []
    for readiness in readiness_rows:
        if not should_include(readiness):
            continue
        version = readiness["version"]
        registry = registry_by_version.get(version, {})
        category = readiness["category"]
        readiness_status = readiness["readiness_status"]
        rows.append(
            {
                "优先级": str(priority(version, readiness_status)),
                "版本": version,
                "类别": category,
                "当前状态": readiness_status,
                "执行环境": environment_for(category, readiness_status),
                "阻塞条件": readiness["blocking_condition"],
                "下一步动作": readiness["required_next_action"],
                "必需回填工件": readiness["required_return_artifacts"],
                "成功/升级门槛": registry.get("success_gate", "必须补齐评测、资源、视频和论文边界后才能升级为正式结果"),
                "完成后重建命令": rebuild_command(version, category),
                "完成后验证命令": ps_command("scripts/verify_experiment_artifacts.py"),
                "论文红线": readiness["paper_boundary"],
            }
        )
    rows.sort(key=lambda row: int(row["优先级"]))
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
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["当前状态"]] = status_counts.get(row["当前状态"], 0) + 1

    lines = [
        "# 剩余实验执行看板",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：把当前还不能写成正式结果的 planned / readiness 实验整理成一张执行看板，明确优先级、执行环境、阻塞条件、必需回填工件、升级门槛和论文红线。它不新增实验结果，只服务后续把剩余实验真正跑完。",
        "",
        "打开本页命令：",
        "",
        "```powershell",
        ps_command("scripts/showcase_launcher.py", ["--target", "remaining-board"]),
        "```",
        "",
        "## 1. 当前剩余状态",
        "",
        "| 状态 | 数量 |",
        "| --- | ---: |",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(md_row([status, str(count)]))

    lines.extend(
        [
            "",
            "## 2. 建议执行顺序",
            "",
            "| 优先级 | 版本 | 类别 | 当前状态 | 执行环境 | 阻塞条件 | 下一步动作 | 必需回填工件 | 成功/升级门槛 | 完成后重建命令 | 完成后验证命令 | 论文红线 |",
            "| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(md_row([row[field] for field in FIELDNAMES]))

    lines.extend(
        [
            "",
            "## 3. 使用原则",
            "",
            "1. 任何 planned 版本只有在回填了评测 CSV/JSON、资源记录、固定视频和论文边界后，才能申请进入正式方法包。",
            "2. `robot_vla_adapter_lite_v1` 和 `robot_vla_lora_lite_v1` 必须等 `robot_vla_action_head_lite_v1` 先完成并通过回填门禁。",
            "3. Isaac 和真实 WidowX 结果不能用 MuJoCo 视频替代；必须保存对应运行环境、原始结果和视频。",
            "4. 完成任一行后，先运行该行的重建命令，再运行总体验证命令。",
            "",
            "## 4. 重建命令",
            "",
            "```powershell",
            ps_command("scripts/build_remaining_experiment_execution_board.py"),
            "```",
            "",
            "## 5. 总体验证命令",
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
    registry_rows = read_csv(args.next_registry)
    readiness_rows = read_csv(args.readiness)
    rows = build_rows(registry_rows, readiness_rows)
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows)
    print(f"remaining_experiment_execution_board_md: {args.output_md}", flush=True)
    print(f"remaining_experiment_execution_board_csv: {args.output_csv}", flush=True)
    print(f"rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
