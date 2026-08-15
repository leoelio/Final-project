from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "robot_vla_remote_result_intake_v1"
TARGET_VERSION = "robot_vla_action_head_lite_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the intake gate for returned Robot VLA remote results.")
    parser.add_argument("--remote-pack-json", type=Path, default=ROOT / "outputs" / "evaluations" / "robot_vla_remote_run_pack_v1.json")
    parser.add_argument("--remote-result", type=Path, default=ROOT / "outputs" / "evaluations" / f"{TARGET_VERSION}.json")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / f"{VERSION}.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "robot_vla_remote_result_intake.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "robot_vla_remote_result_intake.csv")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def ps_command(script: str) -> str:
    return f'& "{PYTHON}" "{ROOT / script}"'


def pattern_exists(pattern: str) -> bool:
    path = ROOT / pattern.replace("/", "\\")
    if "*" in pattern:
        return bool(list(path.parent.glob(path.name)))
    return path.exists()


def build_file_rows(remote_pack: dict[str, Any]) -> list[dict[str, str]]:
    roles = {
        "outputs/robot_vla_action_head/robot_vla_action_head_lite_v1.*": "模型 artifact",
        "outputs/robot_vla_action_head/openvla_feature_cache_v1.*": "真实 VLA feature cache",
        "outputs/evaluations/robot_vla_action_head_lite_v1.json": "评测 JSON",
        "outputs/videos/robot_vla_action_head_lite_v1_seed0.mp4": "主任务视频",
        "outputs/videos/robot_vla_action_head_lite_v1_language_seed200.mp4": "语言/空间视频",
        "docs/robot_vla_action_head_lite_report.md": "中文报告",
    }
    rows = []
    for pattern in remote_pack["required_remote_return_files"]:
        exists = pattern_exists(pattern)
        rows.append(
            {
                "项目": roles.get(pattern, "远端返回文件"),
                "路径或模式": pattern,
                "当前存在": "是" if exists else "否",
                "入包要求": "必须存在",
            }
        )
    return rows


def validate_remote_result(path: Path, schema: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any] | None]:
    if not path.exists():
        return False, [f"缺少远端评测 JSON：{path.relative_to(ROOT).as_posix()}"], None
    data = read_json(path)
    blockers: list[str] = []
    if data.get("version") != TARGET_VERSION:
        blockers.append(f"version 不是 {TARGET_VERSION}")
    if data.get("uses_real_robot_vla_features") is not True:
        blockers.append("uses_real_robot_vla_features 必须显式为 true")
    for field in schema.get("required_top_level_fields", []):
        if field not in data:
            blockers.append(f"缺少顶层字段：{field}")
    for section, section_schema in schema.items():
        if section not in {"hardware", "training", "data_contract", "evaluation", "artifacts"}:
            continue
        section_data = data.get(section, {})
        for field in section_schema.get("required", []):
            if field not in section_data or section_data[field] in ("", None):
                blockers.append(f"缺少 {section}.{field}")
    try:
        gpu_memory = float(data.get("hardware", {}).get("gpu_memory_gb", 0))
    except (TypeError, ValueError):
        gpu_memory = 0
    if gpu_memory < float(schema.get("hardware", {}).get("minimum_gpu_memory_gb", 27)):
        blockers.append("远端 GPU 显存低于 27GB 门槛")
    return not blockers, blockers, data


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def md_row(values: list[object]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"


def write_md(path: Path, intake: dict[str, Any], rows: list[dict[str, str]]) -> None:
    lines = [
        "# Robot VLA 远端结果回填门禁",
        "",
        f"版本：`{VERSION}`",
        "",
        f"目标计划版本：`{TARGET_VERSION}`",
        "",
        "用途：远端 48GB+ GPU 或云端完成真实 robot VLA action-head 后训练后，用本门禁检查返回的模型、feature cache、评测 JSON、视频和报告是否足够把 `robot_vla_action_head_lite_v1` 从 planned 登记为正式方法版本。",
        "",
        "边界：本门禁本身不是策略模型，也不代表 OpenVLA/OFT 后训练已经完成；只有远端结果通过本门禁并补齐正式评测表、资源表、视频证据和失败模式分类后，才能进入正式方法包。",
        "",
        "## 1. 当前状态",
        "",
        f"- 状态：`{intake['status']}`",
        f"- 可登记为正式方法：{'是' if intake['can_register_completed_method'] else '否'}",
        f"- 远端结果 JSON：`{intake['remote_result_path']}`",
        f"- 已完成返回文件：{intake['returned_files_present']}/{intake['returned_files_required']}",
        "",
        "阻塞原因：",
        "",
        "```text",
        *intake["blocking_reasons"],
        "```",
        "",
        "## 2. 返回文件清单",
        "",
        md_row(["项目", "路径或模式", "当前存在", "入包要求"]),
        md_row(["---", "---", "---", "---"]),
    ]
    for row in rows:
        lines.append(md_row([row["项目"], f"`{row['路径或模式']}`", row["当前存在"], row["入包要求"]]))

    lines.extend(
        [
            "",
            "## 3. 正式入包步骤",
            "",
            "远端结果通过本门禁后，仍需执行以下集成步骤：",
            "",
            "```text",
            *intake["formal_registration_steps"],
            "```",
            "",
            "## 4. 重建命令",
            "",
            "```powershell",
            ps_command("scripts/build_robot_vla_remote_result_intake.py"),
            ps_command("scripts/build_next_experiment_registry.py"),
            ps_command("scripts/build_final_artifact_manifest.py"),
            '$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"',
            ps_command("scripts/verify_experiment_artifacts.py"),
            "```",
            "",
            "## 5. 论文边界",
            "",
            "- 可以写：真实 robot VLA action-head 的结果回填门禁已经建立，明确远端结果如何进入统一评测和视频证据体系。",
            "- 不能写：`robot_vla_action_head_lite_v1`、OpenVLA LoRA、OpenVLA-OFT、Isaac 或真实 WidowX 验证已经完成，除非远端结果和视频证据真实回填并通过正式入包 gate。",
            "",
            f"生成时间：{intake['generated_at']}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_intake(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, str]]]:
    remote_pack = read_json(args.remote_pack_json)
    schema_path = ROOT / remote_pack["pack_dir"] / "remote_result_schema.json"
    schema = read_json(schema_path)
    rows = build_file_rows(remote_pack)
    result_ok, result_blockers, _ = validate_remote_result(args.remote_result, schema)
    missing_files = [row["路径或模式"] for row in rows if row["当前存在"] != "是"]
    blockers = result_blockers + [f"缺少远端返回文件：{path}" for path in missing_files]
    can_register = result_ok and not missing_files
    status = "ready_for_formal_registration" if can_register else "waiting_for_remote_result"
    intake = {
        "version": VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "target_planned_version": TARGET_VERSION,
        "remote_pack_version": remote_pack["version"],
        "remote_result_path": args.remote_result.relative_to(ROOT).as_posix(),
        "schema_path": schema_path.relative_to(ROOT).as_posix(),
        "can_register_completed_method": can_register,
        "returned_files_required": len(rows),
        "returned_files_present": sum(1 for row in rows if row["当前存在"] == "是"),
        "blocking_reasons": blockers or ["无"],
        "formal_registration_steps": [
            "把远端模型 artifact、feature cache、评测 JSON、主任务视频、语言视频和中文报告放入本仓库规定路径。",
            "确认远端评测 JSON 中 uses_real_robot_vla_features=true，且硬件、训练、RLDS 数据契约、评测和 artifact 字段完整。",
            "确认 RLDS 数据集已注册，动作表示不是原始 MuJoCo absolute control target，并记录 OpenVLA dataset adapter commit。",
            "将 robot_vla_action_head_lite_v1 加入 docs/experiment_versions.json，并补 evaluation_summary.csv、language_generalization_summary.csv、model_resource_summary.csv。",
            "把远端视频加入 docs/video_evidence_index.csv，并重建 video_quality_audit、failure_mode_taxonomy 和 stage/claim 展示索引。",
            "重新运行 scripts/verify_experiment_artifacts.py，通过后才可写成正式方法结果。",
        ],
        "paper_boundary": "这是远端结果回填门禁，不是 robot_vla_action_head_lite_v1 的完成结果。",
    }
    return intake, rows


def main() -> None:
    args = parse_args()
    intake, rows = build_intake(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(intake, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_csv, rows)
    write_md(args.output_md, intake, rows)
    print(f"robot_vla_remote_result_intake_md: {args.output_md}", flush=True)
    print(f"robot_vla_remote_result_intake_json: {args.output_json}", flush=True)
    print(f"robot_vla_remote_result_intake_csv: {args.output_csv}", flush=True)
    print(f"status: {intake['status']}", flush=True)


if __name__ == "__main__":
    main()
