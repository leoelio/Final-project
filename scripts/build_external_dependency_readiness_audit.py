from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "external_dependency_readiness_audit_v1"

CSV_FIELDS = [
    "version",
    "category",
    "registry_status",
    "readiness_status",
    "formal_method_allowed_now",
    "blocking_condition",
    "required_next_action",
    "required_return_artifacts",
    "source_evidence",
    "paper_boundary",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a readiness audit for planned external VLA, Isaac, and real robot stages.")
    parser.add_argument("--next-registry", type=Path, default=ROOT / "docs" / "next_experiment_registry.csv")
    parser.add_argument("--remote-intake-json", type=Path, default=ROOT / "outputs" / "evaluations" / "robot_vla_remote_result_intake_v1.json")
    parser.add_argument("--isaac-handoff-json", type=Path, default=ROOT / "outputs" / "evaluations" / "isaac_domain_randomization_handoff_v1.json")
    parser.add_argument("--real-widowx-handoff-json", type=Path, default=ROOT / "outputs" / "evaluations" / "real_widowx_validation_handoff_v1.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "external_dependency_readiness_audit.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "external_dependency_readiness_audit.csv")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / f"{VERSION}.json")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def split_items(value: str) -> list[str]:
    return [item.strip() for item in value.split("；") if item.strip()]


def path_or_pattern_exists(value: str) -> bool:
    path = ROOT / value.replace("\\", "/")
    if "*" in value:
        return bool(list(path.parent.glob(path.name)))
    return path.exists()


def count_present_artifacts(items: list[str]) -> int:
    return sum(1 for item in items if item != "not_applicable" and path_or_pattern_exists(item))


def dependency_state(row: dict[str, str], remote_intake: dict, isaac_handoff: dict, real_handoff: dict) -> tuple[str, str, str, str]:
    version = row["version"]
    status = row["status"]
    artifacts = split_items(row["video_outputs"]) + split_items(row["primary_artifact"])
    present = count_present_artifacts(artifacts)
    total = len([item for item in artifacts if item != "not_applicable"])

    if status in {"completed_prerequisite", "completed_diagnostic"}:
        return (
            "supporting_evidence_ready",
            "否",
            "这是前置门禁或诊断负例，不是可登记为正式策略结果的版本。",
            "作为后续阶段依赖继续保留；新增结果后重新运行本审计。",
        )

    if version == "robot_vla_action_head_lite_v1":
        returned = int(remote_intake.get("returned_files_present", 0))
        required = int(remote_intake.get("returned_files_required", 0))
        return (
            "waiting_remote_result",
            "否",
            f"远端真实 VLA 结果尚未回填：{returned}/{required} 个必需文件存在。",
            "把 48GB+ GPU 或云端运行结果放回规定路径，再运行 build_robot_vla_remote_result_intake.py。",
        )

    if version in {"robot_vla_adapter_lite_v1", "robot_vla_lora_lite_v1"}:
        return (
            "waiting_robot_vla_action_head",
            "否",
            "依赖 robot_vla_action_head_lite_v1 先完成并通过回填门禁。",
            "先完成 action-head only 真实 VLA 版本，再用同任务、同 seed 对比 Adapter/LoRA。",
        )

    if version == "isaac_domain_randomization_v1":
        rows = isaac_handoff.get("rows", [])
        runtime_rows = [item for item in rows if item.get("status") == "waiting_external_runtime"]
        return (
            "waiting_isaac_runtime",
            "否",
            f"Isaac/omni/isaacgym 运行时尚不可用；handoff 中 waiting_external_runtime={len(runtime_rows)}。",
            "在安装 Isaac/Isaac Sim 的环境运行同任务，并回填 JSON、CSV、报告和视频。",
        )

    if version == "real_widowx_validation_v1":
        trial_rows = int(real_handoff.get("trial_template_rows", 0))
        return (
            "waiting_real_robot_trials",
            "否",
            f"真实 WidowX trial 尚未采集；当前只有 {trial_rows} 行 trial 模板。",
            "真实机械臂可用后按安全门禁执行 20-50 次 trial，并保存相机视频和 sim-to-real gap。",
        )

    if version == "preference_trajectory_post_training_v1":
        return (
            "ready_for_local_redesign",
            "否",
            "已有 candidate 负例，但正式版本需要更明确的 preference/ranking objective。",
            "先定义 preference 来源和权重策略，再保存正式评测、视频和失败模式。",
        )

    return (
        "planned_missing_artifacts",
        "否",
        f"计划版本仍缺少必需 artifact/video：{present}/{total} 个可见。",
        "补齐评测、资源、视频和报告后再申请进入正式方法包。",
    )


def build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    registry = read_csv(args.next_registry)
    remote_intake = read_json(args.remote_intake_json)
    isaac_handoff = read_json(args.isaac_handoff_json)
    real_handoff = read_json(args.real_widowx_handoff_json)

    rows = []
    for item in registry:
        readiness, allowed, blocker, next_action = dependency_state(item, remote_intake, isaac_handoff, real_handoff)
        rows.append(
            {
                "version": item["version"],
                "category": item["category"],
                "registry_status": item["status"],
                "readiness_status": readiness,
                "formal_method_allowed_now": allowed,
                "blocking_condition": blocker,
                "required_next_action": next_action,
                "required_return_artifacts": item["primary_artifact"] + "；" + item["video_outputs"],
                "source_evidence": item["depends_on"],
                "paper_boundary": item["paper_boundary"],
            }
        )
    return rows


def summarize(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in rows:
        key = item["readiness_status"]
        counts[key] = counts.get(key, 0) + 1
    return counts


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(value.replace("|", "\\|").replace("\n", " ") for value in values) + " |"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, str]], counts: dict[str, int]) -> None:
    data = {
        "version": VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "row_count": len(rows),
        "readiness_counts": counts,
        "formal_method_allowed_now": sum(1 for item in rows if item["formal_method_allowed_now"] == "是"),
        "rows": rows,
        "paper_boundary": "本审计只说明外部依赖阶段的准备状态；不能写成真实 OpenVLA、Isaac 或真实 WidowX 实验已经完成。",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_md(path: Path, rows: list[dict[str, str]], counts: dict[str, int]) -> None:
    lines = [
        "# 外部依赖阶段 Readiness Audit",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：统一审计后续真实 OpenVLA/机器人 VLA、Isaac domain randomization 和真实 WidowX 验证阶段当前是否具备运行与回填条件。本文件不是实验结果表，不是策略成功率结果，也不会把 planned 版本登记为正式方法。",
        "",
        "## 1. 总览",
        "",
        md_row(["readiness_status", "数量"]),
        md_row(["---", "---:"]),
    ]
    for key, value in sorted(counts.items()):
        lines.append(md_row([key, str(value)]))

    lines.extend(
        [
            "",
            "当前没有任何 planned 外部阶段被允许直接登记为正式方法；必须先回填真实评测、资源记录、视频证据和中文报告。",
            "",
            "## 2. 审计表",
            "",
            md_row(CSV_FIELDS),
            md_row(["---"] * len(CSV_FIELDS)),
        ]
    )
    for item in rows:
        lines.append(md_row([item[field] for field in CSV_FIELDS]))

    lines.extend(
        [
            "",
            "## 3. 推荐执行顺序",
            "",
            "1. 先处理 `robot_vla_action_head_lite_v1`：远端或云端运行真实 VLA feature/action-head，回填模型、feature cache、评测 JSON、主任务视频、语言视频和中文报告。",
            "2. `robot_vla_action_head_lite_v1` 通过回填门禁后，再做 `robot_vla_adapter_lite_v1` 和 `robot_vla_lora_lite_v1`，保持同任务、同 seed、同指标比较参数量、时间、显存和成功率。",
            "3. 安装 Isaac/Isaac Sim 后再推进 `isaac_domain_randomization_v1`，不能用 MuJoCo 扰动视频替代 Isaac 结果。",
            "4. 真实 WidowX 可用后再推进 `real_widowx_validation_v1`，必须采集 20-50 次 trial、相机视频、失败原因和 sim-to-real gap。",
            "",
            "## 4. 重建命令",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_external_dependency_readiness_audit.py"}"',
            "```",
            "",
            "## 5. 论文边界",
            "",
            "- 可以写：外部阶段的输入/输出契约、回填门禁和运行顺序已经固定。",
            "- 不能写：真实 OpenVLA、OpenVLA LoRA、Isaac domain randomization 或真实 WidowX 验证已经完成。",
            "- 只有真实运行并回填评测、资源、视频和报告后，planned 版本才能进入正式方法包。",
            "",
            f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    counts = summarize(rows)
    write_csv(args.output_csv, rows)
    write_json(args.output_json, rows, counts)
    write_md(args.output_md, rows, counts)
    print(f"external_dependency_readiness_md: {args.output_md}", flush=True)
    print(f"external_dependency_readiness_csv: {args.output_csv}", flush=True)
    print(f"external_dependency_readiness_json: {args.output_json}", flush=True)
    print(f"external_dependency_readiness_rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
