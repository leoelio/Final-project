from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "real_widowx_validation_handoff_v1"
TARGET_VERSION = "real_widowx_validation_v1"


SCENE_CONTRACT = {
    "robot_family": "WidowX / Trossen WX250s 或同类桌面机械臂；真实型号必须在 trial 中记录",
    "primary_task": "place_blue_cube_blue_pad",
    "primary_instruction": "place the blue cube on the blue pad",
    "heldout_task": "place_blue_cube_red_pad",
    "language_task": "move_leftmost_to_bowl",
    "objects": "blue_cube；red_cube；green_cube；yellow_cube；red_cylinder；blue_cylinder；green_ball；red_pad；blue_pad；bowl",
    "minimum_trials": "20",
    "recommended_trials": "50",
}


SAFETY_GATES = [
    "机械臂急停按钮可用，并在操作者伸手范围内。",
    "真实夹爪、桌面、相机和目标区域完成标定，并记录标定文件路径。",
    "第一次运行只做空场景 dry-run，不放置物体。",
    "第二次运行只放置物体但不闭合夹爪，确认轨迹不会扫出桌面。",
    "真实运行速度不得高于 MuJoCo 慢速 viewer 方案对应的保守速度。",
    "每个 trial 必须有人现场观察；任何异常接触、夹爪卡死、物体飞出桌面立即停机。",
    "同一策略连续 2 次出现危险 stop_reason 后停止该策略的真实测试。",
    "不得用 MuJoCo 或 Isaac 视频替代真实相机视频。",
]


TRIAL_FIELDS = [
    "trial_id",
    "planned_block",
    "method_version",
    "task",
    "instruction",
    "robot_model",
    "camera_model",
    "camera_pose_id",
    "calibration_file",
    "controller_version",
    "control_rate_hz",
    "operator",
    "date",
    "object_start_pose",
    "target_pose",
    "seed_or_layout_id",
    "success",
    "target_distance_m",
    "grasp_success",
    "object_lifted",
    "object_dropped",
    "out_of_workspace",
    "stop_reason",
    "failure_reason",
    "video_path",
    "snapshot_path",
    "notes",
]


REQUIRED_RETURN_FILES = [
    "outputs/real_robot/real_widowx_validation_v1.csv",
    "docs/real_widowx_validation_summary.md",
    "docs/real_widowx_validation_summary.csv",
    "outputs/evaluations/real_widowx_validation_v1.json",
    "outputs/videos/real_widowx_validation_v1_trial001.mp4",
    "outputs/videos/real_widowx_validation_v1_trial001.json",
    "docs/real_widowx_safety_notes.md",
]


PAPER_BOUNDARY = [
    "可以写成：真实 WidowX 验证协议和 trial 记录模板已经建立。",
    "不能写成真实 WidowX 验证已经完成。",
    "不能写成真实机械臂迁移成功或失败已经验证。",
    "不能用 MuJoCo 或 Isaac 视频代替真实相机视频。",
    "只有 20-50 次真实 trial、逐次日志、相机视频和汇总报告回填后，才能把 real_widowx_validation_v1 登记为完成实验。",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the real WidowX validation handoff and trial template.")
    parser.add_argument("--versions", type=Path, default=ROOT / "docs" / "experiment_versions.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "real_widowx_validation_handoff.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "real_widowx_validation_handoff.csv")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / f"{VERSION}.json")
    parser.add_argument("--trial-template", type=Path, default=ROOT / "outputs" / "real_robot" / "real_widowx_validation_v1_trial_template.csv")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def source_summary(path: Path) -> dict[str, Any]:
    versions = read_json(path)
    methods = versions.get("methods", [])
    method_versions = [str(row.get("version")) for row in methods if row.get("version")]
    preferred = [
        version
        for version in (
            "structured_waypoint_policy_v1",
            "trajectory_knn_chunk_bc_v1",
            "object_language_action_head_lite_v1",
            "robot_vla_action_head_lite_v1",
        )
        if version in method_versions or version == "robot_vla_action_head_lite_v1"
    ]
    return {
        "registered_methods": len(methods),
        "candidate_policy_versions": preferred,
        "source_versions_file": "docs/experiment_versions.json",
    }


def build_trial_template() -> list[dict[str, str]]:
    blocks = [
        ("safety_baseline_primary", "structured_waypoint_policy_v1", "place_blue_cube_blue_pad", "place the blue cube on the blue pad", 10),
        ("best_learned_primary", "best_learned_policy_or_robot_vla_policy", "place_blue_cube_blue_pad", "place the blue cube on the blue pad", 10),
        ("heldout_target_region", "best_learned_policy_or_robot_vla_policy", "place_blue_cube_red_pad", "place the blue cube on the red pad", 10),
        ("language_spatial_task", "best_learned_policy_or_robot_vla_policy", "move_leftmost_to_bowl", "move the leftmost object to the bowl", 10),
        ("repeatability_random_layout", "best_safe_policy_after_first_40_trials", "place_blue_cube_blue_pad", "place the blue cube on the blue pad", 10),
    ]
    rows: list[dict[str, str]] = []
    trial_id = 1
    for block, method, task, instruction, count in blocks:
        for _ in range(count):
            row = {field: "" for field in TRIAL_FIELDS}
            row.update(
                {
                    "trial_id": f"trial{trial_id:03d}",
                    "planned_block": block,
                    "method_version": method,
                    "task": task,
                    "instruction": instruction,
                    "success": "pending_real_robot_run",
                    "grasp_success": "pending_real_robot_run",
                    "object_lifted": "pending_real_robot_run",
                    "object_dropped": "pending_real_robot_run",
                    "out_of_workspace": "pending_real_robot_run",
                    "video_path": f"outputs/videos/real_widowx_validation_v1_trial{trial_id:03d}.mp4",
                    "snapshot_path": f"outputs/real_robot/snapshots/real_widowx_validation_v1_trial{trial_id:03d}.png",
                }
            )
            rows.append(row)
            trial_id += 1
    return rows


def build_rows(summary: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = [
        {
            "category": "target",
            "key": TARGET_VERSION,
            "source": VERSION,
            "required_value": "20-50 次真实 trial 回填后才能登记为完成",
            "status": "waiting_real_robot",
            "note": "当前只建立协议和模板，不产生真实机械臂成功率。",
        },
        {
            "category": "source",
            "key": "candidate_policy_versions",
            "source": summary["source_versions_file"],
            "required_value": "；".join(summary["candidate_policy_versions"]),
            "status": "mapped",
            "note": f"当前正式方法数 {summary['registered_methods']}；真实运行前仍需人工选择安全策略。",
        },
    ]

    for key, value in SCENE_CONTRACT.items():
        rows.append(
            {
                "category": "scene_contract",
                "key": key,
                "source": VERSION,
                "required_value": value,
                "status": "required",
                "note": "真实机械臂 trial 必须保持可追溯任务语义。",
            }
        )
    for gate in SAFETY_GATES:
        rows.append(
            {
                "category": "safety_gate",
                "key": gate,
                "source": VERSION,
                "required_value": "must_pass_before_trial",
                "status": "locked",
                "note": "未满足时不允许开始真实机械臂 trial。",
            }
        )
    for field in TRIAL_FIELDS:
        rows.append(
            {
                "category": "trial_field",
                "key": field,
                "source": VERSION,
                "required_value": "required",
                "status": "required",
                "note": "用于逐次记录真实机械臂 trial。",
            }
        )
    for file_path in REQUIRED_RETURN_FILES:
        rows.append(
            {
                "category": "required_return_file",
                "key": file_path,
                "source": TARGET_VERSION,
                "required_value": "must_exist_after_real_robot_run",
                "status": "missing_until_real_robot_run",
                "note": "真实机械臂运行后回填。",
            }
        )
    for boundary in PAPER_BOUNDARY:
        rows.append(
            {
                "category": "paper_boundary",
                "key": boundary,
                "source": VERSION,
                "required_value": "must_follow",
                "status": "locked",
                "note": "论文、答辩和实验日志均需遵守。",
            }
        )
    return rows


def md_row(values: list[object]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in values) + " |"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["category", "key", "source", "required_value", "status", "note"])
        writer.writeheader()
        writer.writerows(rows)


def write_trial_template(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=TRIAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, summary: dict[str, Any], rows: list[dict[str, str]]) -> None:
    lines = [
        "# Real WidowX Validation 运行交接门禁",
        "",
        f"版本：`{VERSION}`",
        "",
        f"目标计划版本：`{TARGET_VERSION}`",
        "",
        "状态：`completed_prerequisite`。该文件只说明真实 WidowX 或同类机械臂验证如何运行、记录和回填，不是真实机械臂评测结果。",
        "",
        "## 1. 当前状态",
        "",
        md_row(["项目", "状态"]),
        md_row(["---", "---"]),
        md_row(["当前正式方法数", summary["registered_methods"]]),
        md_row(["候选策略版本", "；".join(summary["candidate_policy_versions"])]),
        md_row(["真实 trial 数", "0"]),
        md_row(["是否可登记为完成真实验证", "False"]),
        "",
        "## 2. 场景和任务契约",
        "",
        md_row(["字段", "值"]),
        md_row(["---", "---"]),
    ]
    for key, value in SCENE_CONTRACT.items():
        lines.append(md_row([key, value]))

    lines.extend(
        [
            "",
            "## 3. 真实机械臂安全门禁",
            "",
        ]
    )
    lines.extend(f"- {gate}" for gate in SAFETY_GATES)

    lines.extend(
        [
            "",
            "## 4. Trial 必填字段",
            "",
            "```text",
            *TRIAL_FIELDS,
            "```",
            "",
            "## 5. 回填必须文件",
            "",
            "```text",
            *REQUIRED_RETURN_FILES,
            "```",
            "",
            "## 6. Trial 模板",
            "",
            "`outputs/real_robot/real_widowx_validation_v1_trial_template.csv` 已预留 50 条 trial。前 20 条满足最低验证门槛，完整 50 条用于覆盖主任务、留出目标区域、语言/空间任务和重复性测试。所有行当前都标记为 `pending_real_robot_run`。",
            "",
            "## 7. 论文边界",
            "",
        ]
    )
    lines.extend(f"- {boundary}" for boundary in PAPER_BOUNDARY)

    lines.extend(
        [
            "",
            "## 8. 检查表",
            "",
            md_row(["类别", "键", "来源", "要求/取值", "状态", "说明"]),
            md_row(["---", "---", "---", "---", "---", "---"]),
        ]
    )
    for row in rows:
        lines.append(md_row([row["category"], row["key"], row["source"], row["required_value"], row["status"], row["note"]]))

    lines.extend(
        [
            "",
            "## 9. 重建命令",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_real_widowx_validation_handoff.py"}"',
            "```",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = source_summary(args.versions)
    rows = build_rows(summary)
    trial_rows = build_trial_template()
    payload = {
        "version": VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed_prerequisite",
        "target_planned_version": TARGET_VERSION,
        "scene_contract": SCENE_CONTRACT,
        "source_summary": summary,
        "safety_gates": SAFETY_GATES,
        "trial_fields": TRIAL_FIELDS,
        "trial_template": "outputs/real_robot/real_widowx_validation_v1_trial_template.csv",
        "trial_template_rows": len(trial_rows),
        "required_return_files": REQUIRED_RETURN_FILES,
        "paper_boundary": PAPER_BOUNDARY,
        "rows": rows,
        "can_register_completed_real_robot_validation": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.output_csv, rows)
    write_trial_template(args.trial_template, trial_rows)
    write_md(args.output_md, summary, rows)
    print(f"real_widowx_validation_handoff_json: {args.output_json}", flush=True)
    print(f"real_widowx_validation_handoff_md: {args.output_md}", flush=True)
    print(f"real_widowx_validation_handoff_csv: {args.output_csv}", flush=True)
    print(f"real_widowx_trial_template: {args.trial_template}", flush=True)
    print(f"rows: {len(rows)}", flush=True)
    print(f"trial_template_rows: {len(trial_rows)}", flush=True)


if __name__ == "__main__":
    main()
