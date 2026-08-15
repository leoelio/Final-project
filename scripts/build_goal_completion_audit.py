from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

CSV_FIELDS = ["目标要求", "当前状态", "权威证据", "当前数量/结果", "缺口或边界", "下一步"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese completion audit for the active experiment objective.")
    parser.add_argument("--versions", type=Path, default=ROOT / "docs" / "experiment_versions.json")
    parser.add_argument("--evaluation", type=Path, default=ROOT / "docs" / "evaluation_summary.csv")
    parser.add_argument("--language", type=Path, default=ROOT / "docs" / "language_generalization_summary.csv")
    parser.add_argument("--resources", type=Path, default=ROOT / "docs" / "model_resource_summary.csv")
    parser.add_argument("--data-efficiency", type=Path, default=ROOT / "docs" / "data_efficiency_summary.csv")
    parser.add_argument("--domain-randomization", type=Path, default=ROOT / "docs" / "domain_randomization_summary.csv")
    parser.add_argument("--stage-comparison", type=Path, default=ROOT / "docs" / "stage_comparison_report.csv")
    parser.add_argument("--video-evidence", type=Path, default=ROOT / "docs" / "video_evidence_index.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs" / "final_artifact_manifest.json")
    parser.add_argument("--defense-evidence-pack-json", type=Path, default=ROOT / "outputs" / "evaluations" / "defense_evidence_pack_v1.json")
    parser.add_argument("--external-dependency-readiness-json", type=Path, default=ROOT / "outputs" / "evaluations" / "external_dependency_readiness_audit_v1.json")
    parser.add_argument("--version-naming-json", type=Path, default=ROOT / "outputs" / "evaluations" / "version_naming_and_gate_spec_v1.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "goal_completion_audit.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "goal_completion_audit.csv")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def method_versions(methods: list[dict[str, str]], versions: list[str]) -> str:
    present = [version for version in versions if any(method["version"] == version for method in methods)]
    return "；".join(present)


def row(requirement: str, status: str, evidence: str, result: str, boundary: str, next_step: str) -> dict[str, str]:
    return {
        "目标要求": requirement,
        "当前状态": status,
        "权威证据": evidence,
        "当前数量/结果": result,
        "缺口或边界": boundary,
        "下一步": next_step,
    }


def build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    versions = read_json(args.versions)
    manifest = read_json(args.manifest)
    defense_pack = read_json(args.defense_evidence_pack_json)
    external_readiness = read_json(args.external_dependency_readiness_json)
    version_naming = read_json(args.version_naming_json)
    methods = versions["methods"]
    evaluation = read_csv(args.evaluation)
    language = read_csv(args.language)
    resources = read_csv(args.resources)
    data_efficiency = read_csv(args.data_efficiency)
    domain_randomization = read_csv(args.domain_randomization)
    stage_rows = read_csv(args.stage_comparison)
    video_rows = read_csv(args.video_evidence)

    counts = manifest["counts"]
    readiness_counts = external_readiness.get("readiness_counts", {})
    candidate_video_rows = [item for item in video_rows if item.get("视频类型") == "候选诊断片段"]
    domain_methods = sorted({item["method_key"] for item in domain_randomization})
    domain_names = sorted({item["domain"] for item in domain_randomization})

    trajectory_versions = [
        "trajectory_conditioned_chunk_bc_v2",
        "trajectory_knn_chunk_bc_v1",
        "torch_act_state_chunk_v1",
        "torch_act_state_chunk_cuda_v1",
        "phase_conditioned_torch_act_v1",
        "torch_act_cvae_state_chunk_v1",
        "visual_feature_act_lite_v1",
        "visual_act_cnn_cvae_v1",
    ]
    vla_proxy_versions = [
        "object_language_action_head_lite_v1",
        "reward_weighted_action_head_lite_v1",
        "phase_conditioned_action_head_lite_v1",
        "adapter_action_head_lite_v1",
        "lora_action_head_lite_v1",
        "vision_language_action_head_lite_v1",
        "clip_action_head_lite_v1",
        "multi_task_object_action_head_lite_v1",
    ]

    return [
        row(
            "保留不同方法和阶段的版本名称",
            "当前 MuJoCo 阶段已完成",
            "docs/experiment_versions.json；docs/version_lineage_index.html；docs/version_naming_and_gate_spec.md；outputs/evaluations/version_naming_and_gate_spec_v1.json；docs/final_artifact_manifest.json",
            f"{len(methods)} 个正式方法；manifest 登记 {counts['methods']} 个方法；版本谱系索引 {counts.get('version_lineage_rows', 0)} 条；version_naming_and_gate_spec_v1 规则 {version_naming.get('rule_count', 0)} 条",
            "后续 OpenVLA、Isaac、真实机械臂方法还未登记为完成版本",
            "新增方法时先写 version、stage、artifact、clip，再补评测和视频",
        ),
        row(
            "能够按方法和阶段进行说明",
            "当前 MuJoCo 阶段已完成",
            "docs/method_stage_audit.md；docs/stage_comparison_report.md；docs/research_evidence_map.md",
            f"{len(stage_rows)} 个阶段分组；覆盖 scripted、BC、trajectory/ACT、Diffusion、action-head、PEFT/VLM proxy",
            "真实 VLA/Isaac/真实机器人阶段只能作为计划或未完成边界",
            "真实 VLA 接入后沿用阶段字段追加到 stage comparison",
        ),
        row(
            "能够做评测比较",
            "当前 MuJoCo 阶段已完成",
            "docs/core_v2_holdout_comparison_matrix.md；docs/core_task_comparison_matrix.md；docs/evaluation_summary.csv；docs/language_generalization_summary.csv；docs/model_resource_summary.csv；docs/data_efficiency_summary.csv；docs/domain_randomization_summary.csv；docs/method_comparison_dashboard.html",
            f"Core V2 留出矩阵 28 行；历史核心矩阵 24 行；主任务 {len(evaluation)} 行；语言/空间 {len(language)} 行；资源 {len(resources)} 行；数据效率 {len(data_efficiency)} 行；domain randomization {len(domain_randomization)} 行",
            "数据效率仍是快速扫表；真实机器人没有成功率表",
            "真实机器人阶段复用 success_rate、target_distance、train_time、peak_vram 字段",
        ),
        row(
            "能够展示仿真视频片段",
            "当前 MuJoCo 阶段已完成",
            "docs/video_evidence_index.md；docs/video_presentation_storyboard.md；docs/thesis_visual_evidence_index.html；docs/defense_qa_playbook.html；docs/presentation_video_pack.md；outputs/videos；outputs/showcase；outputs/presentation_clips",
            f"{len(video_rows)} 条视频证据，其中候选诊断片段 {len(candidate_video_rows)} 条；{counts['registered_method_videos']} 个单方法视频；{counts['presentation_pack_items']} 个答辩视频包项目；论文图表与视频证据索引 {counts.get('thesis_visual_evidence_rows', 0)} 条；答辩追问 Q&A {counts.get('defense_qa_rows', 0)} 条；已加入 60 秒总览 reel、6 个阶段短片、视频展示讲稿与低摩擦 domain randomization 对比视频",
            "真实机械臂视频尚未采集；Isaac 视频尚未采集",
            "新阶段必须同时保存 mp4、json 元数据、证据用途、讲稿提示和论文红线",
        ),
        row(
            "每次运行具有可视化 viewer 命令",
            "当前 MuJoCo 阶段已完成",
            "docs/reproducible_command_index.md",
            f"{len(methods)} 条主任务 viewer 命令；{len(language)} 条语言/空间 viewer 命令；统一慢速 --duration 60 --speed 0.05",
            "真实机械臂可视化需要相机录制或独立 GUI，不由 MuJoCo viewer 证明",
            "真实机械臂阶段补充相机视角、任务日志和回放索引",
        ),
        row(
            "trajectory-conditioned BC / ACT 作为可靠对照组",
            "当前 MuJoCo 阶段已完成，但策略效果不稳定",
            "docs/evaluation_summary.csv；docs/model_resource_summary.csv；docs/stage_comparison_report.md；docs/trajectory_act_experiment_record.md；docs/video_evidence_index.md",
            method_versions(methods, trajectory_versions),
            "不能写成完整官方 ACT 或完整视觉 ACT；当前是 state-only/pooled visual feature 的轻量 baseline",
            "若继续补强 ACT，应优先做更真实视觉 encoder、更多示范多样性和夹爪接触稳定性",
        ),
        row(
            "轻量 VLA 后训练路线有可比较代理实验",
            "当前 MuJoCo 阶段已完成；真实 OpenVLA/机器人 VLA 后训练不属于当前完成条件",
            "docs/model_resource_summary.csv；docs/language_generalization_summary.csv；docs/research_evidence_map.md；docs/openvla_dataset_bridge_report.md；docs/openvla_feasibility_report.md；docs/robot_vla_action_head_handoff.md；docs/robot_vla_remote_run_pack.md；docs/robot_vla_remote_result_intake.md；docs/external_dependency_readiness_audit.md",
            method_versions(methods, vla_proxy_versions) + f"；openvla_dataset_bridge_v1 样本 {counts.get('openvla_bridge_samples', 0)} 条；openvla_feasibility_check_v1 已完成；robot_vla_action_head_handoff_v1 已完成；robot_vla_remote_run_pack_v1 已完成，包内文件 {counts.get('robot_vla_remote_pack_files', 0)} 个；robot_vla_remote_result_intake_v1 已完成，当前远端返回文件 {counts.get('robot_vla_remote_intake_returned_files', 0)} 个",
            "不能宣称 OpenVLA LoRA、RT-2 或真实 pretrained VLA 后训练已经完成；bridge/feasibility/handoff/remote pack/intake 只是历史扩展模板",
            "当前范围无需执行；未来扩展到远端 GPU 时再按 remote run pack、handoff 和 intake 契约回填",
        ),
        row(
            "外部依赖阶段 readiness 门禁",
            "当前范围外（保留为历史交接模板）",
            "docs/external_dependency_readiness_audit.md；docs/external_dependency_readiness_audit.csv；outputs/evaluations/external_dependency_readiness_audit_v1.json；docs/remaining_experiment_execution_board.md；docs/next_experiment_registry.md；docs/final_artifact_manifest.json",
            f"external_dependency_readiness_audit_v1 行数 {external_readiness.get('row_count', 0)}；supporting_evidence_ready {readiness_counts.get('supporting_evidence_ready', 0)}；waiting_remote_result {readiness_counts.get('waiting_remote_result', 0)}；waiting_robot_vla_action_head {readiness_counts.get('waiting_robot_vla_action_head', 0)}；waiting_isaac_runtime {readiness_counts.get('waiting_isaac_runtime', 0)}；waiting_real_robot_trials {readiness_counts.get('waiting_real_robot_trials', 0)}；formal_method_allowed_now 全部为否",
            "readiness audit 不是策略成功率结果，只说明历史 planned 外部阶段的阻塞条件、回填工件和论文边界；不能把它写成 OpenVLA/Isaac/真实 WidowX 已完成",
            "当前 MuJoCo-only 范围无需执行；保留记录以防未来扩展",
        ),
        row(
            "实验记录尽量中文化并可追溯",
            "当前 MuJoCo 阶段已完成",
            "docs/experiment_log.md；docs/trajectory_act_experiment_record.md；docs/defense_live_runbook.md；docs/final_defense_narrative_script.md；scripts/verify_experiment_artifacts.py",
            f"中文实验日志、研究问题映射、阶段报告、trajectory/ACT 中文台账、答辩现场 Runbook、最终答辩讲解脚本、视频索引和总清单均纳入清洁检查；manifest 中 defense_live_runbook_rows={counts.get('defense_live_runbook_rows', 0)}，final_defense_narrative_rows={counts.get('final_defense_narrative_rows', 0)}，trajectory_act_record_rows={counts.get('trajectory_act_record_rows', 0)}",
            "PowerShell 终端可能显示 mojibake，但文件本身为 UTF-8",
            "后续仍用 UTF-8 写文档，必要时用 Python 按 utf-8-sig 读取验证",
        ),
        row(
            "答辩证据包归档可复制可验证",
            "当前 MuJoCo 阶段已完成",
            "docs/defense_evidence_pack.md；outputs/defense_evidence_pack/defense_evidence_pack_v1；outputs/defense_evidence_pack/defense_evidence_pack_v1.zip；outputs/evaluations/defense_evidence_pack_v1.json；scripts/build_defense_evidence_pack.py",
            f"defense_evidence_pack_v1 已生成；包内文件 {defense_pack.get('file_count', 0)} 个；zip 路径 {defense_pack.get('archive_path', '')}；final manifest 记录文件数 {counts.get('defense_evidence_pack_files', 0)} 个",
            "证据包只归档当前 MuJoCo 实验材料，不能写成真实 OpenVLA、Isaac 或真实 WidowX 已完成",
            "新增真实 VLA/Isaac/真实机械臂结果后，重新运行 build_defense_evidence_pack.py 并更新 final manifest",
        ),
        row(
            "Isaac/domain randomization 与真实机械臂验证",
            "MuJoCo domain randomization 代理已完成；Isaac 与真实 WidowX 为当前范围外的历史交接模板",
            "docs/domain_randomization_summary.md；docs/domain_randomization_summary.csv；outputs/evaluations/domain_randomization_eval_v1.json；docs/isaac_domain_randomization_handoff.md；outputs/evaluations/isaac_domain_randomization_handoff_v1.json；docs/real_widowx_validation_handoff.md；outputs/evaluations/real_widowx_validation_handoff_v1.json；outputs/real_robot/real_widowx_validation_v1_trial_template.csv；docs/external_dependency_readiness_audit.md；docs/next_phase_implementation.md；docs/research_evidence_map.md",
            f"domain_randomization_eval_v1：{len(domain_randomization)} 条 episode，方法 {len(domain_methods)} 个，扰动域 {len(domain_names)} 个；isaac_domain_randomization_handoff_v1 行数 {counts.get('isaac_handoff_rows', 0)}；real_widowx_validation_handoff_v1 行数 {counts.get('real_widowx_handoff_rows', 0)}，trial 模板 {counts.get('real_widowx_trial_template_rows', 0)} 条；Isaac 实验 0；真实 WidowX 测试 0",
            "MuJoCo 扰动评测不能写成高保真 Isaac sim-to-real；真实 WidowX handoff 不能写成真实机械臂迁移成功/失败",
            "当前 MuJoCo-only 范围无需执行；未来扩展时再复用 handoff 字段和 trial 模板",
        ),
        row(
            "整体实验完成后可用于论文和答辩展示",
            "当前 MuJoCo 阶段已完成（MuJoCo-only 正式范围已完成）",
            "docs/mujoco_only_scope.md；docs/final_experiment_package.md；docs/final_defense_narrative_script.md；docs/stage_showcase_index.html；docs/video_presentation_storyboard.html；docs/thesis_visual_evidence_index.html；docs/defense_qa_playbook.html；docs/version_lineage_index.html；docs/defense_live_runbook.md；docs/defense_deck.html；docs/thesis_results_chapter_draft.md；docs/final_artifact_manifest.md；docs/defense_evidence_pack.md",
            "已有 dashboard、阶段展示总索引、视频展示讲稿与时间线、论文图表与视频证据索引、答辩追问 Q&A、实验版本谱系索引、答辩现场 Runbook、HTML deck、论文结果章节草稿、答辩视频包、下一阶段实验注册表、外部依赖 readiness audit、最终 manifest 和答辩证据包",
            "论文结论仅限 MuJoCo 仿真；不能把 proxy 写成真实 OpenVLA、Isaac 或真实 WidowX 结果",
            "本范围内无必做后续步骤；未来扩展时使用历史交接模板",
        ),
    ]


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(value.replace("|", "\\|").replace("\n", " ") for value in values) + " |"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_md(rows: list[dict[str, str]]) -> str:
    done = sum(1 for item in rows if item["当前状态"].startswith("当前 MuJoCo 阶段已完成"))
    incomplete = sum(1 for item in rows if item["当前状态"] == "未完成")
    out_of_scope = sum(1 for item in rows if item["当前状态"].startswith("当前范围外"))
    partial = len(rows) - done - incomplete - out_of_scope
    lines = [
        "# 总目标完成度审计",
        "",
        "版本：`goal_completion_audit_v1`",
        "",
        "用途：逐条审计当前工作是否满足 MuJoCo-only 范围下“按计划进行实验、保留各部分版本名称、说明不同方法和阶段、进行评测比较、展示仿真视频片段”的目标。范围决议见 `docs/mujoco_only_scope.md`。",
        "",
        "## 1. 总体结论",
        "",
        f"- 当前 MuJoCo 阶段已完成项：{done}。",
        f"- 当前部分完成项：{partial}。",
        f"- 当前范围外的历史交接项：{out_of_scope}。",
        f"- 当前未完成项：{incomplete}。",
        "- MuJoCo-only 正式范围已完成，可用于论文和答辩展示。",
        "- 当前 MuJoCo 实验包可用于阶段论文/答辩展示。",
        "- Isaac、真实 WidowX 和真实 OpenVLA 仅保留为历史扩展模板，不属于当前完成条件，也不能写成已经完成。",
        "",
        "## 2. 审计表",
        "",
        md_row(CSV_FIELDS),
        md_row(["---", "---", "---", "---", "---", "---"]),
    ]
    for item in rows:
        lines.append(md_row([item[field] for field in CSV_FIELDS]))

    lines.extend(
        [
            "",
            "## 3. 重建命令",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_goal_completion_audit.py"}"',
            "```",
            "",
            "## 4. 总体验证命令",
            "",
            "```powershell",
            '$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "verify_experiment_artifacts.py"}"',
            "```",
            "",
            f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_md(rows), encoding="utf-8")
    write_csv(args.output_csv, rows)
    print(f"goal_completion_audit_md: {args.output_md}", flush=True)
    print(f"goal_completion_audit_csv: {args.output_csv}", flush=True)
    print(f"audit_rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
