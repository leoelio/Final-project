from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "claim_evidence_traceability_v1"


CLAIMS = [
    {
        "claim_id": "C01",
        "claim_type": "任务与数据链路",
        "usable_claim": "MuJoCo WidowX 桌面抓取/放置任务、示范采集、轨迹回放和固定视频导出链路已经打通。",
        "primary_versions": "expert_scripted_v1；replay_demo_v1；structured_waypoint_policy_v1",
        "quantitative_evidence": "docs/evaluation_summary.csv；docs/task_bc_stage_report.md；docs/method_evidence_gate.md",
        "video_evidence": "outputs/presentation_clips/01_task_data_oracle.mp4；outputs/videos/expert_scripted_v1_seed0.mp4；outputs/videos/replay_demo_v1_seed0.mp4",
        "display_entry": "docs/stage_showcase_index.html；docs/stage_reproduction_runbook.md",
        "paper_redline": "expert、structured waypoint 和 replay 不能写成 learned VLA 或真实机器人验证。",
    },
    {
        "claim_id": "C02",
        "claim_type": "普通 BC 对照",
        "usable_claim": "Linear BC、MLP BC 等单步 imitation baseline 不能稳定完成闭环抓取；kNN 的训练范围成功更接近轨迹记忆。",
        "primary_versions": "linear_bc_v1；knn_bc_v1；mlp_bc_v1",
        "quantitative_evidence": "docs/evaluation_summary.csv；docs/task_bc_stage_report.md；docs/failure_mode_taxonomy.md",
        "video_evidence": "outputs/presentation_clips/02_basic_bc_baselines.mp4；outputs/videos/linear_bc_v1_seed0.mp4；outputs/videos/knn_bc_v1_seed0.mp4",
        "display_entry": "docs/video_evidence_gallery.html；docs/reproducible_command_index.md",
        "paper_redline": "普通 BC 不能写成语言理解、任务泛化或 VLA 后训练结果。",
    },
    {
        "claim_id": "C03",
        "claim_type": "Trajectory / ACT / Diffusion 对照",
        "usable_claim": "trajectory-conditioned BC / ACT-style / Diffusion baseline 已覆盖历史观测、动作块、Transformer、CVAE、小型 CNN 和扩散动作块，但闭环接触与抬升仍不稳定。",
        "primary_versions": "trajectory_conditioned_chunk_bc_v2；trajectory_knn_chunk_bc_v1；torch_act_state_chunk_v1；torch_act_cvae_state_chunk_v1；visual_act_cnn_cvae_v1；torch_diffusion_policy_state_chunk_v1",
        "quantitative_evidence": "docs/trajectory_act_stage_report.md；docs/trajectory_act_stage_report.csv；docs/model_resource_summary.csv",
        "video_evidence": "outputs/presentation_clips/03_trajectory_act_diffusion.mp4；outputs/videos/trajectory_conditioned_chunk_bc_v2_seed0.mp4；outputs/videos/torch_act_state_chunk_v1_seed0.mp4",
        "display_entry": "docs/stage_reproduction_runbook.md；docs/failure_mode_taxonomy.md",
        "paper_redline": "当前是 state-only、pooled visual feature 或小型 CNN 的本地轻量 baseline，不能写成完整官方视觉 ACT 或完整视觉 Diffusion Policy。",
    },
    {
        "claim_id": "C04",
        "claim_type": "Action-head / PEFT proxy",
        "usable_claim": "本地 action-head、Adapter、LoRA-style 和 CLIP proxy 已建立可比较的资源与闭环评测接口，但当前不能证明真实 VLA 后训练成功。",
        "primary_versions": "object_language_action_head_lite_v1；adapter_action_head_lite_v1；lora_action_head_lite_v1；clip_action_head_lite_v1；multi_task_object_action_head_lite_v1",
        "quantitative_evidence": "docs/action_head_stage_report.md；docs/action_head_stage_report.csv；docs/model_resource_summary.csv；docs/data_efficiency_summary.csv",
        "video_evidence": "outputs/presentation_clips/04_action_head_peft_proxy.mp4；outputs/videos/object_language_action_head_lite_v1_seed1_success_example.mp4",
        "display_entry": "docs/method_evidence_gate.md；docs/research_question_showcase_plan.md",
        "paper_redline": "Adapter/LoRA-style 是本地 action-head proxy，不是 OpenVLA LoRA；CLIP 不是机器人 VLA；不能写成真实 pretrained VLA 后训练。",
    },
    {
        "claim_id": "C05",
        "claim_type": "语言/空间泛化",
        "usable_claim": "语言/空间泛化任务已经建立；expert/structured 能完成，普通 learned baseline 和本地 proxy 多数为 0/5，说明当前模型尚未形成真实语言/空间泛化能力。",
        "primary_versions": "expert_scripted_language_v1；structured_waypoint_policy_v1；object_language_action_head_lite_v1；clip_action_head_lite_v1；multi_task_object_action_head_lite_v1",
        "quantitative_evidence": "docs/language_generalization_summary.csv；docs/video_evidence_index.md；docs/failure_mode_taxonomy.md",
        "video_evidence": "outputs/presentation_clips/05_language_generalization.mp4；outputs/showcase/language_generalization_grid.mp4",
        "display_entry": "docs/video_presentation_storyboard.html；docs/video_evidence_gallery.html",
        "paper_redline": "规则解析、对象特征、语言 token 或 frozen CLIP action head 不能写成完整 VLM/VLA 语义泛化能力。",
    },
    {
        "claim_id": "C06",
        "claim_type": "数据效率",
        "usable_claim": "10/25/50/92 条 demonstration 的数据效率扫表已完成，可比较 kNN、trajectory-kNN 和 object action head 的小数据行为。",
        "primary_versions": "knn_bc；trajectory_knn；object_action_head",
        "quantitative_evidence": "docs/data_efficiency_summary.md；docs/data_efficiency_summary.csv；outputs/evaluations/data_efficiency_v2.json；outputs/figures/data_efficiency.svg",
        "video_evidence": "outputs/presentation_clips/02_basic_bc_baselines.mp4；outputs/presentation_clips/04_action_head_peft_proxy.mp4",
        "display_entry": "docs/research_question_showcase_plan.md；docs/experiment_dashboard.html",
        "paper_redline": "不能把 MuJoCo scripted demonstration 的小数据结论直接写成真实机械臂数据效率或真实 VLA 小数据优势。",
    },
    {
        "claim_id": "C07",
        "claim_type": "MuJoCo domain randomization 代理",
        "usable_claim": "MuJoCo 摩擦、执行器增益、力限和夹爪力度扰动代理评测已经完成，并已形成 Isaac domain randomization 运行交接门禁，可作为 sim-to-real 前置鲁棒性检查和后续 Isaac 回填模板。",
        "primary_versions": "structured_waypoint_policy_v1；trajectory_knn_chunk_bc_v1；visual_act_cnn_cvae_v1；domain_randomization_eval_v1；isaac_domain_randomization_handoff_v1",
        "quantitative_evidence": "docs/domain_randomization_summary.md；docs/domain_randomization_summary.csv；outputs/evaluations/domain_randomization_eval_v1.json；docs/isaac_domain_randomization_handoff.md；outputs/evaluations/isaac_domain_randomization_handoff_v1.json",
        "video_evidence": "outputs/presentation_clips/06_domain_randomization_proxy.mp4；outputs/videos/domain_randomization_trajectory_knn_low_friction_seed0.mp4；outputs/videos/domain_randomization_visual_act_cnn_cvae_low_friction_seed0.mp4",
        "display_entry": "docs/research_question_showcase_plan.md；docs/isaac_domain_randomization_handoff.md；docs/video_evidence_gallery.html",
        "paper_redline": "MuJoCo domain randomization 和 isaac_domain_randomization_handoff_v1 不能写成 Isaac domain randomization 已完成，也不能写成真实 WidowX 迁移成功或失败。",
    },
    {
        "claim_id": "C08",
        "claim_type": "视频展示证据",
        "usable_claim": "当前 {video_count} 条视频证据、{method_video_count} 个单方法视频、{showcase_count} 个宫格视频和 {presentation_count} 个答辩阶段视频均已进入索引和质量审计，可用于答辩展示。",
        "primary_versions": "video_evidence_index_v1；video_quality_audit_v1；presentation_video_pack_v1；video_presentation_storyboard_v1",
        "quantitative_evidence": "docs/video_evidence_index.csv；docs/video_quality_audit.csv；docs/final_artifact_manifest.json",
        "video_evidence": "outputs/presentation_clips/00_defense_video_reel.mp4；outputs/showcase/all_registered_methods_grid.mp4；outputs/showcase/core_methods_grid.mp4",
        "display_entry": "docs/video_evidence_gallery.html；docs/video_presentation_storyboard.html",
        "paper_redline": "视频只作为定性展示证据，不替代成功率、目标距离、语言泛化、资源规模和 domain randomization 表。",
    },
    {
        "claim_id": "C09",
        "claim_type": "OpenVLA 前置工作",
        "usable_claim": "OpenVLA 数据桥接、本地可行性检查、robot VLA action-head 交接门禁、远端运行包和结果回填门禁已经完成，说明当前数据能导出为 image/instruction/state/action 样本，本机不适合直接训练真实 OpenVLA LoRA，但后续远端运行的输入/输出契约、命令模板、结果回填 schema 和入包检查已经明确。",
        "primary_versions": "openvla_dataset_bridge_v1；openvla_feasibility_check_v1；openvla_bridge_gallery_v1；robot_vla_action_head_handoff_v1；robot_vla_remote_run_pack_v1；robot_vla_remote_result_intake_v1",
        "quantitative_evidence": "docs/openvla_dataset_bridge_report.md；outputs/evaluations/openvla_dataset_bridge_v1.json；docs/openvla_feasibility_report.md；docs/robot_vla_action_head_handoff.md；outputs/evaluations/robot_vla_action_head_handoff_v1.json；docs/robot_vla_remote_run_pack.md；outputs/evaluations/robot_vla_remote_run_pack_v1.json；docs/robot_vla_remote_result_intake.md；outputs/evaluations/robot_vla_remote_result_intake_v1.json",
        "video_evidence": "data/vla_bridge/openvla_dataset_bridge_v1/preview_grid.png",
        "display_entry": "docs/openvla_bridge_gallery.html；docs/robot_vla_action_head_handoff.md；docs/robot_vla_remote_run_pack.md；docs/robot_vla_remote_result_intake.md；docs/next_experiment_registry.md",
        "paper_redline": "不能写成 OpenVLA LoRA、RT-2、robot_vla_action_head_lite_v1、Isaac 或真实 WidowX 验证已经完成。",
    },
    {
        "claim_id": "C10",
        "claim_type": "方法与阶段统一追踪",
        "usable_claim": "25 个正式方法版本已经通过方法证据门禁，8 个阶段有复现实验手册，6 个研究问题有展示选择表，当前 MuJoCo 实验包可以按方法、阶段和研究问题三条线说明；外部依赖 readiness audit、真实 WidowX 验证协议和 trial 模板也已进入下一阶段门禁。",
        "primary_versions": "method_evidence_gate_v1；stage_reproduction_runbook_v1；research_question_showcase_plan_v1；final_artifact_manifest_v1；external_dependency_readiness_audit_v1；real_widowx_validation_handoff_v1",
        "quantitative_evidence": "docs/method_evidence_gate.csv；docs/stage_reproduction_runbook.csv；docs/research_question_showcase_plan.csv；docs/final_artifact_manifest.json；docs/external_dependency_readiness_audit.md；docs/external_dependency_readiness_audit.csv；docs/real_widowx_validation_handoff.md；outputs/evaluations/real_widowx_validation_handoff_v1.json；outputs/real_robot/real_widowx_validation_v1_trial_template.csv",
        "video_evidence": "outputs/presentation_clips/00_defense_video_reel.mp4；outputs/showcase/all_registered_methods_grid.mp4",
        "display_entry": "docs/final_experiment_package.md；docs/stage_showcase_index.html；docs/external_dependency_readiness_audit.md；docs/real_widowx_validation_handoff.md",
        "paper_redline": "当前完成的是 MuJoCo 实验包；external_dependency_readiness_audit_v1 不是策略成功率结果；真实 OpenVLA、Isaac 和真实 WidowX trial 仍必须作为后续阶段单独登记、评测和保存视频，handoff 不能写成真实验证结果。",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a claim-to-evidence traceability matrix.")
    parser.add_argument("--video-evidence", type=Path, default=ROOT / "docs" / "video_evidence_index.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs" / "final_artifact_manifest.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "claim_evidence_traceability.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "claim_evidence_traceability.md")
    return parser.parse_args()


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", "<br>") for value in values) + " |"


def split_paths(value: str) -> list[str]:
    paths: list[str] = []
    for part in value.replace("；", "\n").splitlines():
        item = part.strip().strip("`")
        if item.startswith(("docs/", "outputs/", "data/")):
            paths.append(item)
    return paths


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def claim_context(args: argparse.Namespace) -> dict[str, str]:
    video_rows = read_csv(args.video_evidence)
    manifest = read_json(args.manifest)
    counts = manifest.get("counts", {})
    return {
        "video_count": str(len(video_rows)),
        "method_video_count": str(counts.get("registered_method_videos", 0)),
        "showcase_count": str(counts.get("showcase_videos", 3) or 3),
        "presentation_count": str(counts.get("presentation_pack_items", 0)),
    }


def build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    context = claim_context(args)
    rows = []
    for claim in CLAIMS:
        paths = (
            split_paths(claim["quantitative_evidence"])
            + split_paths(claim["video_evidence"])
            + split_paths(claim["display_entry"])
        )
        missing = [path for path in paths if not (ROOT / path).exists()]
        row = dict(claim)
        row["usable_claim"] = row["usable_claim"].format(**context)
        row["evidence_status"] = "可写（有证据）" if not missing else "需补证据"
        row["missing_evidence"] = "无" if not missing else "；".join(missing)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "# Claim 证据追踪矩阵",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：把论文和答辩中可以写的阶段性 claim 逐条绑定到量化证据、视频证据、展示入口和论文红线。该矩阵不新增实验结论，只用于防止把 MuJoCo proxy、ACT-lite、action-head proxy 或 OpenVLA 前置工作写过界。",
        "",
        "## 1. 总览",
        "",
        md_row(["项目", "数量"]),
        md_row(["---", "---:"]),
        md_row(["可写 claim", str(len(rows))]),
        md_row(["证据完整", str(sum(1 for row in rows if row["evidence_status"] == "可写（有证据）"))]),
        md_row(["需补证据", str(sum(1 for row in rows if row["evidence_status"] != "可写（有证据）"))]),
        "",
        "## 2. Claim 矩阵",
        "",
        md_row(["ID", "类型", "可写 claim", "关键版本", "量化证据", "视频证据", "论文红线", "状态"]),
        md_row(["---", "---", "---", "---", "---", "---", "---", "---"]),
    ]
    for row in rows:
        lines.append(
            md_row(
                [
                    row["claim_id"],
                    row["claim_type"],
                    row["usable_claim"],
                    row["primary_versions"],
                    row["quantitative_evidence"],
                    row["video_evidence"],
                    row["paper_redline"],
                    row["evidence_status"],
                ]
            )
        )

    lines.extend(["", "## 3. 分条写作提示", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row['claim_id']}：{row['claim_type']}",
                "",
                f"- 可写 claim：{row['usable_claim']}",
                f"- 关键版本：{row['primary_versions']}",
                f"- 量化证据：{row['quantitative_evidence']}",
                f"- 视频证据：{row['video_evidence']}",
                f"- 展示入口：{row['display_entry']}",
                f"- 论文红线：{row['paper_redline']}",
                f"- 证据状态：{row['evidence_status']}",
                "",
            ]
        )

    lines.extend(
        [
            "## 4. 使用边界",
            "",
            "1. 该矩阵是写作和答辩门禁，不替代任何 CSV 评测表。",
            "2. 若某个 claim 需要写进论文正文，必须同时引用量化表和视频/展示入口，并保留论文红线。",
            "3. 新增真实 OpenVLA、Isaac 或真实 WidowX 实验时，必须追加新 claim 或更新现有 claim 的证据状态。",
            "",
            "## 5. 重建命令",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_claim_evidence_traceability.py"}"',
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
    print(f"claim_evidence_traceability_md: {args.output_md}", flush=True)
    print(f"claim_evidence_traceability_csv: {args.output_csv}", flush=True)
    print(f"claim_evidence_rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
