from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


QUESTION_ROWS = [
    {
        "研究问题": "轻量化后训练是否省算力/参数？",
        "status": "已完成（MuJoCo 本地 proxy 证据），真实 OpenVLA/大 VLA 后训练未完成；已补充 VLA 数据桥接、本地可行性检查、robot VLA action-head 远端运行门禁、远端运行包和远端结果回填门禁。",
        "evidence": "docs/model_resource_summary.csv；docs/model_resource_summary.md；docs/stage_comparison_report.md；docs/final_artifact_manifest.json；docs/openvla_dataset_bridge_report.md；docs/openvla_feasibility_report.md；docs/robot_vla_action_head_handoff.md；docs/robot_vla_remote_run_pack.md；docs/robot_vla_remote_result_intake.md",
        "methods": "object_language_action_head_lite_v1；adapter_action_head_lite_v1；lora_action_head_lite_v1；clip_action_head_lite_v1；openvla_dataset_bridge_v1（数据桥接，不是策略）；openvla_feasibility_check_v1（环境审计，不是策略）；robot_vla_action_head_handoff_v1（交接门禁，不是策略）；robot_vla_remote_run_pack_v1（远端运行包，不是策略）；robot_vla_remote_result_intake_v1（结果回填门禁，不是策略）",
        "summary_key": "resource",
        "display": "docs/defense_deck.html；outputs/figures/resource_vs_success.svg；docs/stage_comparison_report.md；docs/openvla_bridge_gallery.html；docs/robot_vla_action_head_handoff.md；docs/robot_vla_remote_run_pack.md；docs/robot_vla_remote_result_intake.md；docs/stage_showcase_index.html",
        "conclusion": "可以写：在当前本地 action-head/PEFT proxy 中，只训练小规模头部或低秩增量，资源开销明显小于重新训练完整策略网络。",
        "redline": "不能宣称 OpenVLA LoRA 或真实 pretrained VLA 微调已经完成；bridge/feasibility/handoff/remote pack/intake 只能称为数据格式、环境审计、远端运行门禁、可迁移运行包和回填门禁，不是策略效果。",
        "next": "本机继续做小型 proxy 和 viewer 评测；真实 OpenVLA/RT-2 类表征建议迁移到 48GB+ GPU 或云端后，按 remote run pack、handoff 和 intake 契约沿用同一资源表字段补充显存、训练时间和成功率。",
    },
    {
        "研究问题": "轻量化后训练是否省数据？",
        "status": "已完成（MuJoCo 数据效率扫表），真实机器人小数据适配未完成。",
        "evidence": "docs/data_efficiency_summary.csv；docs/data_efficiency_summary.md；docs/final_experiment_package.md",
        "methods": "knn_bc；trajectory_knn；object_action_head",
        "summary_key": "data_efficiency",
        "display": "outputs/figures/data_efficiency.svg；docs/experiment_dashboard.html",
        "conclusion": "可以写：当前已经完成 10/25/50/92 条 demonstration 的数据效率对比，可用于说明不同普通 baseline 与 action-head proxy 的小数据行为差异。",
        "redline": "不能把 MuJoCo scripted demonstration 的小数据结论直接等同于真实机械臂数据效率。",
        "next": "真实 WidowX 阶段复用 10/25/50 条示范预算，并保持同一成功率、目标距离和视频证据字段。",
    },
    {
        "研究问题": "语言/空间泛化是否优于普通 BC？",
        "status": "部分完成（语言/空间任务和本地 proxy 已测），真实 VLM 语义理解仍未验证。",
        "evidence": "docs/language_generalization_summary.csv；docs/video_evidence_index.md；outputs/showcase/language_generalization_grid.mp4",
        "methods": "expert_scripted_language_v1；structured_waypoint_policy_v1；object_language_action_head_lite_v1；vision_language_action_head_lite_v1；clip_action_head_lite_v1；multi_task_object_action_head_lite_v1",
        "summary_key": "language",
        "display": "outputs/presentation_clips/05_language_generalization.mp4；outputs/showcase/language_generalization_grid.mp4；docs/defense_deck.html；docs/video_presentation_storyboard.html",
        "conclusion": "可以写：语言/空间泛化评测已形成固定任务和视频证据，普通单任务 BC 与部分 learned proxy 在语言重述或空间目标变化下仍明显不足。",
        "redline": "不能把规则解析、局部语言 token 或 frozen CLIP 头部直接写成完整 VLM/VLA 语义泛化能力。",
        "next": "加入真实 VLM/VLA 表征缓存后，保留相同语言指令集做同表复测；可从 openvla_dataset_bridge_v1 的 image/instruction/state/action 字段开始接入。",
    },
    {
        "研究问题": "仿真适配后能否迁移到真实机械臂？",
        "status": "部分完成；MuJoCo domain randomization 代理评测、Isaac 运行交接门禁和真实 WidowX 运行交接门禁已完成，Isaac 实际运行和真实机械臂 trial 仍未完成。",
        "evidence": "docs/domain_randomization_summary.md；docs/domain_randomization_summary.csv；outputs/evaluations/domain_randomization_eval_v1.json；docs/isaac_domain_randomization_handoff.md；outputs/evaluations/isaac_domain_randomization_handoff_v1.json；docs/real_widowx_validation_handoff.md；outputs/evaluations/real_widowx_validation_handoff_v1.json；outputs/real_robot/real_widowx_validation_v1_trial_template.csv；docs/next_phase_implementation.md；docs/runtime_capability_report.md",
        "methods": "domain_randomization_eval_v1（MuJoCo 代理）；isaac_domain_randomization_handoff_v1（交接门禁，不是结果）；real_widowx_validation_handoff_v1（交接门禁，不是结果）；isaac_domain_randomization_v1（计划）；real_widowx_validation_v1（计划）",
        "summary_key": "sim_to_real",
        "display": "docs/domain_randomization_summary.md；docs/isaac_domain_randomization_handoff.md；docs/real_widowx_validation_handoff.md；outputs/real_robot/real_widowx_validation_v1_trial_template.csv；outputs/videos/domain_randomization_structured_low_friction_seed0.mp4；outputs/videos/domain_randomization_visual_act_cnn_cvae_low_friction_seed0.mp4",
        "conclusion": "可以写：当前已用 MuJoCo 摩擦、执行器增益、力限和夹爪力度扰动完成一个 sim-to-real 前置鲁棒性代理评测，并固定了 Isaac 复现实验与真实 WidowX 20-50 次 trial 的回填字段和论文红线。",
        "redline": "不能写高保真 Isaac domain randomization 已完成，也不能写真实机械臂迁移成功或失败；真实 WidowX handoff 不能写成真实 trial 结果。",
        "next": "若安装 Isaac/Isaac Sim，按 isaac_domain_randomization_handoff_v1 把同一任务、扰动参数和视频证据迁移到 isaac_domain_randomization_v1；真实机械臂按 real_widowx_validation_handoff_v1 执行 20-50 次 trial。",
    },
    {
        "研究问题": "不同阶段和方法能否被统一说明、评测比较和视频展示？",
        "status": "已完成（当前 MuJoCo 实验包）。",
        "evidence": "docs/stage_comparison_report.md；docs/method_comparison_dashboard.html；docs/thesis_visual_evidence_index.html；docs/defense_qa_playbook.html；docs/version_lineage_index.html；docs/stage_showcase_index.md；docs/video_evidence_index.md；docs/video_presentation_storyboard.md；docs/final_artifact_manifest.md；docs/defense_deck.html；docs/next_experiment_registry.md",
        "methods": "全部正式登记方法",
        "summary_key": "package",
        "display": "outputs/presentation_clips/00_defense_video_reel.mp4；outputs/showcase/all_registered_methods_grid.mp4；docs/experiment_dashboard.html；docs/method_comparison_dashboard.html；docs/thesis_visual_evidence_index.html；docs/defense_qa_playbook.html；docs/version_lineage_index.html；docs/stage_showcase_index.html；docs/video_presentation_storyboard.html",
        "conclusion": "可以写：当前每个正式方法都有版本名、结果表、资源表、固定视频或阶段视频证据；方法比较看板、论文图表与视频证据索引、答辩追问 Q&A、版本谱系索引、阶段展示总索引和视频展示讲稿已经把版本、指标、短片、图注/表注、讲稿提示、追问回答、谱系关系和论文红线联到同一套入口，并能通过总体验证脚本检查完整性。",
        "redline": "不能只展示成功视频而隐去失败 baseline；失败结果本身是普通 BC 不足和任务难度的证据。",
        "next": "后续新增任何方法时，必须同步登记版本、评测行、资源行、视频证据和论文红线；计划版本先进入 docs/next_experiment_registry.md，不能直接写成完成方法。",
    },
    {
        "研究问题": "trajectory-conditioned BC / ACT 是否已建立为可靠对照组？",
        "status": "已完成基础实现和闭环评测，但当前效果仍不稳定。",
        "evidence": "docs/evaluation_summary.csv；docs/model_resource_summary.csv；docs/stage_comparison_report.md；docs/video_evidence_index.md",
        "methods": "trajectory_conditioned_chunk_bc_v2；trajectory_knn_chunk_bc_v1；torch_act_state_chunk_v1；torch_act_state_chunk_cuda_v1；phase_conditioned_torch_act_v1；torch_act_cvae_state_chunk_v1；visual_feature_act_lite_v1",
        "summary_key": "trajectory_act",
        "display": "outputs/presentation_clips/03_trajectory_act_diffusion.mp4；outputs/videos/trajectory_conditioned_chunk_bc_v2_seed0.mp4；outputs/videos/torch_act_state_chunk_v1_seed0.mp4；outputs/videos/phase_conditioned_torch_act_v1_seed0.mp4；outputs/videos/visual_act_cnn_cvae_v1_seed0.mp4",
        "conclusion": "可以写：相比线性/MLP 单步 BC，轨迹条件和 ACT-style baseline 已覆盖历史观测、动作块输出、Transformer、CVAE、小型 CNN 视觉编码等变体，但闭环抓取仍没有稳定泛化。",
        "redline": "不能写成完整官方 ACT 或真实机器人视觉 ACT；当前是 state-only、pooled visual feature 或小型 CNN 的本地轻量 baseline。",
        "next": "下一步若继续改进，应优先提升示范多样性、夹爪接触建模、视觉编码和动作时序稳定性，而不是只增加网络规模。",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese map from research questions to current evidence.")
    parser.add_argument("--versions", type=Path, default=ROOT / "docs" / "experiment_versions.json")
    parser.add_argument("--summary", type=Path, default=ROOT / "docs" / "evaluation_summary.csv")
    parser.add_argument("--language-summary", type=Path, default=ROOT / "docs" / "language_generalization_summary.csv")
    parser.add_argument("--resource-summary", type=Path, default=ROOT / "docs" / "model_resource_summary.csv")
    parser.add_argument("--data-efficiency-summary", type=Path, default=ROOT / "docs" / "data_efficiency_summary.csv")
    parser.add_argument("--domain-randomization", type=Path, default=ROOT / "docs" / "domain_randomization_summary.csv")
    parser.add_argument("--stage-comparison", type=Path, default=ROOT / "docs" / "stage_comparison_report.csv")
    parser.add_argument("--video-evidence", type=Path, default=ROOT / "docs" / "video_evidence_index.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs" / "final_artifact_manifest.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "research_evidence_map.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "research_evidence_map.csv")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def success_percent(value: str) -> float:
    if not value or "/" not in value:
        return -1.0
    success, total = value.split("/", 1)
    total_int = int(total)
    if total_int == 0:
        return -1.0
    return 100.0 * int(success) / total_int


def success_text(value: str) -> str:
    pct = success_percent(value)
    return f"{value} ({pct:.0f}%)" if pct >= 0 else value


def by_version(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["version"]: row for row in rows}


def best_row(rows: list[dict[str, str]], key: str, *, percent: bool = False) -> dict[str, str]:
    if percent:
        return max(rows, key=lambda row: float(row.get(key, "0") or 0.0))
    return max(rows, key=lambda row: success_percent(row.get(key, "")))


def format_seconds(value: str) -> str:
    if not value:
        return "-"
    return f"{float(value):.2f}s"


def compact_resource_summary(resource_rows: list[dict[str, str]]) -> str:
    rows = by_version(resource_rows)
    object_head = rows.get("object_language_action_head_lite_v1", {})
    adapter = rows.get("adapter_action_head_lite_v1", {})
    lora = rows.get("lora_action_head_lite_v1", {})
    clip = rows.get("clip_action_head_lite_v1", {})
    return (
        f"资源表覆盖 {len(resource_rows)} 个方法；Object-language action head 为 "
        f"{int(object_head.get('trainable_params', 0)):,} 参数、训练 {format_seconds(object_head.get('train_time_seconds', ''))}；"
        f"Adapter 为 {int(adapter.get('trainable_params', 0)):,} 参数、训练 {format_seconds(adapter.get('train_time_seconds', ''))}；"
        f"LoRA-style 为 {int(lora.get('trainable_params', 0)):,} 参数、训练 {format_seconds(lora.get('train_time_seconds', ''))}；"
        f"CLIP action head 为 {int(clip.get('trainable_params', 0)):,} 参数。"
    )


def compact_data_efficiency_summary(rows: list[dict[str, str]]) -> str:
    heldout = [row for row in rows if row["split"] == "heldout"]
    budgets = sorted({int(row["demo_budget"]) for row in rows})
    best_by_budget = []
    for budget in budgets:
        subset = [row for row in heldout if int(row["demo_budget"]) == budget]
        best = best_row(subset, "success_rate", percent=True)
        best_by_budget.append(f"{budget} 条：{best['method_key']} {float(best['success_rate']) * 100:.0f}%")
    return f"数据效率表覆盖预算 {budgets}，held-out 最好结果为：" + "；".join(best_by_budget) + "。"


def compact_language_summary(rows: list[dict[str, str]]) -> str:
    best = best_row(rows, "success_rate", percent=True)
    selected_versions = {
        "object_language_action_head_lite_v1",
        "vision_language_action_head_lite_v1",
        "clip_action_head_lite_v1",
        "multi_task_object_action_head_lite_v1",
        "structured_waypoint_policy_v1",
    }
    selected = [
        f"{row['version']} {float(row['success_rate']) * 100:.0f}%"
        for row in rows
        if row["version"] in selected_versions
    ]
    return (
        f"语言/空间泛化表覆盖 {len(rows)} 行；最好行为 {best['version']} "
        f"{best['success']}（{float(best['success_rate']) * 100:.0f}%）；"
        f"关键 learned/proxy 对照：" + "；".join(selected) + "。"
    )


def compact_package_summary(methods_count: int, manifest: dict, stage_rows: list[dict[str, str]], video_rows: list[dict[str, str]]) -> str:
    counts = manifest["counts"]
    return (
        f"当前登记 {methods_count} 个正式方法、{len(stage_rows)} 个阶段分组、"
        f"{len(video_rows)} 条视频证据、{counts['presentation_pack_items']} 个答辩视频包项目；"
        f"总清单显示核心交付物和固定视频均存在。"
    )


def compact_trajectory_act_summary(summary_rows: list[dict[str, str]], resource_rows: list[dict[str, str]]) -> str:
    eval_by_version = by_version(summary_rows)
    resource_by_version = by_version(resource_rows)
    versions = [
        "trajectory_conditioned_chunk_bc_v2",
        "trajectory_knn_chunk_bc_v1",
        "torch_act_state_chunk_v1",
        "torch_act_state_chunk_cuda_v1",
        "phase_conditioned_torch_act_v1",
        "torch_act_cvae_state_chunk_v1",
        "visual_feature_act_lite_v1",
        "visual_act_cnn_cvae_v1",
    ]
    parts = []
    for version in versions:
        row = eval_by_version[version]
        params = int(resource_by_version[version].get("trainable_params") or 0)
        parts.append(
            f"{version}: train {success_text(row['train_range_success'])}, held-out {success_text(row['heldout_success'])}, params {params:,}"
        )
    return "；".join(parts) + "。"


def compact_domain_randomization_summary(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "MuJoCo domain randomization 代理评测行数为 0；真实 WidowX 评测次数为 0。"
    methods = sorted({row["method_key"] for row in rows})
    domains = sorted({row["domain"] for row in rows})
    parts = []
    for method in methods:
        for domain in domains:
            subset = [row for row in rows if row["method_key"] == method and row["domain"] == domain]
            if not subset:
                continue
            success = sum(1 for row in subset if row["success"] == "True")
            distance = sum(float(row["target_distance"]) for row in subset) / len(subset)
            parts.append(f"{method}/{domain}: {success}/{len(subset)}, mean_dist {distance:.3f}")
    return (
        f"MuJoCo domain_randomization_eval_v1 覆盖 {len(rows)} 条 episode、"
        f"{len(methods)} 个方法、{len(domains)} 个扰动域；"
        + "；".join(parts)
        + "。真实 WidowX 评测次数仍为 0。"
    )


def build_context(args: argparse.Namespace) -> dict[str, str]:
    versions = read_json(args.versions)
    summary_rows = read_csv(args.summary)
    language_rows = read_csv(args.language_summary)
    resource_rows = read_csv(args.resource_summary)
    data_efficiency_rows = read_csv(args.data_efficiency_summary)
    domain_randomization_rows = read_csv(args.domain_randomization)
    stage_rows = read_csv(args.stage_comparison)
    video_rows = read_csv(args.video_evidence)
    manifest = read_json(args.manifest)

    return {
        "methods_count": str(len(versions["methods"])),
        "resource": compact_resource_summary(resource_rows),
        "data_efficiency": compact_data_efficiency_summary(data_efficiency_rows),
        "language": compact_language_summary(language_rows),
        "sim_to_real": compact_domain_randomization_summary(domain_randomization_rows),
        "package": compact_package_summary(len(versions["methods"]), manifest, stage_rows, video_rows),
        "trajectory_act": compact_trajectory_act_summary(summary_rows, resource_rows),
    }


def materialize_rows(context: dict[str, str]) -> list[dict[str, str]]:
    rows = []
    for row in QUESTION_ROWS:
        rows.append(
            {
                "研究问题": row["研究问题"],
                "当前状态": row["status"],
                "证据文件": row["evidence"],
                "关键版本/方法": row["methods"],
                "量化摘要": context[row["summary_key"]],
                "视频/展示入口": row["display"],
                "可写结论": row["conclusion"],
                "论文红线": row["redline"],
                "下一步": row["next"],
            }
        )
    return rows


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(value.replace("|", "\\|").replace("\n", " ") for value in values) + " |"


def build_md(rows: list[dict[str, str]], context: dict[str, str]) -> str:
    lines = [
        "# 研究问题证据映射",
        "",
        "版本：`research_evidence_map_v1`",
        "",
        "用途：把毕业设计最初提出的研究问题，逐项映射到当前已经生成的评测表、视频证据、展示入口、可写结论和不能夸大的边界。这个文件用于论文写作、答辩讲解和后续补实验排期。",
        "",
        "## 1. 当前总判断",
        "",
        f"- 当前正式登记方法数：{context['methods_count']}。",
        "- MuJoCo 桌面任务、示范数据、普通 BC、trajectory/ACT/Diffusion、action-head/PEFT proxy、语言/空间泛化、domain randomization 代理评测、视频证据和答辩展示链路已经形成闭环。",
        "- 真实 OpenVLA/大 VLA 后训练、高保真 Isaac domain randomization、真实 WidowX 迁移验证还没有完成，论文中必须作为后续工作或实验边界。",
        "",
        "## 2. 证据总表",
        "",
        md_row(["研究问题", "当前状态", "关键证据", "量化摘要", "可写结论", "论文红线"]),
        md_row(["---", "---", "---", "---", "---", "---"]),
    ]
    for row in rows:
        lines.append(
            md_row(
                [
                    row["研究问题"],
                    row["当前状态"],
                    row["证据文件"],
                    row["量化摘要"],
                    row["可写结论"],
                    row["论文红线"],
                ]
            )
        )

    lines.extend(["", "## 3. 分问题说明", ""])
    for index, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"### {index}. {row['研究问题']}",
                "",
                f"当前状态：{row['当前状态']}",
                "",
                f"关键版本/方法：{row['关键版本/方法']}",
                "",
                f"证据文件：{row['证据文件']}",
                "",
                f"量化摘要：{row['量化摘要']}",
                "",
                f"视频/展示入口：{row['视频/展示入口']}",
                "",
                f"可写结论：{row['可写结论']}",
                "",
                f"论文红线：{row['论文红线']}",
                "",
                f"下一步：{row['下一步']}",
                "",
            ]
        )

    lines.extend(
        [
            "## 4. 重建命令",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_research_evidence_map.py"}"',
            "```",
            "",
            "## 5. 验证入口",
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


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["研究问题", "当前状态", "证据文件", "关键版本/方法", "量化摘要", "视频/展示入口", "可写结论", "论文红线", "下一步"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    context = build_context(args)
    rows = materialize_rows(context)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_md(rows, context), encoding="utf-8")
    write_csv(args.output_csv, rows)
    print(f"research_evidence_map_md: {args.output_md}", flush=True)
    print(f"research_evidence_map_csv: {args.output_csv}", flush=True)
    print(f"research_questions: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
