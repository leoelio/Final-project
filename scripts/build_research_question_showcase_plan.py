from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "research_question_showcase_plan_v1"


SHOWCASE_HINTS = {
    "轻量化后训练是否省算力/参数？": {
        "展示顺序": "先展示资源-成功率图，再打开资源表和 robot VLA handoff 门禁，最后播放 action-head/PEFT proxy 阶段短片。",
        "核心图表": "outputs/figures/resource_vs_success.svg；docs/model_resource_summary.csv；docs/method_evidence_gate.md",
        "主视频": "outputs/presentation_clips/04_action_head_peft_proxy.mp4",
        "辅助入口": "docs/action_head_stage_report.md；docs/robot_vla_action_head_handoff.md；docs/stage_showcase_index.html",
        "建议讲稿": "本地 proxy 证明小头部、Adapter/LoRA-style 增量和 CLIP action head 的资源字段已经可比较；robot_vla_action_head_handoff_v1 说明真实 VLA action-head 需迁移到 48GB+ GPU 或云端，真实 OpenVLA LoRA 尚未完成。",
    },
    "轻量化后训练是否省数据？": {
        "展示顺序": "先展示数据效率曲线，再说明 10/25/50/92 条 demonstration 的预算和 held-out split，必要时播放普通 BC 与 action-head 阶段短片。",
        "核心图表": "outputs/figures/data_efficiency.svg；docs/data_efficiency_summary.csv；outputs/evaluations/data_efficiency_v2.json",
        "主视频": "outputs/presentation_clips/02_basic_bc_baselines.mp4；outputs/presentation_clips/04_action_head_peft_proxy.mp4",
        "辅助入口": "docs/data_efficiency_summary.md；docs/experiment_dashboard.html",
        "建议讲稿": "数据效率目前是 MuJoCo scripted demonstration 的快速扫表，可用于比较本地 baseline 行为，不能外推成真实机器人小数据优势。",
    },
    "语言/空间泛化是否优于普通 BC？": {
        "展示顺序": "直接播放语言/空间泛化短片，再打开 language grid，对照 expert/structured 和 learned proxy 的 0/5 结果。",
        "核心图表": "docs/language_generalization_summary.csv；docs/video_evidence_index.md",
        "主视频": "outputs/presentation_clips/05_language_generalization.mp4；outputs/showcase/language_generalization_grid.mp4",
        "辅助入口": "docs/video_presentation_storyboard.html；docs/video_evidence_gallery.html",
        "建议讲稿": "该阶段说明普通单任务 BC 和本地 proxy 尚未形成语言/空间泛化，不能把 CLIP 或语言 token 代理写成真实 VLA 理解。",
    },
    "仿真适配后能否迁移到真实机械臂？": {
        "展示顺序": "播放 domain randomization 代理短片，再展示 3 个扰动域、Isaac handoff 和真实 WidowX trial 模板，最后明确 Isaac/真实 trial 仍未完成。",
        "核心图表": "docs/domain_randomization_summary.csv；outputs/evaluations/domain_randomization_eval_v1.json；docs/isaac_domain_randomization_handoff.md；docs/real_widowx_validation_handoff.md；outputs/real_robot/real_widowx_validation_v1_trial_template.csv；docs/next_experiment_registry.md",
        "主视频": "outputs/presentation_clips/06_domain_randomization_proxy.mp4；outputs/videos/domain_randomization_trajectory_knn_low_friction_seed0.mp4",
        "辅助入口": "docs/domain_randomization_summary.md；docs/isaac_domain_randomization_handoff.md；docs/real_widowx_validation_handoff.md；docs/runtime_capability_report.md",
        "建议讲稿": "当前只完成 MuJoCo 摩擦、力限和夹爪力度扰动代理，以及 Isaac/真实 WidowX 的运行交接门禁；不能写成高保真 Isaac 或真实机械臂迁移。",
    },
    "不同阶段和方法能否被统一说明、评测比较和视频展示？": {
        "展示顺序": "先播放 60 秒总览 reel，再打开阶段展示总索引和方法证据门禁，说明每个正式方法都有版本、结果、资源、视频和红线。",
        "核心图表": "docs/stage_comparison_report.md；docs/method_evidence_gate.md；docs/final_artifact_manifest.md",
        "主视频": "outputs/presentation_clips/00_defense_video_reel.mp4；outputs/showcase/all_registered_methods_grid.mp4",
        "辅助入口": "docs/stage_showcase_index.html；docs/stage_reproduction_runbook.md",
        "建议讲稿": "当前 MuJoCo 实验包已经具备按方法和阶段统一说明、比较和展示的证据链，但未来真实 VLA/Isaac/真实机器人要按同一门禁补齐。",
    },
    "trajectory-conditioned BC / ACT 是否已建立为可靠对照组？": {
        "展示顺序": "播放 Trajectory/ACT/Diffusion 短片，再打开阶段报告和方法门禁，强调 trajectory-kNN 训练范围成功但泛化失败。",
        "核心图表": "docs/trajectory_act_stage_report.md；docs/trajectory_act_stage_report.csv；docs/model_resource_summary.csv",
        "主视频": "outputs/presentation_clips/03_trajectory_act_diffusion.mp4；outputs/videos/trajectory_conditioned_chunk_bc_v2_seed0.mp4；outputs/videos/torch_act_state_chunk_v1_seed0.mp4",
        "辅助入口": "docs/stage_reproduction_runbook.md；docs/failure_mode_taxonomy.md",
        "建议讲稿": "本阶段可写成可靠高级 baseline：覆盖历史观测、动作块、Transformer、CVAE、小型 CNN 和 diffusion，但不是完整官方视觉 ACT。",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a research-question-oriented showcase plan.")
    parser.add_argument("--research-evidence", type=Path, default=ROOT / "docs" / "research_evidence_map.csv")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "research_question_showcase_plan.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "research_question_showcase_plan.md")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", "<br>") for value in values) + " |"


def split_paths(value: str) -> list[str]:
    paths: list[str] = []
    for part in value.replace("；", "\n").splitlines():
        item = part.strip().strip("`")
        if item.startswith(("docs/", "outputs/")):
            paths.append(item)
    return paths


def build_rows(source_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for index, row in enumerate(source_rows, start=1):
        question = row["研究问题"]
        hint = SHOWCASE_HINTS.get(question, {})
        main_video = hint.get("主视频", row["视频/展示入口"])
        evidence_paths = split_paths(hint.get("核心图表", "")) + split_paths(main_video) + split_paths(hint.get("辅助入口", ""))
        missing = [path for path in evidence_paths if not (ROOT / path).exists()]
        rows.append(
            {
                "展示编号": str(index),
                "研究问题": question,
                "当前状态": row["当前状态"],
                "推荐展示顺序": hint.get("展示顺序", "先展示量化表，再展示对应视频证据。"),
                "核心图表": hint.get("核心图表", row["证据文件"]),
                "主视频": main_video,
                "辅助入口": hint.get("辅助入口", row["视频/展示入口"]),
                "建议讲稿": hint.get("建议讲稿", row["可写结论"]),
                "可写结论": row["可写结论"],
                "论文红线": row["论文红线"],
                "缺失证据": "无" if not missing else "；".join(missing),
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
    lines = [
        "# 研究问题展示选择表",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：按最初研究问题整理答辩和论文展示时应优先使用的图、表、视频片段、辅助入口、建议讲稿和论文红线。它不新增实验结果，只把已有证据重新组织成展示路线。",
        "",
        "## 1. 总览",
        "",
        md_row(["编号", "研究问题", "核心图表", "主视频", "缺失证据"]),
        md_row(["---:", "---", "---", "---", "---"]),
    ]
    for row in rows:
        lines.append(md_row([row["展示编号"], row["研究问题"], row["核心图表"], row["主视频"], row["缺失证据"]]))

    lines.extend(["", "## 2. 分问题展示脚本", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row['展示编号']}. {row['研究问题']}",
                "",
                f"- 当前状态：{row['当前状态']}",
                f"- 推荐展示顺序：{row['推荐展示顺序']}",
                f"- 核心图表：{row['核心图表']}",
                f"- 主视频：{row['主视频']}",
                f"- 辅助入口：{row['辅助入口']}",
                f"- 建议讲稿：{row['建议讲稿']}",
                f"- 可写结论：{row['可写结论']}",
                f"- 论文红线：{row['论文红线']}",
                "",
            ]
        )

    lines.extend(
        [
            "## 3. 使用边界",
            "",
            "1. 该表是展示选择表，不替代 `docs/research_evidence_map.md` 和各 CSV 评测表。",
            "2. 视频片段用于辅助说明现象，成功率、资源、语言泛化和 domain randomization 结论仍以量化表为准。",
            "3. OpenVLA、Isaac、真实 WidowX 未完成的部分必须作为后续工作，不得用当前 MuJoCo proxy 视频替代。",
            "",
            "## 4. 重建命令",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_research_question_showcase_plan.py"}"',
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = build_rows(read_csv(args.research_evidence))
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows)
    print(f"research_question_showcase_plan_md: {args.output_md}", flush=True)
    print(f"research_question_showcase_plan_csv: {args.output_csv}", flush=True)
    print(f"research_question_showcase_rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
