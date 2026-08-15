from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "final_showcase_handoff_v1"


FIELDNAMES = ["目标需求", "首选入口", "辅助入口", "打开命令", "用途", "论文边界"]


def ps_command(script: str, args: list[str]) -> str:
    return f'& "{PYTHON}" "{ROOT / script}" ' + " ".join(args)


def start_process(path: str, *, notepad: bool = False) -> str:
    target = ROOT / path
    if notepad:
        return f'Start-Process notepad.exe "{target}"'
    return f'Start-Process "{target}"'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a one-page Chinese handoff index for final showcase, comparison, and video evidence.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs" / "final_artifact_manifest.json")
    parser.add_argument("--goal-audit", type=Path, default=ROOT / "docs" / "goal_completion_audit.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "final_showcase_handoff.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "final_showcase_handoff.csv")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def build_rows(counts: dict) -> list[dict[str, str]]:
    method_count = int(counts.get("methods", 0))
    stage_count = int(counts.get("stage_evidence_rows", 0))
    video_count = int(counts.get("video_evidence_rows", 0))
    return [
        {
            "目标需求": "总入口和当前边界",
            "首选入口": "docs/final_experiment_package.md",
            "辅助入口": "docs/goal_completion_audit.md；docs/final_artifact_manifest.md；docs/defense_evidence_pack.md",
            "打开命令": ps_command("scripts/showcase_launcher.py", ["--target", "package"]),
            "用途": "先确认当前 MuJoCo 实验包有哪些完成项、哪些还只是 planned/readiness。",
            "论文边界": "当前完成的是 MuJoCo 实验包；真实 OpenVLA、Isaac 和真实 WidowX 不能写成已完成。",
        },
        {
            "目标需求": "保留各部分版本名称",
            "首选入口": "docs/version_naming_and_gate_spec.md",
            "辅助入口": "docs/version_lineage_index.html；docs/final_method_version_index.md；docs/experiment_versions.json",
            "打开命令": ps_command("scripts/showcase_launcher.py", ["--target", "lineage"]),
            "用途": "查正式方法、候选诊断、前置门禁、planned 外部版本和 planned→formal 升级规则。",
            "论文边界": "candidate、handoff、readiness、planned 不能混写成正式方法成功率结果。",
        },
        {
            "目标需求": "逐方法说明",
            "首选入口": "docs/final_method_version_index.md",
            "辅助入口": "docs/method_cards.md；docs/method_evidence_gate.md；docs/method_stage_audit.md",
            "打开命令": start_process("docs/final_method_version_index.md", notepad=True),
            "用途": f"逐个查看 {method_count} 个正式方法的阶段、artifact、成功率、固定视频、简介和慢速 viewer 命令。",
            "论文边界": "失败方法也要保留；不能只挑成功样例。",
        },
        {
            "目标需求": "按阶段说明",
            "首选入口": "docs/stage_showcase_index.html",
            "辅助入口": "docs/stage_reproduction_runbook.md；docs/stage_comparison_report.md；docs/stage_evidence_index.md",
            "打开命令": ps_command("scripts/showcase_launcher.py", ["--target", "stage:2", "--action", "print"]),
            "用途": f"按 {stage_count} 个阶段组织任务/数据、普通 BC、Trajectory/ACT、Action-head/PEFT、语言泛化、数据效率、domain randomization 和外部依赖 readiness。",
            "论文边界": "第 8 阶段是 readiness 门禁，不是策略成功率结果。",
        },
        {
            "目标需求": "横向评测比较",
            "首选入口": "docs/core_v2_holdout_comparison_matrix.md；docs/core_v2_pretrained_vlm_action_head_report.md；docs/core_v2_clip_semantic_waypoint_report.md；docs/core_v2_clip_semantic_data_efficiency.md；docs/core_v2_clip_semantic_ood_generalization.md；docs/core_v2_video_evidence.md；docs/method_comparison_dashboard.html",
            "辅助入口": "docs/result_matrix.md；docs/evaluation_summary.csv；docs/language_generalization_summary.csv；docs/model_resource_summary.csv；docs/data_efficiency_summary.md",
            "打开命令": ps_command("scripts/showcase_launcher.py", ["--target", "comparison"]),
            "用途": "先比较主任务、held-out、语言/空间泛化、参数规模和训练资源；随后对照冻结 CLIP 连续动作头的 0/20 与 CLIP 语义-结构化执行的 20/20，并用固定视频解释控制接口差异。",
            "论文边界": "MuJoCo proxy 结果不能直接推广为真实机器人成功率。",
        },
        {
            "目标需求": "仿真视频片段展示",
            "首选入口": "docs/defense_video_playlist.html",
            "辅助入口": "docs/defense_video_cue_sheet.md；docs/video_evidence_gallery.html；docs/video_presentation_storyboard.html；docs/thesis_visual_evidence_index.html",
            "打开命令": ps_command("scripts/showcase_launcher.py", ["--target", "playlist"]),
            "用途": f"按 claim 或答辩顺序播放 {video_count} 条视频证据、阶段短片和候选诊断负例。",
            "论文边界": "视频是定性证据，不能替代成功率、目标距离、资源表和严格抓取审计。",
        },
        {
            "目标需求": "每次运行可视化",
            "首选入口": "docs/showcase_launcher_guide.md",
            "辅助入口": "docs/reproducible_command_index.md；docs/final_method_version_index.md；docs/candidate_diagnostic_video_index.md",
            "打开命令": ps_command("scripts/showcase_launcher.py", ["--target", "method:trajectory_knn_chunk_bc_v1", "--action", "viewer", "--dry-run"]),
            "用途": "查看或启动慢速 MuJoCo viewer；正式方法和候选诊断都保留完整启动命令。",
            "论文边界": "viewer 演示只说明单次过程；正式结论仍以批量评测和审计表为准。",
        },
        {
            "目标需求": "论文写作和答辩讲稿",
            "首选入口": "docs/final_defense_narrative_script.md",
            "辅助入口": "docs/thesis_results_chapter_draft.md；docs/trajectory_act_conclusion_brief.md；docs/defense_live_runbook.md；docs/defense_qa_playbook.html；docs/defense_slide_outline.md；docs/defense_storyboard.md",
            "打开命令": ps_command("scripts/showcase_launcher.py", ["--target", "narrative-script"]),
            "用途": "把研究问题、阶段方法、结果章节、答辩现场顺序、视频播放和论文红线统一成一条中文讲解顺序。",
            "论文边界": "答辩材料不能把 proxy、state-only、lite 或 candidate 改写成完整官方方法。",
        },
        {
            "目标需求": "后续 OpenVLA / Isaac / 真实 WidowX",
            "首选入口": "docs/remaining_experiment_execution_board.md",
            "辅助入口": "docs/external_dependency_readiness_audit.md；docs/next_experiment_registry.md；docs/robot_vla_remote_run_pack.md；docs/isaac_domain_randomization_handoff.md；docs/real_widowx_validation_handoff.md",
            "打开命令": ps_command("scripts/showcase_launcher.py", ["--target", "remaining-board"]),
            "用途": "按优先级查看 planned 外部版本的执行环境、阻塞条件、回填文件、升级门槛和论文红线。",
            "论文边界": "readiness/handoff 只能写成前置门禁，不能写成真实实验结果。",
        },
        {
            "目标需求": "交付归档和复验",
            "首选入口": "docs/defense_evidence_pack.md",
            "辅助入口": "outputs/defense_evidence_pack/defense_evidence_pack_v1.zip；docs/final_artifact_manifest.json；scripts/verify_experiment_artifacts.py",
            "打开命令": ps_command("scripts/showcase_launcher.py", ["--target", "evidence-pack"]),
            "用途": "复制或复验当前 MuJoCo 实验包的文档、CSV/JSON、图表、视频和展示入口。",
            "论文边界": "证据包是归档，不代表外部依赖阶段完成。",
        },
    ]


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(value.replace("|", "\\|").replace("\n", "<br>") for value in values) + " |"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, str]], manifest: dict, goal_rows: list[dict[str, str]]) -> None:
    counts = manifest["counts"]
    complete_rows = sum(1 for row in goal_rows if row["当前状态"].startswith("当前 MuJoCo 阶段已完成"))
    lines = [
        "# 最终展示与交付 Handoff 索引",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：把“版本名称、方法/阶段说明、评测比较、仿真视频片段展示、可视化运行和后续外部依赖”集中到一页，作为写论文、做答辩和继续实验的最短路径索引。它不新增实验结论，只连接已有证据。",
        "",
        "打开本页命令：",
        "",
        "```powershell",
        ps_command("scripts/showcase_launcher.py", ["--target", "handoff"]),
        "```",
        "",
        "## 1. 当前证据计数",
        "",
        md_row(["项目", "数量"]),
        md_row(["---", "---:"]),
        md_row(["正式方法版本", str(counts.get("methods", 0))]),
        md_row(["阶段展示行", str(counts.get("stage_evidence_rows", 0))]),
        md_row(["视频证据", str(counts.get("video_evidence_rows", 0))]),
        md_row(["答辩视频包项目", str(counts.get("presentation_pack_items", 0))]),
        md_row(["版本命名规则", "8"]),
        md_row(["总目标审计行", str(counts.get("goal_completion_rows", len(goal_rows)))]),
        md_row(["MuJoCo 阶段已完成项", str(complete_rows)]),
        md_row(["证据包文件", str(counts.get("defense_evidence_pack_files", 0))]),
        "",
        "## 2. 最短路径表",
        "",
        md_row(FIELDNAMES),
        md_row(["---", "---", "---", "---", "---", "---"]),
    ]
    for row in rows:
        lines.append(md_row([row[field] for field in FIELDNAMES]))

    lines.extend(
        [
            "",
            "## 3. 建议使用顺序",
            "",
            "1. 先打开 `docs/final_experiment_package.md` 明确当前完成范围和论文边界。",
            "2. 写方法部分时打开 `docs/final_method_version_index.md` 和 `docs/version_naming_and_gate_spec.md`。",
            "3. 写结果分析时先打开 `docs/core_v2_holdout_comparison_matrix.md`，再打开 `docs/core_v2_pretrained_vlm_action_head_report.md`、`docs/core_v2_clip_semantic_waypoint_report.md`、`docs/core_v2_clip_semantic_data_efficiency.md`、`docs/core_v2_clip_semantic_ood_generalization.md` 和 `docs/core_v2_video_evidence.md`；历史诊断使用 `docs/core_task_comparison_matrix.md`，补充材料使用 `docs/method_comparison_dashboard.html`、`docs/stage_showcase_index.html` 和 `docs/thesis_results_chapter_draft.md`。",
            "4. 做视频展示时打开 `docs/defense_video_playlist.html`、`docs/defense_video_cue_sheet.md` 或 `docs/video_evidence_gallery.html`。",
            "5. 继续 OpenVLA、Isaac 或真实 WidowX 时先打开 `docs/external_dependency_readiness_audit.md`。",
            "",
            "## 4. 重建命令",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_final_showcase_handoff.py"}"',
            "```",
            "",
            "## 5. 总体验证命令",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "verify_experiment_artifacts.py"}"',
            "```",
            "",
            "论文边界：当前 handoff 证明的是 MuJoCo 实验包的证据入口已经可追溯；真实 OpenVLA、Isaac 和真实 WidowX 仍需要单独运行、回填、登记和保存视频。",
            "",
            f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)
    goal_rows = read_csv(args.goal_audit)
    rows = build_rows(manifest["counts"])
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows, manifest, goal_rows)
    print(f"final_showcase_handoff_md: {args.output_md}", flush=True)
    print(f"final_showcase_handoff_csv: {args.output_csv}", flush=True)
    print(f"final_showcase_handoff_rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
