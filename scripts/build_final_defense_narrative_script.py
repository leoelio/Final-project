from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "final_defense_narrative_script_v1"

FIELDNAMES = [
    "顺序",
    "讲解段落",
    "对应研究问题/阶段",
    "建议时长",
    "推荐打开命令",
    "推荐证据",
    "可说结论",
    "讲解稿",
    "论文红线",
    "承接下一步",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese final defense narrative script from registered experiment evidence.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs" / "final_artifact_manifest.json")
    parser.add_argument("--cue-sheet", type=Path, default=ROOT / "docs" / "defense_video_cue_sheet.csv")
    parser.add_argument("--research-plan", type=Path, default=ROOT / "docs" / "research_question_showcase_plan.csv")
    parser.add_argument("--stage-index", type=Path, default=ROOT / "docs" / "stage_evidence_index.csv")
    parser.add_argument("--handoff", type=Path, default=ROOT / "docs" / "final_showcase_handoff.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "final_defense_narrative_script.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "final_defense_narrative_script.csv")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def ps_command(script: str, args: list[str]) -> str:
    return f'& "{PYTHON}" "{ROOT / script}" ' + " ".join(args)


def cue_by_id(rows: list[dict[str, str]], cue_id: str) -> dict[str, str]:
    for row in rows:
        if row.get("cue_id") == cue_id:
            return row
    return {}


def cue_evidence(cue: dict[str, str]) -> str:
    parts = []
    if cue.get("媒体文件"):
        parts.append(cue["媒体文件"])
    if cue.get("证据引用"):
        parts.append(cue["证据引用"])
    return "；".join(parts)


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(value.replace("|", "\\|").replace("\n", "<br>") for value in values) + " |"


def build_rows(manifest: dict[str, object], cue_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    counts = manifest.get("counts", {})
    if not isinstance(counts, dict):
        counts = {}

    methods = counts.get("methods", 0)
    videos = counts.get("video_evidence_rows", 0)
    pack_files = counts.get("defense_evidence_pack_files", 0)
    trajectory_rows = counts.get("trajectory_act_conclusion_rows", 0)
    cue_count = counts.get("defense_video_cue_sheet_rows", 0)

    c01 = cue_by_id(cue_rows, "C01")
    c02 = cue_by_id(cue_rows, "C02")
    c03 = cue_by_id(cue_rows, "C03")
    c04 = cue_by_id(cue_rows, "C04")
    c05 = cue_by_id(cue_rows, "C05")
    c06 = cue_by_id(cue_rows, "C06")
    c07 = cue_by_id(cue_rows, "C07")
    c08 = cue_by_id(cue_rows, "C08")
    c09 = cue_by_id(cue_rows, "C09")
    c10 = cue_by_id(cue_rows, "C10")

    return [
        {
            "顺序": "1",
            "讲解段落": "总目标、证据范围和论文边界",
            "对应研究问题/阶段": "总目标；所有阶段的入口",
            "建议时长": "45-60 秒",
            "推荐打开命令": ps_command("scripts/showcase_launcher.py", ["--target", "package"]),
            "推荐证据": "docs/final_experiment_package.md；docs/goal_completion_audit.md；docs/final_showcase_handoff.md",
            "可说结论": f"当前归档的是 MuJoCo 阶段实验包：已登记 {methods} 个正式方法、{videos} 条视频证据、{pack_files} 个证据包文件。",
            "讲解稿": "本课题围绕有限算力、小规模示范数据和有限机械臂时间下的轻量化 VLA 后训练展开。当前可完整展示的是 MuJoCo 桌面抓取实验包，包括任务环境、示范采集、普通 BC、trajectory-conditioned BC / ACT-style、Diffusion、action-head/PEFT proxy、语言泛化和 domain randomization 代理评测。",
            "论文红线": "不能把当前材料写成真实 OpenVLA、Isaac 或真实 WidowX 已完成；这些仍是 readiness、handoff 或下一阶段实验。",
            "承接下一步": "转入任务环境和数据链路，先说明比较对象在哪里运行、如何复现。",
        },
        {
            "顺序": "2",
            "讲解段落": "任务、数据和可复现实验链路",
            "对应研究问题/阶段": "阶段 1：任务/数据/普通 BC 前置",
            "建议时长": "60 秒",
            "推荐打开命令": ps_command("scripts/showcase_launcher.py", ["--target", "claim:C01"]),
            "推荐证据": cue_evidence(c01) or "docs/evaluation_summary.csv；docs/task_bc_stage_report.md",
            "可说结论": "桌面任务、物体颜色、目标盘和 scripted/waypoint/replay 链路已经固定，后续方法共享同一类评测指标。",
            "讲解稿": "先展示蓝色方块放到蓝色盘、颜色/位置变化和 replay 复现。这里的重点不是学习算法本身，而是把环境、任务、seed、轨迹、成功失败和 viewer 回放链路固定下来，为后面所有方法提供统一对照。",
            "论文红线": c01.get("论文红线", "expert、structured waypoint 和 replay 不能写成 learned VLA 或真实机器人验证。"),
            "承接下一步": "有了可复现任务后，再看普通模仿学习的下限表现。",
        },
        {
            "顺序": "3",
            "讲解段落": "普通 BC baseline 和失败模式",
            "对应研究问题/阶段": "阶段 1-2：Linear BC、MLP BC、kNN BC",
            "建议时长": "60-75 秒",
            "推荐打开命令": ps_command("scripts/showcase_launcher.py", ["--target", "claim:C02"]),
            "推荐证据": cue_evidence(c02) or "docs/task_bc_stage_report.md；docs/failure_mode_taxonomy.md",
            "可说结论": "普通简单 BC 在接触、夹紧、抬升和放置上不稳定，能作为必要的低阶对照。",
            "讲解稿": "这里重点展示线性 BC、MLP BC 和 kNN/轨迹记忆的差别。简单 BC 的失败不是运行速度问题单独导致的，而是策略缺少稳定的接触阶段表达；因此后面需要历史轨迹、动作块、阶段条件或轻量 action head。",
            "论文红线": c02.get("论文红线", "普通 BC 不能写成语言理解、任务泛化或 VLA 后训练结果。"),
            "承接下一步": "引出 trajectory-conditioned BC / ACT：把历史观测和动作块作为更合理的 baseline。",
        },
        {
            "顺序": "4",
            "讲解段落": "trajectory-conditioned BC / ACT / Diffusion 对照",
            "对应研究问题/阶段": "阶段 3：Trajectory-conditioned BC / ACT-style / Diffusion",
            "建议时长": "90 秒",
            "推荐打开命令": ps_command("scripts/showcase_launcher.py", ["--target", "trajectory-act-brief"]),
            "推荐证据": f"docs/trajectory_act_conclusion_brief.md；docs/trajectory_act_conclusion_brief.csv；{cue_evidence(c03)}",
            "可说结论": f"trajectory-conditioned BC / ACT-style 对照链路已完成并登记 {trajectory_rows} 行结论摘要，但当前结果仍显示抓取接触和泛化不稳定。",
            "讲解稿": "这一段说明已经实现 trajectory-conditioned chunk BC、trajectory-kNN、Torch ACT state chunk、ACT-CVAE、visual ACT-lite 和 Diffusion 相邻对照。核心结论是接口和评测链路已经建立，但 state-only 或轻量视觉特征版本不能自动解决夹爪接触、抬升和稳定抓取问题。",
            "论文红线": c03.get("论文红线", "当前是 state-only、pooled visual feature 或小型 CNN 的本地轻量 baseline，不能写成完整官方视觉 ACT 或完整视觉 Diffusion Policy。"),
            "承接下一步": "在此基础上转向更贴近轻量 VLA 后训练路线的 action-head/PEFT proxy。",
        },
        {
            "顺序": "5",
            "讲解段落": "action head / Adapter / LoRA / VLM proxy",
            "对应研究问题/阶段": "阶段 4：轻量化 VLA 后训练 proxy",
            "建议时长": "75 秒",
            "推荐打开命令": ps_command("scripts/showcase_launcher.py", ["--target", "claim:C04"]),
            "推荐证据": cue_evidence(c04) or "docs/action_head_stage_report.md；docs/model_resource_summary.csv",
            "可说结论": "action-head/Adapter/LoRA/VLM proxy 可以用于比较可训练参数、资源和语言条件接口，但当前仍是本地 proxy，不是真实 OpenVLA 后训练。",
            "讲解稿": "这一段把研究主线拉回轻量化后训练：不是从零训练大模型，而是在现有表征或代理表征上训练 action head、Adapter 或 LoRA 风格模块。展示时强调资源表和成功率表并列，说明为什么它适合有限算力条件下的毕业设计。",
            "论文红线": c04.get("论文红线", "不能把本地 proxy 写成真实 OpenVLA、RT-2 或大规模机器人 VLA 后训练。"),
            "承接下一步": "随后说明语言和空间泛化，回答 VLA 路线为什么值得做。",
        },
        {
            "顺序": "6",
            "讲解段落": "语言/空间泛化能力",
            "对应研究问题/阶段": "研究问题 3：VLA 是否更会理解语言",
            "建议时长": "60 秒",
            "推荐打开命令": ps_command("scripts/showcase_launcher.py", ["--target", "claim:C05"]),
            "推荐证据": cue_evidence(c05) or "docs/language_generalization_summary.csv；docs/research_question_showcase_plan.md",
            "可说结论": "语言/空间泛化测试已经纳入统一评测，但当前可写的是本地语言条件 proxy 的差异，不能扩大到真实 VLA 语义能力。",
            "讲解稿": "这里展示训练指令和测试指令不完全相同的情况，例如颜色同义词、leftmost 目标等。讲解时把成功率表和视频片段分开：视频用于定性说明，真正结论以 CSV 的批量评测为准。",
            "论文红线": c05.get("论文红线", "不能只凭单个视频片段证明语言泛化；必须引用批量评测表。"),
            "承接下一步": "接着回答有限数据条件下，不同方法的数据效率差异。",
        },
        {
            "顺序": "7",
            "讲解段落": "数据效率、资源消耗和横向比较",
            "对应研究问题/阶段": "研究问题 1-2：省算力、省数据",
            "建议时长": "75 秒",
            "推荐打开命令": ps_command("scripts/showcase_launcher.py", ["--target", "comparison"]),
            "推荐证据": f"docs/method_comparison_dashboard.html；docs/data_efficiency_summary.md；docs/model_resource_summary.csv；{cue_evidence(c06)}",
            "可说结论": "当前比较框架能同时看成功率、留出任务、语言/空间泛化、参数量、训练时间和固定视频。",
            "讲解稿": "这一段不播放太多视频，而是打开方法比较 dashboard，说明每种方法不是单独看一次演示，而是按相同任务、seed、指标和资源记录做横向比较。数据效率部分用于支持有限示范数据条件下的研究问题。",
            "论文红线": c06.get("论文红线", "快速数据效率扫表不能写成大规模统计显著性结论。"),
            "承接下一步": "再说明仿真扰动和 sim-to-real 计划，避免把 MuJoCo 结果过度外推。",
        },
        {
            "顺序": "8",
            "讲解段落": "MuJoCo domain randomization 与外部依赖 readiness",
            "对应研究问题/阶段": "研究问题 4：仿真迁移和真实机械臂验证",
            "建议时长": "60-75 秒",
            "推荐打开命令": ps_command("scripts/showcase_launcher.py", ["--target", "claim:C07"]),
            "推荐证据": f"{cue_evidence(c07)}；docs/external_dependency_readiness_audit.md；docs/isaac_domain_randomization_handoff.md；docs/real_widowx_validation_handoff.md",
            "可说结论": "MuJoCo 扰动代理评测已经完成；Isaac、真实 WidowX 和真实 OpenVLA 仍处在 handoff/readiness 阶段。",
            "讲解稿": "这一段要主动划清边界：domain randomization 已经在 MuJoCo 中做了代理测试，但高保真 Isaac 和真实 WidowX 还没有产生真实成功率。可以展示 handoff 文档，说明后续会沿用同一套字段回填真实实验。",
            "论文红线": c07.get("论文红线", "MuJoCo 扰动评测不能写成高保真 Isaac sim-to-real 或真实机械臂迁移成功。"),
            "承接下一步": "最后切到视频证据和答辩展示方式，说明如何复现和抽查。",
        },
        {
            "顺序": "9",
            "讲解段落": "视频证据、cue sheet 和可视化演示",
            "对应研究问题/阶段": "答辩展示层",
            "建议时长": "45-60 秒",
            "推荐打开命令": ps_command("scripts/showcase_launcher.py", ["--target", "playlist"]),
            "推荐证据": f"docs/defense_video_playlist.html；docs/defense_video_cue_sheet.md；docs/video_evidence_index.md；{cue_evidence(c08)}",
            "可说结论": f"答辩视频播放单和 cue sheet 已登记 {cue_count} 个展示条目，覆盖成功样例、失败负例、阶段短片和候选诊断。",
            "讲解稿": "这一段说明所有视频都有对应文件、打开命令、讲解提示和论文红线。答辩时可以先播放总览短片，再按老师追问打开具体方法、claim 或 candidate 的片段。",
            "论文红线": c08.get("论文红线", "视频是定性证据，不能替代成功率、目标距离、资源表和严格抓取审计。"),
            "承接下一步": "收束到最终结论和下一阶段真实 VLA/Isaac/机械臂实验。",
        },
        {
            "顺序": "10",
            "讲解段落": "最终结论和下一阶段计划",
            "对应研究问题/阶段": "OpenVLA / Isaac / 真实 WidowX 后续",
            "建议时长": "60 秒",
            "推荐打开命令": ps_command("scripts/showcase_launcher.py", ["--target", "handoff"]),
            "推荐证据": f"docs/final_showcase_handoff.md；docs/next_experiment_registry.md；{cue_evidence(c09)}；{cue_evidence(c10)}",
            "可说结论": "当前结论可以支撑 MuJoCo 阶段的对照实验、失败分析和展示材料；下一步应按 readiness 文档补真实 VLA、Isaac 和 WidowX 结果。",
            "讲解稿": "最后总结：本阶段已经把轻量化 VLA 后训练路线拆成可复现任务、普通 baseline、trajectory/ACT baseline、action-head/PEFT proxy、语言泛化、数据效率和 domain randomization 代理评测。下一阶段不是推翻现有工作，而是在同一登记和评测格式上补真实外部实验。",
            "论文红线": "不能把 readiness、handoff、remote pack 或数据桥接写成已经完成的真实实验结果；新增结果必须先回填 CSV/JSON/视频再进入正式结论。",
            "承接下一步": "答辩结束后按 next_experiment_registry 继续推进真实 VLA、Isaac 或 WidowX 其中一条线。",
        },
    ]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, str]], manifest: dict[str, object]) -> None:
    counts = manifest.get("counts", {})
    if not isinstance(counts, dict):
        counts = {}

    lines = [
        "# 最终答辩讲解脚本",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：把当前 MuJoCo 实验包中的研究问题、阶段方法、量化指标、视频证据和论文边界整理成一条中文答辩讲解顺序。它不新增实验结果，只把已有证据组织成可讲、可查、可复现的展示脚本。",
        "",
        "打开本页命令：",
        "",
        "```powershell",
        ps_command("scripts/showcase_launcher.py", ["--target", "narrative-script"]),
        "```",
        "",
        "## 1. 当前证据计数",
        "",
        "| 项目 | 数量 |",
        "| --- | ---: |",
        md_row(["正式方法版本", str(counts.get("methods", 0))]),
        md_row(["阶段展示行", str(counts.get("stage_evidence_rows", 0))]),
        md_row(["视频证据", str(counts.get("video_evidence_rows", 0))]),
        md_row(["答辩视频 cue 条目", str(counts.get("defense_video_cue_sheet_rows", 0))]),
        md_row(["Trajectory/ACT 结论摘要行", str(counts.get("trajectory_act_conclusion_rows", 0))]),
        md_row(["证据包文件", str(counts.get("defense_evidence_pack_files", 0))]),
        "",
        "## 2. 推荐讲解顺序",
        "",
        "| 顺序 | 讲解段落 | 对应研究问题/阶段 | 建议时长 | 推荐打开命令 | 推荐证据 | 可说结论 | 讲解稿 | 论文红线 | 承接下一步 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(md_row([row[field] for field in FIELDNAMES]))

    lines.extend(
        [
            "",
            "## 3. 现场使用建议",
            "",
            "1. 先用 `--target package` 说明当前完成范围，再播放 `--target playlist` 或 `--target cue-sheet` 中的短片。",
            "2. 讲 trajectory-conditioned BC / ACT 时优先打开 `--target trajectory-act-brief`，不要只播放单个成功或失败视频。",
            "3. 老师追问某个方法时，用 `--list methods` 查版本名，再用 `--target method:<version> --action viewer --dry-run` 取完整 MuJoCo viewer 命令。",
            "4. 讲真实 OpenVLA、Isaac 或真实 WidowX 时，只引用 readiness/handoff/next registry，不能说已有真实实验成功率。",
            "",
            "## 4. 重建命令",
            "",
            "```powershell",
            ps_command("scripts/build_final_defense_narrative_script.py", []),
            "```",
            "",
            "## 5. 总体验证命令",
            "",
            "```powershell",
            ps_command("scripts/verify_experiment_artifacts.py", []),
            "```",
            "",
            f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)
    cue_rows = read_csv(args.cue_sheet)
    # Read these inputs so the script fails early if any upstream showcase source is missing.
    read_csv(args.research_plan)
    read_csv(args.stage_index)
    read_csv(args.handoff)
    rows = build_rows(manifest, cue_rows)
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows, manifest)
    print(f"final_defense_narrative_script_md: {args.output_md}", flush=True)
    print(f"final_defense_narrative_script_csv: {args.output_csv}", flush=True)
    print(f"rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
