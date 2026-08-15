from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a one-page Chinese defense live demo runbook.")
    parser.add_argument("--versions", type=Path, default=ROOT / "docs" / "experiment_versions.json")
    parser.add_argument("--video-evidence", type=Path, default=ROOT / "docs" / "video_evidence_index.csv")
    parser.add_argument("--candidate-index", type=Path, default=ROOT / "docs" / "candidate_diagnostic_video_index.csv")
    parser.add_argument("--strict-grasp-json", type=Path, default=ROOT / "outputs" / "evaluations" / "strict_grasp_success_audit_v1.json")
    parser.add_argument("--next-registry", type=Path, default=ROOT / "docs" / "next_experiment_registry.csv")
    parser.add_argument("--external-readiness", type=Path, default=ROOT / "docs" / "external_dependency_readiness_audit.csv")
    parser.add_argument("--external-readiness-json", type=Path, default=ROOT / "outputs" / "evaluations" / "external_dependency_readiness_audit_v1.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "defense_live_runbook.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "defense_live_runbook.csv")
    return parser.parse_args()


def q(path: str | Path) -> str:
    return f'"{path}"'


def ps_command(script: str, args: list[str | Path] | None = None, *, cuda: bool = False) -> str:
    args = args or []
    rendered = [q(PYTHON), q(ROOT / script)]
    for arg in args:
        rendered.append(q(arg) if isinstance(arg, Path) else str(arg))
    command = "& " + " ".join(rendered)
    if cuda:
        return f'$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"; {command}'
    return command


def start_process(path: str | Path, *, notepad: bool = False) -> str:
    full = ROOT / path if not Path(path).is_absolute() else Path(path)
    if notepad:
        return f"Start-Process notepad.exe {q(full)}"
    return f"Start-Process {q(full)}"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def build_rows() -> list[dict[str, str]]:
    return [
        {
            "顺序": "0",
            "时间": "开场前 2 分钟",
            "环节": "环境和入口检查",
            "打开内容": "验证脚本、quick/candidate 列表、dashboard/deck、readiness audit",
            "执行命令": "\n".join(
                [
                    ps_command("scripts/verify_experiment_artifacts.py", [], cuda=True),
                    ps_command("scripts/showcase_launcher.py", ["--list", "quick"]),
                    ps_command("scripts/showcase_launcher.py", ["--list", "candidates"]),
                    start_process("docs/experiment_dashboard.html"),
                    start_process("docs/defense_deck.html"),
                    start_process("docs/external_dependency_readiness_audit.md", notepad=True),
                ]
            ),
            "讲解重点": "确认所有登记方法、视频、阶段报告、candidate 诊断、外部依赖 readiness 和中文文档都通过验证。",
            "论文红线": "验证脚本通过只证明当前 MuJoCo 实验包完整，不证明真实 OpenVLA、Isaac 或真实 WidowX 已完成。",
        },
        {
            "顺序": "1",
            "时间": "0-1 分钟",
            "环节": "总览开场",
            "打开内容": "答辩 deck、总览 reel、final package",
            "执行命令": "\n".join(
                [
                    start_process("docs/defense_deck.html"),
                    start_process("outputs/presentation_clips/00_defense_video_reel.mp4"),
                    start_process("docs/final_experiment_package.md", notepad=True),
                ]
            ),
            "讲解重点": "先说明研究问题、MuJoCo WidowX 桌面任务、25 个正式方法版本和视频证据链。",
            "论文红线": "总览视频是定性展示，不能替代成功率、目标距离、资源和语言泛化表。",
        },
        {
            "顺序": "2",
            "时间": "1-3 分钟",
            "环节": "任务、数据和普通 BC",
            "打开内容": "阶段 1/2 短片、任务 BC 阶段报告",
            "执行命令": "\n".join(
                [
                    start_process("outputs/presentation_clips/01_task_data_oracle.mp4"),
                    start_process("outputs/presentation_clips/02_basic_bc_baselines.mp4"),
                    start_process("docs/task_bc_stage_report.md", notepad=True),
                ]
            ),
            "讲解重点": "证明 expert、replay 和 structured waypoint 建立了可复现任务；Linear/MLP BC 失败，kNN 主要是训练范围记忆。",
            "论文红线": "expert、replay 和 structured waypoint 不是 learned VLA；普通 BC 不能写成语言理解或 VLA 后训练。",
        },
        {
            "顺序": "3",
            "时间": "3-5 分钟",
            "环节": "Trajectory / ACT / Diffusion",
            "打开内容": "阶段 3 短片、trajectory/ACT 报告、中文实验台账",
            "执行命令": "\n".join(
                [
                    start_process("outputs/presentation_clips/03_trajectory_act_diffusion.mp4"),
                    start_process("docs/trajectory_act_stage_report.md", notepad=True),
                    start_process("docs/trajectory_act_experiment_record.md", notepad=True),
                ]
            ),
            "讲解重点": "说明历史观测、动作块、Transformer、CVAE、视觉 ACT-lite 和 diffusion baseline 都已建立，但闭环接触和抬升仍不稳定。",
            "论文红线": "当前是本地轻量 baseline 或代理实现，不能写成完整官方 ACT、完整视觉 Diffusion Policy 或稳定抓取成功。",
        },
        {
            "顺序": "4",
            "时间": "5-6 分钟",
            "环节": "严格抓取口径",
            "打开内容": "strict grasp audit、候选诊断总览、候选诊断索引、候选 viewer dry-run",
            "执行命令": "\n".join(
                [
                    start_process("docs/strict_grasp_success_audit.md", notepad=True),
                    start_process("outputs/presentation_clips/07_candidate_diagnostics.mp4"),
                    start_process("docs/candidate_diagnostic_video_index.md", notepad=True),
                    ps_command(
                        "scripts/showcase_launcher.py",
                        ["--target", "candidate:grasp_gated_torch_act_state_chunk_v1_candidate", "--action", "viewer", "--dry-run"],
                        cuda=True,
                    ),
                ]
            ),
            "讲解重点": "先播放候选诊断总览，再把原始放置 success、grasp_success 和 object_z 分开，说明为什么某些 success=True 不能算稳定抓取。",
            "论文红线": "候选诊断总览只说明失败模式和局部现象；不能把目标距离达标或物体被推到盘子附近写成稳定抓取放置。",
        },
        {
            "顺序": "5",
            "时间": "6-8 分钟",
            "环节": "Action-head / PEFT / CLIP proxy",
            "打开内容": "阶段 4 短片、action-head 报告、资源表",
            "执行命令": "\n".join(
                [
                    start_process("outputs/presentation_clips/04_action_head_peft_proxy.mp4"),
                    start_process("docs/action_head_stage_report.md", notepad=True),
                    start_process("docs/model_resource_summary.csv"),
                ]
            ),
            "讲解重点": "说明本地 action head、reward-weighted BC、Adapter、LoRA-style 和 CLIP proxy 有资源对照，但不是 pretrained robot VLA 后训练。",
            "论文红线": "Adapter/LoRA-style 是本地 action-head proxy，不是 OpenVLA LoRA；CLIP 也不是机器人 VLA。",
        },
        {
            "顺序": "6",
            "时间": "8-10 分钟",
            "环节": "语言/空间泛化与数据效率",
            "打开内容": "语言短片、语言宫格、数据效率图",
            "执行命令": "\n".join(
                [
                    start_process("outputs/presentation_clips/05_language_generalization.mp4"),
                    start_process("outputs/showcase/language_generalization_grid.mp4"),
                    start_process("outputs/figures/data_efficiency.svg"),
                    start_process("docs/research_question_showcase_plan.md", notepad=True),
                ]
            ),
            "讲解重点": "展示 leftmost-to-bowl 任务和小数据曲线，强调规则/结构化可解，但 learned proxy 还没有可靠语言泛化。",
            "论文红线": "不能把 MuJoCo scripted demonstration 的小数据结论直接写成真实机械臂数据效率或真实 VLA 小数据优势。",
        },
        {
            "顺序": "7",
            "时间": "10-12 分钟",
            "环节": "Domain randomization 与后续真实验证",
            "打开内容": "阶段 6 短片、domain randomization 报告、readiness audit、Isaac/真实 WidowX handoff",
            "执行命令": "\n".join(
                [
                    start_process("outputs/presentation_clips/06_domain_randomization_proxy.mp4"),
                    start_process("docs/domain_randomization_summary.md", notepad=True),
                    start_process("docs/external_dependency_readiness_audit.md", notepad=True),
                    start_process("docs/isaac_domain_randomization_handoff.md", notepad=True),
                    start_process("docs/real_widowx_validation_handoff.md", notepad=True),
                ]
            ),
            "讲解重点": "说明 MuJoCo 扰动域是 Isaac/真实机器人前置检查，readiness audit 已固定阻塞条件、回填字段和 trial 模板。",
            "论文红线": "readiness audit 不是策略成功率结果；不能写成 Isaac domain randomization 已完成，也不能写成真实 WidowX 迁移验证已完成。",
        },
        {
            "顺序": "8",
            "时间": "12-14 分钟",
            "环节": "OpenVLA / 机器人 VLA 后续入口",
            "打开内容": "OpenVLA bridge 预览、handoff、remote run pack、result intake、readiness audit",
            "执行命令": "\n".join(
                [
                    start_process("data/vla_bridge/openvla_dataset_bridge_v1/preview_grid.png"),
                    start_process("docs/openvla_bridge_gallery.html"),
                    start_process("docs/robot_vla_action_head_handoff.md", notepad=True),
                    start_process("docs/robot_vla_remote_run_pack.md", notepad=True),
                    start_process("docs/robot_vla_remote_result_intake.md", notepad=True),
                    start_process("docs/external_dependency_readiness_audit.md", notepad=True),
                ]
            ),
            "讲解重点": "收束到毕业设计主线：当前 MuJoCo 对照组和数据桥接已完成，下一阶段按 readiness 门禁接真实 robot VLA 表征 + action head/Adapter/LoRA。",
            "论文红线": "不能写成 OpenVLA LoRA、RT-2、robot_vla_action_head_lite_v1 训练结果已经完成；planned 版本必须先回填真实评测、资源、视频和报告。",
        },
        {
            "顺序": "9",
            "时间": "追问时",
            "环节": "现场追问快速入口",
            "打开内容": "showcase launcher、stage/method/candidate viewer",
            "执行命令": "\n".join(
                [
                    ps_command("scripts/showcase_launcher.py", ["--target", "claim:C03"]),
                    ps_command("scripts/showcase_launcher.py", ["--target", "method:trajectory_knn_chunk_bc_v1", "--action", "viewer", "--dry-run"]),
                    ps_command("scripts/showcase_launcher.py", ["--target", "candidate:grasp_gated_trajectory_conditioned_chunk_bc_v1_candidate", "--action", "open-all"]),
                    ps_command("scripts/showcase_launcher.py", ["--target", "candidate:grasp_gated_trajectory_conditioned_chunk_bc_v1_candidate", "--action", "viewer"]),
                ]
            ),
            "讲解重点": "评委追问某个方法时，用 method/candidate 入口打开固定视频、报告、元数据或慢速 MuJoCo viewer。",
            "论文红线": "现场 viewer 是可视化复现入口，不改变已登记成功率；所有结论仍以 CSV/JSON 和固定视频审计为准。",
        },
    ]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(value.replace("|", "\\|").replace("\n", "<br>") for value in values) + " |"


def build_markdown(args: argparse.Namespace, rows: list[dict[str, str]]) -> str:
    versions = read_json(args.versions)
    video_rows = read_csv(args.video_evidence)
    candidate_rows = read_csv(args.candidate_index)
    next_rows = read_csv(args.next_registry)
    external_rows = read_csv(args.external_readiness)
    external_readiness = read_json(args.external_readiness_json)
    strict = read_json(args.strict_grasp_json).get("summary", {})
    completed_next = sum(1 for row in next_rows if row.get("status", row.get("状态", "")).startswith("completed"))
    planned_next = len(next_rows) - completed_next
    formal_allowed = sum(1 for row in external_rows if row.get("formal_method_allowed_now") == "是")
    readiness_counts = external_readiness.get("readiness_counts", {})
    loose = f"{strict.get('loose_successes', '?')}/{strict.get('episodes', '?')}"
    strict_success = f"{strict.get('strict_grasp_successes', '?')}/{strict.get('episodes', '?')}"

    lines = [
        "# 答辩现场展示 Runbook",
        "",
        "版本：`defense_live_runbook_v1`",
        "",
        "用途：把当前实验包转成答辩当天可直接执行的一页式操作清单。它只组织现有证据，不新增实验结果；所有成功率和边界仍以 CSV/JSON、阶段报告和固定视频审计为准。",
        "",
        "## 1. 当前证据计数",
        "",
        f"- 正式方法版本：`{len(versions.get('methods', []))}`。",
        f"- 视频证据条目：`{len(video_rows)}`。",
        f"- 候选诊断视频：`{len(candidate_rows)}`。",
        f"- 严格抓取审计：原始放置成功 `{loose}`，严格抓取成功 `{strict_success}`。",
        f"- 下一阶段实验注册表：`{len(next_rows)}` 行，其中 completed/prerequisite/diagnostic `{completed_next}` 行，planned `{planned_next}` 行。",
        f"- 外部依赖 readiness audit：`{len(external_rows)}` 行，formal_method_allowed_now 为 `是` 的行数 `{formal_allowed}`，waiting_remote_result `{readiness_counts.get('waiting_remote_result', 0)}`，waiting_isaac_runtime `{readiness_counts.get('waiting_isaac_runtime', 0)}`，waiting_real_robot_trials `{readiness_counts.get('waiting_real_robot_trials', 0)}`。",
        "",
        "## 2. 开场前检查",
        "",
        "```powershell",
        ps_command("scripts/verify_experiment_artifacts.py", [], cuda=True),
        ps_command("scripts/showcase_launcher.py", ["--list", "quick"]),
        ps_command("scripts/showcase_launcher.py", ["--list", "candidates"]),
        start_process("docs/experiment_dashboard.html"),
        start_process("docs/defense_deck.html"),
        "```",
        "",
        "## 3. 推荐现场顺序",
        "",
        "| 顺序 | 时间 | 环节 | 打开内容 | 讲解重点 | 论文红线 |",
        "| ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(md_row([row["顺序"], row["时间"], row["环节"], row["打开内容"], row["讲解重点"], row["论文红线"]]))

    lines.extend(["", "## 4. 分步命令", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row['顺序']}. {row['环节']}",
                "",
                f"- 时间：{row['时间']}",
                f"- 打开内容：{row['打开内容']}",
                f"- 讲解重点：{row['讲解重点']}",
                f"- 论文红线：{row['论文红线']}",
                "",
                "```powershell",
                row["执行命令"],
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## 5. 现场必须坚持的边界",
            "",
            "- 不能写成稳定抓取成功，除非同时有批量 `success`、`grasp_success`、`object_z` 和视频证据支持。",
            "- 不能写成真实 OpenVLA/RT-2/机器人 VLA 后训练完成。",
            "- 不能写成 Isaac domain randomization 已完成。",
            "- 不能写成真实 WidowX 机械臂验证已完成。",
            "- 不能把 `external_dependency_readiness_audit_v1` 写成策略成功率结果；它只说明外部阶段门禁、阻塞条件和回填要求。",
            "- 不能只展示成功视频而隐去失败 baseline；失败结果本身是普通 BC 不足和任务难度的证据。",
            "",
            "## 6. 最短应急展示",
            "",
            "只有 3 分钟时按这个顺序：",
            "",
            "```powershell",
            start_process("outputs/presentation_clips/00_defense_video_reel.mp4"),
            start_process("docs/result_matrix.md", notepad=True),
            start_process("docs/strict_grasp_success_audit.md", notepad=True),
            start_process("docs/external_dependency_readiness_audit.md", notepad=True),
            start_process("docs/next_experiment_registry.md", notepad=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = build_rows()
    write_csv(args.output_csv, rows)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_markdown(args, rows), encoding="utf-8")
    print(f"defense_live_runbook_rows: {len(rows)}", flush=True)
    print(f"defense_live_runbook_md: {args.output_md}", flush=True)
    print(f"defense_live_runbook_csv: {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
