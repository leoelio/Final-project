from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


FIELDNAMES = [
    "阶段编号",
    "阶段名称",
    "覆盖数量",
    "关键版本",
    "阶段报告",
    "量化证据",
    "视频证据",
    "展示入口",
    "论文可写结论",
    "论文红线",
    "推荐讲解",
    "重建命令",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese stage-level evidence index for the final project.")
    parser.add_argument("--versions", type=Path, default=ROOT / "docs" / "experiment_versions.json")
    parser.add_argument("--task-bc-stage", type=Path, default=ROOT / "docs" / "task_bc_stage_report.csv")
    parser.add_argument("--trajectory-act-stage", type=Path, default=ROOT / "docs" / "trajectory_act_stage_report.csv")
    parser.add_argument("--action-head-stage", type=Path, default=ROOT / "docs" / "action_head_stage_report.csv")
    parser.add_argument("--language-summary", type=Path, default=ROOT / "docs" / "language_generalization_summary.csv")
    parser.add_argument("--data-efficiency-summary", type=Path, default=ROOT / "docs" / "data_efficiency_summary.csv")
    parser.add_argument("--domain-randomization", type=Path, default=ROOT / "docs" / "domain_randomization_summary.csv")
    parser.add_argument("--video-evidence", type=Path, default=ROOT / "docs" / "video_evidence_index.csv")
    parser.add_argument("--external-readiness", type=Path, default=ROOT / "docs" / "external_dependency_readiness_audit.csv")
    parser.add_argument("--external-readiness-json", type=Path, default=ROOT / "outputs" / "evaluations" / "external_dependency_readiness_audit_v1.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "stage_evidence_index.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "stage_evidence_index.md")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def ps_command(script: str, extra: list[str] | None = None, *, cuda: bool = False) -> str:
    lines = []
    if cuda:
        lines.append('$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"')
    parts = [f'& "{PYTHON}" "{ROOT / script}"']
    if extra:
        parts.extend(extra)
    lines.append(" ".join(parts))
    return "\n".join(lines)


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", "<br>") for value in values) + " |"


def assert_existing(paths: list[str]) -> None:
    missing = [path for path in paths if not (ROOT / path).exists()]
    if missing:
        raise FileNotFoundError("\n".join(missing))


def count_success(rows: list[dict[str, str]], field: str = "success") -> str:
    if not rows:
        return "0/0"
    successes = sum(1 for row in rows if row.get(field) in {"True", "true", "1", "success=True"})
    return f"{successes}/{len(rows)}"


def build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    versions = read_json(args.versions)["methods"]
    task_bc_rows = read_csv(args.task_bc_stage)
    trajectory_rows = read_csv(args.trajectory_act_stage)
    action_head_rows = read_csv(args.action_head_stage)
    language_rows = read_csv(args.language_summary)
    data_rows = read_csv(args.data_efficiency_summary)
    domain_rows = read_csv(args.domain_randomization)
    video_rows = read_csv(args.video_evidence)
    external_rows = read_csv(args.external_readiness)
    external_readiness = read_json(args.external_readiness_json)
    formal_allowed = sum(1 for row in external_rows if row.get("formal_method_allowed_now") == "是")
    readiness_counts = external_readiness.get("readiness_counts", {})

    rows = [
        {
            "阶段编号": "1",
            "阶段名称": "任务/数据/普通 BC",
            "覆盖数量": f"{len(task_bc_rows)} 个版本",
            "关键版本": "expert_scripted_v1；structured_waypoint_policy_v1；replay_demo_v1；linear_bc_v1；knn_bc_v1；mlp_bc_v1",
            "阶段报告": "docs/task_bc_stage_report.md；docs/task_bc_stage_report.csv",
            "量化证据": "docs/evaluation_summary.csv；docs/model_resource_summary.csv；docs/data_efficiency_summary.csv；docs/failure_mode_taxonomy.csv",
            "视频证据": "outputs/presentation_clips/01_task_data_oracle.mp4；outputs/presentation_clips/02_basic_bc_baselines.mp4",
            "展示入口": "docs/video_evidence_gallery.html；docs/reproducible_command_index.md",
            "论文可写结论": "任务在 MuJoCo 中可解，示范回放可复现，结构化 waypoint 是强对照；linear/MLP 单步 BC 不足，kNN 的训练范围成功更接近轨迹记忆。",
            "论文红线": "expert/structured/replay 不能写成学习出的 VLA；普通 BC 不能写成语言理解或泛化策略。",
            "推荐讲解": "先展示任务、物体和盘子，再展示 expert/replay 证明数据可复现，最后用 linear/kNN/MLP 说明普通 baseline 的局限。",
            "重建命令": ps_command("scripts/build_task_bc_stage_report.py"),
        },
        {
            "阶段编号": "2",
            "阶段名称": "Trajectory / ACT / Diffusion",
            "覆盖数量": f"{len(trajectory_rows)} 个版本",
            "关键版本": "trajectory_conditioned_chunk_bc_v2；trajectory_knn_chunk_bc_v1；torch_act_state_chunk_v1；phase_conditioned_torch_act_v1；torch_act_cvae_state_chunk_v1；visual_act_cnn_cvae_v1；torch_diffusion_policy_state_chunk_v1",
            "阶段报告": "docs/trajectory_act_stage_report.md；docs/trajectory_act_stage_report.csv；docs/trajectory_prior_residual_bc_report.md",
            "量化证据": "docs/evaluation_summary.csv；docs/model_resource_summary.csv；docs/language_generalization_summary.csv；docs/domain_randomization_summary.csv；docs/failure_mode_taxonomy.csv；docs/trajectory_prior_residual_bc_report.csv",
            "视频证据": "outputs/presentation_clips/03_trajectory_act_diffusion.mp4；outputs/videos/domain_randomization_trajectory_knn_low_friction_seed0.mp4；outputs/videos/domain_randomization_visual_act_cnn_cvae_low_friction_seed0.mp4",
            "展示入口": "docs/trajectory_act_stage_report.md；docs/video_evidence_gallery.html",
            "论文可写结论": "trajectory/action-chunk/ACT-style/Diffusion 组已经构成可靠对照；trajectory-kNN 在训练范围能成功但留出和语言任务失败；phase-conditioned ACT 失败说明显式阶段条件本身不足以解决接触和抬升。",
            "论文红线": "当前 ACT-lite/state ACT/ACT-CVAE/Visual ACT 是本地轻量 baseline，不能写成完整官方 ACT；Diffusion-lite 不能写成完整 Diffusion Policy；不能写成 OpenVLA/RT-2。",
            "推荐讲解": "把这一组作为普通模仿学习的高级 baseline：先讲 action chunk 和 history，再讲 ACT/CVAE/Diffusion，最后用 trajectory-prior residual 候选说明结构化接触阶段先验能改善闭环，但不能写成纯 ACT 成功。",
            "重建命令": ps_command("scripts/build_trajectory_act_stage_report.py"),
        },
        {
            "阶段编号": "3",
            "阶段名称": "Action-Head / PEFT / CLIP",
            "覆盖数量": f"{len(action_head_rows)} 个版本",
            "关键版本": "object_language_action_head_lite_v1；reward_weighted_action_head_lite_v1；adapter_action_head_lite_v1；lora_action_head_lite_v1；clip_action_head_lite_v1；multi_task_object_action_head_lite_v1",
            "阶段报告": "docs/action_head_stage_report.md；docs/action_head_stage_report.csv",
            "量化证据": "docs/evaluation_summary.csv；docs/model_resource_summary.csv；docs/language_generalization_summary.csv；docs/data_efficiency_summary.csv；docs/failure_mode_taxonomy.csv",
            "视频证据": "outputs/presentation_clips/04_action_head_peft_proxy.mp4；outputs/videos/object_language_action_head_lite_v1_seed1_success_example.mp4",
            "展示入口": "docs/action_head_stage_report.md；docs/video_evidence_gallery.html",
            "论文可写结论": "本地 action-head/PEFT/CLIP proxy 链路已经建立，可用于参数量、训练时间、模型大小和小数据行为的对照；当前成功仍不稳定。",
            "论文红线": "不能写成真实 pretrained VLA 后训练；LoRA/Adapter 是本地 action-head proxy，不是 OpenVLA LoRA；CLIP 不是机器人 VLA。",
            "推荐讲解": "强调这是轻量化 VLA 路线的前置代理实验：先证明 action head 接口和资源统计可跑，再说明它还没有达到真实 VLA 的语义与动作泛化。",
            "重建命令": ps_command("scripts/build_action_head_stage_report.py"),
        },
        {
            "阶段编号": "4",
            "阶段名称": "语言/空间泛化",
            "覆盖数量": f"{len(language_rows)} 条语言评测",
            "关键版本": "expert_scripted_language_v1；structured_waypoint_policy_v1；object_language_action_head_lite_v1；torch_act_cvae_state_chunk_v1；clip_action_head_lite_v1；multi_task_object_action_head_lite_v1",
            "阶段报告": "docs/language_generalization_summary.csv；docs/video_evidence_index.md",
            "量化证据": "docs/language_generalization_summary.csv；docs/evaluation_summary.csv；docs/failure_mode_taxonomy.csv",
            "视频证据": "outputs/presentation_clips/05_language_generalization.mp4；outputs/showcase/language_generalization_grid.mp4",
            "展示入口": "docs/video_evidence_gallery.html；outputs/showcase/language_generalization_grid.mp4",
            "论文可写结论": "expert/structured 能完成语言任务；多数学习型单任务 baseline 在 move_leftmost_to_bowl 上失败，说明它们没有形成语言/空间泛化能力。",
            "论文红线": "语言 token、对象特征或 CLIP 代理不能等同于真实 VLA 语言理解；不能把 local proxy 写成 OpenVLA/RT-2 泛化。",
            "推荐讲解": "用同一 viewer 任务展示从颜色目标到空间目标的变化，再用 0/5 的语言结果说明普通方法与 VLA 目标之间的差距。",
            "重建命令": ps_command("scripts/build_video_evidence_index.py") + "\n" + ps_command("scripts/build_video_evidence_gallery.py"),
        },
        {
            "阶段编号": "5",
            "阶段名称": "数据效率",
            "覆盖数量": f"{len(data_rows)} 条预算评测",
            "关键版本": "knn_bc；trajectory_knn；object_action_head",
            "阶段报告": "docs/data_efficiency_summary.md；docs/data_efficiency_summary.csv",
            "量化证据": "docs/data_efficiency_summary.csv；outputs/evaluations/data_efficiency_v2.json；outputs/figures/data_efficiency.svg",
            "视频证据": "docs/video_evidence_index.md；docs/video_evidence_gallery.html",
            "展示入口": "docs/data_efficiency_summary.md；docs/experiment_dashboard.html",
            "论文可写结论": "已经完成 10/25/50/92 条 demonstration 的数据效率扫表；kNN/trajectory-kNN 的优势主要发生在训练范围，不能直接说明真实小数据泛化。",
            "论文红线": "不能把 MuJoCo scripted demonstration 的小数据结论写成真实机械臂数据效率；当前还不能证明真实 VLA 小数据优势。",
            "推荐讲解": "把数据效率作为研究问题的量化维度，而不是成功率主结论；重点讲预算、split 和 held-out 的差异。",
            "重建命令": ps_command("scripts/evaluate_data_efficiency.py", ["--episodes", "3", "--steps", "2840"]) + "\n" + ps_command("scripts/summarize_experiments.py"),
        },
        {
            "阶段编号": "6",
            "阶段名称": "MuJoCo domain randomization",
            "覆盖数量": f"{len(domain_rows)} 条扰动评测，成功 {count_success(domain_rows)}",
            "关键版本": "structured_waypoint_policy_v1；trajectory_knn_chunk_bc_v1；visual_act_cnn_cvae_v1",
            "阶段报告": "docs/domain_randomization_summary.md；docs/domain_randomization_summary.csv",
            "量化证据": "docs/domain_randomization_summary.csv；outputs/evaluations/domain_randomization_eval_v1.json；docs/failure_mode_taxonomy.csv",
            "视频证据": "outputs/presentation_clips/06_domain_randomization_proxy.mp4；outputs/videos/domain_randomization_structured_low_friction_seed0.mp4；outputs/videos/domain_randomization_trajectory_knn_low_friction_seed0.mp4；outputs/videos/domain_randomization_visual_act_cnn_cvae_low_friction_seed0.mp4",
            "展示入口": "docs/domain_randomization_summary.md；docs/video_evidence_gallery.html",
            "论文可写结论": "结构化 waypoint 在 MuJoCo 扰动域下稳定，trajectory-kNN 在低摩擦弱抓取域下降，visual ACT-CNN-CVAE 在当前数据规模下仍失败。",
            "论文红线": "MuJoCo domain randomization 只能写成 sim-to-real 代理鲁棒性证据，不能写成 Isaac domain randomization，也不能写成真实机器人验证。",
            "推荐讲解": "把它放在真实机械臂前的风险预检：说明哪些方法一碰到摩擦、力限或抓力扰动就退化。",
            "重建命令": ps_command("scripts/evaluate_domain_randomization.py", ["--episodes", "2", "--steps", "2840"], cuda=True),
        },
        {
            "阶段编号": "7",
            "阶段名称": "最终展示/答辩入口",
            "覆盖数量": f"{len(versions)} 个方法版本，{len(video_rows)} 条视频证据",
            "关键版本": "final_experiment_package_v1；video_evidence_gallery_v1；final_artifact_manifest_v1；defense_deck_v1；defense_video_playlist_v1；candidate_diagnostic_montage_v1",
            "阶段报告": "docs/final_experiment_package.md；docs/final_artifact_manifest.md；docs/reproducible_command_index.md；docs/goal_completion_audit.md；docs/defense_video_playlist.md；docs/video_presentation_storyboard.md",
            "量化证据": "docs/result_matrix.md；docs/stage_comparison_report.csv；docs/research_evidence_map.csv；docs/final_artifact_manifest.json",
            "视频证据": "outputs/presentation_clips/00_defense_video_reel.mp4；outputs/presentation_clips/07_candidate_diagnostics.mp4；outputs/showcase/all_registered_methods_grid.mp4；outputs/showcase/core_methods_grid.mp4",
            "展示入口": "docs/video_evidence_gallery.html；docs/defense_video_playlist.html；docs/defense_deck.html；docs/final_experiment_package.md",
            "论文可写结论": "当前 MuJoCo 实验包可以支撑普通 BC、trajectory/ACT/Diffusion baseline、action-head proxy、语言泛化、数据效率、扰动域代理评测和候选诊断失败模式的阶段性对比。",
            "论文红线": "展示视频不是额外实验结论；最终论文若要写真实 OpenVLA、Isaac 或真实 WidowX，必须补对应实验，不能用当前 MuJoCo proxy 代替。",
            "推荐讲解": "答辩时从总览视频进入，再用候选诊断总览解释失败模式和严格抓取口径，然后按阶段报告讲量化表和红线，最后落到 next_phase_implementation 说明后续真实 VLA/Isaac/真实机械臂工作。",
            "重建命令": ps_command("scripts/build_final_artifact_manifest.py") + "\n" + ps_command("scripts/verify_experiment_artifacts.py"),
        },
        {
            "阶段编号": "8",
            "阶段名称": "外部依赖 readiness 门禁",
            "覆盖数量": f"{len(external_rows)} 条 readiness 审计，formal_method_allowed_now=是 为 {formal_allowed} 条",
            "关键版本": "robot_vla_action_head_lite_v1；robot_vla_adapter_lite_v1；robot_vla_lora_lite_v1；isaac_domain_randomization_v1；real_widowx_validation_v1",
            "阶段报告": "docs/external_dependency_readiness_audit.md；docs/next_experiment_registry.md；docs/robot_vla_remote_result_intake.md；docs/isaac_domain_randomization_handoff.md；docs/real_widowx_validation_handoff.md",
            "量化证据": "docs/external_dependency_readiness_audit.csv；outputs/evaluations/external_dependency_readiness_audit_v1.json；docs/final_artifact_manifest.json",
            "视频证据": "outputs/presentation_clips/00_defense_video_reel.mp4",
            "展示入口": "docs/external_dependency_readiness_audit.md；docs/openvla_bridge_gallery.html；docs/defense_deck.html；docs/final_experiment_package.md",
            "论文可写结论": f"真实 robot VLA、Isaac 和真实 WidowX planned 版本已经有阻塞条件、回填文件和入包门禁；当前 waiting_remote_result {readiness_counts.get('waiting_remote_result', 0)}、waiting_isaac_runtime {readiness_counts.get('waiting_isaac_runtime', 0)}、waiting_real_robot_trials {readiness_counts.get('waiting_real_robot_trials', 0)}，没有任何 planned 外部版本可直接写入正式方法成功率。",
            "论文红线": "external_dependency_readiness_audit_v1 不是策略成功率结果；不能写成真实 OpenVLA/机器人 VLA 后训练、Isaac domain randomization 或真实 WidowX 验证已经完成。",
            "推荐讲解": "把它作为从当前 MuJoCo 实验包走向真实 VLA/Isaac/真实机械臂的门禁页：先说明哪些文件已经准备好，再说明哪些外部运行结果必须回填后才能变成正式方法版本。",
            "重建命令": ps_command("scripts/build_external_dependency_readiness_audit.py") + "\n" + ps_command("scripts/build_stage_evidence_index.py"),
        },
    ]

    declared_paths: list[str] = []
    for row in rows:
        for field in ("阶段报告", "量化证据", "视频证据", "展示入口"):
            declared_paths.extend(path.strip() for path in row[field].split("；") if path.strip().startswith(("docs/", "outputs/")))
    assert_existing(declared_paths)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 阶段证据总表",
        "",
        "版本：`stage_evidence_index_v1`",
        "",
        "用途：把当前毕业设计实验包按“研究阶段”重新组织，集中记录每个阶段的版本数量、关键版本、量化表、视频证据、论文可写结论、论文红线和推荐讲解顺序。这个文档不新增方法，只作为中文实验记录和答辩导航入口。",
        "",
        "当前定位：已经完成 MuJoCo WidowX 桌面抓取/放置环境、普通 BC、trajectory-conditioned BC / ACT-style / Diffusion-lite、action-head/PEFT proxy、语言/空间泛化、数据效率、MuJoCo domain randomization 代理评测和外部依赖 readiness 门禁。真实 OpenVLA/RT-2 后训练、Isaac 高保真随机化和真实机械臂验证仍是后续阶段。",
        "",
        "快速打开入口：",
        "",
        "```powershell",
        f'Start-Process "{ROOT / "docs" / "video_evidence_gallery.html"}"',
        f'Start-Process "{ROOT / "docs" / "defense_deck.html"}"',
        f'notepad.exe "{ROOT / "docs" / "stage_evidence_index.md"}"',
        "```",
        "",
        "## 1. 阶段总览",
        "",
        md_row(["阶段", "覆盖数量", "关键版本", "阶段报告", "视频证据", "论文红线"]),
        md_row(["---", "---:", "---", "---", "---", "---"]),
    ]
    for row in rows:
        lines.append(
            md_row(
                [
                    f"{row['阶段编号']}. {row['阶段名称']}",
                    row["覆盖数量"],
                    row["关键版本"],
                    row["阶段报告"],
                    row["视频证据"],
                    row["论文红线"],
                ]
            )
        )

    lines.extend(["", "## 2. 推荐讲解顺序", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row['阶段编号']}. {row['阶段名称']}",
                "",
                f"- 覆盖数量：{row['覆盖数量']}",
                f"- 关键版本：{row['关键版本']}",
                f"- 量化证据：{row['量化证据']}",
                f"- 视频证据：{row['视频证据']}",
                f"- 展示入口：{row['展示入口']}",
                f"- 论文可写结论：{row['论文可写结论']}",
                f"- 论文红线：{row['论文红线']}",
                f"- 推荐讲解：{row['推荐讲解']}",
                "",
                "重建命令：",
                "",
                "```powershell",
                row["重建命令"],
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## 3. 全量重建与验证",
            "",
            "重建本阶段证据总表：",
            "",
            "```powershell",
            ps_command("scripts/build_stage_evidence_index.py"),
            "```",
            "",
            "重建总 manifest 后运行完整验证：",
            "",
            "```powershell",
            ps_command("scripts/build_final_artifact_manifest.py"),
            '$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"',
            ps_command("scripts/verify_experiment_artifacts.py"),
            "```",
            "",
            "论文写作提醒：本文档中的“不能写成真实机器人验证”“不能写成 OpenVLA/RT-2”“不能写成完整官方 ACT”等红线需要保留到最终论文和答辩材料中，避免把 MuJoCo proxy 或本地轻量 baseline 夸大成真实 VLA 结论。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows)
    print(f"stage_evidence_rows: {len(rows)}", flush=True)
    print(f"stage_evidence_csv: {args.output_csv}", flush=True)
    print(f"stage_evidence_md: {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
