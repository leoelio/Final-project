from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_experiment_command_index as command_index  # noqa: E402


TRAJECTORY_ACT_VERSIONS = (
    "act_lite_chunk_bc_v1",
    "trajectory_conditioned_chunk_bc_v2",
    "trajectory_knn_chunk_bc_v1",
    "torch_act_state_chunk_v1",
    "torch_act_state_chunk_cuda_v1",
    "phase_conditioned_torch_act_v1",
    "torch_act_cvae_state_chunk_v1",
    "visual_feature_act_lite_v1",
    "visual_act_cnn_cvae_v1",
)

INPUT_TYPES = {
    "act_lite_chunk_bc_v1": "当前状态",
    "trajectory_conditioned_chunk_bc_v2": "8 帧历史状态",
    "trajectory_knn_chunk_bc_v1": "8 帧历史状态检索",
    "torch_act_state_chunk_v1": "8 帧历史状态",
    "torch_act_state_chunk_cuda_v1": "8 帧历史状态",
    "phase_conditioned_torch_act_v1": "8 帧历史状态 + 阶段 one-hot",
    "torch_act_cvae_state_chunk_v1": "8 帧历史状态 + CVAE latent",
    "visual_feature_act_lite_v1": "RGB pooled feature + 语言 token + 本体状态",
    "visual_act_cnn_cvae_v1": "RGB 小 CNN + 语言 token + 本体状态",
}

ACTION_TYPES = {
    "act_lite_chunk_bc_v1": "8 步动作块 MLP",
    "trajectory_conditioned_chunk_bc_v2": "历史条件 8 步动作块 MLP",
    "trajectory_knn_chunk_bc_v1": "历史条件 kNN 动作块记忆",
    "torch_act_state_chunk_v1": "Transformer ACT-style 动作块",
    "torch_act_state_chunk_cuda_v1": "Transformer ACT-style 动作块 CUDA 资源对照",
    "phase_conditioned_torch_act_v1": "阶段条件 Transformer ACT-style 动作块",
    "torch_act_cvae_state_chunk_v1": "Transformer ACT-CVAE 动作块",
    "visual_feature_act_lite_v1": "视觉特征 Transformer ACT-lite 动作块",
    "visual_act_cnn_cvae_v1": "CNN 视觉 ACT-CVAE-lite 动作块",
}

PAPER_BOUNDARIES = {
    "act_lite_chunk_bc_v1": "不能写成完整 ACT，只能写成短动作块 BC baseline。",
    "trajectory_conditioned_chunk_bc_v2": "不能写成完整 ACT 或稳定抓取成功；它只是历史状态条件动作块 baseline。",
    "trajectory_knn_chunk_bc_v1": "训练范围成功不能写成泛化能力；它更接近轨迹记忆。",
    "torch_act_state_chunk_v1": "不能写成完整视觉 ACT；这是 state-only ACT-style baseline。",
    "torch_act_state_chunk_cuda_v1": "CUDA 版本只能写成资源对照，不能写成策略结构改进。",
    "phase_conditioned_torch_act_v1": "阶段 one-hot 失败不能证明 ACT 无效，只能说明当前阶段条件代理不足。",
    "torch_act_cvae_state_chunk_v1": "不能写成官方 ACT 复现；这是 state-only ACT-CVAE-lite。",
    "visual_feature_act_lite_v1": "pooled RGB 特征不能写成端到端视觉 ACT。",
    "visual_act_cnn_cvae_v1": "小型 CNN 视觉链路不能写成官方完整 ACT 或真实机器人视觉 ACT。",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese experiment record for trajectory-conditioned BC / ACT baselines.")
    parser.add_argument("--versions", type=Path, default=ROOT / "docs" / "experiment_versions.json")
    parser.add_argument("--resource-summary", type=Path, default=ROOT / "docs" / "model_resource_summary.csv")
    parser.add_argument("--language-summary", type=Path, default=ROOT / "docs" / "language_generalization_summary.csv")
    parser.add_argument("--video-evidence", type=Path, default=ROOT / "docs" / "video_evidence_index.csv")
    parser.add_argument("--strict-grasp-json", type=Path, default=ROOT / "outputs" / "evaluations" / "strict_grasp_success_audit_v1.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "trajectory_act_experiment_record.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "trajectory_act_experiment_record.md")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def rel(path_text: str) -> str:
    path = Path(path_text)
    if path.is_absolute():
        try:
            return path.relative_to(ROOT).as_posix()
        except ValueError:
            return path.as_posix()
    return path.as_posix()


def fraction_value(text: str) -> float:
    if "/" not in text:
        return 0.0
    numerator, denominator = text.split("/", 1)
    try:
        return float(numerator) / float(denominator)
    except ValueError:
        return 0.0


def md_cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", "<br>")


def build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    versions = read_json(args.versions)["methods"]
    methods = {method["version"]: method for method in versions}
    resources = {row["version"]: row for row in read_csv(args.resource_summary)}
    language = {row["version"]: row for row in read_csv(args.language_summary)}
    videos = {
        row["版本"]: row
        for row in read_csv(args.video_evidence)
        if row.get("视频类型") == "主任务固定片段"
    }

    rows: list[dict[str, str]] = []
    for version in TRAJECTORY_ACT_VERSIONS:
        method = methods[version]
        resource = resources.get(version, {})
        lang = language.get(version, {})
        video = videos.get(version, {})
        main_viewer = command_index.viewer_command(method, task="place_blue_cube_blue_pad", complexity="medium", seed=0)
        language_viewer = command_index.viewer_command(method, task="move_leftmost_to_bowl", complexity="language", seed=200)
        train_command = command_index.training_command(version) or ""

        rows.append(
            {
                "版本": version,
                "方法": method["method"],
                "阶段": method["stage"],
                "输入形式": INPUT_TYPES[version],
                "动作形式": ACTION_TYPES[version],
                "最终模型": rel(method["artifact"]),
                "训练范围成功率": method["train_range_success"],
                "留出范围成功率": method["heldout_success"],
                "语言/空间泛化": lang.get("success", "未登记"),
                "语言平均目标距离": lang.get("mean_target_distance", "未登记"),
                "可训练参数": resource.get("trainable_params", ""),
                "模型大小MB": resource.get("artifact_size_mb", ""),
                "训练时间秒": resource.get("train_time_seconds", ""),
                "峰值显存MB": resource.get("peak_vram_mb", ""),
                "固定视频": rel(method["clip"]),
                "固定视频结果": video.get("结果", ""),
                "目标距离": video.get("目标距离", ""),
                "抓取标志": video.get("抓取标志", ""),
                "物体高度": video.get("物体高度", ""),
                "实验结论": method["note"],
                "论文红线": PAPER_BOUNDARIES[version],
                "主任务Viewer命令": main_viewer,
                "语言Viewer命令": language_viewer,
                "训练命令": train_command,
            }
        )
    return rows


def build_markdown(rows: list[dict[str, str]], strict_grasp: dict) -> str:
    train_total = sum(fraction_value(row["训练范围成功率"]) for row in rows)
    heldout_total = sum(fraction_value(row["留出范围成功率"]) for row in rows)
    language_total = sum(fraction_value(row["语言/空间泛化"]) for row in rows)
    strict_summary = strict_grasp.get("summary", {})
    loose = f"{strict_summary.get('loose_successes', '?')}/{strict_summary.get('episodes', '?')}"
    strict = f"{strict_summary.get('strict_grasp_successes', '?')}/{strict_summary.get('episodes', '?')}"

    lines = [
        "# Trajectory-conditioned BC / ACT 中文实验台账",
        "",
        "版本：`trajectory_act_experiment_record_v1`",
        "",
        "用途：把 trajectory-conditioned BC / ACT 阶段的最终版本、模型结构、量化结果、固定视频、慢速 MuJoCo viewer 命令和论文红线整理成中文实验记录。代码标识、版本名和命令参数保持英文，便于复现实验。",
        "",
        "数据来源：`docs/experiment_versions.json`、`docs/model_resource_summary.csv`、`docs/language_generalization_summary.csv`、`docs/video_evidence_index.csv`、`outputs/evaluations/strict_grasp_success_audit_v1.json`。",
        "",
        "## 1. 阶段口径",
        "",
        f"- 记录版本数：{len(rows)}。",
        f"- 主任务训练范围成功率等价总和：{train_total:.2f} / {len(rows)}。",
        f"- 主任务留出范围成功率等价总和：{heldout_total:.2f} / {len(rows)}。",
        f"- 语言/空间泛化成功率等价总和：{language_total:.2f} / {len(rows)}。",
        f"- 严格抓取审计：原始放置成功 `{loose}`，严格抓取成功 `{strict}`。",
        "- 慢速可视化统一使用 `--viewer --duration 60 --speed 0.05`。",
        "",
        "结论口径：本阶段可以写成“高级普通模仿学习对照组已经建立”，不能写成“完整官方 ACT 已复现成功”，也不能写成“已经稳定抓取”。`success=True` 必须和 `grasp_success/object_z` 分开解释。",
        "",
        "## 2. 中文台账表",
        "",
        "| 版本 | 方法 | 输入形式 | 动作形式 | Train | Held-out | Language | 参数 | 固定视频结果 | 论文红线 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                md_cell(row[key])
                for key in (
                    "版本",
                    "方法",
                    "输入形式",
                    "动作形式",
                    "训练范围成功率",
                    "留出范围成功率",
                    "语言/空间泛化",
                    "可训练参数",
                    "固定视频结果",
                    "论文红线",
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 3. 逐版本记录与完整命令",
            "",
        ]
    )
    for row in rows:
        lines.extend(
            [
                f"### `{row['版本']}`",
                "",
                f"- 方法：{row['方法']}。",
                f"- 输入形式：{row['输入形式']}。",
                f"- 动作形式：{row['动作形式']}。",
                f"- 最终模型：`{row['最终模型']}`。",
                f"- 主任务：train `{row['训练范围成功率']}`，held-out `{row['留出范围成功率']}`。",
                f"- 语言/空间泛化：`{row['语言/空间泛化']}`，平均目标距离 `{row['语言平均目标距离']}`。",
                f"- 固定视频：`{row['固定视频']}`；视频结果 `{row['固定视频结果']}`，抓取标志 `{row['抓取标志']}`，物体高度 `{row['物体高度']}`。",
                f"- 实验结论：{row['实验结论']}",
                f"- 论文红线：{row['论文红线']}",
                "",
                "主任务慢速 viewer：",
                "",
                "```powershell",
                row["主任务Viewer命令"],
                "```",
                "",
                "语言/空间泛化慢速 viewer：",
                "",
                "```powershell",
                row["语言Viewer命令"],
                "```",
                "",
            ]
        )
        if row["训练命令"]:
            lines.extend(
                [
                    "训练命令：",
                    "",
                    "```powershell",
                    row["训练命令"],
                    "```",
                    "",
                ]
            )

    lines.extend(
        [
            "## 4. 候选诊断入口",
            "",
            "如果需要继续观察“慢速、限幅、抓取门控之后仍失败”的过程，优先打开候选诊断视频或候选 viewer。",
            "",
            "```powershell",
            command_index.ps_command("scripts/showcase_launcher.py", ["--list", "candidates"]),
            command_index.ps_command(
                "scripts/showcase_launcher.py",
                ["--target", "candidate:grasp_gated_trajectory_conditioned_chunk_bc_v1_candidate", "--action", "viewer", "--dry-run"],
            ),
            command_index.ps_command(
                "scripts/showcase_launcher.py",
                ["--target", "candidate:grasp_gated_torch_act_state_chunk_v1_candidate", "--action", "viewer", "--dry-run"],
            ),
            command_index.ps_command(
                "scripts/showcase_launcher.py",
                ["--target", "candidate:contact_aware_trajectory_knn_v1_candidate", "--action", "viewer", "--dry-run"],
            ),
            "```",
            "",
            "相关报告：",
            "",
            "```text",
            "docs\\trajectory_act_stage_report.md",
            "docs\\trajectory_act_failure_diagnosis.md",
            "docs\\grasp_gated_trajectory_act_report.md",
            "docs\\strict_grasp_success_audit.md",
            "docs\\candidate_diagnostic_video_index.md",
            "docs\\contact_aware_trajectory_knn_report.md",
            "```",
            "",
            "## 5. 论文写法边界",
            "",
            "- 可以写：trajectory-conditioned BC / ACT-style baseline 的训练、闭环运行、视频证据和资源记录已经打通。",
            "- 可以写：trajectory-kNN 在训练范围成功率高，但 held-out 和语言/空间泛化失败，说明它更像轨迹记忆。",
            "- 可以写：state-only ACT、ACT-CVAE 和视觉 ACT-lite 结构升级没有自动解决接触、夹紧和抬升。",
            "- 不能写：当前结果已经证明完整官方 ACT 成功。",
            "- 不能写：当前结果已经证明稳定抓取成功。",
            "- 不能写：当前结果已经是 OpenVLA、RT-2 或真实机器人 VLA 后训练结果。",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    strict_grasp = read_json(args.strict_grasp_json)
    write_csv(args.output_csv, rows)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_markdown(rows, strict_grasp), encoding="utf-8")
    print(f"trajectory_act_experiment_record_rows: {len(rows)}", flush=True)
    print(f"trajectory_act_experiment_record_csv: {args.output_csv}", flush=True)
    print(f"trajectory_act_experiment_record_md: {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
