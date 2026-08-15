from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_experiment_command_index import training_command, viewer_command  # noqa: E402


STAGE_VERSIONS = [
    "act_lite_chunk_bc_v1",
    "diffusion_policy_lite_v1",
    "torch_diffusion_policy_state_chunk_v1",
    "trajectory_conditioned_chunk_bc_v2",
    "trajectory_knn_chunk_bc_v1",
    "torch_act_state_chunk_v1",
    "torch_act_state_chunk_cuda_v1",
    "phase_conditioned_torch_act_v1",
    "torch_act_cvae_state_chunk_v1",
    "visual_feature_act_lite_v1",
    "visual_act_cnn_cvae_v1",
]


STRUCTURE_NOTES = {
    "act_lite_chunk_bc_v1": "轻量 MLP action-chunk / ACT-lite",
    "diffusion_policy_lite_v1": "NumPy DDPM 风格动作块扩散 baseline",
    "torch_diffusion_policy_state_chunk_v1": "PyTorch state-only 条件扩散动作块",
    "trajectory_conditioned_chunk_bc_v2": "8 帧历史状态条件 + 8 步动作块",
    "trajectory_knn_chunk_bc_v1": "8 帧历史状态检索 + kNN 动作块记忆",
    "torch_act_state_chunk_v1": "state-only Transformer encoder/decoder ACT-style",
    "torch_act_state_chunk_cuda_v1": "同结构 CUDA 训练资源对照",
    "phase_conditioned_torch_act_v1": "state-only Transformer ACT + 离散阶段 one-hot",
    "torch_act_cvae_state_chunk_v1": "state-only Transformer ACT + CVAE latent",
    "visual_feature_act_lite_v1": "MuJoCo RGB pooled features + 语言 token + 本体状态",
    "visual_act_cnn_cvae_v1": "小型 CNN RGB encoder + ACT-CVAE 动作块",
}


PAPER_CONCLUSIONS = {
    "act_lite_chunk_bc_v1": "短动作块回归不能稳定完成闭环抓取，适合作为普通 action-chunk baseline。",
    "diffusion_policy_lite_v1": "NumPy-lite 扩散动作块仍无法解决接触与抬升，不能代表完整 Diffusion Policy。",
    "torch_diffusion_policy_state_chunk_v1": "升级到 PyTorch state-only 条件扩散后闭环仍失败，说明缺少视觉/任务结构时扩散建模不足。",
    "trajectory_conditioned_chunk_bc_v2": "加入历史轨迹让动作更平滑，但仍主要失败在接触、夹紧和抬升阶段。",
    "trajectory_knn_chunk_bc_v1": "训练范围成功率高，但留出和语言任务失败，说明它更像轨迹记忆而不是泛化策略。",
    "torch_act_state_chunk_v1": "state-only Transformer ACT-style 有个别训练范围成功，但总体仍不稳定。",
    "torch_act_state_chunk_cuda_v1": "CUDA 训练改善的是运行资源记录，不代表策略效果提升。",
    "phase_conditioned_torch_act_v1": "加入离散阶段 one-hot 后闭环仍失败，说明阶段条件本身不能解决接触、夹紧和抬升控制。",
    "torch_act_cvae_state_chunk_v1": "CVAE latent 降低离线重构误差，但没有转化为稳定闭环成功。",
    "visual_feature_act_lite_v1": "pooled RGB 视觉代理不等价于完整 CNN/Transformer 视觉 ACT。",
    "visual_act_cnn_cvae_v1": "小型 CNN + ACT-CVAE 更接近视觉 ACT 形态，但当前数据规模下仍不能稳定抬升。",
}


FIELDNAMES = [
    "版本",
    "阶段",
    "方法",
    "结构定位",
    "artifact",
    "主任务训练范围",
    "主任务留出范围",
    "语言/空间泛化",
    "可训练参数",
    "模型大小MB",
    "feature_dim",
    "history",
    "horizon",
    "训练时间秒",
    "峰值显存MB",
    "主任务视频",
    "语言视频",
    "失败模式",
    "论文结论",
    "论文红线",
    "主任务viewer命令",
    "语言viewer命令",
    "训练命令",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese stage report for trajectory-conditioned BC / ACT / Diffusion baselines.")
    parser.add_argument("--versions", type=Path, default=ROOT / "docs" / "experiment_versions.json")
    parser.add_argument("--evaluation", type=Path, default=ROOT / "docs" / "evaluation_summary.csv")
    parser.add_argument("--language-summary", type=Path, default=ROOT / "docs" / "language_generalization_summary.csv")
    parser.add_argument("--resource-summary", type=Path, default=ROOT / "docs" / "model_resource_summary.csv")
    parser.add_argument("--video-evidence", type=Path, default=ROOT / "docs" / "video_evidence_index.csv")
    parser.add_argument("--failure-taxonomy", type=Path, default=ROOT / "docs" / "failure_mode_taxonomy.csv")
    parser.add_argument("--domain-randomization", type=Path, default=ROOT / "docs" / "domain_randomization_summary.csv")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "trajectory_act_stage_report.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "trajectory_act_stage_report.md")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def by_version(rows: list[dict[str, str]], key: str = "version") -> dict[str, dict[str, str]]:
    return {row[key]: row for row in rows}


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", "<br>") for value in values) + " |"


def success_rate(text: str) -> float:
    if "/" not in text:
        return 0.0
    left, right = text.split("/", 1)
    try:
        return float(left) / max(1.0, float(right))
    except ValueError:
        return 0.0


def aggregate_failure_modes(rows: list[dict[str, str]]) -> dict[str, str]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[row["版本"]][row["失败模式"]] += 1
    return {
        version: "；".join(f"{mode} x{count}" for mode, count in counter.most_common())
        for version, counter in grouped.items()
    }


def first_video(rows: list[dict[str, str]], version: str, video_type: str) -> str:
    for row in rows:
        if row.get("版本") == version and row.get("视频类型") == video_type:
            return row.get("视频文件", "")
    return ""


def build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    evaluations = by_version(read_csv(args.evaluation), "version")
    language = by_version(read_csv(args.language_summary), "version")
    resources = by_version(read_csv(args.resource_summary), "version")
    video_rows = read_csv(args.video_evidence)
    failure_modes = aggregate_failure_modes(read_csv(args.failure_taxonomy))

    rows: list[dict[str, str]] = []
    for version in STAGE_VERSIONS:
        evaluation = evaluations[version]
        resource = resources.get(version, {})
        language_row = language.get(version, {})
        method = {
            "version": version,
            "stage": evaluation["stage"],
            "method": evaluation["method"],
            "artifact": evaluation["artifact"],
            "clip": evaluation["clip"],
        }
        main_viewer = viewer_command(method, task="place_blue_cube_blue_pad", complexity="medium", seed=0)
        language_viewer = viewer_command(method, task="move_leftmost_to_bowl", complexity="language", seed=200) if language_row else ""
        train = training_command(version)
        rows.append(
            {
                "版本": version,
                "阶段": evaluation["stage"],
                "方法": evaluation["method"],
                "结构定位": STRUCTURE_NOTES[version],
                "artifact": evaluation["artifact"],
                "主任务训练范围": evaluation["train_range_success"],
                "主任务留出范围": evaluation["heldout_success"],
                "语言/空间泛化": language_row.get("success", "未登记"),
                "可训练参数": resource.get("trainable_params", ""),
                "模型大小MB": resource.get("artifact_size_mb", ""),
                "feature_dim": resource.get("feature_dim", ""),
                "history": resource.get("history", ""),
                "horizon": resource.get("horizon", ""),
                "训练时间秒": resource.get("train_time_seconds", ""),
                "峰值显存MB": resource.get("peak_vram_mb", ""),
                "主任务视频": evaluation["clip"],
                "语言视频": first_video(video_rows, version, "语言/空间泛化片段"),
                "失败模式": failure_modes.get(version, ""),
                "论文结论": PAPER_CONCLUSIONS[version],
                "论文红线": evaluation["note"],
                "主任务viewer命令": main_viewer,
                "语言viewer命令": language_viewer,
                "训练命令": train,
            }
        )
    return rows


def domain_randomization_note(rows: list[dict[str, str]]) -> list[str]:
    stage_rows = [row for row in rows if row.get("method_version") in {"trajectory_knn_chunk_bc_v1", "visual_act_cnn_cvae_v1"}]
    by_method_domain: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in stage_rows:
        by_method_domain[(row["method_version"], row["domain"])].append(row)

    lines = [
        "## 5. 扰动域补充证据",
        "",
        "`domain_randomization_eval_v1` 中只把 `trajectory_knn_chunk_bc_v1` 和 `visual_act_cnn_cvae_v1` 纳入本阶段扰动对照。该部分是 MuJoCo 代理评测，不能写成 Isaac 或真实机器人验证。",
        "",
        md_row(["方法版本", "扰动域", "成功率", "平均目标距离"]),
        md_row(["---", "---", "---:", "---:"]),
    ]
    for (version, domain), items in sorted(by_method_domain.items()):
        successes = sum(1 for item in items if item.get("success") == "True")
        distances = [float(item["target_distance"]) for item in items]
        lines.append(md_row([f"`{version}`", domain, f"{successes}/{len(items)}", f"{sum(distances) / len(distances):.4f}"]))
    lines.extend(
        [
            "",
            "补充视频：",
            "",
            "```text",
            "outputs/videos/domain_randomization_trajectory_knn_low_friction_seed0.mp4",
            "outputs/videos/domain_randomization_visual_act_cnn_cvae_low_friction_seed0.mp4",
            "outputs/presentation_clips/06_domain_randomization_proxy.mp4",
            "```",
        ]
    )
    return lines


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, str]], domain_rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    train_success = sum(success_rate(row["主任务训练范围"]) for row in rows)
    heldout_success = sum(success_rate(row["主任务留出范围"]) for row in rows)
    language_success = sum(success_rate(row["语言/空间泛化"]) for row in rows if "/" in row["语言/空间泛化"])

    lines = [
        "# Trajectory / ACT / Diffusion 阶段报告",
        "",
        "版本：`trajectory_act_stage_report_v1`",
        "",
        "用途：把动作块、轨迹条件、ACT-style、ACT-CVAE、视觉 ACT-lite 和 Diffusion-lite 这些可靠对照组集中成一份中文阶段报告，方便后续论文撰写、答辩讲解和可视化复现。",
        "",
        "数据来源：`docs/evaluation_summary.csv`、`docs/model_resource_summary.csv`、`docs/language_generalization_summary.csv`、`docs/video_evidence_index.csv`、`docs/failure_mode_taxonomy.csv`。",
        "",
        "阶段展示视频：`outputs/presentation_clips/03_trajectory_act_diffusion.mp4`。完整视频浏览页：`docs/video_evidence_gallery.html`。",
        "",
        "论文边界：本阶段方法都是本地轻量 baseline 或代理实现，不能写成完整官方 ACT，不能写成完整视觉 Diffusion Policy，也不能写成 OpenVLA/RT-2 后训练或真实机器人验证。",
        "",
        "## 1. 阶段结论",
        "",
        f"- 覆盖版本数：{len(rows)}。",
        f"- 主任务训练范围成功率总和：{train_success:.2f} / {len(rows)} 个方法等价满分；真正明显成功的是 `trajectory_knn_chunk_bc_v1`，`torch_act_state_chunk_v1` 只有少量成功。",
        f"- 主任务留出范围成功率总和：{heldout_success:.2f} / {len(rows)}，说明这一阶段尚不能作为泛化策略。",
        f"- 语言/空间泛化成功率总和：{language_success:.2f} / {len(rows)}，说明单任务轨迹/ACT/Diffusion baseline 基本没有语言泛化能力。",
        "- 主要失败模式是未形成有效抓取/未抬升，其次是语言/空间泛化失败；这说明问题集中在闭环接触、夹紧、抬升和任务条件化，而不是单纯动作速度过快。",
        "",
        "## 2. 方法对比表",
        "",
        md_row(["版本", "结构定位", "Train", "Held-out", "Language", "参数", "大小MB", "失败模式"]),
        md_row(["---", "---", "---:", "---:", "---:", "---:", "---:", "---"]),
    ]
    for row in rows:
        lines.append(
            md_row(
                [
                    f"`{row['版本']}`",
                    row["结构定位"],
                    row["主任务训练范围"],
                    row["主任务留出范围"],
                    row["语言/空间泛化"],
                    row["可训练参数"],
                    row["模型大小MB"],
                    row["失败模式"],
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 3. 论文可写结论",
            "",
            md_row(["版本", "可写结论", "论文红线"]),
            md_row(["---", "---", "---"]),
        ]
    )
    for row in rows:
        lines.append(md_row([f"`{row['版本']}`", row["论文结论"], row["论文红线"]]))

    lines.extend(
        [
            "",
            "## 4. 视频证据入口",
            "",
            md_row(["版本", "主任务视频", "语言视频"]),
            md_row(["---", "---", "---"]),
        ]
    )
    for row in rows:
        lines.append(md_row([f"`{row['版本']}`", f"`{row['主任务视频']}`", f"`{row['语言视频']}`" if row["语言视频"] else "未导出"]))

    lines.extend(["", *domain_randomization_note(domain_rows)])

    lines.extend(
        [
            "",
            "## 6. 主任务慢速 Viewer 命令",
            "",
            "以下命令都会打开 MuJoCo viewer，统一使用 `--duration 60 --speed 0.05`。",
            "",
        ]
    )
    for row in rows:
        lines.extend([f"### `{row['版本']}`", "", "```powershell", row["主任务viewer命令"], "```", ""])

    lines.extend(["## 7. 语言/空间泛化慢速 Viewer 命令", ""])
    for row in rows:
        if row["语言viewer命令"]:
            lines.extend([f"### `{row['版本']}`", "", "```powershell", row["语言viewer命令"], "```", ""])

    lines.extend(["## 8. 训练/重建命令", ""])
    for row in rows:
        if row["训练命令"]:
            lines.extend([f"### `{row['版本']}`", "", "```powershell", row["训练命令"], "```", ""])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    domain_rows = read_csv(args.domain_randomization)
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows, domain_rows)
    print(f"trajectory_act_stage_rows: {len(rows)}", flush=True)
    print(f"trajectory_act_stage_csv: {args.output_csv}", flush=True)
    print(f"trajectory_act_stage_md: {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
