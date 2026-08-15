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
    "object_language_action_head_lite_v1",
    "reward_weighted_action_head_lite_v1",
    "phase_conditioned_action_head_lite_v1",
    "adapter_action_head_lite_v1",
    "lora_action_head_lite_v1",
    "vision_language_action_head_lite_v1",
    "clip_action_head_lite_v1",
    "multi_task_object_action_head_lite_v1",
]


STRUCTURE_NOTES = {
    "object_language_action_head_lite_v1": "符号对象/目标特征 + 语言 token + MLP action head",
    "reward_weighted_action_head_lite_v1": "object-language 特征 + dense reward / attempt 权重的离线加权 BC",
    "phase_conditioned_action_head_lite_v1": "approach、grasp、lift、transfer、place/release 五阶段动作头",
    "adapter_action_head_lite_v1": "冻结 action-head 主干，仅训练 Adapter 残差模块",
    "lora_action_head_lite_v1": "冻结 action-head 主干，仅训练 LoRA-style 低秩残差",
    "vision_language_action_head_lite_v1": "MuJoCo RGB 统计特征 + 语言 token + 本体状态 action head",
    "clip_action_head_lite_v1": "冻结通用 CLIP 图文编码器 + 轻量 MLP action head",
    "multi_task_object_action_head_lite_v1": "blue-pad、red-pad、leftmost-to-bowl 多任务数据混合 action head",
}


PAPER_CONCLUSIONS = {
    "object_language_action_head_lite_v1": "训练范围出现少量成功，说明 action-head 代理链路可运行；held-out 和语言任务失败，不能写成真实 VLA 后训练。",
    "reward_weighted_action_head_lite_v1": "离线 reward-weighted BC 没有解决接触与阶段问题，不能写成在线 RL 或真实偏好优化。",
    "phase_conditioned_action_head_lite_v1": "显式阶段拆分带来少量 held-out 成功，但 train-range 和语言泛化仍失败，说明阶段标签本身不足。",
    "adapter_action_head_lite_v1": "约 2.1k 可训练参数的 Adapter 代理能记录 PEFT 对照，但不是 pretrained VLA Adapter。",
    "lora_action_head_lite_v1": "约 2.1k 可训练参数的 LoRA-style 代理能记录低秩后训练对照，但不是 pretrained VLA LoRA。",
    "vision_language_action_head_lite_v1": "简单 RGB 统计特征不能替代 pretrained VLM/VLA 表征。",
    "clip_action_head_lite_v1": "通用 CLIP 表征加小动作头仍无法稳定闭环操作，CLIP 不是机器人 VLA。",
    "multi_task_object_action_head_lite_v1": "naive 多任务数据混合没有带来闭环成功，后续需要更明确的任务阶段建模和真实 VLA 表征。",
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
    "stored_samples",
    "训练时间秒",
    "峰值显存MB",
    "主任务视频",
    "语言视频",
    "补充成功视频",
    "失败模式",
    "论文结论",
    "论文红线",
    "主任务viewer命令",
    "语言viewer命令",
    "训练命令",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese stage report for VLA/action-head, PEFT, CLIP, and multi-task proxy baselines.")
    parser.add_argument("--evaluation", type=Path, default=ROOT / "docs" / "evaluation_summary.csv")
    parser.add_argument("--language-summary", type=Path, default=ROOT / "docs" / "language_generalization_summary.csv")
    parser.add_argument("--resource-summary", type=Path, default=ROOT / "docs" / "model_resource_summary.csv")
    parser.add_argument("--video-evidence", type=Path, default=ROOT / "docs" / "video_evidence_index.csv")
    parser.add_argument("--failure-taxonomy", type=Path, default=ROOT / "docs" / "failure_mode_taxonomy.csv")
    parser.add_argument("--data-efficiency", type=Path, default=ROOT / "docs" / "data_efficiency_summary.csv")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "action_head_stage_report.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "action_head_stage_report.md")
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
        version = row["版本"]
        if version == "object_language_action_head_lite_v1_success_example":
            version = "object_language_action_head_lite_v1"
        grouped[version][row["失败模式"]] += 1
    return {
        version: "；".join(f"{mode} x{count}" for mode, count in counter.most_common())
        for version, counter in grouped.items()
    }


def first_video(rows: list[dict[str, str]], version: str, video_type: str) -> str:
    for row in rows:
        if row.get("版本") == version and row.get("视频类型") == video_type:
            return row.get("视频文件", "")
    return ""


def success_example_video(rows: list[dict[str, str]]) -> str:
    for row in rows:
        if row.get("版本") == "object_language_action_head_lite_v1_success_example":
            return row.get("视频文件", "")
    return ""


def build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    evaluations = by_version(read_csv(args.evaluation), "version")
    language = by_version(read_csv(args.language_summary), "version")
    resources = by_version(read_csv(args.resource_summary), "version")
    video_rows = read_csv(args.video_evidence)
    failure_modes = aggregate_failure_modes(read_csv(args.failure_taxonomy))
    object_success_example = success_example_video(video_rows)

    rows: list[dict[str, str]] = []
    for version in STAGE_VERSIONS:
        evaluation = evaluations[version]
        language_row = language.get(version, {})
        resource = resources.get(version, {})
        method = {
            "version": version,
            "stage": evaluation["stage"],
            "method": evaluation["method"],
            "artifact": evaluation["artifact"],
            "clip": evaluation["clip"],
        }
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
                "stored_samples": resource.get("stored_samples", ""),
                "训练时间秒": resource.get("train_time_seconds", ""),
                "峰值显存MB": resource.get("peak_vram_mb", ""),
                "主任务视频": evaluation["clip"],
                "语言视频": first_video(video_rows, version, "语言/空间泛化片段"),
                "补充成功视频": object_success_example if version == "object_language_action_head_lite_v1" else "",
                "失败模式": failure_modes.get(version, ""),
                "论文结论": PAPER_CONCLUSIONS[version],
                "论文红线": evaluation["note"],
                "主任务viewer命令": viewer_command(method, task="place_blue_cube_blue_pad", complexity="medium", seed=0),
                "语言viewer命令": viewer_command(method, task="move_leftmost_to_bowl", complexity="language", seed=200) if language_row else "",
                "训练命令": training_command(version),
            }
        )
    return rows


def data_efficiency_lines(rows: list[dict[str, str]]) -> list[str]:
    object_rows = [row for row in rows if row.get("method_key") == "object_action_head"]
    if not object_rows:
        return []
    lines = [
        "## 5. 数据效率补充",
        "",
        "`data_efficiency_v2` 中纳入了 `object_action_head` 小数据曲线。当前 10/25/50/92 条示范下 train-range 和 held-out 都为 0/3，说明符号 object-language action head 的小数据训练仍不稳定。",
        "",
        md_row(["示范预算", "split", "成功率", "平均目标距离"]),
        md_row(["---:", "---", "---:", "---:"]),
    ]
    for row in object_rows:
        lines.append(md_row([row["demo_budget"], row["split"], row["success"], f"{float(row['mean_target_distance']):.4f}"]))
    lines.extend(
        [
            "",
            "该补充不能写成 VLA 小数据优势，只能写成当前本地 action-head 代理尚未体现数据效率优势。",
        ]
    )
    return lines


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, str]], data_efficiency: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    train_success = sum(success_rate(row["主任务训练范围"]) for row in rows)
    heldout_success = sum(success_rate(row["主任务留出范围"]) for row in rows)
    language_success = sum(success_rate(row["语言/空间泛化"]) for row in rows if "/" in row["语言/空间泛化"])
    peft_params = [int(row["可训练参数"]) for row in rows if row["版本"] in {"adapter_action_head_lite_v1", "lora_action_head_lite_v1"}]

    lines = [
        "# Action-Head / PEFT / CLIP 阶段报告",
        "",
        "版本：`action_head_stage_report_v1`",
        "",
        "用途：把 VLA/action-head 代理、reward-weighted BC 后训练代理、阶段条件 action-head、LoRA/Adapter PEFT 代理、Frozen CLIP action-head 和多任务 action-head 集中成一份中文阶段报告。",
        "",
        "数据来源：`docs/evaluation_summary.csv`、`docs/model_resource_summary.csv`、`docs/language_generalization_summary.csv`、`docs/video_evidence_index.csv`、`docs/failure_mode_taxonomy.csv`、`docs/data_efficiency_summary.csv`。",
        "",
        "阶段展示视频：`outputs/presentation_clips/04_action_head_peft_proxy.mp4`。完整视频浏览页：`docs/video_evidence_gallery.html`。",
        "",
        "论文边界：本阶段方法都是本地 action-head / PEFT / CLIP proxy，不能写成真实 pretrained VLA 后训练，不能写成 OpenVLA/RT-2，不能写成 pretrained VLA LoRA/Adapter，CLIP 也不能写成机器人 VLA。",
        "",
        "## 1. 阶段结论",
        "",
        f"- 覆盖版本数：{len(rows)}。",
        f"- 主任务训练范围成功率总和：{train_success:.2f} / {len(rows)}；只有 `object_language_action_head_lite_v1` 出现 1/5。",
        f"- 主任务留出范围成功率总和：{heldout_success:.2f} / {len(rows)}；`phase_conditioned_action_head_lite_v1`、`adapter_action_head_lite_v1`、`lora_action_head_lite_v1` 各有 1/5 held-out 成功，但不稳定。",
        f"- 语言/空间泛化成功率总和：{language_success:.2f} / {len(rows)}，说明本地 action-head/PEFT/CLIP proxy 尚无可靠语言泛化。",
        f"- Adapter/LoRA-style 代理的可训练参数约为 {min(peft_params)}-{max(peft_params)}，能支撑“参数高效后训练代理”的资源对比，但不能宣称真实 VLA LoRA/Adapter。",
        "- 主要失败模式是未形成有效抓取/未抬升和语言/空间泛化失败；这说明轻量 head 本身不能自动解决接触控制和语言落地。",
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
            md_row(["版本", "主任务视频", "语言视频", "补充成功视频"]),
            md_row(["---", "---", "---", "---"]),
        ]
    )
    for row in rows:
        lines.append(
            md_row(
                [
                    f"`{row['版本']}`",
                    f"`{row['主任务视频']}`",
                    f"`{row['语言视频']}`" if row["语言视频"] else "未导出",
                    f"`{row['补充成功视频']}`" if row["补充成功视频"] else "",
                ]
            )
        )

    lines.extend(["", *data_efficiency_lines(data_efficiency)])

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
    data_efficiency = read_csv(args.data_efficiency)
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows, data_efficiency)
    print(f"action_head_stage_rows: {len(rows)}", flush=True)
    print(f"action_head_stage_csv: {args.output_csv}", flush=True)
    print(f"action_head_stage_md: {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
