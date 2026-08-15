from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Chinese experiment comparison report.")
    parser.add_argument("--versions", type=Path, default=ROOT / "docs" / "experiment_versions.json")
    parser.add_argument("--summary", type=Path, default=ROOT / "docs" / "evaluation_summary.csv")
    parser.add_argument("--language-summary", type=Path, default=ROOT / "docs" / "language_generalization_summary.csv")
    parser.add_argument("--data-efficiency", type=Path, default=ROOT / "docs" / "data_efficiency_summary.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "evaluation_report.md")
    return parser.parse_args()


def read_versions(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def read_summary(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return {row["version"]: row for row in csv.DictReader(file)}


def write_summary(path: Path, methods: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "version",
        "stage",
        "method",
        "artifact",
        "train_range_success",
        "heldout_success",
        "clip",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for method in methods:
            writer.writerow({key: method.get(key, "") for key in fieldnames})


def read_language_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_data_efficiency(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_video_metadata(version: str) -> dict | None:
    path = ROOT / "outputs" / "videos" / f"{version}_seed0.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def success_value(value: str) -> float | None:
    if "/" not in value:
        return None
    left, right = value.split("/", 1)
    try:
        return float(left) / float(right)
    except ValueError:
        return None


def format_rate(value: str) -> str:
    rate = success_value(value)
    if rate is None:
        return value
    return f"{value} ({rate:.0%})"


def main() -> None:
    args = parse_args()
    versions = read_versions(args.versions)
    methods = versions["methods"]
    write_summary(args.summary, methods)
    rows = read_summary(args.summary)
    language_rows = read_language_summary(args.language_summary)
    data_efficiency_rows = read_data_efficiency(args.data_efficiency)

    lines = [
        "# 实验评测对比报告",
        "",
        "本报告由 `scripts/summarize_experiments.py` 根据版本登记表、评测汇总表和视频元数据自动生成。",
        "",
        "## 任务设置",
        "",
        f"- 主任务：`{versions['task']}`",
        f"- 复杂度：`{versions['complexity']}`",
        f"- 数据版本：`{versions['dataset']['version']}`",
        f"- 数据路径：`{versions['dataset']['path']}`",
        f"- 示范成功率：{versions['dataset']['successes']}/{versions['dataset']['episodes']} = {versions['dataset']['success_rate']:.2f}",
        "",
        "## 方法结果总表",
        "",
        "| 版本名 | 阶段 | 方法 | 训练范围 | 留出范围 | seed0 视频结果 | 展示片段 |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]

    for method in methods:
        row = rows.get(method["version"], {})
        video_meta = read_video_metadata(method["version"])
        if video_meta is None:
            video_result = "missing"
        else:
            summary = video_meta["summary"]
            video_result = str(summary.get("success", summary.get("steps_replayed", "recorded")))
        lines.append(
            "| "
            f"`{method['version']}` | "
            f"{method['stage']} | "
            f"{method['method']} | "
            f"{format_rate(row.get('train_range_success', method['train_range_success']))} | "
            f"{format_rate(row.get('heldout_success', method['heldout_success']))} | "
            f"{video_result} | "
            f"`{method['clip']}` |"
        )

    if language_rows:
        lines.extend(
            [
                "",
                "## 语言/任务泛化评测",
                "",
                "任务：`move_leftmost_to_bowl`，复杂度：`language`，seeds：`200-204`。",
                "",
                "| 方法 | 阶段 | 成功率 | 平均目标距离 |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for row in language_rows:
            lines.append(
                "| "
                f"`{row['version']}` | "
                f"{row['stage']} | "
                f"{format_rate(row['success'])} | "
                f"{float(row['mean_target_distance']):.4f} |"
            )

    if data_efficiency_rows:
        lines.extend(
            [
                "",
                "## 数据效率评测",
                "",
                "快速评测：预算 `10,25,50,92` 条成功示范；方法为 `knn_bc` 和 `trajectory_knn`；每个条件 3 个训练范围 seed 与 3 个留出 seed。",
                "",
                "| 方法 | 示范预算 | 范围 | 成功率 | 平均目标距离 |",
                "| --- | ---: | --- | ---: | ---: |",
            ]
        )
        for row in data_efficiency_rows:
            lines.append(
                "| "
                f"`{row['method_key']}` | "
                f"{row['demo_budget']} | "
                f"{row['split']} | "
                f"{format_rate(row['success'])} | "
                f"{float(row['mean_target_distance']):.4f} |"
            )

    lines.extend(
        [
            "",
            "## 阶段说明",
            "",
            "1. Scripted expert 和 replay 阶段证明任务、环境和示范数据链路可用。",
            "2. Linear BC、MLP BC、Action-Chunk BC、Trajectory-conditioned ACT-lite v2、PyTorch State Transformer ACT、PyTorch State ACT-CVAE-lite、Visual-Feature ACT-lite 和 Diffusion Policy-lite 均未在当前任务上形成稳定抓取，说明简单低维状态回归、小型状态 Transformer、CVAE latent、pooled RGB 视觉代理与轻量动作块建模仍不够。",
            "3. kNN BC 在训练范围 seed 上成功，但 held-out 表现明显下降，说明它更偏记忆相近轨迹而不是泛化策略。",
            "4. Object-Language Action Head-lite 在训练范围有少量成功，但 held-out 仍失败，说明符号对象条件 action head 还不能替代真实 VLM/VLA 表征。",
            "5. PyTorch State ACT-CVAE-lite 加入 latent posterior 后离线 zero-latent 误差下降，但闭环仍没有稳定抬升，说明 state-only ACT-CVAE 仍不足。",
            "6. Visual-Feature ACT-lite 使用 MuJoCo RGB pooled features 后仍没有稳定抬升，说明手工视觉代理输入不能替代完整视觉 ACT。",
            "7. Phase-Conditioned Action Head-lite 将动作头按阶段拆分，离线误差下降但闭环仍不稳定，说明显式阶段标签本身还不足以解决接触控制。",
            "8. Frozen CLIP Action Head-lite 使用 pretrained CLIP 表征但闭环仍失败，说明通用 VLM encoder 加小动作头还不能直接解决机器人阶段和接触控制。",
            "9. Multi-task Object-Language Action Head-lite 在加入多任务示范后仍闭环失败，说明 naive 数据混合和单一 MLP action head 不足。",
            "10. 当前结果支持下一阶段转向端到端视觉 ACT、更多多任务数据，或机器人预训练 VLA 表征加轻量 action head/Adapter/LoRA。",
            "",
            "## 展示建议",
            "",
            "- 用 `expert_scripted_v1_seed0.mp4` 展示任务理想执行过程。",
            "- 用 `replay_demo_v1_seed0.mp4` 展示采集轨迹可复现。",
            "- 用 `knn_bc_v1_seed0.mp4` 展示强记忆型 baseline 在训练范围内能成功。",
            "- 用 `linear_bc_v1_seed0.mp4`、`mlp_bc_v1_seed0.mp4`、`act_lite_chunk_bc_v1_seed0.mp4`、`trajectory_conditioned_chunk_bc_v2_seed0.mp4`、`diffusion_policy_lite_v1_seed0.mp4` 对比普通学习方法失败现象。",
            "- 用 `torch_act_cvae_state_chunk_v1_seed0.mp4` 展示加入 CVAE latent 后 state-only ACT 仍没有稳定抓取。",
            "- 用 `visual_feature_act_lite_v1_seed0.mp4` 展示加入 pooled RGB 视觉代理后 ACT-lite 仍没有稳定抓取。",
            "- 用 `object_language_action_head_lite_v1_seed0.mp4` 展示 action-head 代理基线的 seed0 失败，用 `object_language_action_head_lite_v1_seed1_success_example.mp4` 展示其局部成功样例。",
            "- 用 `phase_conditioned_action_head_lite_v1_seed0.mp4` 展示显式阶段拆分后仍未形成稳定抓取。",
            "- 用 `clip_action_head_lite_v1_seed0.mp4` 展示 frozen CLIP 表征加轻量 action head 的失败现象。",
            "- 用 `multi_task_object_action_head_lite_v1_seed0.mp4` 和 `multi_task_object_action_head_lite_v1_language_seed400.mp4` 展示 naive 多任务 action-head 失败现象。",
            "",
            "## 注意事项",
            "",
            "- `Diffusion Policy-lite` 是 NumPy DDPM 风格 baseline，不是官方完整 PyTorch Diffusion Policy。",
            "- `PyTorch State Transformer ACT` 是 state-only lightweight ACT-style baseline，不是视觉输入完整 ACT。",
            "- `PyTorch State ACT-CVAE-lite` 含 CVAE latent，但仍是 state-only baseline，不是视觉输入完整 ACT。",
            "- `Visual-Feature ACT-lite` 使用 pooled RGB 视觉代理特征，不是端到端 CNN/Transformer 视觉 ACT。",
            "- `Object-Language Action Head-lite` 是符号特征 action-head 代理基线，不是 pretrained VLM/VLA。",
            "- `Phase-Conditioned Action Head-lite` 是本地阶段条件 action-head 代理，不是层级 VLA 或真实任务规划器。",
            "- `Frozen CLIP Action Head-lite` 使用 pretrained CLIP，但 CLIP 不是机器人 VLA，不能写成 OpenVLA 后训练。",
            "- 当前评测任务只覆盖 `place_blue_cube_blue_pad`，后续语言泛化和多任务适配仍需新增数据和评测。",
            "- PowerShell 可能按本地编码显示中文乱码；Markdown 文件本身按 UTF-8 保存，VS Code 中应正常显示。",
            "",
        ]
    )

    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"report_path: {args.output}", flush=True)
    print(f"methods: {len(methods)}", flush=True)


if __name__ == "__main__":
    main()
