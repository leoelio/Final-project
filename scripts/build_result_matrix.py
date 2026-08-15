from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


STAGE_GROUPS = [
    (
        "1. 任务、环境与数据链路",
        "证明 MuJoCo WidowX 桌面任务、示范采集、轨迹回放和结构化强对照可用。",
        {
            "scripted_oracle",
            "data_verification",
            "structured_control_baseline",
        },
    ),
    (
        "2. 普通 imitation learning baseline",
        "回答普通 BC / kNN / MLP 在小规模示范下的闭环表现。",
        {
            "weak_bc_baseline",
            "non_neural_baseline",
            "neural_bc_baseline",
        },
    ),
    (
        "3. 动作块、Trajectory、ACT 与 Diffusion baseline",
        "回答序列动作建模、轨迹历史和扩散动作块是否能改善闭环控制。",
        {
            "trajectory_conditioned_baseline",
            "trajectory_memory_baseline",
            "torch_act_baseline",
            "torch_act_cvae_baseline",
            "visual_feature_act_baseline",
            "visual_act_cnn_cvae_baseline",
            "diffusion_policy_baseline",
            "torch_diffusion_policy_baseline",
        },
    ),
    (
        "4. 轻量 VLA / action-head 代理",
        "回答符号对象条件、RGB 视觉代理、pretrained CLIP 表征和多任务 action head 是否足够。",
        {
            "vla_action_head_proxy",
            "phase_conditioned_action_head_proxy",
            "pretrained_vlm_action_head_proxy",
            "multi_task_action_head_proxy",
        },
    ),
    (
        "5. 参数高效与偏好/奖励加权后训练代理",
        "回答轻量后训练、Adapter-style、LoRA-style 和 reward-weighted BC 是否能以少量参数改善策略。",
        {
            "peft_action_head_proxy",
            "reward_weighted_bc_post_training",
        },
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese staged result matrix for thesis/report writing.")
    parser.add_argument("--versions", type=Path, default=ROOT / "docs" / "experiment_versions.json")
    parser.add_argument("--summary", type=Path, default=ROOT / "docs" / "evaluation_summary.csv")
    parser.add_argument("--language-summary", type=Path, default=ROOT / "docs" / "language_generalization_summary.csv")
    parser.add_argument("--resources", type=Path, default=ROOT / "docs" / "model_resource_summary.csv")
    parser.add_argument("--data-efficiency", type=Path, default=ROOT / "docs" / "data_efficiency_summary.csv")
    parser.add_argument("--video-evidence", type=Path, default=ROOT / "docs" / "video_evidence_index.csv")
    parser.add_argument("--showcase", type=Path, default=ROOT / "outputs" / "showcase" / "video_showcase_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "result_matrix.md")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def by_version(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["version"]: row for row in rows if row.get("version")}


def rate(value: str) -> float | None:
    if not value or "/" not in value:
        return None
    left, right = value.split("/", 1)
    try:
        denominator = float(right)
        return float(left) / denominator if denominator else None
    except ValueError:
        return None


def rate_text(value: str) -> str:
    parsed = rate(value)
    if parsed is None:
        return value or "未记录"
    return f"{value} ({parsed:.0%})"


def number_text(value: str) -> str:
    if value in ("", None):
        return "未记录"
    try:
        return f"{int(float(value)):,}"
    except ValueError:
        return str(value)


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(values) + " |"


def language_for(version: str, language_rows: dict[str, dict[str, str]]) -> str:
    language_aliases = {
        "expert_scripted_v1": "expert_scripted_language_v1",
    }
    row = language_rows.get(version) or language_rows.get(language_aliases.get(version, ""))
    if not row:
        return "未评测"
    return rate_text(row["success"])


def best_rows(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    candidates = [row for row in rows if row.get("split") == split]
    return sorted(candidates, key=lambda row: (row.get("method_key", ""), int(row.get("demo_budget", "0"))))


def write_matrix(args: argparse.Namespace) -> None:
    versions = read_json(args.versions)
    methods = versions["methods"]
    summary = by_version(read_csv(args.summary))
    language = by_version(read_csv(args.language_summary))
    resources = by_version(read_csv(args.resources))
    data_efficiency = read_csv(args.data_efficiency)
    video_rows = read_csv(args.video_evidence)
    showcase = read_json(args.showcase) if args.showcase.exists() else {"showcases": []}

    lines = [
        "# 阶段结果矩阵",
        "",
        "版本：`result_matrix_v1`",
        "",
        "用途：按毕业设计研究流程，把当前所有已登记方法的版本名、阶段定位、主任务闭环结果、语言泛化结果、资源规模和视频片段整理到同一张中文矩阵中。该文档由 `scripts/build_result_matrix.py` 自动生成。",
        "",
        "## 总体状态",
        "",
        f"- 已登记正式方法：`{len(methods)}`",
        f"- 主任务：`{versions['task']}` / `{versions['complexity']}`",
        f"- 主数据集：`{versions['dataset']['path']}`",
        f"- 视频证据：`{len(video_rows)}` 条，固定视频和展示短片仅作为定性证据，成功率仍以 CSV/JSON 评测表为准。",
        "- 已覆盖：环境与示范、普通 BC、trajectory/ACT/Diffusion、action-head/VLM 代理、PEFT 代理、reward-weighted BC 后训练代理。",
        "- 未完成：Isaac 高保真/domain randomization、真实机械臂验证、真实机器人 VLA/OpenVLA 后训练。",
        "",
        "## 分阶段方法矩阵",
        "",
    ]

    used_versions: set[str] = set()
    for title, description, stages in STAGE_GROUPS:
        lines.extend([f"### {title}", "", description, ""])
        lines.append(md_row(["版本", "方法", "主任务 train", "主任务 held-out", "语言泛化", "可训练参数", "固定视频", "结论用途"]))
        lines.append(md_row(["---", "---", "---:", "---:", "---:", "---:", "---", "---"]))
        for method in methods:
            if method["stage"] not in stages:
                continue
            used_versions.add(method["version"])
            row = summary.get(method["version"], method)
            resource = resources.get(method["version"], {})
            lines.append(
                md_row(
                    [
                        f"`{method['version']}`",
                        method["method"],
                        rate_text(row.get("train_range_success", method.get("train_range_success", ""))),
                        rate_text(row.get("heldout_success", method.get("heldout_success", ""))),
                        language_for(method["version"], language),
                        number_text(resource.get("trainable_params", "")),
                        f"`{method['clip']}`",
                        method["note"],
                    ]
                )
            )
        lines.append("")

    remaining = [method for method in methods if method["version"] not in used_versions]
    if remaining:
        lines.extend(["### 其它已登记版本", ""])
        lines.append(md_row(["版本", "阶段", "方法", "固定视频"]))
        lines.append(md_row(["---", "---", "---", "---"]))
        for method in remaining:
            lines.append(md_row([f"`{method['version']}`", method["stage"], method["method"], f"`{method['clip']}`"]))
        lines.append("")

    lines.extend(
        [
            "## 研究问题对应证据",
            "",
            "### Q1：轻量化后训练是否省算力？",
            "",
            "- `Adapter Action Head-lite` 和 `LoRA-style Action Head-lite` 可训练参数约 2.1k，显著小于普通 action-head 的 106k 级别。",
        "- `Frozen CLIP Action Head-lite` 冻结 151M 级 CLIP encoder，只训练 302,599 个 action-head 参数；这可作为 pretrained VLM 表征代理的资源对照。",
        "- `PyTorch State ACT-CVAE-lite` 在 state-only ACT 上加入 16 维 latent posterior，记录 zero-latent 离线误差、KL 和参数量，用于对比更接近标准 ACT 的结构成本。",
        "- `Visual-Feature ACT-lite` 加入 MuJoCo RGB pooled features、语言 token 和本体状态，记录视觉代理输入下的动作块学习成本。",
        "- `Phase-Conditioned Action Head-lite` 将单一动作头拆成 5 个阶段头，参数量仍在轻量 MLP 范围内，用于检验阶段条件是否能缓解动作冲突。",
        "- 当前所有 learned/action-head 代理闭环仍不稳定，因此“省参数”已成立为资源事实，但“省参数且效果更好”尚未成立。",
            "",
            "### Q2：轻量化后训练是否省数据？",
            "",
            "数据效率快速评测来自 `docs/data_efficiency_summary.csv`。关键行：",
            "",
            md_row(["方法", "预算", "范围", "成功率", "平均目标距离"]),
            md_row(["---", "---:", "---", "---:", "---:"]),
        ]
    )
    for row in best_rows(data_efficiency, "train_range") + best_rows(data_efficiency, "heldout"):
        if row.get("demo_budget") not in {"10", "92"}:
            continue
        lines.append(
            md_row(
                [
                    f"`{row['method_key']}`",
                    row["demo_budget"],
                    row["split"],
                    rate_text(row["success"]),
                    f"{float(row['mean_target_distance']):.4f}",
                ]
            )
        )

    lines.extend(
        [
            "",
            "结论：`kNN` 和 `trajectory-kNN` 在训练范围随数据增加表现更好，但 held-out 仍弱；当前 action-head 小数据曲线没有形成稳定优势。",
            "",
            "### Q3：VLA/action-head 是否更理解语言？",
            "",
            "语言任务为 `move_leftmost_to_bowl / language`，seeds `200-204`。当前只有规则/结构化策略能完成，learned/action-head/VLM 代理均为 0/5。这说明当前模型还没有学出空间语言泛化能力。",
            "",
            "### Q4：仿真到真实能否迁移？",
            "",
            "当前证据只覆盖 MuJoCo。真实机械臂和 Isaac/domain randomization 还没有做，论文中不能写 sim-to-real 已验证；只能写“已建立 MuJoCo 可视化、数据、评测和视频证据链”。",
            "",
            "## 视频展示矩阵",
            "",
            "| 展示集合 | 文件 | 用途 |",
            "| --- | --- | --- |",
        ]
    )
    showcase_note = {
        "core": "核心方法快速对比",
        "registered": "全部正式登记方法并排展示",
        "language": "语言/空间泛化任务对比",
    }
    for item in showcase.get("showcases", []):
        lines.append(
            md_row(
                [
                    f"`{item['preset']}`",
                    f"`{item['output']}`",
                    showcase_note.get(item["preset"], "展示视频"),
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 论文表述边界",
            "",
            "- `structured_waypoint_policy_v1` 是显式状态和阶段控制，不是 learned VLA。",
            "- `act_lite_chunk_bc_v1`、`trajectory_conditioned_chunk_bc_v2` 是轻量 ACT-like 动作块 baseline，不是完整 ACT。",
            "- `torch_act_state_chunk_v1` 是 state-only ACT-style baseline，不含视觉 encoder/CVAE。",
            "- `phase_conditioned_torch_act_v1` 是 state-only ACT-style baseline 加离散阶段 one-hot，不是完整视觉 ACT 或层级 VLA。",
            "- `torch_act_cvae_state_chunk_v1` 是 state-only ACT-CVAE-lite baseline，含 CVAE latent，但不含视觉 encoder，仍不能写成完整视觉 ACT。",
            "- `visual_feature_act_lite_v1` 使用 pooled RGB 视觉代理特征，不是端到端 CNN/Transformer 视觉 ACT。",
            "- `diffusion_policy_lite_v1` 是 NumPy DDPM 风格 baseline，不是官方完整 Diffusion Policy。",
            "- `torch_diffusion_policy_state_chunk_v1` 是 state-only PyTorch diffusion action-chunk baseline，不含视觉 encoder，不能写成完整视觉 Diffusion Policy。",
            "- `reward_weighted_action_head_lite_v1` 是 reward-weighted BC，不是在线 RL。",
            "- `phase_conditioned_action_head_lite_v1` 是显式阶段条件 action-head 代理，不是层级 VLA 或真实任务规划器。",
            "- `adapter_action_head_lite_v1` 和 `lora_action_head_lite_v1` 是本地 PEFT 代理，不是真实 pretrained VLA LoRA/Adapter。",
            "- `clip_action_head_lite_v1` 使用 pretrained CLIP，但 CLIP 不是机器人 VLA，不能写成 OpenVLA 后训练。",
            "",
            "## 推荐后续实验",
            "",
            "1. 接入机器人动作数据预训练 VLA/OpenVLA 类表征，保留当前 action-head/Adapter/LoRA 评测模板。",
            "2. 若继续 preference 路线，做轨迹级 preference ranking 或 trajectory selection，而不是只做样本加权。",
            "3. 扩展 Isaac/domain randomization 和真实机械臂 20-50 次小规模验证。",
            "",
            "## 重新生成命令",
            "",
            "```powershell",
            f'& "{ROOT / ".venv" / "Scripts" / "python.exe"}" "{ROOT / "scripts" / "build_result_matrix.py"}"',
            "```",
            "",
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    write_matrix(args)
    print(f"result_matrix_path: {args.output}", flush=True)


if __name__ == "__main__":
    main()
