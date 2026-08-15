from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


STAGE_TITLES = {
    "scripted_oracle": "任务 oracle / 示范生成",
    "structured_control_baseline": "结构化控制强基线",
    "data_verification": "示范数据验证",
    "weak_bc_baseline": "普通 BC baseline",
    "non_neural_baseline": "普通 BC baseline",
    "neural_bc_baseline": "普通 BC baseline",
    "trajectory_conditioned_baseline": "Trajectory / ACT-lite baseline",
    "trajectory_memory_baseline": "Trajectory / ACT-lite baseline",
    "torch_act_baseline": "PyTorch ACT-style baseline",
    "torch_act_cvae_baseline": "PyTorch ACT-CVAE-lite baseline",
    "visual_feature_act_baseline": "视觉特征 ACT-lite baseline",
    "diffusion_policy_baseline": "Diffusion Policy-lite baseline",
    "torch_diffusion_policy_baseline": "PyTorch Diffusion Policy baseline",
    "vla_action_head_proxy": "VLA/action-head 代理",
    "pretrained_vlm_action_head_proxy": "pretrained VLM 表征代理",
    "phase_conditioned_action_head_proxy": "阶段条件 action-head 代理",
    "reward_weighted_bc_post_training": "reward-weighted BC 后训练代理",
    "peft_action_head_proxy": "Adapter/LoRA-style PEFT 代理",
    "multi_task_action_head_proxy": "多任务 action-head 代理",
}


KEY_MAIN_VERSIONS = [
    "expert_scripted_v1",
    "structured_waypoint_policy_v1",
    "linear_bc_v1",
    "knn_bc_v1",
    "mlp_bc_v1",
    "trajectory_knn_chunk_bc_v1",
    "torch_act_state_chunk_v1",
    "phase_conditioned_torch_act_v1",
    "torch_act_cvae_state_chunk_v1",
    "torch_diffusion_policy_state_chunk_v1",
    "object_language_action_head_lite_v1",
    "adapter_action_head_lite_v1",
    "lora_action_head_lite_v1",
    "clip_action_head_lite_v1",
    "multi_task_object_action_head_lite_v1",
]


KEY_RESOURCE_VERSIONS = [
    "linear_bc_v1",
    "mlp_bc_v1",
    "torch_act_state_chunk_cuda_v1",
    "phase_conditioned_torch_act_v1",
    "torch_act_cvae_state_chunk_v1",
    "torch_diffusion_policy_state_chunk_v1",
    "visual_feature_act_lite_v1",
    "object_language_action_head_lite_v1",
    "adapter_action_head_lite_v1",
    "lora_action_head_lite_v1",
    "clip_action_head_lite_v1",
]


LANGUAGE_ALIASES = {
    "expert_scripted_v1": "expert_scripted_language_v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese thesis results chapter draft from current experiment artifacts.")
    parser.add_argument("--versions", type=Path, default=ROOT / "docs" / "experiment_versions.json")
    parser.add_argument("--summary", type=Path, default=ROOT / "docs" / "evaluation_summary.csv")
    parser.add_argument("--language-summary", type=Path, default=ROOT / "docs" / "language_generalization_summary.csv")
    parser.add_argument("--resource-summary", type=Path, default=ROOT / "docs" / "model_resource_summary.csv")
    parser.add_argument("--data-efficiency", type=Path, default=ROOT / "docs" / "data_efficiency_summary.csv")
    parser.add_argument("--domain-randomization", type=Path, default=ROOT / "docs" / "domain_randomization_summary.csv")
    parser.add_argument("--strict-grasp-audit", type=Path, default=ROOT / "docs" / "strict_grasp_success_audit.csv")
    parser.add_argument("--method-audit", type=Path, default=ROOT / "docs" / "method_stage_audit.md")
    parser.add_argument("--presentation-pack", type=Path, default=ROOT / "docs" / "presentation_video_pack.md")
    parser.add_argument("--video-gallery", type=Path, default=ROOT / "docs" / "video_evidence_gallery.html")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "thesis_results_chapter_draft.md")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as file:
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


def number_text(value: str, decimals: int = 0) -> str:
    if value in ("", None):
        return "未记录"
    try:
        number = float(value)
    except ValueError:
        return str(value)
    if decimals == 0:
        return f"{int(round(number)):,}"
    return f"{number:,.{decimals}f}"


def md_escape(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(md_escape(value) for value in values) + " |"


def row_value(row: dict[str, str] | None, key: str, default: str = "未记录") -> str:
    if not row:
        return default
    value = row.get(key, "")
    return value if value not in ("", None) else default


def language_row_for(version: str, language_rows: dict[str, dict[str, str]]) -> dict[str, str] | None:
    return language_rows.get(version) or language_rows.get(LANGUAGE_ALIASES.get(version, ""))


def method_lookup(versions: dict) -> dict[str, dict]:
    return {method["version"]: method for method in versions["methods"]}


def stage_overview(methods: list[dict]) -> list[str]:
    grouped: dict[str, list[str]] = {}
    for method in methods:
        grouped.setdefault(STAGE_TITLES.get(method["stage"], method["stage"]), []).append(method["version"])
    lines = [
        md_row(["阶段", "版本数量", "代表版本", "论文用途"]),
        md_row(["---", "---:", "---", "---"]),
    ]
    for stage, versions in grouped.items():
        representative = ", ".join(f"`{version}`" for version in versions[:4])
        if len(versions) > 4:
            representative += " ..."
        purpose = "建立对照、解释失败模式或提供后续 VLA 后训练比较基线"
        if "oracle" in stage or "结构化" in stage:
            purpose = "证明任务可行性，并提供非学习上界"
        if "代理" in stage:
            purpose = "建立 action-head / VLM / PEFT 轻量后训练代理对照"
        lines.append(md_row([stage, str(len(versions)), representative, purpose]))
    return lines


def main_results_table(methods: dict[str, dict], summary: dict[str, dict[str, str]], language: dict[str, dict[str, str]], resources: dict[str, dict[str, str]]) -> list[str]:
    lines = [
        md_row(["版本", "方法", "阶段", "train-range", "held-out", "language", "可训练参数", "固定视频"]),
        md_row(["---", "---", "---", "---:", "---:", "---:", "---:", "---"]),
    ]
    for version in KEY_MAIN_VERSIONS:
        method = methods.get(version)
        if not method:
            continue
        main = summary.get(version, method)
        lang = language_row_for(version, language)
        resource = resources.get(version, {})
        lines.append(
            md_row(
                [
                    f"`{version}`",
                    method["method"],
                    STAGE_TITLES.get(method["stage"], method["stage"]),
                    rate_text(row_value(main, "train_range_success", method.get("train_range_success", ""))),
                    rate_text(row_value(main, "heldout_success", method.get("heldout_success", ""))),
                    rate_text(row_value(lang, "success", "未评测")),
                    number_text(row_value(resource, "trainable_params", "0")),
                    f"`{method['clip']}`",
                ]
            )
        )
    return lines


def language_table(language_rows: list[dict[str, str]]) -> list[str]:
    important = [
        "expert_scripted_language_v1",
        "structured_waypoint_policy_v1",
        "linear_bc_v1",
        "knn_bc_v1",
        "trajectory_knn_chunk_bc_v1",
        "torch_act_state_chunk_cuda_v1",
        "torch_diffusion_policy_state_chunk_v1",
        "object_language_action_head_lite_v1",
        "clip_action_head_lite_v1",
        "multi_task_object_action_head_lite_v1",
    ]
    by = by_version(language_rows)
    lines = [
        md_row(["版本", "阶段", "成功率", "平均目标距离", "seeds"]),
        md_row(["---", "---", "---:", "---:", "---"]),
    ]
    for version in important:
        row = by.get(version)
        if not row:
            continue
        lines.append(
            md_row(
                [
                    f"`{version}`",
                    row_value(row, "stage"),
                    rate_text(row_value(row, "success")),
                    number_text(row_value(row, "mean_target_distance"), 4),
                    row_value(row, "seeds"),
                ]
            )
        )
    return lines


def resource_table(resources: dict[str, dict[str, str]]) -> list[str]:
    lines = [
        md_row(["版本", "可训练参数", "存储样本", "特征维度", "训练时间(s)", "峰值显存(MB)", "说明"]),
        md_row(["---", "---:", "---:", "---:", "---:", "---:", "---"]),
    ]
    for version in KEY_RESOURCE_VERSIONS:
        row = resources.get(version)
        if not row:
            continue
        note = ""
        if version in {"adapter_action_head_lite_v1", "lora_action_head_lite_v1"}:
            note = "参数高效 proxy"
        elif version == "clip_action_head_lite_v1":
            note = "冻结 CLIP，只训练 action head"
        elif "torch" in version:
            note = "PyTorch 动作块 baseline"
        lines.append(
            md_row(
                [
                    f"`{version}`",
                    number_text(row_value(row, "trainable_params", "0")),
                    number_text(row_value(row, "stored_samples")),
                    number_text(row_value(row, "feature_dim")),
                    number_text(row_value(row, "train_time_seconds"), 2),
                    number_text(row_value(row, "peak_vram_mb"), 2),
                    note,
                ]
            )
        )
    return lines


def data_efficiency_table(rows: list[dict[str, str]]) -> list[str]:
    keep = [row for row in rows if row.get("demo_budget") in {"10", "92"}]
    keep.sort(key=lambda row: (row.get("method_key", ""), row.get("split", ""), int(row.get("demo_budget", "0"))))
    lines = [
        md_row(["方法", "示范预算", "范围", "成功率", "平均目标距离", "存储样本"]),
        md_row(["---", "---:", "---", "---:", "---:", "---:"]),
    ]
    for row in keep:
        lines.append(
            md_row(
                [
                    f"`{row_value(row, 'method_key')}`",
                    row_value(row, "demo_budget"),
                    row_value(row, "split"),
                    rate_text(row_value(row, "success")),
                    number_text(row_value(row, "mean_target_distance"), 4),
                    number_text(row_value(row, "stored_samples")),
                ]
            )
        )
    return lines


def domain_randomization_table(rows: list[dict[str, str]]) -> list[str]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["method_key"], row["method_version"], row["domain"])
        grouped.setdefault(key, []).append(row)

    lines = [
        md_row(["方法", "版本", "扰动域", "成功率", "平均目标距离", "摩擦", "arm kp/force", "gripper kp/force"]),
        md_row(["---", "---", "---", "---:", "---:", "---:", "---", "---"]),
    ]
    for (method, version, domain), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][2])):
        success = sum(1 for item in items if item.get("success") == "True")
        total = len(items)
        mean_distance = sum(float(item["target_distance"]) for item in items) / total
        first = items[0]
        lines.append(
            md_row(
                [
                    method,
                    f"`{version}`",
                    domain,
                    f"{success}/{total}",
                    f"{mean_distance:.4f}",
                    first["friction"],
                    f"{first['arm_kp']} / {first['arm_force']}",
                    f"{first['gripper_kp']} / {first['gripper_force']}",
                ]
            )
        )
    return lines


def strict_grasp_summary(rows: list[dict[str, str]]) -> dict[str, int]:
    summary = {
        "rows": len(rows),
        "episodes": 0,
        "loose_successes": 0,
        "strict_successes": 0,
        "loose_without_grasp_rows": 0,
    }
    for row in rows:
        episodes = int(row["episodes"])
        loose = int(row["loose_successes"])
        strict = int(row["strict_grasp_successes"])
        summary["episodes"] += episodes
        summary["loose_successes"] += loose
        summary["strict_successes"] += strict
        summary["loose_without_grasp_rows"] += int(loose > 0 and strict == 0)
    return summary


def success_summary(summary: dict[str, dict[str, str]]) -> tuple[str, str]:
    best_train = []
    best_held = []
    for version, row in summary.items():
        train_rate = rate(row.get("train_range_success", ""))
        held_rate = rate(row.get("heldout_success", ""))
        if train_rate is not None:
            best_train.append((train_rate, version, row.get("train_range_success", "")))
        if held_rate is not None:
            best_held.append((held_rate, version, row.get("heldout_success", "")))
    best_train.sort(reverse=True)
    best_held.sort(reverse=True)
    train_text = ", ".join(f"`{version}` {success}" for _rate, version, success in best_train[:3])
    held_text = ", ".join(f"`{version}` {success}" for _rate, version, success in best_held[:3])
    return train_text, held_text


def write_doc(args: argparse.Namespace) -> None:
    versions = read_json(args.versions)
    methods = method_lookup(versions)
    methods_list = versions["methods"]
    summary = by_version(read_csv(args.summary))
    language_rows = read_csv(args.language_summary)
    language = by_version(language_rows)
    resources = by_version(read_csv(args.resource_summary))
    data_efficiency = read_csv(args.data_efficiency)
    domain_randomization = read_csv(args.domain_randomization)
    strict_grasp = strict_grasp_summary(read_csv(args.strict_grasp_audit))
    best_train, best_held = success_summary(summary)

    lines = [
        "# 论文结果章节草稿",
        "",
        "版本：`thesis_results_chapter_draft_v1`",
        "",
        "用途：把当前 MuJoCo WidowX 桌面操作实验整理成毕业论文“实验与结果分析”章节草稿。本文档由 `scripts/build_thesis_results_chapter.py` 自动生成，可直接作为论文写作底稿，再按学校模板调整编号、图题和格式。",
        "",
        "## 5.1 实验设置",
        "",
        f"本文在 MuJoCo 中构建 WidowX 桌面机械臂操作任务，主任务为 `{versions['task']}`，复杂度为 `{versions['complexity']}`。桌面环境包含目标物体、干扰物体和目标放置区域，用于观察不同策略在接触、夹取、移动和释放阶段的稳定性。",
        "",
        f"主示范数据集为 `{versions['dataset']['path']}`，共采集 `{versions['dataset']['episodes']}` 条 demonstration，其中成功 `{versions['dataset']['successes']}` 条，成功率为 `{versions['dataset']['success_rate']:.0%}`。所有策略统一保存版本名、模型 artifact、主任务闭环结果、语言泛化结果、资源统计和固定视频片段。",
        "",
        "评价指标包括：主任务训练范围成功率、主任务留出 seed 成功率、语言/空间泛化成功率、平均目标距离、可训练参数量、训练时间、峰值显存和固定 rollout 视频。主任务对比使用训练范围 seeds `0-4` 与留出 seeds `100-104`；语言泛化任务使用 `move_leftmost_to_bowl / language`，seeds `200-204`。",
        "",
        "## 5.2 方法分组",
        "",
        *stage_overview(methods_list),
        "",
        "## 5.3 主任务闭环结果",
        "",
        f"当前正式登记方法共有 `{len(methods_list)}` 个。训练范围表现最好的方法包括：{best_train}。留出范围表现最好的方法包括：{best_held}。整体上，结构化控制与轨迹记忆方法在局部范围内表现更好，但 learned/action-head proxy 尚未形成稳定泛化。",
        "",
        *main_results_table(methods, summary, language, resources),
        "",
        "### 5.3.1 严格抓取/抬升口径补充审计",
        "",
        f"由于桌面放置任务中的原始 `success` 可以由目标距离达标触发，本文额外引入 `strict_grasp_success_audit_v1`，同时检查 `success`、`grasp_success` 和 `object_z`。该审计汇总 `control_safety_sweep_v1`、`action_head_control_safety_sweep_v1` 与候选诊断视频，共 `{strict_grasp['rows']}` 行、`{strict_grasp['episodes']}` 个 rollout/episode 单元；其中原始放置成功为 `{strict_grasp['loose_successes']}/{strict_grasp['episodes']}`，严格抓取成功为 `{strict_grasp['strict_successes']}/{strict_grasp['episodes']}`，存在“放置成功但抓取失败”的行数为 `{strict_grasp['loose_without_grasp_rows']}`。",
        "",
        "因此，在论文表述中，trajectory-kNN、ACT 或候选后训练样例如果只满足目标距离指标，而 `grasp_success=False` 且 `object_z` 未超过抬升阈值，只能写成“目标距离指标达标但未稳定抓取/抬升”，不能写成稳定抓取成功。对应证据文件为 `docs/strict_grasp_success_audit.md`、`docs/strict_grasp_success_audit.csv` 和 `outputs/evaluations/strict_grasp_success_audit_v1.json`。",
        "",
        "结果表明，`linear_bc_v1` 与 `mlp_bc_v1` 的单步动作回归不能稳定完成闭环抓取；`knn_bc_v1` 与 `trajectory_knn_chunk_bc_v1` 在训练范围内可以成功，但留出范围下降明显，说明它们更接近轨迹记忆。`torch_act_state_chunk_v1`、`phase_conditioned_torch_act_v1`、`torch_act_cvae_state_chunk_v1` 和 `torch_diffusion_policy_state_chunk_v1` 更接近动作块策略结构，但在当前 state-only、小规模数据条件下仍未稳定完成接触和抬升；其中阶段 one-hot 版本闭环仍为 `0/5`，说明显式阶段条件本身不能解决夹紧与抬升。",
        "",
        "## 5.4 语言/空间泛化结果",
        "",
        "语言泛化任务为 `move_leftmost_to_bowl`，要求策略根据场景关系选择最左侧物体并移动到碗中。该测试用于区分显式结构化控制、普通 BC、轨迹记忆和 VLA/action-head proxy 是否具备空间语言泛化能力。",
        "",
        *language_table(language_rows),
        "",
        "结果显示，`expert_scripted_language_v1` 和 `structured_waypoint_policy_v1` 可达到 `4/5`，说明任务本身可执行；普通 learned baseline、ACT/Diffusion baseline、CLIP/action-head proxy 和多任务 action-head 当前均为 `0/5`。因此本阶段不能声称模型已经学到真实语言理解，只能说明语言泛化评测链路已经建立。",
        "",
        "## 5.5 数据效率结果",
        "",
        "为回答小规模示范数据下是否省数据，项目在 10、25、50、92 条成功示范预算下评测 kNN BC、trajectory-kNN 和 object-language action head。下表保留 10 条与 92 条两个端点，用于论文主体展示；完整表见 `docs/data_efficiency_summary.csv`。",
        "",
        *data_efficiency_table(data_efficiency),
        "",
        "数据效率结果说明，kNN 和 trajectory-kNN 随示范数量增加在训练范围内改善明显，但留出范围仍弱；object-language action head 在这些预算下没有稳定成功。现阶段不能证明轻量 action-head proxy 已经省数据，只能证明评测模板和对照组已建立。",
        "",
        "## 5.6 算力与参数效率",
        "",
        "轻量化路线的关键问题之一是可训练参数、训练时间和显存是否可控。下表选取普通 BC、ACT/Diffusion、视觉代理和 PEFT proxy 的代表版本。",
        "",
        *resource_table(resources),
        "",
        "`adapter_action_head_lite_v1` 和 `lora_action_head_lite_v1` 的可训练参数约为 2.1k，明显少于 106k 级 object-language action head；`clip_action_head_lite_v1` 冻结 CLIP encoder，只训练 action head。资源事实支持“参数更少”的论点，但由于闭环成功率没有同步提升，论文中应避免写成“轻量化必然更高效”。",
        "",
        "## 5.7 MuJoCo Domain Randomization 代理评测",
        "",
        "由于本机当前没有可用的 Isaac/Isaac Sim/IsaacGym，本阶段先在 MuJoCo 中做 domain randomization 代理评测。评测扰动包括桌面接触摩擦、机械臂执行器增益、执行器力限和夹爪力度，用于观察策略对简单动力学偏差和接触不稳定的敏感性。该评测版本为 `domain_randomization_eval_v1`，不能写成高保真 Isaac domain randomization，也不能写成真实机械臂迁移验证。后续 `isaac_domain_randomization_handoff_v1` 和 `real_widowx_validation_handoff_v1` 已经把 Isaac 复现实验字段、真实机械臂安全门禁、50 条 trial 模板和论文红线固定下来，但它们仍然不是 Isaac 或真实机械臂运行结果。",
        "",
        *domain_randomization_table(domain_randomization),
        "",
        "结果显示，`structured_waypoint_policy_v1` 在 nominal、低摩擦弱夹爪和高摩擦强执行器三种域下均为 `2/2`；`trajectory_knn_chunk_bc_v1` 在低摩擦弱夹爪域下降为 `1/2`；`visual_act_cnn_cvae_v1` 在三种域均为 `0/2`。这说明结构化强对照对当前扰动更稳，轨迹记忆 baseline 对接触条件变化敏感，而小型视觉 ACT-CNN-CVAE-lite 在当前数据规模下仍没有形成可靠闭环控制。",
        "",
        "对应报告和视频证据为 `docs/domain_randomization_summary.md`、`outputs/presentation_clips/06_domain_randomization_proxy.mp4`、`outputs/videos/domain_randomization_structured_low_friction_seed0.mp4`、`outputs/videos/domain_randomization_trajectory_knn_low_friction_seed0.mp4` 和 `outputs/videos/domain_randomization_visual_act_cnn_cvae_low_friction_seed0.mp4`。",
        "",
        "## 5.8 可视化证据",
        "",
        "本项目为每个正式方法保存固定 rollout 视频，并额外生成答辩阶段短视频包。论文或答辩中建议使用以下材料：",
        "",
        "- 单方法视频索引：`docs/video_clips.md`",
        "- 视频证据浏览页：`docs/video_evidence_gallery.html`",
        "- 全量宫格视频：`outputs/showcase/all_registered_methods_grid.mp4`",
        "- 语言泛化宫格：`outputs/showcase/language_generalization_grid.mp4`",
        "- 答辩总览 reel：`outputs/presentation_clips/00_defense_video_reel.mp4`",
        "- Domain randomization 阶段短片：`outputs/presentation_clips/06_domain_randomization_proxy.mp4`",
        "- 阶段视频包说明：`docs/presentation_video_pack.md`",
        "- OpenVLA 数据桥接报告：`docs/openvla_dataset_bridge_report.md`",
        "- OpenVLA 本地可行性检查：`docs/openvla_feasibility_report.md`",
        "- Robot VLA action-head 交接门禁：`docs/robot_vla_action_head_handoff.md`",
        "- Robot VLA 远端运行包：`docs/robot_vla_remote_run_pack.md`",
        "- Robot VLA 远端结果回填门禁：`docs/robot_vla_remote_result_intake.md`",
        "- Isaac domain randomization 交接门禁：`docs/isaac_domain_randomization_handoff.md`",
        "- 真实 WidowX 验证交接门禁：`docs/real_widowx_validation_handoff.md`",
        "- 外部依赖阶段 readiness audit：`docs/external_dependency_readiness_audit.md`",
        "",
        "这些视频只作为定性证据和讲解材料，定量结论仍以 CSV 评测表和 `docs/method_stage_audit.md` 为准。`external_dependency_readiness_audit_v1` 只用于说明真实 robot VLA、Isaac 和真实 WidowX planned 版本当前的阻塞条件、回填文件和论文边界，不是策略成功率结果。",
        "",
        "## 5.9 阶段性结论与边界",
        "",
        "第一，MuJoCo WidowX 桌面操作、示范采集、轨迹回放、闭环评测和视频导出链路已经打通。第二，普通 BC、ACT-lite、state-only ACT、阶段条件 state-only ACT、state-only ACT-CVAE、state-only diffusion action-chunk 和当前 action-head proxy 都还不能稳定超过结构化强基线。第三，kNN/trajectory-kNN 的训练范围成功主要体现轨迹检索能力，不代表真实泛化。第四，MuJoCo domain randomization 代理评测显示接触和动力学扰动会放大轨迹记忆与视觉 ACT-lite 的弱点，但它不能替代 Isaac 或真实机械臂验证。第五，CLIP/action-head 和 Adapter/LoRA-style proxy 证明了轻量后训练实验框架，OpenVLA bridge、feasibility、robot VLA action-head handoff、远端运行包和结果回填门禁已经把真实 VLA 接入的输入、远端运行和回填门禁明确下来，但还不能写成真实 pretrained VLA/OpenVLA 后训练。第六，Isaac 和真实 WidowX 的运行交接门禁已经完成，可作为后续外部验证模板，但不能写成实际运行结果。第七，`external_dependency_readiness_audit_v1` 已把 15 个外部依赖/候选阶段统一标为当前不可直接进入正式方法统计，其中 `formal_method_allowed_now` 全部为 `否`。",
        "",
        "论文表述红线：`structured_waypoint_policy_v1` 是显式状态控制，不是 learned VLA；`torch_act_*` 是 state-only ACT-style baseline，不能写成完整视觉 ACT；`torch_diffusion_policy_state_chunk_v1` 是 state-only diffusion baseline，不能写成完整视觉 Diffusion Policy；`clip_action_head_lite_v1` 使用通用 CLIP，不是机器人 VLA；`adapter_action_head_lite_v1` 和 `lora_action_head_lite_v1` 是本地 PEFT proxy，不是 pretrained VLA LoRA/Adapter；`robot_vla_action_head_handoff_v1`、`robot_vla_remote_run_pack_v1` 和 `robot_vla_remote_result_intake_v1` 是运行/回填门禁，不是 `robot_vla_action_head_lite_v1` 策略结果；`external_dependency_readiness_audit_v1` 是外部依赖门禁审计，不是策略成功率结果；`domain_randomization_eval_v1` 是 MuJoCo 代理评测，不是 Isaac domain randomization 或真实 WidowX 迁移验证；`isaac_domain_randomization_handoff_v1` 和 `real_widowx_validation_handoff_v1` 是交接门禁，不是实际外部验证结果。",
        "",
        "## 5.10 后续工作",
        "",
        "下一阶段应在保持当前评测、资源统计和视频导出接口不变的前提下，先复用 `openvla_dataset_bridge_v1` 的 `image + instruction + state + action` 字段，并依据 `openvla_feasibility_check_v1`、`robot_vla_action_head_handoff_v1`、`robot_vla_remote_run_pack_v1`、`robot_vla_remote_result_intake_v1` 和 `external_dependency_readiness_audit_v1` 的结论把真实机器人预训练 VLA/OpenVLA 训练迁移到 48GB+ GPU 或云端环境，再比较 action head only、Adapter 和 LoRA 的参数量、训练时间、显存和闭环成功率；随后按 `isaac_domain_randomization_handoff_v1` 在 Isaac/Isaac Sim 可用时复现当前 domain randomization 字段，最后按 `real_widowx_validation_handoff_v1` 执行真实 WidowX 20-50 次小规模验证。",
        "",
        "## 重新生成命令",
        "",
        "```powershell",
        f'& "{ROOT / ".venv" / "Scripts" / "python.exe"}" "{ROOT / "scripts" / "build_thesis_results_chapter.py"}"',
        "```",
        "",
        "## 关联材料",
        "",
        f"- 方法阶段审计：`{args.method_audit.relative_to(ROOT)}`",
        f"- 答辩视频片段包：`{args.presentation_pack.relative_to(ROOT)}`",
        "- 结果矩阵：`docs/result_matrix.md`",
        "- 方法卡片：`docs/method_cards.md`",
        "- Domain randomization 代理评测：`docs/domain_randomization_summary.md`",
        "- OpenVLA 数据桥接：`docs/openvla_dataset_bridge_report.md`",
        "- OpenVLA 本地可行性检查：`docs/openvla_feasibility_report.md`",
        "- Robot VLA action-head 交接门禁：`docs/robot_vla_action_head_handoff.md`",
        "- Robot VLA 远端运行包：`docs/robot_vla_remote_run_pack.md`",
        "- Robot VLA 远端结果回填门禁：`docs/robot_vla_remote_result_intake.md`",
        "- Isaac domain randomization 交接门禁：`docs/isaac_domain_randomization_handoff.md`",
        "- 真实 WidowX 验证交接门禁：`docs/real_widowx_validation_handoff.md`",
        "- 外部依赖阶段 readiness audit：`docs/external_dependency_readiness_audit.md`",
        f"- 视频证据浏览页：`{args.video_gallery.relative_to(ROOT)}`",
        "- 总交付入口：`docs/final_experiment_package.md`",
        "",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    write_doc(args)
    print(f"thesis_results_chapter: {args.output}", flush=True)


if __name__ == "__main__":
    main()
