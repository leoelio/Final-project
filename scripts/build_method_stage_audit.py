from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


STAGE_META = {
    "scripted_oracle": {
        "group": "环境与示范",
        "nature": "规则 oracle",
        "vision": "无学习视觉输入",
        "language": "任务参数由脚本显式给出",
        "training": "不训练",
        "claim": "可作为任务可行性和示范生成上界",
        "red_line": "不能写成 learned policy 或 VLA 泛化能力",
    },
    "data_verification": {
        "group": "数据验证",
        "nature": "轨迹回放",
        "vision": "无学习视觉输入",
        "language": "不涉及语言理解",
        "training": "不训练",
        "claim": "可证明保存的动作轨迹能复现 MuJoCo 执行过程",
        "red_line": "不能作为策略学习方法对比",
    },
    "structured_control_baseline": {
        "group": "结构化控制强基线",
        "nature": "显式状态控制",
        "vision": "直接使用目标状态",
        "language": "规则解析任务目标",
        "training": "不训练或仅保存参数",
        "claim": "可作为任务可解性和阶段化控制上界",
        "red_line": "不能写成 learned VLA 或语言泛化模型",
    },
    "weak_bc_baseline": {
        "group": "普通模仿学习",
        "nature": "线性 BC",
        "vision": "低维状态",
        "language": "无语言泛化",
        "training": "单步 MSE 行为克隆",
        "claim": "可作为最弱普通 BC 对照",
        "red_line": "不能用离线 MSE 替代闭环成功率",
    },
    "non_neural_baseline": {
        "group": "普通模仿学习",
        "nature": "非参数 kNN",
        "vision": "低维状态",
        "language": "无语言泛化",
        "training": "示范样本检索",
        "claim": "可说明训练范围记忆效果",
        "red_line": "训练范围成功不能写成泛化能力",
    },
    "neural_bc_baseline": {
        "group": "普通模仿学习",
        "nature": "MLP BC",
        "vision": "低维状态",
        "language": "无语言泛化",
        "training": "单步 MSE 行为克隆",
        "claim": "可作为标准神经网络 BC 对照",
        "red_line": "不能写成序列策略或 VLA",
    },
    "trajectory_conditioned_baseline": {
        "group": "动作块 / 轨迹条件",
        "nature": "ACT-lite 动作块",
        "vision": "低维状态历史",
        "language": "无真实语言理解",
        "training": "历史窗口到动作块的监督学习",
        "claim": "可作为 trajectory-conditioned / ACT-like baseline",
        "red_line": "不能写成完整 ACT",
    },
    "trajectory_memory_baseline": {
        "group": "动作块 / 轨迹条件",
        "nature": "轨迹窗口 kNN",
        "vision": "低维状态历史",
        "language": "无真实语言理解",
        "training": "历史轨迹检索",
        "claim": "可说明动作块记忆与泛化差距",
        "red_line": "训练范围成功不能写成策略泛化",
    },
    "torch_act_baseline": {
        "group": "ACT-style",
        "nature": "PyTorch Transformer ACT-style",
        "vision": "state-only",
        "language": "无真实语言理解",
        "training": "Transformer 动作块监督学习",
        "claim": "可作为更接近 ACT 的 state-only baseline",
        "red_line": "不能写成完整视觉 ACT",
    },
    "torch_act_cvae_baseline": {
        "group": "ACT-style",
        "nature": "ACT-CVAE-lite",
        "vision": "state-only",
        "language": "无真实语言理解",
        "training": "Transformer + CVAE latent 动作块监督学习",
        "claim": "可说明加入 CVAE latent 后的资源和闭环效果",
        "red_line": "不能写成完整视觉 ACT 或机器人 ACT 复现",
    },
    "visual_feature_act_baseline": {
        "group": "ACT-style",
        "nature": "视觉特征 ACT-lite",
        "vision": "MuJoCo RGB pooled features",
        "language": "任务 token",
        "training": "视觉代理特征 + Transformer 动作块",
        "claim": "可作为视觉代理输入的 ACT-lite 对照",
        "red_line": "不能写成端到端 CNN/Transformer 视觉 ACT",
    },
    "visual_act_cnn_cvae_baseline": {
        "group": "ACT-style",
        "nature": "CNN visual ACT-CVAE-lite",
        "vision": "MuJoCo RGB + 小型 CNN encoder",
        "language": "任务 token",
        "training": "CNN 视觉编码 + Transformer ACT-CVAE 动作块",
        "claim": "可作为更接近完整视觉 ACT 的本地轻量对照",
        "red_line": "不能写成官方完整 ACT 或真实机器人视觉 ACT 复现",
    },
    "diffusion_policy_baseline": {
        "group": "Diffusion Policy",
        "nature": "NumPy DDPM-lite",
        "vision": "低维状态",
        "language": "无真实语言理解",
        "training": "扩散式动作块代理训练",
        "claim": "可作为 diffusion action-chunk 思路的轻量对照",
        "red_line": "不能写成完整官方 Diffusion Policy",
    },
    "torch_diffusion_policy_baseline": {
        "group": "Diffusion Policy",
        "nature": "PyTorch state diffusion",
        "vision": "state-only",
        "language": "无真实语言理解",
        "training": "Transformer 条件去噪动作块",
        "claim": "可作为 state-only PyTorch Diffusion Policy baseline",
        "red_line": "不能写成完整视觉 Diffusion Policy",
    },
    "vla_action_head_proxy": {
        "group": "VLA/action-head 代理",
        "nature": "轻量 action head",
        "vision": "符号对象或 RGB 统计代理",
        "language": "任务 token / 符号语言条件",
        "training": "冻结特征 + MLP action head",
        "claim": "可作为本地 VLA action-head proxy",
        "red_line": "不能写成 pretrained VLA 后训练",
    },
    "pretrained_vlm_action_head_proxy": {
        "group": "VLM 表征代理",
        "nature": "Frozen CLIP action head",
        "vision": "pretrained CLIP image embedding",
        "language": "pretrained CLIP text embedding",
        "training": "冻结 CLIP + 轻量 MLP action head",
        "claim": "可作为已有 VLM 表征 + action head 的本地代理",
        "red_line": "CLIP 不是机器人 VLA，不能写成 OpenVLA/RT-2 后训练",
    },
    "phase_conditioned_action_head_proxy": {
        "group": "VLA/action-head 代理",
        "nature": "阶段条件 action head",
        "vision": "符号对象代理",
        "language": "任务 token / 阶段条件",
        "training": "五阶段 MLP action head",
        "claim": "可说明显式阶段条件的增益和局限",
        "red_line": "不能写成学出了完整任务规划",
    },
    "reward_weighted_bc_post_training": {
        "group": "轻量后训练代理",
        "nature": "reward-weighted BC",
        "vision": "符号对象代理",
        "language": "任务 token",
        "training": "按 attempt 偏好和 dense reward 加权",
        "claim": "可作为偏好/奖励加权后训练代理",
        "red_line": "不能写成在线 RL 或真实 preference optimization",
    },
    "peft_action_head_proxy": {
        "group": "参数高效后训练代理",
        "nature": "Adapter / LoRA-style 残差",
        "vision": "符号对象代理",
        "language": "任务 token",
        "training": "冻结主干，仅训练小残差模块",
        "claim": "可对比 action head only / Adapter / LoRA-style 参数量",
        "red_line": "不能写成 pretrained VLA LoRA/Adapter",
    },
    "multi_task_action_head_proxy": {
        "group": "多任务 action-head 代理",
        "nature": "naive 多任务 MLP action head",
        "vision": "符号对象代理",
        "language": "任务 token",
        "training": "混合多个任务数据源监督训练",
        "claim": "可说明 naive 多任务混合不等于语言泛化",
        "red_line": "不能写成真正多任务 VLA 泛化",
    },
}


LANGUAGE_ALIASES = {
    "expert_scripted_v1": "expert_scripted_language_v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese audit table for experiment stages and thesis claims.")
    parser.add_argument("--versions", type=Path, default=ROOT / "docs" / "experiment_versions.json")
    parser.add_argument("--summary", type=Path, default=ROOT / "docs" / "evaluation_summary.csv")
    parser.add_argument("--language-summary", type=Path, default=ROOT / "docs" / "language_generalization_summary.csv")
    parser.add_argument("--resources", type=Path, default=ROOT / "docs" / "model_resource_summary.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "method_stage_audit.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "method_stage_audit.csv")
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


def success_rate_text(value: str) -> str:
    if not value or "/" not in value:
        return value or "未记录"
    left, right = value.split("/", 1)
    try:
        rate = float(left) / float(right)
    except (ValueError, ZeroDivisionError):
        return value
    return f"{value} ({rate:.0%})"


def field(row: dict[str, str], key: str, default: str = "未记录") -> str:
    value = row.get(key, "")
    return value if value not in ("", None) else default


def md_escape(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(md_escape(value) for value in values) + " |"


def merged_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    versions = read_json(args.versions)
    summary = by_version(read_csv(args.summary))
    language = by_version(read_csv(args.language_summary))
    resources = by_version(read_csv(args.resources))
    rows = []
    for item in versions["methods"]:
        version = item["version"]
        stage = item["stage"]
        meta = STAGE_META.get(stage, {})
        main = summary.get(version, item)
        lang = language.get(version) or language.get(LANGUAGE_ALIASES.get(version, ""), {})
        res = resources.get(version, {})
        rows.append(
            {
                "版本": version,
                "阶段": stage,
                "阶段分组": field(meta, "group", stage),
                "方法": item["method"],
                "方法性质": field(meta, "nature"),
                "视觉/状态输入": field(meta, "vision"),
                "语言输入": field(meta, "language"),
                "训练方式": field(meta, "training"),
                "主任务训练范围": success_rate_text(field(main, "train_range_success", field(item, "train_range_success"))),
                "主任务留出范围": success_rate_text(field(main, "heldout_success", field(item, "heldout_success"))),
                "语言泛化": success_rate_text(field(lang, "success", "未评测")),
                "可训练参数": field(res, "trainable_params", "0"),
                "训练时间秒": field(res, "train_time_seconds"),
                "峰值显存MB": field(res, "peak_vram_mb"),
                "artifact": item["artifact"],
                "固定视频": item["clip"],
                "论文可写": field(meta, "claim"),
                "论文红线": field(meta, "red_line"),
                "当前结论": item.get("note", ""),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, str]]) -> None:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["阶段分组"], []).append(row)

    lines = [
        "# 方法阶段审计表",
        "",
        "版本：`method_stage_audit_v1`",
        "",
        "用途：把当前所有正式实验版本按阶段、输入类型、训练方式、可写结论和论文红线统一审计。后续写毕业论文或答辩 PPT 时，优先以本表限制表述范围，避免把本地 proxy 写成真实 VLA/OpenVLA/ACT/Diffusion Policy。",
        "",
        "数据来源：",
        "",
        "- `docs/experiment_versions.json`",
        "- `docs/evaluation_summary.csv`",
        "- `docs/language_generalization_summary.csv`",
        "- `docs/model_resource_summary.csv`",
        "- `outputs/videos/*.mp4`",
        "",
        "## 总体边界",
        "",
        "- 已完成：MuJoCo WidowX 桌面任务、示范采集/回放、普通 BC、trajectory-conditioned BC、ACT-style、Diffusion-style、action-head、CLIP/VLM proxy、Adapter/LoRA-style proxy、reward-weighted BC proxy 的统一登记和可视化对比。",
        "- 尚未完成：真实 OpenVLA/RT-2/机器人 VLA 后训练、Isaac domain randomization、真实 WidowX 机械臂验证。",
        "- 论文主张应聚焦：在当前有限数据和算力下，简单 BC/ACT/Diffusion/action-head proxy 的闭环成功率不足，结构化阶段控制和轨迹记忆提供了有用对照，下一步需要真正机器人预训练表征与更强阶段/接触建模。",
        "",
    ]

    compact_columns = [
        "版本",
        "方法",
        "方法性质",
        "视觉/状态输入",
        "语言输入",
        "训练方式",
        "主任务训练范围",
        "主任务留出范围",
        "语言泛化",
        "可训练参数",
        "论文红线",
    ]
    for group, group_rows in groups.items():
        lines.extend([f"## {group}", ""])
        lines.append(md_row(compact_columns))
        lines.append(md_row(["---"] * len(compact_columns)))
        for row in group_rows:
            values = [f"`{row[col]}`" if col == "版本" else row[col] for col in compact_columns]
            lines.append(md_row(values))
        lines.append("")

    lines.extend(
        [
            "## 视频证据索引",
            "",
            md_row(["版本", "固定视频", "artifact", "论文可写"]),
            md_row(["---", "---", "---", "---"]),
        ]
    )
    for row in rows:
        lines.append(md_row([f"`{row['版本']}`", f"`{row['固定视频']}`", f"`{row['artifact']}`", row["论文可写"]]))

    lines.extend(
        [
            "",
            "## 重新生成命令",
            "",
            "```powershell",
            f'& "{ROOT / ".venv" / "Scripts" / "python.exe"}" "{ROOT / "scripts" / "build_method_stage_audit.py"}"',
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = merged_rows(args)
    if not rows:
        raise RuntimeError("no methods found")
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows)
    print(f"method_stage_audit_md: {args.output_md}", flush=True)
    print(f"method_stage_audit_csv: {args.output_csv}", flush=True)
    print(f"methods: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
