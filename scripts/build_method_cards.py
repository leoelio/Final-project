from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "method_cards.md"


INTERPRETATIONS = {
    "expert_scripted_v1": "规则 oracle，证明任务链路、IK、抓取接触和示范数据生成可用；不应作为 learned policy 对比中的可学习方法。",
    "structured_waypoint_policy_v1": "强结构化控制 baseline，显式访问目标物和目标区域状态。它说明任务本身可解，也说明 learned 方法需要学出阶段结构和接触控制才可能接近上界。",
    "replay_demo_v1": "数据复现检查，证明保存的动作轨迹能在 MuJoCo 中复现执行过程；不是策略模型。",
    "linear_bc_v1": "最弱的行为克隆 baseline。离线回归可以拟合动作均值，但闭环控制失败，说明低 MSE 不等于机器人任务成功。",
    "knn_bc_v1": "非参数轨迹记忆 baseline。训练范围内成功，但 held-out 明显下降，说明它主要依赖相似轨迹检索。",
    "mlp_bc_v1": "标准小型神经网络 BC baseline。比线性模型表达力更强，但单步状态到动作回归仍不能稳定完成抓取和放置。",
    "act_lite_chunk_bc_v1": "轻量动作块 baseline。输出短动作序列，但缺少足够的阶段/接触建模，闭环仍失败。",
    "diffusion_policy_lite_v1": "NumPy DDPM 风格动作块 baseline，用于对比扩散式动作生成思路；不是官方完整 PyTorch Diffusion Policy。",
    "torch_diffusion_policy_state_chunk_v1": "PyTorch state-only conditional Diffusion Policy baseline，使用历史状态条件、Transformer encoder/decoder 去噪器和 DDPM 动作块训练；不是完整视觉 Diffusion Policy。",
    "trajectory_conditioned_chunk_bc_v2": "加入历史观测窗口的动作块模型。动作更平滑，但没有解决闭环抓取失败。",
    "trajectory_knn_chunk_bc_v1": "历史轨迹窗口 + kNN 动作块检索。训练范围成功、held-out 失败，进一步说明轨迹记忆不等于泛化策略。",
    "torch_act_state_chunk_v1": "PyTorch Transformer ACT-style baseline，输入状态历史并输出动作块。比 MLP 更接近 ACT，但当前 state-only 版本仍不稳定。",
    "torch_act_state_chunk_cuda_v1": "与 torch_act_state_chunk_v1 同结构、同数据设置的 CUDA 训练版本。它用于记录 GPU 训练时间和峰值显存，不代表策略结构改进。",
    "phase_conditioned_torch_act_v1": "在 state-only PyTorch ACT 的历史状态输入上追加离散阶段 one-hot，用于检验显式阶段条件是否能改善动作块闭环控制；当前结果显示它仍未解决接触、夹紧和抬升。",
    "torch_act_cvae_state_chunk_v1": "PyTorch Transformer ACT-CVAE-lite baseline，训练时用动作块 posterior 学 latent，执行时 zero latent 解码；更接近标准 ACT，但当前 state-only 版本仍不稳定。",
    "visual_feature_act_lite_v1": "MuJoCo 离线重渲染 RGB pooled features + 语言/本体状态的 Transformer ACT-lite。它比纯状态 ACT 多了视觉代理输入，但不是完整 CNN 视觉 ACT。",
    "object_language_action_head_lite_v1": "符号对象/目标特征 + 语言任务 token + 轻量 action head 的 VLA 代理基线；不是 pretrained VLM/VLA。",
    "reward_weighted_action_head_lite_v1": "在 object-language action-head 特征上使用 attempt 偏好和 dense shaping 权重训练；属于轻量后训练代理，不是 RL。",
    "phase_conditioned_action_head_lite_v1": "显式拆分 approach、grasp、lift、transfer、place/release 五个阶段动作头；用于检验阶段条件是否能改善轻量 action-head。",
    "adapter_action_head_lite_v1": "冻结 object-language action-head 主干，仅训练小型 Adapter 残差模块；用于本地 PEFT 对照，不是 pretrained VLA Adapter。",
    "lora_action_head_lite_v1": "冻结 object-language action-head 主干，仅训练 LoRA-style 低秩输出残差；用于本地 PEFT 对照，不是 pretrained VLA LoRA。",
    "vision_language_action_head_lite_v1": "MuJoCo RGB 统计视觉特征 + 语言 token + 轻量 action head 的视觉-语言代理基线；结果说明手工视觉统计不等于 VLA 表征。",
    "clip_action_head_lite_v1": "冻结 pretrained CLIP 图像/文本编码器，只训练轻量 action head；这是 VLM 表征代理基线，不是 OpenVLA/机器人 VLA。",
    "multi_task_object_action_head_lite_v1": "多任务 action-head 代理基线。加入多数据源后仍失败，说明 naive 多任务混合和单一 MLP head 不足。",
}


TYPES = {
    "scripted_oracle": "oracle / 数据生成器",
    "structured_control_baseline": "强结构化控制 baseline",
    "data_verification": "数据复现验证",
    "weak_bc_baseline": "普通模仿学习 baseline",
    "non_neural_baseline": "非神经检索 baseline",
    "neural_bc_baseline": "神经网络 BC baseline",
    "trajectory_conditioned_baseline": "动作块 / 轨迹条件 baseline",
    "diffusion_policy_baseline": "扩散式动作块 baseline",
    "torch_diffusion_policy_baseline": "PyTorch 扩散式动作块 baseline",
    "trajectory_memory_baseline": "轨迹记忆 baseline",
    "torch_act_baseline": "Transformer ACT-style baseline",
    "torch_act_cvae_baseline": "Transformer ACT-CVAE-lite baseline",
    "visual_feature_act_baseline": "视觉特征 ACT-lite baseline",
    "vla_action_head_proxy": "轻量 VLA/action-head 代理",
    "reward_weighted_bc_post_training": "reward-weighted BC 后训练代理",
    "phase_conditioned_action_head_proxy": "阶段条件 action-head 代理",
    "pretrained_vlm_action_head_proxy": "pretrained VLM action-head 代理",
    "peft_action_head_proxy": "参数高效 action-head 代理",
    "multi_task_action_head_proxy": "多任务 action-head 代理",
}


LANGUAGE_VERSION_ALIASES = {
    "expert_scripted_v1": "expert_scripted_language_v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chinese method cards for all registered experiment versions.")
    parser.add_argument("--versions", type=Path, default=ROOT / "docs" / "experiment_versions.json")
    parser.add_argument("--summary", type=Path, default=ROOT / "docs" / "evaluation_summary.csv")
    parser.add_argument("--language-summary", type=Path, default=ROOT / "docs" / "language_generalization_summary.csv")
    parser.add_argument("--resource-summary", type=Path, default=ROOT / "docs" / "model_resource_summary.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def row_by_version(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["version"]: row for row in rows}


def format_value(value: str | None, empty: str = "未记录") -> str:
    if value is None or value == "":
        return empty
    return value


def learned_tag(stage: str, trainable_params: str) -> str:
    params = int(float(trainable_params or 0))
    if stage in {"scripted_oracle", "structured_control_baseline", "data_verification"}:
        return "否"
    if params > 0:
        return "是"
    return "非参数方法"


def risk_note(version: str, stage: str) -> str:
    if version == "expert_scripted_v1":
        return "论文中只能写作 oracle，不可写成学习策略。"
    if version == "structured_waypoint_policy_v1":
        return "显式使用状态和任务结构，不可写成 VLA 泛化能力。"
    if version == "phase_conditioned_torch_act_v1":
        return "当前是 state-only ACT-style baseline 加阶段 one-hot，不是完整视觉 ACT，也没有学出稳定抓取。"
    if "phase_conditioned" in version:
        return "当前只是显式阶段条件的本地 action-head 代理，不是层级 VLA，也没有学出稳定抓取。"
    if "clip_action_head" in version:
        return "当前冻结的是通用 CLIP VLM encoder，不是机器人 VLA；只能写作 pretrained VLM 表征代理。"
    if "action_head" in version:
        return "当前是本地代理基线，不是 pretrained VLM/VLA、LoRA 或 Adapter。"
    if "torch_diffusion_policy" in version:
        return "当前是 state-only PyTorch diffusion action-chunk baseline，不含视觉 encoder，不能写成完整视觉 Diffusion Policy。"
    if "diffusion" in version:
        return "当前是轻量 NumPy 版本，不是官方完整 Diffusion Policy。"
    if "torch_act_cvae" in version:
        return "当前是 state-only ACT-CVAE-lite baseline，不含视觉 encoder；不能写成完整视觉 ACT。"
    if "visual_feature_act" in version:
        return "当前使用的是 pooled RGB 视觉代理特征，不是端到端 CNN/Transformer 视觉 ACT。"
    if "act" in version and "torch" not in version:
        return "当前是 ACT-lite，不是论文完整 ACT。"
    if "torch_act" in version:
        return "当前是 state-only ACT-style baseline，不含视觉 encoder/CVAE。"
    if "knn" in version:
        return "成功更可能来自轨迹记忆，需和泛化能力区分。"
    return "按当前评测结果陈述，不扩展到未验证任务。"


def write_cards(args: argparse.Namespace, versions: dict, summary: dict, language: dict, resources: dict) -> None:
    lines = [
        "# 方法卡片 Method Cards",
        "",
        "版本：`method_cards_v1`",
        "",
        "用途：把当前实验登记表、主任务评测、语言泛化评测、资源统计和视频片段整理成论文/答辩可直接使用的中文方法卡片。该文档由 `scripts/build_method_cards.py` 自动生成。",
        "",
        "## 总体结论",
        "",
        "1. `expert_scripted_v1` 和 `structured_waypoint_policy_v1` 表明环境和任务是可解的。",
        "2. 普通 BC、ACT-lite、Diffusion-lite 和当前 action-head 代理方法在 held-out 与 language 任务上普遍不足。",
        "3. `kNN` 与 `trajectory-kNN` 在训练范围有效，但更接近轨迹记忆，不代表真正泛化。",
        "4. `torch_act_cvae_state_chunk_v1` 补上了 ACT 的 CVAE latent 结构，但 state-only 小数据闭环仍失败。",
        "5. `visual_feature_act_lite_v1` 补上了 MuJoCo 视觉代理输入，但 pooled RGB 特征仍不能替代完整视觉 ACT。",
        "6. `phase_conditioned_action_head_lite_v1` 说明显式阶段拆分能降低离线误差，但仍不能自动解决接触和抓取闭环。",
        "7. 当前 vision-language proxy 说明手工 RGB 统计特征不等价于 pretrained VLM/VLA 表征。",
        "",
        "## 卡片列表",
        "",
    ]

    for item in versions["methods"]:
        version = item["version"]
        stage = item["stage"]
        res = resources.get(version, {})
        lang = language.get(version) or language.get(LANGUAGE_VERSION_ALIASES.get(version, ""), {})
        trainable_params = res.get("trainable_params", "0")
        lines.extend(
            [
                f"### `{version}`",
                "",
                f"- 方法名称：{item['method']}",
                f"- 阶段定位：{TYPES.get(stage, stage)}",
                f"- 是否可学习策略：{learned_tag(stage, trainable_params)}",
                f"- 主任务训练范围：`{item['train_range_success']}`",
                f"- 主任务 held-out：`{item['heldout_success']}`",
                f"- 语言/空间泛化：`{format_value(lang.get('success'), '未纳入该项')}`",
                f"- 可训练参数：`{format_value(trainable_params, '0')}`",
                f"- 存储样本/轨迹片段：`{format_value(res.get('stored_samples'))}`",
                f"- 特征/观测维度：`{format_value(res.get('feature_dim'))}`",
                f"- 模型/脚本 artifact：`{item['artifact']}`",
                f"- 固定展示视频：`{item['clip']}`",
                f"- 论文表述：{INTERPRETATIONS.get(version, item.get('note', '按当前评测结果解释。'))}",
                f"- 注意事项：{risk_note(version, stage)}",
                "",
            ]
        )

    lines.extend(
        [
            "## 推荐讲解顺序",
            "",
            "1. 先展示 `expert_scripted_v1` 和 `replay_demo_v1`，证明环境、示范和回放链路可靠。",
            "2. 再展示 `structured_waypoint_policy_v1`，说明任务可由显式阶段结构解决。",
            "3. 展示 `linear_bc_v1`、`mlp_bc_v1`、`act_lite_chunk_bc_v1`、`diffusion_policy_lite_v1` 的失败，说明普通学习 baseline 不够。",
            "4. 展示 `knn_bc_v1` 和 `trajectory_knn_chunk_bc_v1`，强调训练范围成功与 held-out 泛化的差距。",
            "5. 再展示 `torch_act_cvae_state_chunk_v1`，说明加入 CVAE latent 以后仍没有解决 state-only 闭环抓取。",
            "6. 展示 `visual_feature_act_lite_v1`，说明 pooled RGB 视觉代理输入仍不能替代完整视觉 ACT。",
            "7. 最后展示 action-head、phase-conditioned action-head 与 vision-language proxy，说明当前轻量 VLA 代理路线已搭好，但还需要 pretrained VLM/VLA 表征、LoRA/Adapter 或更强阶段建模。",
            "",
            "## 重新生成命令",
            "",
            "```powershell",
            f'& "{ROOT / ".venv" / "Scripts" / "python.exe"}" "{ROOT / "scripts" / "build_method_cards.py"}"',
            "```",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    versions = read_json(args.versions)
    summary = row_by_version(read_csv(args.summary))
    language = row_by_version(read_csv(args.language_summary))
    resources = row_by_version(read_csv(args.resource_summary))
    write_cards(args, versions, summary, language, resources)
    print(f"method_cards_path: {args.output}", flush=True)
    print(f"methods: {len(versions['methods'])}", flush=True)


if __name__ == "__main__":
    main()
