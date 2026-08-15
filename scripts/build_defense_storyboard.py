from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "defense_storyboard.md"


CHAPTERS = [
    {
        "title": "1. 研究问题与任务入口",
        "versions": ["expert_scripted_v1", "replay_demo_v1"],
        "message": "先证明环境、任务和数据链路可用：专家能完成任务，保存的示范轨迹能回放。",
        "materials": [
            "outputs/videos/expert_scripted_v1_seed0.mp4",
            "outputs/videos/replay_demo_v1_seed0.mp4",
            "outputs/showcase/core_methods_grid.mp4",
        ],
    },
    {
        "title": "2. 结构化强对照组",
        "versions": ["structured_waypoint_policy_v1"],
        "message": "显式阶段分解和目标状态访问可以高成功率完成任务，说明 learned policy 的困难主要在阶段、接触和泛化，而不是仿真任务不可解。",
        "materials": [
            "outputs/videos/structured_waypoint_policy_v1_seed0.mp4",
            "outputs/videos/structured_waypoint_policy_v1_language_seed200.mp4",
        ],
    },
    {
        "title": "3. 普通模仿学习 baseline",
        "versions": ["linear_bc_v1", "knn_bc_v1", "mlp_bc_v1"],
        "message": "Linear/MLP 单步回归闭环不稳定；kNN 在训练范围能成功但 held-out 明显下降，说明轨迹记忆不等于泛化策略。",
        "materials": [
            "outputs/videos/linear_bc_v1_seed0.mp4",
            "outputs/videos/knn_bc_v1_seed0.mp4",
            "outputs/videos/mlp_bc_v1_seed0.mp4",
        ],
    },
    {
        "title": "4. 动作块、Trajectory 与 ACT-style baseline",
        "versions": [
            "act_lite_chunk_bc_v1",
            "trajectory_conditioned_chunk_bc_v2",
            "trajectory_knn_chunk_bc_v1",
            "torch_act_state_chunk_v1",
            "torch_act_state_chunk_cuda_v1",
            "phase_conditioned_torch_act_v1",
            "torch_act_cvae_state_chunk_v1",
            "visual_feature_act_lite_v1",
            "visual_act_cnn_cvae_v1",
        ],
        "message": "动作块、历史轨迹、CUDA 资源对照、离散阶段条件、CVAE latent、pooled RGB 视觉代理特征和小型 CNN 视觉 encoder 让控制结构更接近 ACT，但当前小数据版本仍不能稳定泛化；这些版本只能作为轻量 ACT baseline，不是完整官方视觉 ACT。",
        "materials": [
            "outputs/videos/act_lite_chunk_bc_v1_seed0.mp4",
            "outputs/videos/trajectory_conditioned_chunk_bc_v2_seed0.mp4",
            "outputs/videos/trajectory_knn_chunk_bc_v1_seed0.mp4",
            "outputs/videos/torch_act_state_chunk_v1_seed0.mp4",
            "outputs/videos/torch_act_state_chunk_cuda_v1_seed0.mp4",
            "outputs/videos/phase_conditioned_torch_act_v1_seed0.mp4",
            "outputs/videos/torch_act_cvae_state_chunk_v1_seed0.mp4",
            "outputs/videos/visual_feature_act_lite_v1_seed0.mp4",
            "outputs/videos/visual_act_cnn_cvae_v1_seed0.mp4",
        ],
    },
    {
        "title": "5. Diffusion Policy-lite baseline",
        "versions": ["diffusion_policy_lite_v1", "torch_diffusion_policy_state_chunk_v1"],
        "message": "NumPy Diffusion-lite 和 PyTorch state-only Diffusion Policy 动作块 baseline 都未形成稳定抓取；PyTorch 版本更接近扩散式动作块训练，但仍不含视觉 encoder，不能写作完整视觉 Diffusion Policy。",
        "materials": [
            "outputs/videos/diffusion_policy_lite_v1_seed0.mp4",
            "outputs/videos/torch_diffusion_policy_state_chunk_v1_seed0.mp4",
        ],
    },
    {
        "title": "6. 轻量 VLA/action-head 代理路线",
        "versions": [
            "object_language_action_head_lite_v1",
            "reward_weighted_action_head_lite_v1",
            "phase_conditioned_action_head_lite_v1",
            "adapter_action_head_lite_v1",
            "lora_action_head_lite_v1",
            "vision_language_action_head_lite_v1",
            "clip_action_head_lite_v1",
            "multi_task_object_action_head_lite_v1",
        ],
        "message": "本地 action head、reward-weighted 后训练代理和冻结 CLIP 表征代理链路已经搭好；结果显示加权 BC 或通用 VLM 表征加小动作头仍不能自动补足阶段和接触控制。",
        "materials": [
            "outputs/videos/object_language_action_head_lite_v1_seed0.mp4",
            "outputs/videos/object_language_action_head_lite_v1_seed1_success_example.mp4",
            "outputs/videos/reward_weighted_action_head_lite_v1_seed0.mp4",
            "outputs/videos/phase_conditioned_action_head_lite_v1_seed0.mp4",
            "outputs/videos/adapter_action_head_lite_v1_seed0.mp4",
            "outputs/videos/lora_action_head_lite_v1_seed0.mp4",
            "outputs/videos/vision_language_action_head_lite_v1_seed0.mp4",
            "outputs/videos/clip_action_head_lite_v1_seed0.mp4",
            "outputs/videos/multi_task_object_action_head_lite_v1_seed0.mp4",
        ],
    },
    {
        "title": "7. 语言/空间泛化与数据效率",
        "versions": [
            "expert_scripted_v1",
            "structured_waypoint_policy_v1",
            "object_language_action_head_lite_v1",
            "reward_weighted_action_head_lite_v1",
            "torch_act_state_chunk_cuda_v1",
            "torch_act_cvae_state_chunk_v1",
            "torch_diffusion_policy_state_chunk_v1",
            "visual_feature_act_lite_v1",
            "visual_act_cnn_cvae_v1",
            "adapter_action_head_lite_v1",
            "lora_action_head_lite_v1",
            "vision_language_action_head_lite_v1",
            "clip_action_head_lite_v1",
            "multi_task_object_action_head_lite_v1",
        ],
        "message": "结构化策略能处理 leftmost -> bowl，当前 learned/action-head 代理方法不能；数据效率曲线也显示 kNN/trajectory-kNN 主要是训练范围记忆。",
        "materials": [
            "outputs/showcase/language_generalization_grid.mp4",
            "outputs/figures/language_success.svg",
            "outputs/figures/data_efficiency.svg",
        ],
    },
    {
        "title": "8. MuJoCo Domain Randomization 代理评测",
        "versions": ["structured_waypoint_policy_v1", "trajectory_knn_chunk_bc_v1", "visual_act_cnn_cvae_v1"],
        "message": "在低摩擦、弱夹爪和执行器扰动下，结构化策略仍稳定，trajectory-kNN 出现接触鲁棒性下降，Visual ACT-CNN-CVAE-lite 仍未闭环成功。该阶段只能写成 MuJoCo 代理评测；Isaac 和真实 WidowX handoff 已建立，但不能写成实际外部验证结果。",
        "materials": [
            "docs/domain_randomization_summary.md",
            "outputs/presentation_clips/06_domain_randomization_proxy.mp4",
            "outputs/videos/domain_randomization_structured_low_friction_seed0.mp4",
            "outputs/videos/domain_randomization_trajectory_knn_low_friction_seed0.mp4",
            "outputs/videos/domain_randomization_visual_act_cnn_cvae_low_friction_seed0.mp4",
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese defense/storyboard guide from current experiment artifacts.")
    parser.add_argument("--versions", type=Path, default=ROOT / "docs" / "experiment_versions.json")
    parser.add_argument("--summary", type=Path, default=ROOT / "docs" / "evaluation_summary.csv")
    parser.add_argument("--language-summary", type=Path, default=ROOT / "docs" / "language_generalization_summary.csv")
    parser.add_argument("--resources", type=Path, default=ROOT / "docs" / "model_resource_summary.csv")
    parser.add_argument("--data-efficiency", type=Path, default=ROOT / "docs" / "data_efficiency_summary.csv")
    parser.add_argument("--showcase-manifest", type=Path, default=ROOT / "outputs" / "showcase" / "video_showcase_manifest.json")
    parser.add_argument("--strict-grasp-json", type=Path, default=ROOT / "outputs" / "evaluations" / "strict_grasp_success_audit_v1.json")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def by_version(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["version"]: row for row in rows}


def command(script: str, extra: str = "") -> str:
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    target = ROOT / "scripts" / script
    suffix = f" {extra}" if extra else ""
    return f'& "{python}" "{target}"{suffix}'


def method_table(
    version_ids: list[str],
    methods: dict[str, dict],
    summary: dict[str, dict[str, str]],
    language: dict[str, dict[str, str]],
    resources: dict[str, dict[str, str]],
) -> list[str]:
    lines = [
        "| 版本 | 阶段 | 主任务 train | 主任务 held-out | 语言泛化 | 参数量 | 固定视频 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for version in version_ids:
        item = methods[version]
        row = summary.get(version, {})
        lang_key = "expert_scripted_language_v1" if version == "expert_scripted_v1" else version
        lang = language.get(lang_key, {})
        res = resources.get(version, {})
        lines.append(
            "| "
            f"`{version}` | "
            f"{item['stage']} | "
            f"{row.get('train_range_success', item.get('train_range_success', ''))} | "
            f"{row.get('heldout_success', item.get('heldout_success', ''))} | "
            f"{lang.get('success', '未评测')} | "
            f"{int(float(res.get('trainable_params', 0))):,} | "
            f"`{item['clip']}` |"
        )
    return lines


def write_storyboard(args: argparse.Namespace) -> None:
    versions = read_json(args.versions)
    methods_list = versions["methods"]
    methods = {item["version"]: item for item in methods_list}
    summary = by_version(read_csv(args.summary))
    language = by_version(read_csv(args.language_summary))
    resources = by_version(read_csv(args.resources))
    data_efficiency_rows = read_csv(args.data_efficiency)
    showcase = read_json(args.showcase_manifest) if args.showcase_manifest.exists() else {"showcases": []}
    strict_grasp = read_json(args.strict_grasp_json).get("summary", {})
    strict_episodes = strict_grasp.get("episodes", "?")
    strict_loose = f"{strict_grasp.get('loose_successes', '?')}/{strict_episodes}"
    strict_success = f"{strict_grasp.get('strict_grasp_successes', '?')}/{strict_episodes}"

    lines = [
        "# 答辩与论文叙事 Storyboard",
        "",
        "版本：`defense_storyboard_v1`",
        "",
        "用途：把当前 MuJoCo WidowX 桌面操作实验整理成可直接用于论文结果章节、阶段汇报和答辩演示的讲解顺序。该文档由 `scripts/build_defense_storyboard.py` 从当前实验登记表、评测表、资源表和视频 manifest 自动生成。",
        "",
        "## 总论点",
        "",
        "在当前小规模示范数据和有限算力条件下，普通 BC、轻量动作块、state-only ACT-style、Diffusion-lite、本地 action-head 代理、phase-conditioned action-head、reward-weighted BC 后训练代理和冻结 CLIP 表征 action head 都还不能稳定超过结构化强对照组。已有结果支持下一阶段继续做机器人预训练 VLA 表征 + action head / Adapter / LoRA，而不是把当前 CLIP 或 RGB 统计特征 action head 直接写成真实 VLA。",
        "",
        "## 现有证据包",
        "",
        f"- 已登记方法数：`{len(methods_list)}`",
        f"- 主任务：`{versions['task']}` / `{versions['complexity']}`",
        f"- 数据集：`{versions['dataset']['path']}`",
        "- 总览 dashboard：`docs/experiment_dashboard.html`",
        "- 阶段结果矩阵：`docs/result_matrix.md`",
        "- 方法卡片：`docs/method_cards.md`",
        "- Robot VLA action-head 交接门禁：`docs/robot_vla_action_head_handoff.md`",
        "- Robot VLA 远端运行包：`docs/robot_vla_remote_run_pack.md`",
        "- Robot VLA 远端结果回填门禁：`docs/robot_vla_remote_result_intake.md`",
        "- Isaac domain randomization 交接门禁：`docs/isaac_domain_randomization_handoff.md`",
        "- 真实 WidowX 验证交接门禁：`docs/real_widowx_validation_handoff.md`",
        "- 严格抓取成功口径审计：`docs/strict_grasp_success_audit.md`",
        "- 固定视频索引：`docs/video_showcase.md` 与 `docs/video_clips.md`",
        "- 评测表：`docs/evaluation_summary.csv`、`docs/language_generalization_summary.csv`、`docs/data_efficiency_summary.csv`、`docs/domain_randomization_summary.csv`、`docs/model_resource_summary.csv`",
        "",
        "## 推荐讲解顺序",
        "",
    ]

    for chapter in CHAPTERS:
        lines.extend([f"### {chapter['title']}", "", f"讲法：{chapter['message']}", ""])
        lines.extend(method_table(chapter["versions"], methods, summary, language, resources))
        lines.extend(["", "展示素材："])
        for material in chapter["materials"]:
            lines.append(f"- `{material}`")
        lines.append("")

    lines.extend(
        [
            "## 总览视频",
            "",
            "| 预设 | 输出文件 | 分辨率 | 时长 | 片段数 |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for item in showcase.get("showcases", []):
        lines.append(
            "| "
            f"`{item['preset']}` | "
            f"`{item['output']}` | "
            f"{item['width']}x{item['height']} | "
            f"{float(item['duration']):.1f}s | "
            f"{len(item['clips'])} |"
        )

    lines.extend(
        [
            "",
            "## 数据效率讲法",
            "",
            "数据效率表用于回答“少量示范下是否省数据”。当前预算为 10、25、50、92 条成功示范，评测 `knn_bc`、`trajectory_knn` 和 `object_action_head`。关键结论是：kNN/trajectory-kNN 在训练范围随数据增加而改善，但 held-out 仍弱；object_action_head 在这些预算下仍没有稳定成功。",
            "",
            "| 方法 | 预算 | 范围 | 成功率 | 平均目标距离 |",
            "| --- | ---: | --- | ---: | ---: |",
        ]
    )
    for row in data_efficiency_rows:
        if row["demo_budget"] in {"10", "92"}:
            lines.append(
                "| "
                f"`{row['method_key']}` | "
                f"{row['demo_budget']} | "
                f"{row['split']} | "
                f"{row['success']} | "
                f"{float(row['mean_target_distance']):.4f} |"
            )

    lines.extend(
        [
            "",
            "## 论文表述红线",
            "",
            "- `structured_waypoint_policy_v1` 是显式状态和阶段控制 baseline，不是 learned VLA。",
            "- `act_lite_chunk_bc_v1` 和 `trajectory_conditioned_chunk_bc_v2` 是 NumPy MLP 动作块 baseline，不是完整 ACT。",
            "- `torch_act_state_chunk_v1` 是 state-only Transformer ACT-style baseline，不含视觉 encoder 或 CVAE latent。",
            "- `phase_conditioned_torch_act_v1` 是 state-only ACT-style baseline 追加离散阶段 one-hot，不是完整视觉 ACT，也不是层级任务规划器。",
            "- `torch_act_cvae_state_chunk_v1` 是 state-only ACT-CVAE-lite baseline，含 CVAE latent，但不含视觉 encoder，不能写成完整视觉 ACT。",
            "- `visual_feature_act_lite_v1` 使用 pooled RGB 视觉代理特征，不是端到端 CNN/Transformer 视觉 ACT。",
            "- `visual_act_cnn_cvae_v1` 使用小型 CNN RGB encoder + ACT-CVAE 动作块，是本地轻量视觉 ACT baseline，不能写成官方完整 ACT 或真实机器人视觉 ACT 复现。",
            "- `diffusion_policy_lite_v1` 是 NumPy DDPM 风格动作块 baseline，不是官方完整 PyTorch Diffusion Policy。",
            "- `reward_weighted_action_head_lite_v1` 是 attempt 偏好和 dense shaping 加权 BC，不是真正在线 RL。",
            "- `phase_conditioned_action_head_lite_v1` 是显式阶段条件 action-head 代理，不是层级 VLA 或真实任务规划器。",
            "- `object_language_action_head_lite_v1`、`adapter_action_head_lite_v1`、`lora_action_head_lite_v1`、`vision_language_action_head_lite_v1` 和 `multi_task_object_action_head_lite_v1` 是本地 VLA/action-head/PEFT 代理实验，不是 pretrained VLM/VLA、真实 LoRA 或真实 Adapter。",
            "- `clip_action_head_lite_v1` 使用 frozen pretrained CLIP 图像/文本 encoder，但 CLIP 不是机器人 VLA；该版本只能写作 pretrained VLM 表征代理。",
            f"- `strict_grasp_success_audit_v1` 显示原始放置成功 `{strict_loose}`、严格抓取成功 `{strict_success}`，因此不能把目标距离达标样例写成稳定抓取成功。",
            "",
            "## 下一阶段实验入口",
            "",
            "优先级建议：",
            "",
            "1. 按 `docs/robot_vla_remote_run_pack.md` 和 `docs/robot_vla_remote_result_intake.md` 接入机器人动作数据预训练的 VLA/OpenVLA 类表征，冻结主干训练 action head，并保持当前 `docs/model_resource_summary.csv` 的参数量、训练耗时和显存字段。",
            "2. 在同一任务上比较 CLIP action-head only、机器人 VLA action-head only、Adapter 和 LoRA 的 trainable parameters、训练时间、显存和成功率。",
            "3. 再把 `move_leftmost_to_bowl`、`place_blue_cube_red_pad` 等任务纳入同一评测表，避免只在单任务上报告结果。",
            "4. 如果补 Isaac 或真实 WidowX，必须按 `docs/isaac_domain_randomization_handoff.md` 和 `docs/real_widowx_validation_handoff.md` 回填结果，并和当前 MuJoCo domain randomization 分开命名。",
            "5. 后续所有 ACT/VLA/真实机械臂结果必须同时报告 `success`、`grasp_success`、`object_z` 和视频证据，沿用 `docs/strict_grasp_success_audit.md` 的红线。",
            "",
            "## 重新生成命令",
            "",
            "```powershell",
            command("build_defense_storyboard.py"),
            command("build_video_showcase.py", "--preset all --duration 12 --tile-width 320 --tile-height 240 --fps 24"),
            command("build_experiment_figures.py"),
            command("build_method_cards.py"),
            command("build_result_matrix.py"),
            command("verify_experiment_artifacts.py"),
            "```",
            "",
            "## 慢速可视化入口",
            "",
            "结构化强对照：",
            "",
            "```powershell",
            command(
                "run_structured_waypoint_policy.py",
                '--model "C:\\Users\\Administrator\\Desktop\\final project-robot ic\\vla_robot_grasping\\outputs\\structured_waypoint_policy\\structured_waypoint_policy_20260720_065456.npz" --seed 0 --episodes 1 --viewer --duration 60 --speed 0.05',
            ),
            "```",
            "",
            "PyTorch State Transformer ACT：",
            "",
            "```powershell",
            command(
                "run_torch_act_policy.py",
                '--model "C:\\Users\\Administrator\\Desktop\\final project-robot ic\\vla_robot_grasping\\outputs\\torch_act\\torch_act_state_chunk_20260720_055409.pt" --seed 0 --episodes 1 --steps 2840 --viewer --duration 60 --speed 0.05 --action-alpha 0.25 --max-arm-delta 0.012 --max-gripper-delta 0.0005 --replan-interval 1 --temporal-ensemble --ensemble-decay 0.1 --stop-on-unsafe --log-every 500',
            ),
            "```",
            "",
            "Phase-conditioned PyTorch State ACT：",
            "",
            "```powershell",
            '$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"',
            command(
                "run_torch_act_policy.py",
                '--model "C:\\Users\\Administrator\\Desktop\\final project-robot ic\\vla_robot_grasping\\outputs\\torch_act\\phase_conditioned_torch_act_20260720_132228.pt" --seed 0 --episodes 1 --steps 2840 --viewer --duration 60 --speed 0.05 --action-alpha 0.25 --max-arm-delta 0.012 --max-gripper-delta 0.0005 --replan-interval 4 --temporal-ensemble --ensemble-decay 0.1 --stop-on-unsafe --log-every 500',
            ),
            "```",
            "",
            "PyTorch State ACT-CVAE-lite：",
            "",
            "```powershell",
            command(
                "run_torch_act_cvae_policy.py",
                '--model "C:\\Users\\Administrator\\Desktop\\final project-robot ic\\vla_robot_grasping\\outputs\\torch_act_cvae\\torch_act_cvae_state_chunk_20260720_084842.pt" --seed 0 --episodes 1 --steps 2840 --viewer --duration 60 --speed 0.05 --action-alpha 0.25 --max-arm-delta 0.012 --max-gripper-delta 0.0005 --replan-interval 4 --temporal-ensemble --ensemble-decay 0.1 --stop-on-unsafe --log-every 500',
            ),
            "```",
            "",
            "Visual-Feature ACT-lite：",
            "",
            "```powershell",
            command(
                "run_visual_feature_act_policy.py",
                '--model "C:\\Users\\Administrator\\Desktop\\final project-robot ic\\vla_robot_grasping\\outputs\\visual_feature_act\\visual_feature_act_lite_20260720_091256.pt" --seed 0 --episodes 1 --steps 2840 --viewer --duration 60 --speed 0.05 --action-alpha 0.25 --max-arm-delta 0.012 --max-gripper-delta 0.0005 --replan-interval 4 --temporal-ensemble --ensemble-decay 0.1 --stop-on-unsafe --log-every 500',
            ),
            "```",
            "",
            "Visual ACT-CNN-CVAE-lite：",
            "",
            "```powershell",
            '$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"',
            command(
                "run_visual_act_cnn_cvae_policy.py",
                '--model "C:\\Users\\Administrator\\Desktop\\final project-robot ic\\vla_robot_grasping\\outputs\\visual_act_cnn_cvae\\visual_act_cnn_cvae_20260720_115104.pt" --seed 0 --episodes 1 --steps 2840 --viewer --duration 60 --speed 0.05 --action-alpha 0.25 --max-arm-delta 0.012 --max-gripper-delta 0.0005 --replan-interval 4 --temporal-ensemble --ensemble-decay 0.1 --stop-on-unsafe --log-every 500',
            ),
            "```",
            "",
            "Vision-Language Action Head-lite：",
            "",
            "```powershell",
            command(
                "run_vision_language_action_head.py",
                '--model "C:\\Users\\Administrator\\Desktop\\final project-robot ic\\vla_robot_grasping\\outputs\\vision_language_action_head\\vision_language_action_head_lite_20260720_063123.npz" --seed 0 --episodes 1 --steps 2840 --viewer --duration 60 --speed 0.05 --action-alpha 0.2 --max-arm-delta 0.01 --max-gripper-delta 0.0005 --stop-on-unsafe --log-every 500',
            ),
            "```",
            "",
            "Reward-Weighted Action Head-lite：",
            "",
            "```powershell",
            command(
                "run_object_action_head.py",
                '--model "C:\\Users\\Administrator\\Desktop\\final project-robot ic\\vla_robot_grasping\\outputs\\reward_weighted_action_head\\reward_weighted_action_head_lite_20260720_080912.npz" --seed 0 --episodes 1 --steps 2840 --viewer --duration 60 --speed 0.05 --action-alpha 0.2 --max-arm-delta 0.01 --max-gripper-delta 0.0005 --stop-on-unsafe --log-every 500',
            ),
            "```",
            "",
            "Phase-Conditioned Action Head-lite：",
            "",
            "```powershell",
            command(
                "run_phase_action_head.py",
                '--model "C:\\Users\\Administrator\\Desktop\\final project-robot ic\\vla_robot_grasping\\outputs\\phase_action_head\\phase_conditioned_action_head_lite_20260720_082827.npz" --seed 0 --episodes 1 --steps 2840 --phase-mode progress --viewer --duration 60 --speed 0.05 --action-alpha 0.12 --max-arm-delta 0.006 --max-gripper-delta 0.0003 --stop-on-unsafe --log-every 500',
            ),
            "```",
            "",
            "Frozen CLIP Action Head-lite：",
            "",
            "```powershell",
            command(
                "run_clip_action_head.py",
                '--model "C:\\Users\\Administrator\\Desktop\\final project-robot ic\\vla_robot_grasping\\outputs\\clip_action_head\\clip_action_head_lite_20260720_074716.npz" --seed 0 --episodes 1 --steps 2840 --vision-interval 64 --viewer --duration 60 --speed 0.05 --action-alpha 0.2 --max-arm-delta 0.01 --max-gripper-delta 0.0005 --stop-on-unsafe --log-every 500',
            ),
            "```",
            "",
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    write_storyboard(args)
    print(f"storyboard_path: {args.output}", flush=True)


if __name__ == "__main__":
    main()
