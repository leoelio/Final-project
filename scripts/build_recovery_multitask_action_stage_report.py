from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_line(metrics: dict, view: str) -> str:
    item = next(row for row in metrics["results"] if row["view"] == view)
    meta = item["metadata"]
    test = meta["test_metrics"]
    return f"| `{view}` | {meta['trainable_adapter_params']:,} | {meta['feature_dim']} | {test['accuracy']:.3f} | {test['balanced_accuracy_present_classes']:.3f} |"


def main() -> None:
    bank = load(ROOT / "data" / "clip_recovery_bank" / "clip_recovery_multitask_training_v2_summary.json")
    visual = load(ROOT / "outputs" / "clip_recovery_value_multitask_v2" / "clip_recovery_value_multitask_v2_metrics.json")
    visual_language = load(ROOT / "outputs" / "clip_recovery_value_multitask_v2_visual_language" / "clip_recovery_value_multitask_v2_visual_language_metrics.json")
    probe = load(ROOT / "outputs" / "probes" / "visual_recovery_profiles_multitask_v2.json")
    terminal_diagnostic = load(ROOT / "outputs" / "evaluations" / "recovery_profile_heldout_v1.json")
    rgb_eval = load(ROOT / "outputs" / "evaluations" / "rgb_recovery_profile_heldout_v1.json")
    action_bank = load(ROOT / "data" / "action_profile_bank" / "action_profile_bank_v1_summary.json")
    action_selector = load(ROOT / "outputs" / "evaluations" / "action_profile_selector_offline_v1.json")
    action_train = action_bank["split_class_counts"]["train"]
    action_test = action_bank["split_class_counts"]["test"]
    model_results = visual["results"] + visual_language["results"]
    best_accuracy = max(item["metadata"]["test_metrics"]["accuracy"] for item in model_results)
    test_counts = bank["split_class_counts"]["test"]
    majority_accuracy = max(test_counts.values()) / sum(test_counts.values())

    lines = [
        "# 跨任务恢复与候选动作阶段报告",
        "",
        "版本：`recovery_multitask_action_stage_v1`",
        "",
        "## 阶段目标",
        "",
        "上一阶段的单任务 CLIP 恢复头在很小的留出集上表现较高，但无法说明跨任务泛化。本阶段扩展到颜色匹配、颜色错配和空间语言任务，依次验证：图像恢复头、图文恢复头、候选恢复动作。所有闭环结果均仅限 MuJoCo。",
        "",
        "## 跨任务数据",
        "",
        f"- 总计 {bank['samples']} 个真实首轮失败且可视觉重定位的状态；训练 {bank['split_class_counts']['train']['stop']} `stop` / {bank['split_class_counts']['train']['retry']} `retry`，独立测试 {bank['split_class_counts']['test']['stop']} / {bank['split_class_counts']['test']['retry']}。",
        f"- 训练 seed：`{bank['train_seed_ranges']}`；测试 seed：`{bank['test_seed_ranges']}`；旧的 `800-819` 测试段被明确排除，避免测试复用。",
        "- 训练中 `blue -> red` 没有 `retry` 样本，因而该细分类不应被单独宣称具有泛化能力。",
        "",
        "## 冻结 CLIP 轻量恢复头",
        "",
        "| 输入 | 视图 | 可训练参数 | 特征维度 | 独立测试准确率 | 平衡准确率 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        f"| 图像 | `top` | {metric_line(visual, 'top').split('|', 2)[2]}",
        f"| 图像 | `top_front` | {metric_line(visual, 'top_front').split('|', 2)[2]}",
        f"| 图像 + 冻结指令文本 | `top` | {metric_line(visual_language, 'top').split('|', 2)[2]}",
        f"| 图像 + 冻结指令文本 | `top_front` | {metric_line(visual_language, 'top_front').split('|', 2)[2]}",
        "",
        f"四个模型均未超过多数类基准 {majority_accuracy:.3f}，最佳准确率为 {best_accuracy:.3f}。加入文本没有解决问题，因此这里的结论不是“文本无用”，而是当前 61 条训练样本和二元重复/停止动作空间不足以学习跨任务恢复价值。不会将它们部署为默认策略，也不再为这个失败的局部模型申请 Kaggle 扩展训练。",
        "",
        "## 候选动作反事实探针",
        "",
        f"在 probe seed `1200-1219` 的 {probe['candidate_states']} 个真实失败快照中，`standard` 成功 {probe['profile_successes']['standard']} 次，`deep_tight_slow` 成功 {probe['profile_successes']['deep_tight_slow']} 次。每个 profile 从同一失败快照开始，轨迹由 RGB 重定位生成；MuJoCo 快照只用于离线反事实对照。",
        "",
        "这个 5/10 到 8/10 的内部探针只用于选择待验证的候选动作，不能作为最终性能结论。",
        "",
        "## 两个终局门的区分",
        "",
        f"- 旧 learned terminal gate 的诊断评测为 `standard={terminal_diagnostic['profile_successes']['standard']}/{terminal_diagnostic['episodes_per_profile']}`、`deep_tight_slow={terminal_diagnostic['profile_successes']['deep_tight_slow']}/{terminal_diagnostic['episodes_per_profile']}`，但它在部分真实失败上误判 `complete`，且可能对已成功轨迹误触发重试。因此该结果只用于定位门控失配，不用于主对照。",
        "- 主对照改用 RGB 目标确认：只有目标不在视觉目标区域且源物体可重定位时才允许一次重试。这个规则不读取 MuJoCo 物体真值。",
        "",
        "## 全新 Seed 主对照",
        "",
        f"极端接触扰动下，`blue -> red pad` 与 `leftmost cube -> bowl` 两个 RGB 可验证任务，seed `{rgb_eval['seed_range']}`，每个方法 {rgb_eval['episodes_per_profile']} 个配对 episode。",
        "",
        "| 恢复轨迹 | 成功 | RGB 触发重试 | 配对变化 |",
        "| --- | ---: | ---: | --- |",
        f"| `standard` | {rgb_eval['profile_successes']['standard']}/{rgb_eval['episodes_per_profile']} | {rgb_eval['profile_retries']['standard']} | 默认 |",
        f"| `deep_tight_slow` | {rgb_eval['profile_successes']['deep_tight_slow']}/{rgb_eval['episodes_per_profile']} | {rgb_eval['profile_retries']['deep_tight_slow']} | 改进 {rgb_eval['paired']['improved']}，回退 {rgb_eval['paired']['regressed']}，p={rgb_eval['paired']['exact_two_sided_p']:.4f} |",
        "",
        "`deep_tight_slow` 未复现探针优势，且在 seed 1308 由标准恢复成功变为失败。样本只有一个不一致对，不能给出统计显著性判断；工程上应保留 `standard`。",
        "",
        "## 视频证据",
        "",
        "- `videos/recovery_profile_v1/seed1308_standard_success.mp4`：标准 RGB 重试成功。",
        "- `videos/recovery_profile_v1/seed1308_deep_tight_slow_failure.mp4`：同一场景中改进轨迹失败，作为反例保留。",
        "",
        "## 当前结论与下一步",
        "",
        "当前可部署方案仍是：冻结 CLIP 语义意图 + 顶视 RGB 定位 + 结构化标准抓取/放置 + RGB 目标确认后的一次标准重试。冻结 CLIP 恢复头、图文恢复头和固定深抓取轨迹均作为有边界的消融结果保留。",
        "",
        "## Action-Conditioned 选择器可行性",
        "",
        f"随后采集了 {action_bank['candidate_samples']} 个同状态双轨迹反事实样本：训练 {action_train['stop']} `stop` / {action_train['standard']} `standard` / {action_train['deep_tight_slow']} `deep_tight_slow`；独立测试 {action_test['stop']} / {action_test['standard']} / {action_test['deep_tight_slow']}。测试段中固定 `standard` 与固定 `deep_tight_slow` 均为 {action_selector['fixed_profiles']['standard']}/{action_selector['candidate_states']}，oracle 为 {action_selector['fixed_profiles']['oracle']}/{action_selector['candidate_states']}。",
        "",
        f"冻结 CLIP 图文 3 类动作头的结果级评测：顶视 {action_selector['selectors']['top']['successes']}/{action_selector['candidate_states']}，双视图 {action_selector['selectors']['top_front']['successes']}/{action_selector['candidate_states']}。它们均未超过固定策略，因此不进入新的闭环部署。",
        "",
        "这说明在当前极端接触扰动中，标准/深抓取哪个更好主要由 RGB 难以观测的接触状态决定；即使完美 oracle 也只有一个 episode 的理论提升。后续若继续，应改为增加具有动作差异的困难状态，并引入真实可获得的夹爪电流、关节力矩或触觉反馈；不能把 MuJoCo 物体真值当作运行时输入。",
        "",
        "## 可视化复现命令",
        "",
        "```powershell",
        'cd "C:\\Users\\Administrator\\Desktop\\final project-robot ic\\vla_robot_grasping"',
        '$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"',
        '& .\\.venv\\Scripts\\python.exe .\\scripts\\run_clip_semantic_rgb_feedback.py `',
        '  --model .\\outputs\\clip_semantic_waypoint\\clip_semantic_waypoint_core_v2_v1_20260721_110325.npz `',
        '  --task move_leftmost_cube_to_bowl --complexity language --seed 1308 `',
        '  --feedback-attempts 1 --recovery-profile standard `',
        '  --arm-kp 105 --arm-force 70 --gripper-kp 550 --gripper-force 75 --friction 0.8 `',
        '  --viewer --duration 35 --speed 0.5 `',
        '  --video-path .\\videos\\recovery_profile_v1\\seed1308_standard_success.mp4',
        "```",
    ]
    output = ROOT / "docs" / "recovery_multitask_action_stage_v1.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report_path: {output}")


if __name__ == "__main__":
    main()
