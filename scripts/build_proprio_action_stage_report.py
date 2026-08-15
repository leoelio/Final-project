from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(metrics: dict, view: str) -> dict:
    return next(item["metadata"] for item in metrics["results"] if item["view"] == view)


def main() -> None:
    bank = load(ROOT / "data" / "action_profile_bank" / "action_profile_bank_proprio_v2_summary.json")
    visual_language = load(ROOT / "outputs" / "action_profile_value_proprio_v2_visual_language" / "metrics.json")
    proprio = load(ROOT / "outputs" / "action_profile_value_proprio_v2_visual_language_proprio" / "metrics.json")
    selector_visual_language = load(ROOT / "outputs" / "evaluations" / "action_profile_selector_visual_language_v2.json")
    selector_proprio = load(ROOT / "outputs" / "evaluations" / "action_profile_selector_visual_language_proprio_v2.json")
    vl_top_front = metric(visual_language, "top_front")
    prop_top_front = metric(proprio, "top_front")
    fixed = selector_proprio["fixed_profiles"]

    lines = [
        "# 本体感知接触恢复扩展报告",
        "",
        "版本：`proprio_action_stage_v1`",
        "",
        "## 问题",
        "",
        "前一阶段表明失败后 RGB 和语言不能稳定判断标准重试与深抓取重试哪个更好。本阶段只加入真实机器人可获得的本体量，不加入 MuJoCo 物体位置、速度、接触数或目标真值。",
        "",
        "## 运行时输入边界",
        "",
        "- 关节位置：6 个机械臂关节和 2 个手指关节，`qpos[0:8]`。",
        "- 关节速度：`qvel[0:8]`。",
        "- 执行器控制与力：`ctrl[0:7]`、`actuator_force[0:7]`。",
        "- 合计 30 维。物体 free-joint 从 `qpos[8]` 开始，未进入特征；MuJoCo `ncon` 亦未使用。",
        "",
        "## 数据协议",
        "",
        f"- 极端接触扰动下，扫描 {bank['counters']['scanned']} 个 episode，得到 {bank['candidate_samples']} 个 RGB 可验证的首轮失败状态。",
        f"- 训练：`stop={bank['split_class_counts']['train']['stop']}`、`standard={bank['split_class_counts']['train']['standard']}`、`deep_tight_slow={bank['split_class_counts']['train']['deep_tight_slow']}`。",
        f"- 独立测试：`stop={bank['split_class_counts']['test']['stop']}`、`standard={bank['split_class_counts']['test']['standard']}`、`deep_tight_slow={bank['split_class_counts']['test']['deep_tight_slow']}`。",
        "- 每个样本的两条候选轨迹从同一首轮失败快照反事实执行；快照仅用于离线标签，模型输入为保存的 RGB、指令和本体量。",
        "",
        "## 冻结 CLIP 轻量动作头",
        "",
        "| 模型 | 双视图特征维度 | 可训练参数 | 测试准确率 | 平衡准确率 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| RGB + instruction | {vl_top_front['feature_dim']} | {vl_top_front['trainable_adapter_params']:,} | {vl_top_front['test_metrics']['accuracy']:.3f} | {vl_top_front['test_metrics']['balanced_accuracy_present_classes']:.3f} |",
        f"| RGB + instruction + proprio | {prop_top_front['feature_dim']} | {prop_top_front['trainable_adapter_params']:,} | {prop_top_front['test_metrics']['accuracy']:.3f} | {prop_top_front['test_metrics']['balanced_accuracy_present_classes']:.3f} |",
        "",
        "本体感知提高了分类指标，但测试中 `deep_tight_slow` 只有 1 条，不能把该变化写成稳定泛化。",
        "",
        "## 结果级选择评测",
        "",
        f"在 {selector_proprio['candidate_states']} 个独立候选状态中，固定 `standard` 为 {fixed['standard']}/{selector_proprio['candidate_states']}，固定 `deep_tight_slow` 为 {fixed['deep_tight_slow']}/{selector_proprio['candidate_states']}，oracle 为 {fixed['oracle']}/{selector_proprio['candidate_states']}。",
        "",
        "| 选择器 | 成功 | 结论 |",
        "| --- | ---: | --- |",
        f"| RGB + instruction 双视图 | {selector_visual_language['selectors']['top_front']['successes']}/{selector_visual_language['candidate_states']} | 低于固定轨迹 |",
        f"| RGB + instruction + proprio 双视图 | {selector_proprio['selectors']['top_front']['successes']}/{selector_proprio['candidate_states']} | 有增量，但仍低于固定深抓取 |",
        "",
        "因此本阶段不把本体感知选择器部署到新的闭环轨迹。固定深抓取在该条件候选集上较好，但与此前完整 RGB 闭环的留出回退证据冲突，不能据此替换全局默认方案。",
        "",
        "## 结论与下一步",
        "",
        "合理结论是：机器人本体量可能携带接触恢复信息，但当前候选数据中动作差异样本太少，无法训练可验证的选择器。下一步应按摩擦、抓夹力和控制增益做参数敏感性扫描，主动采集 `standard_only_success` 与 `deep_only_success` 两类状态，再进行新的 seed-disjoint 训练/测试。",
    ]
    output = ROOT / "docs" / "proprio_action_stage_v1.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report_path: {output}")


if __name__ == "__main__":
    main()
