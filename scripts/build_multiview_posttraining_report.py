from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def model_line(metrics: dict, view: str) -> str:
    item = next(row for row in metrics["results"] if row["view"] == view)
    meta = item["metadata"]
    test = item["test"]
    return (
        f"| `{view}` | {meta['trainable_parameters']:,} | {meta['train_samples']} | {meta['test_samples']} | "
        f"{test['accuracy']:.3f} | {test['balanced_accuracy_present_classes']:.3f} |"
    )


def main() -> None:
    terminal = load(ROOT / "outputs" / "multiview_value_head" / "terminal_mixed_training_metrics_v1.json")
    recovery = load(ROOT / "outputs" / "multiview_value_head" / "recovery_value_training_metrics_v1.json")
    candidate = load(ROOT / "outputs" / "multiview_value_head" / "recovery_candidate_training_metrics_v1.json")
    evaluation = load(ROOT / "outputs" / "evaluations" / "clip_semantic_multiview_gate_v1.json")
    terminal_data = load(ROOT / "data" / "multiview_recovery" / "terminal_mixed_v1_summary.json")
    recovery_data = load(ROOT / "data" / "multiview_recovery" / "spatial_recovery_v2_summary.json")
    final = {row["mode"]: (sum(int(item["success"]) for item in evaluation["rows"] if item["mode"] == row["mode"]), sum(1 for item in evaluation["rows"] if item["mode"] == row["mode"])) for row in evaluation["summary"]}
    lines = [
        "# 轻量视觉终局与恢复后训练阶段报告",
        "",
        "版本：`multiview_posttraining_stage_v1`",
        "",
        "## 问题与边界",
        "",
        "本阶段不训练端到端动作策略。冻结 CLIP 语义意图、RGB 平面定位和结构化 waypoint 执行器保持不变；新增的小型头只在第一次动作结束后，从 `top_rgb`、可选 `front_rgb` 与静态任务配置中决定终局是否完成，以及是否允许一次恢复。MuJoCo 物体真值只用于离线标签和最终评分。",
        "",
        "这不是 OpenVLA LoRA，也不是端到端 VLA。它是 MuJoCo-only 条件下的轻量后训练对照，用来验证有限数据是否足以改善闭环决策。",
        "",
        "## 数据协议",
        "",
        f"- 混合终局集：{terminal_data['samples']} 条，训练 `{terminal_data['split_class_counts']['train']['not_complete']} not_complete / {terminal_data['split_class_counts']['train']['complete']} complete`，测试 `{terminal_data['split_class_counts']['test']['not_complete']} / {terminal_data['split_class_counts']['test']['complete']}`。它合并了受控同色终局场景与真实 MuJoCo 动作终局。",
        f"- 恢复集：{recovery_data['samples']} 条空间语言 episode，训练 `59 accept / 10 retry / 11 stop`，测试 `38 accept / 2 retry / 0 stop`。测试缺少 `stop`，因此不对 stop 泛化作结论。",
        "- 恢复标签来自真实反事实执行：第一次失败后仅在 RGB 能重新定位源物体时执行一次第二轨迹，标签为该第二轨迹是否严格成功。",
        "- 全部 train/test 按 seed 范围分离；最终闭环评测使用 `750–754`，不与恢复集训练 `720–739` 或验证 `740–749` 重叠。",
        "",
        "## 终局头消融",
        "",
        "| 视图 | 可训练参数 | 训练样本 | 测试样本 | 准确率 | 已出现类别的平衡准确率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        model_line(terminal, "top"),
        model_line(terminal, "top_front"),
        "",
        "结论：顶视图优于 top+front（0.953 对 0.938）。在当前固定相机、小样本桌面场景中，前视图没有带来可验证增益，因此保留为消融项，不作为默认部署配置。",
        "",
        "## 恢复价值头",
        "",
        "| 视图 | 可训练参数 | 训练样本 | 测试样本 | 准确率 | 已出现类别的平衡准确率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        model_line(recovery, "top"),
        model_line(recovery, "top_front"),
        "",
        "三分类恢复头在测试集中对 `retry` 仅召回 1/2，且在完全独立的低接触 seed 750 中把一次可恢复失败误拒绝为 `stop`；它不能作为部署策略。进一步将问题限制到“已视觉重定位的失败样本”后，候选集只有 18 条，留出两条均为 retry，顶视图为 0/2、双视图为 1/2。这里的主要瓶颈是有效失败样本，而不是算力或模型参数量。",
        "",
        "## 最终闭环评测",
        "",
        "| 方法 | 留出成功 | 说明 |",
        "| --- | ---: | --- |",
        f"| `rule_rgb_retry` | {final['rule_rgb_retry'][0]}/{final['rule_rgb_retry'][1]} | 既有 RGB 规则重定位基线 |",
        f"| `learned_top_rule_recovery` | {final['learned_top_rule_recovery'][0]}/{final['learned_top_rule_recovery'][1]} | 顶视图终局头 + 已验证的 RGB 恢复规则 |",
        f"| `learned_top_front_rule_recovery` | {final['learned_top_front_rule_recovery'][0]}/{final['learned_top_front_rule_recovery'][1]} | 双视图终局头 + 相同恢复规则 |",
        "",
        "三个方法在 35 个留出 episode 上均为 33/35；同 seed 比较没有改善也没有回退，精确双侧检验均为 p=1。终局头当前提供的是可审计的同色完成信号，而不是成功率增益。",
        "",
        "## 当前最优方案",
        "",
        "推荐用于论文主结果和演示的策略是：冻结 CLIP 语义意图适配器 + top RGB 物体定位 + 结构化抓取/放置轨迹 + 一次 RGB 规则重定位。顶视图终局头可以作为“同色终局可观测性”消融保留；直接学习 recovery value head 必须标为失败/数据不足消融，不能写成主方法改进。",
        "",
        "## Kaggle 决策",
        "",
        "Kaggle 已配置且现有冻结 CLIP 语义 adapter 的训练包可复用。但本阶段头部仅 2,226–3,459 个可训练参数，本机训练为秒级；将它搬到 Kaggle 不会解决 18 条恢复候选样本造成的泛化问题。下一次只有在扩展到数百条真实失败/恢复轨迹，或训练冻结 CLIP 特征上的头部时，才应启动 Kaggle 远程训练并把远程产物导回本机 MuJoCo 评测。",
        "",
        "## 可复现命令",
        "",
        "```powershell",
        'cd "C:\\Users\\Administrator\\Desktop\\final project-robot ic\\vla_robot_grasping"',
        '$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"',
        '& ".\\.venv\\Scripts\\python.exe" ".\\scripts\\evaluate_clip_semantic_multiview_gate.py" `',
        '  --model ".\\outputs\\clip_semantic_waypoint\\clip_semantic_waypoint_core_v2_v1_20260721_110325.npz" `',
        '  --terminal-top ".\\outputs\\multiview_value_head\\terminal_mixed_top_v1.npz" `',
        '  --recovery-top ".\\outputs\\multiview_value_head\\recovery_value_top_v1.npz" `',
        '  --terminal-top-front ".\\outputs\\multiview_value_head\\terminal_mixed_top_front_v1.npz" `',
        '  --recovery-top-front ".\\outputs\\multiview_value_head\\recovery_value_top_front_v1.npz" `',
        '  --seed 750 --episodes 5 --recovery-mode rule',
        "```",
        "",
        "可视化单例（低接触恢复）使用：",
        "",
        "```powershell",
        'cd "C:\\Users\\Administrator\\Desktop\\final project-robot ic\\vla_robot_grasping"',
        '$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"',
        '& ".\\.venv\\Scripts\\python.exe" ".\\scripts\\run_clip_semantic_multiview_gate.py" `',
        '  --model ".\\outputs\\clip_semantic_waypoint\\clip_semantic_waypoint_core_v2_v1_20260721_110325.npz" `',
        '  --terminal-head ".\\outputs\\multiview_value_head\\terminal_mixed_top_v1.npz" `',
        '  --recovery-head ".\\outputs\\multiview_value_head\\recovery_value_top_v1.npz" `',
        '  --task move_leftmost_cube_to_bowl --complexity language --seed 724 `',
        '  --arm-kp 120 --arm-force 80 --gripper-kp 750 --gripper-force 110 --friction 1.5 `',
        '  --recovery-mode rule --viewer --duration 30 --speed 0.5',
        "```",
    ]
    output = ROOT / "docs" / "multiview_posttraining_stage_v1.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report_path: {output}")


if __name__ == "__main__":
    main()
