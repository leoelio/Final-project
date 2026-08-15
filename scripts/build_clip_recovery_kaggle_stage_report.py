from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def method_totals(rows: list[dict]) -> dict[str, dict]:
    totals: dict[str, dict] = {}
    for row in rows:
        item = totals.setdefault(row["mode"], {"episodes": 0, "successes": 0, "retries": 0, "attempts": 0})
        item["episodes"] += 1
        item["successes"] += int(row["success"])
        item["retries"] += int(row["retry_executed"])
        item["attempts"] += int(row["attempt_count"])
    return totals


def offline_line(item: dict) -> str:
    meta = item["metadata"]
    test = meta["test_metrics"]
    return (
        f"| `{meta['view']}` | {meta['trainable_adapter_params']:,} | {meta['feature_dim']} | "
        f"{meta['train_samples']} | {meta['test_samples']} | {test['accuracy']:.3f} | "
        f"{test['balanced_accuracy_present_classes']:.3f} |"
    )


def main() -> None:
    bank = load(ROOT / "data" / "clip_recovery_bank" / "clip_recovery_training_v1_summary.json")
    remote = load(ROOT / "outputs" / "kaggle_remote" / "widowx_mujoco_clip_recovery_value_v1" / "kaggle_clip_recovery_value_v1_metrics.json")
    evaluation = load(ROOT / "outputs" / "evaluations" / "clip_semantic_clip_recovery_kaggle_v1.json")
    rows = evaluation["rows"]
    totals = method_totals(rows)
    failures = [row for row in rows if not row["success"]]
    retried = [row for row in rows if row["retry_executed"]]
    top = next(item for item in remote["results"] if item["view"] == "top")
    top_front = next(item for item in remote["results"] if item["view"] == "top_front")
    top_meta = top["metadata"]

    lines = [
        "# CLIP 恢复头 Kaggle 后训练阶段报告",
        "",
        "版本：`clip_recovery_kaggle_stage_v1`",
        "",
        "## 本阶段问题",
        "",
        "首轮抓取或放置失败且 RGB 能重新定位源物体时，是否应执行一次恢复轨迹？本阶段冻结 CLIP 视觉编码器，只训练一个很小的二分类恢复头输出 `stop` 或 `retry`。它不预测机械臂动作，因此不是端到端 VLA、OpenVLA LoRA 或真实机器人实验。",
        "",
        "运行时输入仅为顶视/前视 RGB、静态任务配置与冻结的语义意图；MuJoCo 状态只用于离线反事实标签和最终评分。",
        "",
        "## 数据与划分",
        "",
        f"- 只保留真实首轮失败、且 RGB 可重新定位源物体的样本，共 {bank['samples']} 条；每条都实际执行一次候选恢复轨迹，成功记为 `retry`，失败记为 `stop`。",
        f"- 训练集：{bank['split_class_counts']['train']['stop']} `stop` / {bank['split_class_counts']['train']['retry']} `retry`，seed `{bank['train_seed_ranges']}`。",
        f"- 独立测试集：{bank['split_class_counts']['test']['stop']} `stop` / {bank['split_class_counts']['test']['retry']} `retry`，seed `{bank['test_seed_range']}`。",
        "- 闭环评测使用更新的 seed `880-884`，不与上述恢复头数据重叠。",
        "",
        "## 轻量后训练与 Kaggle 复现",
        "",
        f"冻结编码器为 `{top_meta['clip_model']}`，约 {top_meta['frozen_encoder_params']:,} 个参数；训练部分仅是一个 8 维瓶颈分类器。Kaggle 导出的模型被导回本地 MuJoCo 闭环评测。",
        "",
        "| 视图 | 可训练参数 | 特征维度 | 训练样本 | 独立测试样本 | 测试准确率 | 平衡准确率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        offline_line(top),
        offline_line(top_front),
        "",
        f"Kaggle 本次实际使用 `{top_meta['device']}`：环境声明 CUDA 可用，但设备兼容性回退，`gpu_execution=false`，因此不能把本次训练表述为 GPU 加速结果。顶视模型训练耗时 {top_meta['train_time_seconds']:.1f} 秒；11 条测试样本只能说明该小型离线划分上的可行性，不能证明稳定泛化。",
        "",
        "## 全新 Seed 的 MuJoCo 闭环对照",
        "",
        "评测覆盖 35 个 episode/方法：名义域 4 个任务各 5 个 seed（蓝块到蓝盘、蓝块到红盘、红块到红盘、最左方块到碗），再加低/中/极端接触扰动下的空间语言任务各 5 个 seed。所有方法使用相同 seed 和同一结构化执行器。",
        "",
        "| 方法 | 成功 | 恢复重试 | 平均尝试次数 | 结论 |",
        "| --- | ---: | ---: | ---: | --- |",
        f"| `rule_rgb_retry` | {totals['rule_rgb_retry']['successes']}/{totals['rule_rgb_retry']['episodes']} | {totals['rule_rgb_retry']['retries']} | {totals['rule_rgb_retry']['attempts'] / totals['rule_rgb_retry']['episodes']:.3f} | 当前默认基线 |",
        f"| `learned_top_clip_recovery` | {totals['learned_top_clip_recovery']['successes']}/{totals['learned_top_clip_recovery']['episodes']} | {totals['learned_top_clip_recovery']['retries']} | {totals['learned_top_clip_recovery']['attempts'] / totals['learned_top_clip_recovery']['episodes']:.3f} | 与规则结果相同，无增益 |",
        f"| `learned_top_front_clip_recovery` | {totals['learned_top_front_clip_recovery']['successes']}/{totals['learned_top_front_clip_recovery']['episodes']} | {totals['learned_top_front_clip_recovery']['retries']} | {totals['learned_top_front_clip_recovery']['attempts'] / totals['learned_top_front_clip_recovery']['episodes']:.3f} | 极端域少 1 次成功 |",
        "",
        "顶视恢复头与规则基线的成功/失败在 35 个配对 episode 中完全一致；没有可报告的成功率提升。双视角在 `severe_contact_shift / seed 883` 把一次可恢复失败误判为 `stop`，从而由成功回退为失败。该条件下只有一个不一致样本，精确双侧检验 `p=1.0`，不应宣称统计显著。",
        "",
        "## 失败案例审计",
        "",
        "| seed | 规则重试 | 顶视 CLIP | 双视角 CLIP | 解释 |",
        "| ---: | --- | --- | --- | --- |",
        "| 881 | 重试后仍失败 | 重试后仍失败 | 重试后仍失败 | 恢复轨迹本身不足，决策头无法修复执行误差。 |",
        "| 883 | 重试后成功 | 重试后成功 | 错误停止，失败 | 前视附加信息在小样本下产生了不利决策。 |",
        "",
        "## 最终决策",
        "",
        "本阶段的默认部署方案不切换为学习恢复头，仍采用：冻结 CLIP 语义适配 + 顶视 RGB 物体定位 + 结构化抓取/放置轨迹 + 一次 RGB 规则重试。顶视 CLIP 恢复头作为“真实失败样本上的轻量后训练可行性与负结果”保留；双视角作为失败消融保留，不能用于主结果。",
        "",
        "下一轮最有价值的改进不是扩大模型，而是采集更多极端接触域的首轮失败及多种恢复轨迹，使 `retry/stop` 选择有足够的有效样本；随后再评估是否能超过规则基线。",
        "",
        "## 视频证据",
        "",
        "`videos/clip_recovery_kaggle_v1/severe_seed883_clip_top_recovery_success.mp4`：极端接触扰动下，首轮失败，顶视 CLIP 头输出 `retry=0.879`，第二次执行成功。该视频仅展示一个成功恢复案例，不能替代上表中的 35 episode 定量评测。",
        "",
        "## 完整复现命令",
        "",
        "```powershell",
        'cd "C:\\Users\\Administrator\\Desktop\\final project-robot ic\\vla_robot_grasping"',
        '$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"',
        '& .\\.venv\\Scripts\\python.exe .\\scripts\\evaluate_clip_semantic_multiview_gate.py `',
        '  --model .\\outputs\\clip_semantic_waypoint\\clip_semantic_waypoint_core_v2_v1_20260721_110325.npz `',
        '  --terminal-top .\\outputs\\multiview_value_head\\terminal_mixed_top_v1.npz `',
        '  --recovery-top .\\outputs\\multiview_value_head\\recovery_value_top_v1.npz `',
        '  --terminal-top-front .\\outputs\\multiview_value_head\\terminal_mixed_top_front_v1.npz `',
        '  --recovery-top-front .\\outputs\\multiview_value_head\\recovery_value_top_front_v1.npz `',
        '  --clip-recovery-top .\\outputs\\kaggle_remote\\widowx_mujoco_clip_recovery_value_v1\\kaggle_clip_recovery_value_top_v1.npz `',
        '  --clip-recovery-top-front .\\outputs\\kaggle_remote\\widowx_mujoco_clip_recovery_value_v1\\kaggle_clip_recovery_value_top_front_v1.npz `',
        '  --recovery-mode clip_head --seed 880 --episodes 5 `',
        '  --recovery-train-seeds "760-799,820-869" --recovery-test-seeds "800-819" `',
        '  --output-json .\\outputs\\evaluations\\clip_semantic_clip_recovery_kaggle_v1.json `',
        '  --output-csv .\\docs\\clip_semantic_clip_recovery_kaggle_v1.csv `',
        '  --output-md .\\docs\\clip_semantic_clip_recovery_kaggle_v1.md',
        "```",
        "",
        "可视化复现实例：",
        "",
        "```powershell",
        'cd "C:\\Users\\Administrator\\Desktop\\final project-robot ic\\vla_robot_grasping"',
        '$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"',
        '& .\\.venv\\Scripts\\python.exe .\\scripts\\run_clip_semantic_multiview_gate.py `',
        '  --model .\\outputs\\clip_semantic_waypoint\\clip_semantic_waypoint_core_v2_v1_20260721_110325.npz `',
        '  --terminal-head .\\outputs\\multiview_value_head\\terminal_mixed_top_v1.npz `',
        '  --recovery-head .\\outputs\\multiview_value_head\\recovery_value_top_v1.npz `',
        '  --clip-recovery-head .\\outputs\\kaggle_remote\\widowx_mujoco_clip_recovery_value_v1\\kaggle_clip_recovery_value_top_v1.npz `',
        '  --recovery-mode clip_head --task move_leftmost_cube_to_bowl --complexity language --seed 883 `',
        '  --arm-kp 105 --arm-force 70 --gripper-kp 550 --gripper-force 75 --friction 0.8 `',
        '  --viewer --duration 35 --speed 0.5 `',
        '  --video-path .\\videos\\clip_recovery_kaggle_v1\\severe_seed883_clip_top_recovery_success.mp4',
        "```",
    ]
    output = ROOT / "docs" / "clip_recovery_kaggle_stage_v1.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report_path: {output}")


if __name__ == "__main__":
    main()
