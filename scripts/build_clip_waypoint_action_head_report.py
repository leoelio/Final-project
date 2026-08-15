from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the exploratory frozen-CLIP waypoint action-head data-efficiency report.")
    parser.add_argument("--d5-model-json", type=Path, default=ROOT / "outputs" / "clip_waypoint_action_head" / "clip_waypoint_action_head_core_v2_d5_v1_20260729_101425.json")
    parser.add_argument("--d10-model-json", type=Path, default=ROOT / "outputs" / "clip_waypoint_action_head" / "clip_waypoint_action_head_core_v2_d10_v1_20260729_101440.json")
    parser.add_argument("--d20-model-json", type=Path, default=ROOT / "outputs" / "clip_waypoint_action_head" / "clip_waypoint_action_head_core_v2_v1_20260729_101013.json")
    parser.add_argument("--d5-evaluation", type=Path, default=ROOT / "outputs" / "evaluations" / "clip_waypoint_action_head_core_v2_d5_holdout_v1.json")
    parser.add_argument("--d10-evaluation", type=Path, default=ROOT / "outputs" / "evaluations" / "clip_waypoint_action_head_core_v2_d10_holdout_v1.json")
    parser.add_argument("--d20-evaluation", type=Path, default=ROOT / "outputs" / "evaluations" / "clip_waypoint_action_head_core_v2_holdout_v1.json")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / "clip_waypoint_action_head_data_efficiency_v1.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "clip_waypoint_action_head_stage_v1.md")
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    inputs = ((5, args.d5_model_json, args.d5_evaluation), (10, args.d10_model_json, args.d10_evaluation), (20, args.d20_model_json, args.d20_evaluation))
    rows = []
    for budget, model_path, evaluation_path in inputs:
        model = load(model_path)["metadata"]
        evaluation = load(evaluation_path)
        rows.append(
            {
                "demonstration_cap_per_task": budget,
                "effective_successful_episodes_by_task": [source["successful_episodes"] for source in model["sources"]],
                "augmented_samples": model["samples"],
                "train_samples": model["train_samples"],
                "validation_samples": model["validation_samples"],
                "validation_intent_accuracy": model["validation_metrics"]["intent_accuracy"],
                "validation_waypoint_rmse_m": model["validation_metrics"]["waypoint_rmse_m"],
                "closed_loop": evaluation["overall"],
                "model_path": load(model_path)["model"],
                "evaluation_path": str(evaluation_path),
            }
        )
    result = {
        "version": "clip_waypoint_action_head_data_efficiency_v1",
        "stage": "exploratory_frozen_clip_waypoint_action_head",
        "method_boundary": "Frozen CLIP image-language features feed a 65,990-parameter intent and 2D source-waypoint head. A structured expert executes continuous contact control. This is not an end-to-end VLA, OpenVLA, LoRA, or deployment candidate.",
        "offline_truth_boundary": "Initial object positions supervise training and calculate offline waypoint error only. Runtime planning uses top RGB, instruction, fixed scene target configuration, and predicted intent/waypoint.",
        "score_fix": "All reported holdout waypoint errors are measured against the pre-action object position. Earlier post-action values were superseded before this report was generated.",
        "rows": rows,
        "decision": "Do not deploy. The 20-demo cap reduces mean waypoint error relative to lower caps but yields 0/20 task success; the isolated 10-demo success is not stable across data budgets.",
        "videos": {
            "isolated_success": "videos/clip_waypoint_action_head_v1/seed24_d10_success.mp4",
            "near_miss_failure": "videos/clip_waypoint_action_head_v1/seed24_near_miss_failure.mp4",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    table = "\n".join(
        f"| {row['demonstration_cap_per_task']} | {'/'.join(map(str, row['effective_successful_episodes_by_task']))} | {row['augmented_samples']} | {row['validation_intent_accuracy']:.3f} | {row['validation_waypoint_rmse_m']:.4f} | {row['closed_loop']['semantic_correct']}/{row['closed_loop']['episodes']} | {row['closed_loop']['strict_grasp_success']}/{row['closed_loop']['episodes']} | {row['closed_loop']['task_success']}/{row['closed_loop']['episodes']} | {row['closed_loop']['mean_source_error_m']:.4f} |"
        for row in rows
    )
    markdown = f"""# 冻结 CLIP 二维抓取点动作头：探索性数据效率报告

版本：`clip_waypoint_action_head_data_efficiency_v1`

## 方法边界

该基线冻结 `openai/clip-vit-base-patch32`，以顶视 RGB 和任务指令的图文特征作为输入，训练一个 `65,990` 参数的 joint head，同时预测四类任务意图与源物体二维抓取点。预测抓取点和固定场景目标随后交给结构化 MuJoCo 抓取/放置执行器。

它是“冻结 VLM 表征 + 轻量参数化动作头”的探索性基线，不是端到端 VLA、OpenVLA、LoRA 或部署候选。连续接触控制仍由结构化执行器负责。物体真值只用于离线训练标签和评分；运行时不读取物体位置。

## 数据效率与独立闭环

训练每条示范使用 3 个任务保持的语言变体。四任务的实际成功示范数以 `蓝到蓝/蓝到红/红到红/最左到碗` 顺序列出；20 条上限中的空间任务只有 19 条成功示范。

| 每任务示范上限 | 实际示范数 | 图文样本 | 验证意图 | 验证抓取点 RMSE (m) | 独立意图 | 严格抓取 | 独立任务成功 | 独立平均抓取点误差 (m) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

独立闭环集固定为四任务各 5 个、共 20 个 seed-disjoint 场景。抓取点误差均相对**执行前**的真实源物体位置计算；此前一次后置评分时刻的实现错误已在生成本报告前修正并重跑三档评估。

## 结论

更多示范使验证抓取点 RMSE 从约 `8.30 cm` 降至 `5.80 cm`，独立平均误差从 `8.68 cm` 降至 `8.20 cm`，但没有形成稳定闭环收益：5 条和 20 条上限均为 `0/20`，10 条上限只有 `1/20`。这条唯一成功不是可部署证据，因为相邻数据预算未复现。

这个结果支持一个具体的技术判断：冻结 CLIP 的全局图文表征足以支持任务意图，但在当前小数据设置中，单个 MLP 无法把它转换为机械臂稳定抓取所需的厘米级空间精度。当前应继续使用经过审计的 RGB 几何定位作为空间前端；该动作头保留为失败基线和 Kaggle/更大数据训练的对照，而不进入默认策略。

## 可视化复核

- 孤立成功：`videos/clip_waypoint_action_head_v1/seed24_d10_success.mp4`。10 条/任务模型在独立 seed 24 中误差 `5.7 mm`，正确抓取并放入蓝盘；它仅对应 `1/20` 统计结果。
- 近失失败：`videos/clip_waypoint_action_head_v1/seed24_near_miss_failure.mp4`。20 条/任务模型在相同 seed 中误差约 `1.8 cm`，未形成严格抓取。

## 复现

```powershell
cd "C:\\Users\\Administrator\\Desktop\\final project-robot ic\\vla_robot_grasping"
$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"
& .\\.venv\\Scripts\\python.exe .\\scripts\\train_clip_waypoint_action_head.py --run-dirs .\\data\\demos\\core_v2_place_blue_cube_blue_pad_medium_train20_v1 .\\data\\demos\\core_v2_place_blue_cube_red_pad_medium_train20_v1 .\\data\\demos\\core_v2_place_red_cube_red_pad_medium_train20_v1 .\\data\\demos\\core_v2_move_leftmost_cube_to_bowl_language_train20_v1 --workspace-profile core_v2 --max-episodes-per-run 20 --language-augmentation semantic_alias_v1 --seed 20260729
& .\\.venv\\Scripts\\python.exe .\\scripts\\evaluate_clip_waypoint_action_head.py --model .\\outputs\\clip_waypoint_action_head\\clip_waypoint_action_head_core_v2_v1_20260729_101013.pt --episodes 5 --workspace-profile core_v2 --arm-kp 150 --arm-force 100 --gripper-kp 1200 --gripper-force 200 --friction 5 --place-tcp-z 0.041 --output-json .\\outputs\\evaluations\\clip_waypoint_action_head_core_v2_holdout_v1.json --output-csv .\\docs\\clip_waypoint_action_head_core_v2_holdout_v1.csv
& .\\.venv\\Scripts\\python.exe .\\scripts\\build_clip_waypoint_action_head_report.py
```
"""
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md), "decision": result["decision"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
