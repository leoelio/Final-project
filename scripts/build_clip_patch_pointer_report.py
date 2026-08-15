from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the frozen CLIP patch-pointer stage report.")
    parser.add_argument("--model-json", type=Path, default=ROOT / "outputs" / "clip_patch_pointer" / "clip_patch_pointer_core_v2_v1_20260729_103453.json")
    parser.add_argument("--evaluation-json", type=Path, default=ROOT / "outputs" / "evaluations" / "clip_patch_pointer_core_v2_holdout_v1.json")
    parser.add_argument("--global-baseline-json", type=Path, default=ROOT / "outputs" / "evaluations" / "clip_waypoint_action_head_core_v2_holdout_v1.json")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / "clip_patch_pointer_stage_v1.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "clip_patch_pointer_stage_v1.md")
    parser.add_argument("--stage-version", default="clip_patch_pointer_stage_v1")
    parser.add_argument("--preregistration", default="docs/clip_patch_pointer_preregistered_v1.md")
    parser.add_argument("--protocol-note", default="The preregistration specifies 300 epochs. The desktop runner interrupted a 300-epoch run before artifact saving, so the completed 80-epoch model is a local candidate run, not confirmation of the 300-epoch configuration.")
    parser.add_argument("--video-success", default="videos/clip_patch_pointer_v1/seed23_blue_success.mp4")
    parser.add_argument("--video-failure", default="videos/clip_patch_pointer_v1/seed221_red_failure.mp4")
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    model = load(args.model_json)["metadata"]
    evaluation = load(args.evaluation_json)
    global_baseline = load(args.global_baseline_json)["overall"]
    overall = evaluation["overall"]
    candidate_gate = overall["mean_source_error_m"] <= 0.03 and overall["task_success"] >= 8
    result = {
        "version": args.stage_version,
        "stage": "frozen_clip_patch_token_language_conditioned_pointer",
        "method_boundary": model["method_boundary"],
        "offline_truth_boundary": model["offline_truth_boundary"],
        "preregistration": args.preregistration,
        "protocol_note": args.protocol_note,
        "model": model,
        "closed_loop": evaluation,
        "comparison_to_global_clip_head": {
            "global_clip_20_demo_overall": global_baseline,
            "patch_pointer_25_cap_overall": overall,
            "interpretation": "The patch pointer has lower mean source error and more successes, but the budgets and head architectures differ; this is directional evidence, not a paired causal comparison.",
        },
        "candidate_gate": {
            "max_mean_source_error_m": 0.03,
            "min_task_success": 8,
            "passed": candidate_gate,
        },
        "decision": "Do not deploy and do not present as a completed VLA fine-tuning result. The 80-epoch patch-pointer candidate improves over the global CLIP MLP directionally but misses both preregistered closed-loop thresholds. Scale seed-diverse MuJoCo demonstrations before a Kaggle long-training run.",
        "videos": {
            "success": args.video_success,
            "failure": args.video_failure,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    task_rows = "\n".join(
        f"| {task} | {row['semantic_correct']}/{row['episodes']} | {row['strict_grasp_success']}/{row['episodes']} | {row['task_success']}/{row['episodes']} | {row['mean_source_error_m']:.4f} | {row['mean_target_distance_m']:.4f} |"
        for task, row in evaluation["by_task"].items()
    )
    markdown = f"""# 冻结 CLIP Patch 指针头：独立闭环阶段报告

版本：`{args.stage_version}`

## 问题与方法

上一阶段的全局 CLIP 图文 embedding 接 MLP 能识别意图，却不能稳定定位抓取点。本实验保留冻结 CLIP ViT-B/32 的 `7 x 7` 图像 patch token，使用文本特征条件化一个 `92,551` 参数的指针头，直接预测初始源物体像素位置；固定平面标定将像素转为机械臂 XY，结构化执行器完成连续接触控制。

运行时只使用顶视 RGB、语言、固定标定和模型预测。MuJoCo 对象真值只参与示范标签、执行前误差评分与任务判定，因此它是轻量 VLM 空间动作参数头，而不是端到端 VLA、OpenVLA 或 LoRA。

## 已完成训练

- 成功示范：蓝到蓝/蓝到红/红到红/最左到碗为 `{[item['successful_episodes'] for item in model['sources']]}` 条；每条使用原句和 3 条语言变体，共 `{model['samples']}` 图文样本。
- 冻结编码器：`{model['frozen_encoder_params']:,}` 参数；训练头：`{model['trainable_head_params']:,}` 参数。
- 本地训练：`{model['epochs']}` epoch、`{model['train_time_seconds']:.2f}` 秒、峰值 `{model['peak_vram_mb']:.1f} MB`。
- episode-disjoint 验证：意图准确率 `{model['validation_metrics']['intent_accuracy']:.3f}`，二维指针 RMSE `{model['validation_metrics']['pointer_rmse_m']:.4f} m`。

{args.protocol_note}

## 固定独立闭环（20 条）

固定留出为四任务各 5 个 seed，均与训练示范分离。

| 任务 | 语义正确 | 严格抓取 | 任务成功 | 平均源点误差 (m) | 平均目标距离 (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
{task_rows}

总体：语义 `{overall['semantic_correct']}/{overall['episodes']}`，严格抓取 `{overall['strict_grasp_success']}/{overall['episodes']}`，严格任务成功 `{overall['task_success']}/{overall['episodes']}`，平均源点误差 `{overall['mean_source_error_m']:.4f} m`，平均目标距离 `{overall['mean_target_distance_m']:.4f} m`。

## 对照与决策

相对全局 CLIP + MLP 的 20-demo 基线（`0/20`、平均源点误差 `{global_baseline['mean_source_error_m']:.4f} m`），patch 指针候选达到 `4/20`、`{overall['mean_source_error_m']:.4f} m`。这说明保留局部视觉 token 有方向性收益，但两者的示范上限和网络结构不同，不能表述为配对因果提升。

预注册候选门槛是平均源点误差不高于 `0.03 m` 且严格任务成功至少 `8/20`。本模型未通过：`{str(candidate_gate).lower()}`。因此不部署、不替换当前 RGB 几何前端，也不把它写成真实 VLA 微调成果。

下一步应优先收集更多 seed-diverse 的 MuJoCo 成功示范，并在 Kaggle 对同一空间指针模型做长训练；仅增加本地 epoch 没有证据能解决目前的验证泛化误差。

## Viewer 复核

- 成功：`{args.video_success}`。该 viewer 复核对应无泄漏独立成功 episode。
- 失败：`{args.video_failure}`。该 viewer 复核对应无泄漏独立失败 episode。

## 复现

```powershell
cd "C:\\Users\\Administrator\\Desktop\\final project-robot ic\\vla_robot_grasping"
$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"
& .\\.venv\\Scripts\\python.exe .\\scripts\\train_clip_patch_pointer.py --run-dirs .\\data\\demos\\core_v2_place_blue_cube_blue_pad_medium_25d_v1 .\\data\\demos\\core_v2_place_blue_cube_red_pad_medium_25d_v1 .\\data\\demos\\core_v2_place_red_cube_red_pad_medium_25d_v1 .\\data\\demos\\core_v2_move_leftmost_cube_to_bowl_language_25d_v1 --workspace-profile core_v2 --max-episodes-per-run 25 --epochs 80 --seed 20260729
& .\\.venv\\Scripts\\python.exe .\\scripts\\evaluate_clip_patch_pointer.py --model .\\outputs\\clip_patch_pointer\\clip_patch_pointer_core_v2_v1_20260729_103453.pt --episodes 5 --workspace-profile core_v2 --arm-kp 150 --arm-force 100 --gripper-kp 1200 --gripper-force 200 --friction 5 --place-tcp-z 0.041 --output-json .\\outputs\\evaluations\\clip_patch_pointer_core_v2_holdout_v1.json --output-csv .\\docs\\clip_patch_pointer_core_v2_holdout_v1.csv
& .\\.venv\\Scripts\\python.exe .\\scripts\\build_clip_patch_pointer_report.py
```
"""
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md), "candidate_gate_passed": candidate_gate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
