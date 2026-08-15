from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "outputs" / "kaggle_runs" / "clip_lora_gpu_v3_complete" / "clip_lora_patch_pointer_kaggle_v1.pt"
MODEL_METRICS = MODEL.with_suffix(".json")
LORA_HOLDOUT = ROOT / "outputs" / "evaluations" / "clip_lora_patch_pointer_kaggle_v1_holdout.json"
FROZEN_HOLDOUT = ROOT / "outputs" / "evaluations" / "clip_patch_pointer_kaggle_v2_cpu_holdout.json"
RGB_HOLDOUT = ROOT / "outputs" / "evaluations" / "clip_semantic_rgb_feedback_patch_pointer_holdout_v1.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def successes(rows: list[dict], key: str) -> int:
    return sum(bool(row[key]) for row in rows)


def ratio(value: int, total: int) -> str:
    return f"{value}/{total}"


def main() -> None:
    model = read(MODEL_METRICS)
    lora = read(LORA_HOLDOUT)
    frozen = read(FROZEN_HOLDOUT)
    rgb = read(RGB_HOLDOUT)
    metadata = model["metadata"]
    lora_rows = lora["rows"]
    rgb_retry = [row for row in rgb["rows"] if row["mode"] == "rgb_visual_retry"]
    seed_pairs = sorted((row["task"], int(row["seed"])) for row in lora_rows)
    if seed_pairs != sorted((row["task"], int(row["seed"])) for row in frozen["rows"]):
        raise ValueError("LoRA and frozen pointer holdouts are not seed-paired")
    if seed_pairs != sorted((row["task"], int(row["seed"])) for row in rgb_retry):
        raise ValueError("LoRA and RGB holdouts are not seed-paired")

    total = int(lora["overall"]["episodes"])
    lora_summary = {
        "semantic_correct": successes(lora_rows, "semantic_correct"),
        "strict_grasp_success": successes(lora_rows, "strict_grasp_success"),
        "task_success": successes(lora_rows, "task_success"),
        "mean_source_error_m": float(lora["overall"]["mean_source_error_m"]),
    }
    frozen_summary = {
        "semantic_correct": int(frozen["overall"]["semantic_correct"]),
        "strict_grasp_success": int(frozen["overall"]["strict_grasp_success"]),
        "task_success": int(frozen["overall"]["task_success"]),
        "mean_source_error_m": float(frozen["overall"]["mean_source_error_m"]),
    }
    rgb_summary = {
        "semantic_correct": successes(rgb_retry, "semantic_correct"),
        "visual_selection_correct": successes(rgb_retry, "visual_selection_correct"),
        "strict_grasp_success": successes(rgb_retry, "strict_grasp_success"),
        "task_success": successes(rgb_retry, "success"),
    }
    gate = lora_summary["mean_source_error_m"] <= 0.03 and lora_summary["task_success"] >= 8
    payload = {
        "version": "clip_lora_patch_pointer_stage_v1",
        "method": metadata["method"],
        "method_boundary": metadata["method_boundary"],
        "training": {
            "platform": "Kaggle Tesla P100 GPU",
            "pytorch": "2.6.0+cu118 with sm_60 verified before training",
            "checkpoint": str(MODEL.relative_to(ROOT)),
            "dataset_hash": metadata["dataset_content_sha256"],
            "epochs": int(metadata["epochs"]),
            "best_epoch": int(metadata["best_epoch"]),
            "best_validation_pointer_rmse_m": float(metadata["best_validation_pointer_rmse_m"]),
            "frozen_encoder_params": int(metadata["frozen_encoder_params"]),
            "trainable_lora_params": int(metadata["trainable_lora_params"]),
            "trainable_head_params": int(metadata["trainable_head_params"]),
            "trainable_total_params": int(metadata["trainable_total_params"]),
            "lora_config": metadata["lora_config"],
            "train_time_seconds": float(metadata["train_time_seconds"]),
            "peak_vram_mb": float(metadata["peak_vram_mb"]),
        },
        "holdout": {
            "episodes": total,
            "seed_disjoint_from_training": True,
            "same_seed_frozen_pointer": True,
            "same_seed_rgb_baseline": True,
            "lora": lora_summary,
            "frozen_393_scene_pointer": frozen_summary,
            "rgb_visual_retry": rgb_summary,
            "by_task": lora["by_task"],
        },
        "decision": {
            "candidate_gate_passed": gate,
            "deployment": "Do not deploy. Keep frozen CLIP intent + RGB geometry + structured executor as the default.",
            "interpretation": "LoRA adds one strict task success over the frozen 393-scene pointer but does not improve mean source error and misses both preregistered thresholds. It is a controlled low-rank adaptation result, not a successful VLA fine-tuning claim.",
        },
        "videos": [
            "videos/clip_lora_patch_pointer_kaggle_v1/seed23_blue_success.mp4",
            "videos/clip_lora_patch_pointer_kaggle_v1/seed420_leftmost_success.mp4",
        ],
    }
    output_json = ROOT / "outputs" / "evaluations" / "clip_lora_patch_pointer_stage_v1.json"
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# CLIP 视觉 LoRA 空间指针头：Kaggle GPU 与独立闭环",
        "",
        "版本：`clip_lora_patch_pointer_stage_v1`",
        "",
        "## 方法边界",
        "",
        "该候选只在 `openai/clip-vit-base-patch32` 视觉 Transformer 最后两层的 `q_proj/v_proj` 注入 rank-4、alpha-8 LoRA；文本编码器冻结，二维 pointer head、固定标定和结构化 MuJoCo 执行器不变。运行时只用顶视 RGB、语言和固定标定；MuJoCo 对象真值只用于离线标签与评分。因此它是轻量 VLM 的低秩视觉适配，不是端到端 VLA、OpenVLA、动作序列 LoRA 或真实机械臂实验。",
        "",
        "## Kaggle 训练",
        "",
        f"- GPU：Kaggle Tesla P100。使用官方 CUDA 11.8 PyTorch `2.6.0+cu118`，训练前确认 wheel 包含 `sm_60`。",
        f"- 数据：与冻结 393 场景 pointer 相同的数据包，hash `{metadata['dataset_content_sha256']}`；训练/验证为 `{metadata['train_samples']}/{metadata['validation_samples']}` 个扩增图文样本。",
        f"- 参数：冻结编码器 `{metadata['frozen_encoder_params']:,}`；LoRA `{metadata['trainable_lora_params']:,}`；pointer head `{metadata['trainable_head_params']:,}`；合计可训练 `{metadata['trainable_total_params']:,}`。",
        f"- 最优 checkpoint：epoch `{metadata['best_epoch']}`，验证 pointer RMSE `{metadata['best_validation_pointer_rmse_m'] * 100:.2f} cm`；训练计算时长 `{metadata['train_time_seconds']:.1f} s`，峰值 VRAM `{metadata['peak_vram_mb']:.1f} MB`。",
        "",
        "## 固定 20 条留出集与同 seed 对照",
        "",
        "| 方法 | 语义正确 | 严格抓取 | 任务成功 | 平均源点误差 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| LoRA Patch 指针 | {ratio(lora_summary['semantic_correct'], total)} | {ratio(lora_summary['strict_grasp_success'], total)} | {ratio(lora_summary['task_success'], total)} | {lora_summary['mean_source_error_m'] * 100:.2f} cm |",
        f"| 冻结 393 场景 Patch 指针 | {ratio(frozen_summary['semantic_correct'], total)} | {ratio(frozen_summary['strict_grasp_success'], total)} | {ratio(frozen_summary['task_success'], total)} | {frozen_summary['mean_source_error_m'] * 100:.2f} cm |",
        f"| 冻结 CLIP + RGB 几何 + 一次重试 | {ratio(rgb_summary['semantic_correct'], total)} | {ratio(rgb_summary['strict_grasp_success'], total)} | {ratio(rgb_summary['task_success'], total)} | RGB 几何定位 |",
        "",
        "四类任务的留出 seed 固定为 `20-24/120-124/220-224/420-424`，三个方法一一配对。LoRA 在蓝到蓝、红到红和最左到碗各成功 1 条，蓝到红为 0 条；同一 RGB 基线为 19/20。",
        "",
        "## 决策",
        "",
        f"LoRA 将任务成功从 `{frozen_summary['task_success']}/{total}` 增至 `{lora_summary['task_success']}/{total}`，但平均源点误差从 `{frozen_summary['mean_source_error_m'] * 100:.2f} cm` 变为 `{lora_summary['mean_source_error_m'] * 100:.2f} cm`，没有达到预注册的 `3 cm` 与 `8/20` 主门槛。通过：`{str(gate).lower()}`。不部署、不替换 RGB 几何方案，也不将该结果称为 VLA 微调成功。",
        "",
        "这条负结果仍然有研究价值：在相同数据、同一动作头和同一 holdout 下，视觉注意力 LoRA 并没有自动解决 7 x 7 patch 的厘米级定位与接触执行问题。后续若继续学习路线，应改进高分辨率空间表征或引入闭环视觉纠偏，而不是单纯扩大 rank 或 epoch。",
        "",
        "## Viewer 视频",
        "",
        "- `videos/clip_lora_patch_pointer_kaggle_v1/seed23_blue_success.mp4`：蓝方块到蓝盘，源点误差 7.9 mm，成功。",
        "- `videos/clip_lora_patch_pointer_kaggle_v1/seed420_leftmost_success.mp4`：最左方块到碗，源点误差 2.9 cm，成功。",
        "视频仅供定性复核；主结论以 20 条固定留出统计为准。",
        "",
        "## 完整复现命令",
        "",
        "```powershell",
        f'cd "{ROOT}"',
        '$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"',
        f'& ".\\.venv\\Scripts\\python.exe" ".\\scripts\\evaluate_clip_patch_pointer.py" --model ".\\{MODEL.relative_to(ROOT)}" --episodes 5 --workspace-profile core_v2 --arm-kp 150 --arm-force 100 --gripper-kp 1200 --gripper-force 200 --friction 5 --place-tcp-z 0.041 --output-json ".\\outputs\\evaluations\\clip_lora_patch_pointer_kaggle_v1_holdout.json" --output-csv ".\\outputs\\evaluations\\clip_lora_patch_pointer_kaggle_v1_holdout.csv"',
        f'& ".\\.venv\\Scripts\\python.exe" ".\\scripts\\build_clip_lora_patch_pointer_stage_report.py"',
        "```",
    ]
    output_md = ROOT / "docs" / "clip_lora_patch_pointer_stage_v1.md"
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {output_md}")
    print(f"artifact: {output_json}")


if __name__ == "__main__":
    main()
