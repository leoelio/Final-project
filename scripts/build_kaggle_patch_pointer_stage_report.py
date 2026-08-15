from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KAGGLE_MODEL = ROOT / "outputs" / "kaggle_runs" / "patch_pointer_v2_cpu_complete" / "clip_patch_pointer_kaggle_v2_cpu.pt"
KAGGLE_META = KAGGLE_MODEL.with_suffix(".json")
KAGGLE_HOLDOUT = ROOT / "outputs" / "evaluations" / "clip_patch_pointer_kaggle_v2_cpu_holdout.json"
LOCAL_HOLDOUT = ROOT / "outputs" / "evaluations" / "clip_patch_pointer_core_v2_v2_holdout.json"
RGB_HOLDOUT = ROOT / "outputs" / "evaluations" / "clip_semantic_rgb_feedback_patch_pointer_holdout_v1.json"
DATA_AUDIT = ROOT / "outputs" / "evaluations" / "kaggle_spatial_data_collection_v1.json"
PACK = ROOT / "outputs" / "evaluations" / "kaggle_patch_pointer_pack_v2.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def count(rows: list[dict], key: str) -> int:
    return sum(bool(row[key]) for row in rows)


def format_ratio(value: int, total: int) -> str:
    return f"{value}/{total}"


def main() -> None:
    model = read_json(KAGGLE_META)
    kaggle = read_json(KAGGLE_HOLDOUT)
    local = read_json(LOCAL_HOLDOUT)
    rgb = read_json(RGB_HOLDOUT)
    data_audit = read_json(DATA_AUDIT)
    pack = read_json(PACK)

    metadata = model["metadata"]
    if metadata["dataset_content_sha256"] != pack["dataset_content_sha256"]:
        raise ValueError("Kaggle checkpoint and dataset pack hashes do not match")
    if int(data_audit["total_successful_episodes"]) != int(pack["samples"]):
        raise ValueError("Kaggle data audit and pack sample count do not match")

    patch_rows = kaggle["rows"]
    rgb_open = [row for row in rgb["rows"] if row["mode"] == "rgb_open_loop"]
    rgb_retry = [row for row in rgb["rows"] if row["mode"] == "rgb_visual_retry"]
    patch_seeds = sorted((row["task"], int(row["seed"])) for row in patch_rows)
    if patch_seeds != sorted((row["task"], int(row["seed"])) for row in rgb_open):
        raise ValueError("patch-pointer and RGB open-loop holdouts are not seed-paired")
    if patch_seeds != sorted((row["task"], int(row["seed"])) for row in rgb_retry):
        raise ValueError("patch-pointer and RGB retry holdouts are not seed-paired")

    total = len(patch_rows)
    local_total = int(local["overall"]["episodes"])
    paired_rows = {
        "patch_pointer": {
            "semantic_correct": count(patch_rows, "semantic_correct"),
            "strict_grasp_success": count(patch_rows, "strict_grasp_success"),
            "task_success": count(patch_rows, "task_success"),
            "mean_source_error_m": float(kaggle["overall"]["mean_source_error_m"]),
        },
        "rgb_open_loop": {
            "semantic_correct": count(rgb_open, "semantic_correct"),
            "visual_selection_correct": count(rgb_open, "visual_selection_correct"),
            "strict_grasp_success": count(rgb_open, "strict_grasp_success"),
            "task_success": count(rgb_open, "success"),
        },
        "rgb_visual_retry": {
            "semantic_correct": count(rgb_retry, "semantic_correct"),
            "visual_selection_correct": count(rgb_retry, "visual_selection_correct"),
            "strict_grasp_success": count(rgb_retry, "strict_grasp_success"),
            "task_success": count(rgb_retry, "success"),
        },
    }
    pointer_gate = (
        paired_rows["patch_pointer"]["mean_source_error_m"] <= 0.03
        and paired_rows["patch_pointer"]["task_success"] >= 8
    )
    payload = {
        "version": "kaggle_frozen_clip_patch_pointer_stage_v2",
        "method": metadata["method"],
        "method_boundary": metadata["method_boundary"],
        "training": {
            "platform": "Kaggle",
            "execution": "CPU fallback; the initial P100 run was incompatible with the supplied PyTorch build, so this result is not a GPU-speed claim.",
            "checkpoint": str(KAGGLE_MODEL.relative_to(ROOT)),
            "dataset_hash": metadata["dataset_content_sha256"],
            "successful_mujoco_episodes": int(data_audit["total_successful_episodes"]),
            "train_validation_scenes": {"train": int(pack["per_split"]["train"]), "validation": int(pack["per_split"]["validation"])},
            "expanded_language_image_samples": int(metadata["expanded_samples"]),
            "frozen_encoder_params": int(metadata["frozen_encoder_params"]),
            "trainable_head_params": int(metadata["trainable_head_params"]),
            "best_epoch": int(metadata["best_epoch"]),
            "best_validation_pointer_rmse_m": float(metadata["best_validation_pointer_rmse_m"]),
            "train_time_seconds": float(metadata["train_time_seconds"]),
        },
        "holdout": {
            "episodes": total,
            "seed_disjoint_from_training": True,
            "same_seed_paired_rgb_baseline": True,
            "patch_pointer": paired_rows["patch_pointer"],
            "local_20_demo_patch_pointer": {
                "episodes": local_total,
                "task_success": int(local["overall"]["task_success"]),
                "mean_source_error_m": float(local["overall"]["mean_source_error_m"]),
            },
            "rgb_open_loop": paired_rows["rgb_open_loop"],
            "rgb_visual_retry": paired_rows["rgb_visual_retry"],
        },
        "decision": {
            "candidate_gate_passed": pointer_gate,
            "deployment": "Keep frozen CLIP intent + RGB geometry + structured executor as the default. Do not deploy the learned patch pointer.",
            "interpretation": "The 393-scene checkpoint lowers point error against the local 20-demo pointer, but its strict end-to-end success does not improve. This is evidence of a spatial-resolution/contact bottleneck, not a VLA fine-tuning success claim.",
        },
        "videos": [
            "videos/clip_patch_pointer_kaggle_v2/seed23_blue_success.mp4",
            "videos/clip_patch_pointer_kaggle_v2/seed423_leftmost_success.mp4",
        ],
    }
    output_json = ROOT / "outputs" / "evaluations" / "kaggle_frozen_clip_patch_pointer_stage_v2.json"
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    pointer = paired_rows["patch_pointer"]
    local_pointer = payload["holdout"]["local_20_demo_patch_pointer"]
    lines = [
        "# Kaggle 冻结 CLIP Patch 空间头：独立闭环与同 seed 对照",
        "",
        "版本：`kaggle_frozen_clip_patch_pointer_stage_v2`",
        "",
        "## 方法边界",
        "",
        "本阶段训练的是冻结 `openai/clip-vit-base-patch32` 的 `7 x 7` 图像 patch token 与语言特征之上的轻量二维指针/动作参数头。运行时输入仅为顶视 RGB、语言和固定相机-平面标定；MuJoCo 对象真值仅用于离线示范标签、误差评分与成功判定。它是轻量 VLM 空间后训练实验，不是端到端 VLA、OpenVLA LoRA、OFT 或真实机械臂实验。",
        "",
        "## 数据与训练",
        "",
        f"- MuJoCo 成功示范：`{data_audit['total_successful_episodes']}` 条，四任务分别为 `99 / 99 / 99 / 96`，固定 20 条留出 seed 未进入训练。",
        f"- Kaggle 数据包：`{pack['per_split']['train']}` 个训练场景、`{pack['per_split']['validation']}` 个验证场景；内容 hash 为 `{metadata['dataset_content_sha256']}`。",
        f"- 语言扩增后训练输入：`{metadata['expanded_samples']}` 个图文样本；冻结编码器 `{metadata['frozen_encoder_params']:,}` 参数，可训练头 `{metadata['trainable_head_params']:,}` 参数。",
        f"- 远端 checkpoint：第 `{metadata['best_epoch']}` 个 epoch 的最低记录验证 RMSE，`{metadata['best_validation_pointer_rmse_m']:.4f} m`；训练时长 `{metadata['train_time_seconds']:.1f} s`。",
        "- 远端执行为 Kaggle CPU 回退：初始 P100 任务与当时的 PyTorch 二进制不兼容，因此本轮不把它写成 GPU 加速结果。",
        "",
        "## 固定独立留出集（20 条）",
        "",
        "| 方法 | 语义正确 | 视觉选物正确 | 严格抓取 | 任务成功 | 平均源点误差 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Kaggle 冻结 CLIP patch 指针 | {format_ratio(pointer['semantic_correct'], total)} | - | {format_ratio(pointer['strict_grasp_success'], total)} | {format_ratio(pointer['task_success'], total)} | {pointer['mean_source_error_m'] * 100:.2f} cm |",
        f"| 本地 20-demo patch 指针 | {format_ratio(local['overall']['semantic_correct'], local_total)} | - | {format_ratio(local['overall']['strict_grasp_success'], local_total)} | {format_ratio(local_pointer['task_success'], local_total)} | {local_pointer['mean_source_error_m'] * 100:.2f} cm |",
        f"| 冻结 CLIP + RGB 几何（单次） | {format_ratio(paired_rows['rgb_open_loop']['semantic_correct'], total)} | {format_ratio(paired_rows['rgb_open_loop']['visual_selection_correct'], total)} | {format_ratio(paired_rows['rgb_open_loop']['strict_grasp_success'], total)} | {format_ratio(paired_rows['rgb_open_loop']['task_success'], total)} | RGB 几何定位 |",
        f"| 冻结 CLIP + RGB 几何（最多一次重试） | {format_ratio(paired_rows['rgb_visual_retry']['semantic_correct'], total)} | {format_ratio(paired_rows['rgb_visual_retry']['visual_selection_correct'], total)} | {format_ratio(paired_rows['rgb_visual_retry']['strict_grasp_success'], total)} | {format_ratio(paired_rows['rgb_visual_retry']['task_success'], total)} | RGB 几何定位 |",
        "",
        "RGB 两个模式使用与 patch 指针完全相同的 20 个 `(task, seed)` 初始状态：蓝到蓝 `20-24`、蓝到红 `120-124`、红到红 `220-224`、最左到碗 `420-424`。RGB 对照的唯一失败为红方块任务 seed `222` 的初始 RGB 定位失败；重试不可能修复“尚未执行首轮动作”的失效，因此该固定子集上单次与重试均为 `19/20`。",
        "",
        "## 结论与下一步",
        "",
        f"393 场景 checkpoint 的源点误差由 `{local_pointer['mean_source_error_m'] * 100:.2f} cm` 降至 `{pointer['mean_source_error_m'] * 100:.2f} cm`，但严格任务成功由 `{local_pointer['task_success']}/{local_total}` 变为 `{pointer['task_success']}/{total}`。两次训练的数据规模与训练轮数不同，因此这是方向性数据规模证据，不应表述为严格因果消融。",
        f"预注册候选门槛为平均源点误差不高于 `3 cm` 且严格任务成功至少 `8/20`；本轮通过：`{str(pointer_gate).lower()}`。学习空间头不部署，默认方案继续使用冻结 CLIP 语义、RGB 几何定位、结构化执行和最多一次 RGB 重试。",
        "下一阶段应在同一 393 场景协议下加入低秩视觉适配，并与冻结 patch 指针使用同一留出集比较，以验证能否突破 `7 x 7` patch 空间分辨率和接触执行的双重瓶颈。",
        "",
        "## Viewer 视频",
        "",
        "- 成功：`videos/clip_patch_pointer_kaggle_v2/seed23_blue_success.mp4`，蓝色方块放入蓝色垫。",
        "- 成功：`videos/clip_patch_pointer_kaggle_v2/seed423_leftmost_success.mp4`，最左方块放入碗。",
        "视频只用于定性复核，结论以固定留出集汇总表为准。",
        "",
        "## 完整复现命令",
        "",
        "```powershell",
        f'cd "{ROOT}"',
        '$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"',
        f'& ".\\.venv\\Scripts\\python.exe" ".\\scripts\\evaluate_clip_patch_pointer.py" --model ".\\{KAGGLE_MODEL.relative_to(ROOT)}" --episodes 5 --workspace-profile core_v2 --arm-kp 150 --arm-force 100 --gripper-kp 1200 --gripper-force 200 --friction 5 --place-tcp-z 0.041 --output-json ".\\outputs\\evaluations\\clip_patch_pointer_kaggle_v2_cpu_holdout.json" --output-csv ".\\outputs\\evaluations\\clip_patch_pointer_kaggle_v2_cpu_holdout.csv"',
        f'& ".\\.venv\\Scripts\\python.exe" ".\\scripts\\evaluate_clip_semantic_rgb_feedback.py" --model ".\\outputs\\clip_semantic_waypoint\\clip_semantic_waypoint_core_v2_v1_20260721_110325.npz" --seed 20 --task-seed-offsets 0,100,200,400 --episodes 5 --domains nominal --modes rgb_open_loop,rgb_visual_retry --output-json ".\\outputs\\evaluations\\clip_semantic_rgb_feedback_patch_pointer_holdout_v1.json" --output-csv ".\\docs\\clip_semantic_rgb_feedback_patch_pointer_holdout_v1.csv" --output-md ".\\docs\\clip_semantic_rgb_feedback_patch_pointer_holdout_v1.md" --log-every 0',
        f'& ".\\.venv\\Scripts\\python.exe" ".\\scripts\\build_kaggle_patch_pointer_stage_report.py"',
        "```",
    ]
    output_md = ROOT / "docs" / "kaggle_frozen_clip_patch_pointer_stage_v2.md"
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {output_md}")
    print(f"artifact: {output_json}")


if __name__ == "__main__":
    main()
