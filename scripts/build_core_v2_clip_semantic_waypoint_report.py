from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VERSION = "core_v2_clip_semantic_waypoint_v1"
MODEL = ROOT / "outputs" / "clip_semantic_waypoint" / "clip_semantic_waypoint_core_v2_v1_20260721_110325.npz"
TASKS = [
    ("blue_to_blue", "蓝色立方体 -> 蓝盘", "主任务留出", "core_v2_clip_semantic_holdout_blue_cube_blue_pad"),
    ("blue_to_red", "蓝色立方体 -> 红盘", "目标区域迁移", "core_v2_clip_semantic_holdout_blue_cube_red_pad"),
    ("red_to_red", "红色立方体 -> 红盘", "目标物体迁移", "core_v2_clip_semantic_holdout_red_cube_red_pad"),
    ("leftmost_cube", "最左立方体 -> 碗", "空间关系与语言", "core_v2_clip_semantic_holdout_leftmost_cube_to_bowl"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Core V2 CLIP semantic-waypoint report.")
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "core_v2_clip_semantic_waypoint_report.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "core_v2_clip_semantic_waypoint_report.csv")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / f"{VERSION}.json")
    return parser.parse_args()


def md_row(values: list[object]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"


def main() -> None:
    args = parse_args()
    with np.load(args.model) as data:
        metadata = json.loads(data["metadata"].item())
    rows = []
    for key, label, role, stem in TASKS:
        source = ROOT / "outputs" / "evaluations" / f"{stem}.json"
        data = json.loads(source.read_text(encoding="utf-8"))
        rows.append({
            "任务key": key,
            "任务": label,
            "任务定位": role,
            "成功": data["success"],
            "语义正确": data["semantic_correct"],
            "严格抓取": data["strict_grasp_success"],
            "平均目标距离": float(data["mean_target_distance"]),
            "证据CSV": f"docs/{stem}.csv",
            "证据JSON": source.relative_to(ROOT).as_posix(),
        })
    success = sum(int(row["成功"].split("/", 1)[0]) for row in rows)
    semantic = sum(int(row["语义正确"].split("/", 1)[0]) for row in rows)
    strict_grasps = sum(int(row["严格抓取"].split("/", 1)[0]) for row in rows)
    total = sum(int(row["成功"].split("/", 1)[1]) for row in rows)
    payload = {
        "version": VERSION,
        "method_key": "clip_semantic_waypoint",
        "stage": "frozen_pretrained_vlm_semantic_hierarchical_policy",
        "model": args.model.relative_to(ROOT).as_posix(),
        "model_metadata": metadata,
        "protocol": {"workspace_profile": "core_v2", "gripper": "kp=1200, force=200, friction=5.0", "place_tcp_z": 0.041},
        "rows": rows,
        "summary": {"success": f"{success}/{total}", "semantic_correct": f"{semantic}/{total}", "strict_grasp_success": f"{strict_grasps}/{total}", "mean_target_distance": float(np.mean([row["平均目标距离"] for row in rows]))},
        "video": "outputs/videos/clip_semantic_waypoint_core_v2_v1_leftmost_cube_seed420.mp4",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    dirs = " ".join(f'"{ROOT / "data" / "demos" / item}"' for item in (
        "core_v2_place_blue_cube_blue_pad_medium_train20_v1",
        "core_v2_place_blue_cube_red_pad_medium_train20_v1",
        "core_v2_place_red_cube_red_pad_medium_train20_v1",
        "core_v2_move_leftmost_cube_to_bowl_language_train20_v1",
    ))
    lines = [
        "# Core V2 CLIP 语义-结构化执行报告",
        "",
        f"版本：`{VERSION}`",
        "",
        "## 方法与边界",
        "",
        "- 冻结 `openai/clip-vit-base-patch32`，训练一个 4 类任务意图线性 adapter；输入为初始俯视图和指令。",
        "- 执行阶段依据预测的意图，使用 MuJoCo 场景状态选择对象，并交由 scripted waypoint expert 完成接触、抓取和放置。严格抓取要求物体抬升至少 `0.06 m`，且在 TCP `0.06 m` 内持续至少 50 个仿真步。",
        "- 因此它是 VLM 语义决策与结构化控制的分层候选，不是端到端 VLA、连续 action-head、OpenVLA、LoRA 或真实机器人结果。",
        f"- adapter 训练样本 `{metadata['samples']}`，验证意图准确率 `{metadata['val_accuracy']:.3f}`，冻结编码器参数 `{metadata['frozen_encoder_params']:,}`，训练时间 `{metadata['train_time_seconds']:.2f} s`。",
        "",
        "## Core V2 留出集结果",
        "",
        md_row(["任务", "任务定位", "严格抓放成功", "意图正确", "严格抓取", "平均目标距离"]),
        md_row(["---", "---", "---:", "---:", "---:", "---:"]),
    ]
    lines.extend(md_row([row["任务"], row["任务定位"], row["成功"], row["语义正确"], row["严格抓取"], f"{row['平均目标距离']:.4f}"]) for row in rows)
    lines.extend([
        "",
        f"总计：严格抓放成功 `{success}/{total}`；意图预测正确 `{semantic}/{total}`；严格抓取 `{strict_grasps}/{total}`；平均目标距离 `{payload['summary']['mean_target_distance']:.4f} m`。",
        "",
        "## 对比结论",
        "",
        "- `core_v2_pretrained_vlm_action_head_v1` 的冻结 CLIP 连续 action-head 为 `0/20`；本方法为 `20/20`。差异说明连续单帧动作回归在接触阶段失败，而不是 CLIP 图文任务意图完全不可用。",
        "- 本方法的成功不能用来宣称端到端学习控制成功，因为拾取、抬升、释放由结构化 expert 执行，空间关系的具体对象还读取了仿真状态。",
        "- 论文中可将其用作“语义选择与接触执行解耦”的诊断上界；端到端 VLA 仍需以动作块、视觉闭环和真实机器人评测验证。",
        "",
        "## 视频证据",
        "",
        "- `outputs/videos/clip_semantic_waypoint_core_v2_v1_leftmost_cube_seed420.mp4`：空间关系任务的成功片段；预测为 leftmost-cube 意图，选择绿色方块并放入碗中。",
        "",
        "## 完整训练命令",
        "",
        "```powershell",
        "$env:VLA_TORCH_PACKAGE_DIR=\"D:\\vla_torch_cuda_pkgs\"",
        f'& "{ROOT / ".venv" / "Scripts" / "python.exe"}" "{ROOT / "scripts" / "train_clip_semantic_waypoint.py"}" --run-dirs {dirs} --output "{ROOT / "outputs" / "clip_semantic_waypoint"}" --model-prefix clip_semantic_waypoint_core_v2_v1 --workspace-profile core_v2 --epochs 200 --batch-size 32 --lr 0.02 --weight-decay 0.0001 --seed 0',
        "```",
        "",
        "## 交互式 viewer 命令",
        "",
        "```powershell",
        "$env:VLA_TORCH_PACKAGE_DIR=\"D:\\vla_torch_cuda_pkgs\"",
        f'& "{ROOT / ".venv" / "Scripts" / "python.exe"}" "{ROOT / "scripts" / "run_clip_semantic_waypoint.py"}" --model "{args.model}" --task move_leftmost_cube_to_bowl --complexity language --workspace-profile core_v2 --seed 420 --episodes 1 --viewer --duration 45 --speed 0.25 --image-size 224 --camera top_rgb --arm-kp 150 --arm-force 100 --gripper-kp 1200 --gripper-force 200 --friction 5 --place-tcp-z 0.041',
        "```",
    ])
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report_md: {args.output_md}", flush=True)
    print(f"success: {success}/{total}", flush=True)


if __name__ == "__main__":
    main()
