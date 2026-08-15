from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VERSION = "core_v2_pretrained_vlm_action_head_v1"
MODEL = ROOT / "outputs" / "clip_action_head" / "clip_core_v2_multitask_v1_20260721_104743.npz"

TASKS = [
    ("blue_to_blue", "蓝色立方体 -> 蓝盘", "主任务留出", "core_v2_clip_holdout_blue_cube_blue_pad"),
    ("blue_to_red", "蓝色立方体 -> 红盘", "目标区域迁移", "core_v2_clip_holdout_blue_cube_red_pad"),
    ("red_to_red", "红色立方体 -> 红盘", "目标物体迁移", "core_v2_clip_holdout_red_cube_red_pad"),
    ("leftmost_cube", "最左立方体 -> 碗", "空间关系与语言", "core_v2_clip_holdout_leftmost_cube_to_bowl"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Core V2 frozen pretrained VLM action-head report.")
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "core_v2_pretrained_vlm_action_head_report.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "core_v2_pretrained_vlm_action_head_report.md")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / f"{VERSION}.json")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def md_row(values: list[object]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"


def main() -> None:
    args = parse_args()
    with np.load(args.model) as data:
        metadata = json.loads(data["metadata"].item())

    rows: list[dict[str, object]] = []
    total_success = 0
    total_episodes = 0
    all_distances: list[float] = []
    total_out_of_table = 0
    total_grasp = 0
    evidence: list[dict[str, str]] = []
    for task_key, label, role, stem in TASKS:
        csv_path = ROOT / "docs" / f"{stem}.csv"
        json_path = ROOT / "outputs" / "evaluations" / f"{stem}.json"
        summary = read_csv(csv_path)[0]
        episodes = json.loads(json_path.read_text(encoding="utf-8"))["episodes_by_method"]["clip_core_v2_action_head"]
        successes = sum(int(item["success"]) for item in episodes)
        distances = [float(item["target_distance"]) for item in episodes]
        out_of_table = sum(int(item["out_of_table"]) for item in episodes)
        grasps = sum(int(item["grasp_success"]) for item in episodes)
        rows.append(
            {
                "任务key": task_key,
                "任务": label,
                "任务定位": role,
                "成功": summary["success"],
                "成功率": float(summary["success_rate"]),
                "平均目标距离": float(summary["mean_target_distance"]),
                "抓取成功次数": grasps,
                "物体出界次数": out_of_table,
                "证据CSV": csv_path.relative_to(ROOT).as_posix(),
                "证据JSON": json_path.relative_to(ROOT).as_posix(),
            }
        )
        total_success += successes
        total_episodes += len(episodes)
        all_distances.extend(distances)
        total_out_of_table += out_of_table
        total_grasp += grasps
        evidence.append({"csv": csv_path.relative_to(ROOT).as_posix(), "json": json_path.relative_to(ROOT).as_posix()})

    payload = {
        "version": VERSION,
        "method_key": "clip_core_v2_action_head",
        "method": "Frozen pretrained CLIP + lightweight continuous action head",
        "stage": "frozen_pretrained_vlm_action_head",
        "model_path": args.model.relative_to(ROOT).as_posix(),
        "model_metadata": metadata,
        "protocol": {
            "workspace_profile": "core_v2",
            "gripper": "kp=1200, force=200, friction=5.0",
            "place_tcp_z": 0.041,
            "split": "每项任务前 20 条 episode 训练，后 5 个 seed 留出；空间任务训练子集保留 19 条成功示范。",
        },
        "rows": rows,
        "summary": {
            "success": f"{total_success}/{total_episodes}",
            "success_rate": total_success / max(1, total_episodes),
            "mean_target_distance": float(np.mean(all_distances)),
            "grasp_successes": total_grasp,
            "out_of_table": total_out_of_table,
        },
        "evidence": evidence,
        "video": "outputs/videos/clip_core_v2_multitask_v1_leftmost_cube_seed420.mp4",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    train_dirs = " ".join(
        f'"{ROOT / "data" / "demos" / name}"'
        for name in (
            "core_v2_place_blue_cube_blue_pad_medium_train20_v1",
            "core_v2_place_blue_cube_red_pad_medium_train20_v1",
            "core_v2_place_red_cube_red_pad_medium_train20_v1",
            "core_v2_move_leftmost_cube_to_bowl_language_train20_v1",
        )
    )
    lines = [
        "# Core V2 预训练 VLM 动作头报告",
        "",
        f"版本：`{VERSION}`",
        "",
        "## 方法边界",
        "",
        "- 视觉语言编码器为真实预训练的 `openai/clip-vit-base-patch32`，参数冻结；仅训练 256-128 的连续动作头。",
        f"- 冻结编码器参数：`{metadata['frozen_encoder_params']:,}`；动作头训练样本：`{metadata['samples']}`；训练耗时：`{metadata['train_time_seconds']:.2f} s`；峰值显存：`{metadata['peak_vram_mb']:.1f} MB`。",
        "- 这是一项预训练 VLM 特征后训练实验，不是 OpenVLA、RT-2、LoRA 或端到端 VLA；本机 6GB GPU 不具备真实 OpenVLA LoRA 条件。",
        "",
        "## Core V2 留出集结果",
        "",
        md_row(["任务", "任务定位", "成功", "平均目标距离", "抓取成功", "物体出界"]),
        md_row(["---", "---", "---:", "---:", "---:", "---:"]),
    ]
    lines.extend(
        md_row([
            row["任务"],
            row["任务定位"],
            row["成功"],
            f"{float(row['平均目标距离']):.4f}",
            row["抓取成功次数"],
            row["物体出界次数"],
        ])
        for row in rows
    )
    lines.extend([
        "",
        f"总计：成功 `{total_success}/{total_episodes}`；平均目标距离 `{float(np.mean(all_distances)):.4f} m`；标准抓取成功 `{total_grasp}/{total_episodes}`；物体出界 `{total_out_of_table}/{total_episodes}`。",
        "",
        "## 可写结论",
        "",
        "- 在相同 Core V2 任务、相同 20/5 数据划分和相同物理参数下，低离线误差没有转化为任何一次闭环抓取或放置成功。",
        "- 与 `docs/core_v2_holdout_comparison_matrix.md` 中的普通 object-language action head 相同，连续单帧回归未解决接触保持、阶段转换和误差累积；预训练 CLIP 特征本身不能替代闭环轨迹建模。",
        "- 因此下一候选应比较“预训练 VLM 目标/阶段表征 + 结构化轨迹或动作块执行器”，并把该模型保留为真实预训练特征 action-head 的负对照。",
        "",
        "## 视频证据",
        "",
        "- 固定任务 `move_leftmost_cube_to_bowl`、seed `420` 的唯一代表性失败视频：`outputs/videos/clip_core_v2_multitask_v1_leftmost_cube_seed420.mp4`。",
        "- 视频显示未抓取、未到达目标；它只解释 20 次量化结果，不能替代该表的成功率结论。",
        "",
        "## 完整训练命令",
        "",
        "```powershell",
        "$env:VLA_TORCH_PACKAGE_DIR=\"D:\\vla_torch_cuda_pkgs\"",
        f'& "{ROOT / ".venv" / "Scripts" / "python.exe"}" "{ROOT / "scripts" / "train_clip_action_head.py"}" --run-dirs {train_dirs} --output "{ROOT / "outputs" / "clip_action_head"}" --model-prefix clip_core_v2_multitask_v1 --hidden-sizes "256,128" --epochs 12 --batch-size 128 --lr 0.0005 --weight-decay 0.000001 --gripper-loss-weight 4 --sample-stride 128 --max-samples 0 --image-size 224 --camera top_rgb --workspace-profile core_v2 --clip-batch-size 32 --device cuda --seed 0 --log-every-episodes 5',
        "```",
        "",
        "## 交互式 viewer 复查命令",
        "",
        "```powershell",
        "$env:VLA_TORCH_PACKAGE_DIR=\"D:\\vla_torch_cuda_pkgs\"",
        f'& "{ROOT / ".venv" / "Scripts" / "python.exe"}" "{ROOT / "scripts" / "run_clip_action_head.py"}" --model "{args.model}" --task move_leftmost_cube_to_bowl --complexity language --workspace-profile core_v2 --seed 420 --episodes 1 --steps 2840 --viewer --duration 45 --speed 0.25 --image-size 224 --camera top_rgb --vision-interval 64 --arm-kp 150 --arm-force 100 --gripper-kp 1200 --gripper-force 200 --friction 5 --clip-actions --action-alpha 0.2 --max-arm-delta 0.01 --max-gripper-delta 0.0005 --stop-on-unsafe --log-every 500',
        "```",
    ])
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report_md: {args.output_md}", flush=True)
    print(f"report_csv: {args.output_csv}", flush=True)
    print(f"report_json: {args.output_json}", flush=True)
    print(f"success: {total_success}/{total_episodes}", flush=True)


if __name__ == "__main__":
    main()
