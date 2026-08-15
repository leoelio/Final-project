from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
VERSION = "contact_aware_phase_gated_torch_act_v1_candidate"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_contact_phase_gated_torch_act as base  # noqa: E402
from run_torch_act_policy import load_model  # noqa: E402


base.VERSION = VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the contact-aware phase-gated Torch ACT diagnostic candidate.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--task", default="place_blue_cube_blue_pad")
    parser.add_argument("--complexity", default="medium")
    parser.add_argument("--steps", type=int, default=2840)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--heldout-seed-start", type=int, default=100)
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / f"{VERSION}.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "contact_aware_phase_gated_torch_act_report.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "contact_aware_phase_gated_torch_act_report.md")
    return parser.parse_args()


def latest_model() -> Path:
    candidates = sorted((ROOT / "outputs" / "torch_act").glob(f"{VERSION}_*.pt"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no {VERSION} model found under outputs/torch_act")
    return candidates[-1]


def q(path: str | Path) -> str:
    return f'"{path}"'


def ps_command(script: str, script_args: list[str | Path]) -> str:
    rendered = [q(ROOT / ".venv" / "Scripts" / "python.exe"), q(ROOT / script)]
    for value in script_args:
        rendered.append(q(value) if isinstance(value, Path) else str(value))
    return "& " + " ".join(rendered)


def write_md(path: Path, args: argparse.Namespace, model_path: Path, model: dict, rows: list[dict], summary_rows: list[dict]) -> None:
    metadata = model["metadata"]
    train_command = ps_command(
        "scripts/train_torch_act.py",
        [
            "--run-dir",
            ROOT / "data" / "demos" / "place_blue_cube_blue_pad_medium_20260702_051752",
            "--horizon",
            "8",
            "--history",
            "8",
            "--sample-stride",
            "16",
            "--d-model",
            "64",
            "--nhead",
            "4",
            "--encoder-layers",
            "2",
            "--decoder-layers",
            "2",
            "--dim-feedforward",
            "128",
            "--epochs",
            "8",
            "--batch-size",
            "256",
            "--lr",
            "0.0003",
            "--gripper-loss-weight",
            "8",
            "--phase-one-hot",
            "--augment-relative",
            "--phase-loss-weights",
            '"grasp:5,lift:5,transfer:2,place_release:3"',
            "--model-prefix",
            VERSION,
        ],
    )
    eval_command = ps_command("scripts/evaluate_contact_aware_phase_gated_torch_act.py", ["--model", model_path])
    viewer_command = ps_command(
        "scripts/run_torch_act_policy.py",
        [
            "--model",
            model_path,
            "--task",
            args.task,
            "--complexity",
            args.complexity,
            "--seed",
            "0",
            "--episodes",
            "1",
            "--steps",
            str(args.steps),
            "--viewer",
            "--duration",
            "60",
            "--speed",
            "0.05",
            "--grasp-gate",
            "--action-alpha",
            "0.12",
            "--max-arm-delta",
            "0.006",
            "--max-gripper-delta",
            "0.00025",
            "--replan-interval",
            "4",
            "--temporal-ensemble",
            "--ensemble-decay",
            "0.1",
            "--gripper-kp",
            "900",
            "--gripper-force",
            "180",
            "--friction",
            "3.0",
            "--stop-on-unsafe",
            "--log-every",
            "500",
        ],
    )
    export_command = ps_command(
        "scripts/export_video.py",
        [
            "--method",
            "grasp_gated_torch_act",
            "--version",
            VERSION,
            "--model",
            model_path,
            "--task",
            args.task,
            "--complexity",
            args.complexity,
            "--seed",
            "0",
            "--steps",
            str(args.steps),
            "--camera",
            "top_rgb",
            "--fps",
            "24",
            "--frame-stride",
            "12",
            "--width",
            "640",
            "--height",
            "480",
            "--gripper-kp",
            "900",
            "--gripper-force",
            "180",
            "--friction",
            "3.0",
            "--log-every",
            "0",
        ],
    )
    video_path = ROOT / "outputs" / "videos" / f"{VERSION}_seed0.mp4"

    lines = [
        "# Contact-aware Phase-gated Torch ACT 候选诊断报告",
        "",
        f"版本：`{VERSION}`",
        "",
        "定位：这是 trajectory-conditioned BC / ACT 路线的一个诊断候选。它在 Torch ACT 的历史状态输入上追加相对几何特征，同时保留 phase one-hot、阶段损失加权、较高 gripper loss 和保守 grasp gate。它不是完整官方 ACT，也不是 VLA 后训练结果。",
        "",
        "## 1. 模型与训练配置",
        "",
        f"- 模型：`{model_path.relative_to(ROOT).as_posix()}`",
        f"- observation_dim：`{metadata.get('observation_dim')}`",
        f"- raw_observation_dim：`{metadata.get('raw_observation_dim')}`",
        f"- augment_relative：`{metadata.get('augment_relative')}`",
        f"- phase_one_hot：`{metadata.get('phase_one_hot')}`",
        f"- trainable_params：`{metadata.get('trainable_params')}`",
        f"- train_time_seconds：`{float(metadata.get('train_time_seconds', 0.0)):.2f}`",
        f"- peak_vram_mb：`{float(metadata.get('peak_vram_mb', 0.0)):.2f}`",
        f"- device：`{metadata.get('device')}`",
        f"- phase_loss_weights：`{json.dumps(metadata.get('phase_loss_weights', {}), ensure_ascii=False)}`",
        f"- train_mse_norm：`{float(metadata.get('train_mse_norm', 0.0)):.8f}`",
        f"- val_mse_norm：`{float(metadata.get('val_mse_norm', 0.0)):.8f}`",
        "",
        "## 2. 汇总结论",
        "",
        base.md_row(["范围", "放置 success", "TCP 抓取抬升", "曾经标准抓取", "严格抓取+高度", "高度达标", "出界", "平均目标距离", "平均最高高度"]),
        base.md_row(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for item in summary_rows:
        lines.append(
            base.md_row(
                [
                    item["split"],
                    f"{item['successes']}/{item['episodes']}",
                    f"{item['tcp_grasp_lift_successes']}/{item['episodes']}",
                    f"{item['ever_grasp_successes']}/{item['episodes']}",
                    f"{item['strict_grasp_lift_successes']}/{item['episodes']}",
                    f"{item['height_threshold_hits']}/{item['episodes']}",
                    f"{item['out_of_table']}/{item['episodes']}",
                    f"{float(item['mean_target_distance']):.4f}",
                    f"{float(item['mean_max_object_z']):.4f}",
                ]
            )
        )

    lines.extend(["", "## 3. 单次明细", ""])
    lines.append(base.md_row(["范围", "seed", "success", "tcp_lift", "standard_ever_grasp", "strict", "max_object_z", "target_distance", "gate_closed_steps"]))
    lines.append(base.md_row(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
    for row in rows:
        lines.append(
            base.md_row(
                [
                    row["split"],
                    row["seed"],
                    row["success"],
                    row["tcp_grasp_lift_success"],
                    row["ever_grasp_success"],
                    row["strict_grasp_lift_success"],
                    f"{float(row['max_object_z']):.4f}",
                    f"{float(row['target_distance']):.4f}",
                    row["gate_closed_steps"],
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 4. 完整命令",
            "",
            "固定视频证据：",
            "",
            "```text",
            video_path.relative_to(ROOT).as_posix(),
            video_path.with_suffix(".json").relative_to(ROOT).as_posix(),
            "```",
            "",
            "训练命令：",
            "",
            "```powershell",
            train_command,
            "```",
            "",
            "评估命令：",
            "",
            "```powershell",
            eval_command,
            "```",
            "",
            "慢速 viewer 命令：",
            "",
            "```powershell",
            viewer_command,
            "```",
            "",
            "重新导出固定视频命令：",
            "",
            "```powershell",
            export_command,
            "```",
            "",
            "## 5. 阶段判断",
            "",
            "如果该候选仍不能稳定获得 held-out TCP 抓取抬升和严格抓取成功，就说明当前 ACT-style baseline 的瓶颈不只是阶段标签或末端相对几何，而是接触反馈、视觉闭环和可泛化动作表示不足。论文中应把它写成 trajectory-conditioned / ACT 诊断负例，而不是写成稳定抓取成功方法。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    model_path = args.model or latest_model()
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    model = load_model(model_path)

    rows = []
    rows.extend(base.run_split(args, model, "train_range", args.train_seed_start))
    rows.extend(base.run_split(args, model, "heldout", args.heldout_seed_start))
    summary_rows = base.aggregate(rows)

    payload = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": model_path.relative_to(ROOT).as_posix(),
        "task": args.task,
        "complexity": args.complexity,
        "fixed_video": f"outputs/videos/{VERSION}_seed0.mp4",
        "rows": rows,
        "summary": summary_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    base.write_csv(args.output_csv, rows)
    write_md(args.output_md, args, model_path, model, rows, summary_rows)
    print(f"json_path: {args.output_json}", flush=True)
    print(f"csv_path: {args.output_csv}", flush=True)
    print(f"md_path: {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
