from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "contact_hold_weighted_torch_act_v1_candidate"
DEMO_RUN = ROOT / "data" / "demos" / "contact_stage_demo_place_blue_cube_blue_pad_medium_v1"
FIXED_VIDEO_SEED = 0

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_contact_phase_gated_torch_act as base  # noqa: E402
from run_torch_act_policy import load_model  # noqa: E402


base.VERSION = VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the contact-hold weighted Torch ACT diagnostic candidate.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--task", default="place_blue_cube_blue_pad")
    parser.add_argument("--complexity", default="medium")
    parser.add_argument("--steps", type=int, default=2840)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--heldout-seed-start", type=int, default=100)
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / f"{VERSION}.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "contact_hold_weighted_torch_act_report.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "contact_hold_weighted_torch_act_report.md")
    return parser.parse_args()


def latest_model() -> Path:
    candidates = sorted((ROOT / "outputs" / "torch_act").glob(f"{VERSION}_*.pt"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no {VERSION} model found under outputs/torch_act")
    return candidates[-1]


def q(path: str | Path) -> str:
    return f'"{path}"'


def ps_command(script: str, script_args: list[str | Path]) -> str:
    rendered = [q(PYTHON), q(ROOT / script)]
    for value in script_args:
        rendered.append(q(value) if isinstance(value, Path) else str(value))
    return "& " + " ".join(rendered)


def write_md(path: Path, args: argparse.Namespace, model_path: Path, model: dict, rows: list[dict], summary_rows: list[dict]) -> None:
    metadata = model["metadata"]
    train_command = ps_command(
        "scripts/train_torch_act.py",
        [
            "--run-dir",
            DEMO_RUN,
            "--horizon",
            "12",
            "--history",
            "12",
            "--sample-stride",
            "8",
            "--d-model",
            "64",
            "--nhead",
            "4",
            "--encoder-layers",
            "2",
            "--decoder-layers",
            "2",
            "--dim-feedforward",
            "160",
            "--epochs",
            "10",
            "--batch-size",
            "256",
            "--lr",
            "0.00025",
            "--gripper-loss-weight",
            "12",
            "--phase-one-hot",
            "--augment-relative",
            "--phase-loss-weights",
            '"grasp:10,lift:10,transfer:4,place_release:3"',
            "--model-prefix",
            VERSION,
        ],
    )
    eval_command = ps_command("scripts/evaluate_contact_hold_weighted_torch_act.py", ["--model", model_path])
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
            str(FIXED_VIDEO_SEED),
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
            str(FIXED_VIDEO_SEED),
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
    video_path = ROOT / "outputs" / "videos" / f"{VERSION}_seed{FIXED_VIDEO_SEED}.mp4"

    lines = [
        "# Contact-hold Weighted Torch ACT 候选诊断报告",
        "",
        f"版本：`{VERSION}`",
        "",
        "定位：该版本仍属于 trajectory-conditioned BC / ACT 对照组。它使用 `contact_stage_demo_v1` 的 12 条成功 scripted contact-stage 示范，在 Torch ACT-style 动作块模型上加入相对几何特征、phase one-hot、grasp/lift/transfer 接触保持阶段损失加权和更高 gripper loss。它不是完整官方 ACT，也不是真实 VLA 后训练。",
        "",
        "## 1. 数据与模型",
        "",
        f"- 示范数据：`{DEMO_RUN.relative_to(ROOT).as_posix()}`",
        "- 示范采集：`contact_stage_demo_v1`，12/12 成功。",
        f"- 模型：`{model_path.relative_to(ROOT).as_posix()}`",
        f"- trainable_params：`{metadata.get('trainable_params')}`",
        f"- source_samples：`{metadata.get('source_samples')}`",
        f"- train_chunks：`{metadata.get('train_chunks')}`",
        f"- val_chunks：`{metadata.get('val_chunks')}`",
        f"- horizon/history：`{metadata.get('horizon')}/{metadata.get('history')}`",
        f"- augment_relative：`{metadata.get('augment_relative')}`",
        f"- phase_one_hot：`{metadata.get('phase_one_hot')}`",
        f"- gripper_loss_weight：`{metadata.get('gripper_loss_weight')}`",
        f"- phase_loss_weights：`{json.dumps(metadata.get('phase_loss_weights', {}), ensure_ascii=False)}`",
        f"- train_time_seconds：`{float(metadata.get('train_time_seconds', 0.0)):.2f}`",
        f"- train_mse_norm：`{float(metadata.get('train_mse_norm', 0.0)):.8f}`",
        f"- val_mse_norm：`{float(metadata.get('val_mse_norm', 0.0)):.8f}`",
        "",
        "## 2. 汇总结果",
        "",
        base.md_row(["范围", "放置 success", "TCP 抬升", "标准曾经抓取", "严格抓取+高度", "高度达标", "出界", "平均目标距离", "平均最高高度"]),
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
            "训练命令：",
            "",
            "```powershell",
            train_command,
            "```",
            "",
            "评测命令：",
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
            "固定视频导出命令：",
            "",
            "```powershell",
            export_command,
            "```",
            "",
            "固定视频证据：",
            "",
            "```text",
            video_path.relative_to(ROOT).as_posix(),
            video_path.with_suffix(".json").relative_to(ROOT).as_posix(),
            "```",
            "",
            "## 5. 论文边界",
            "",
            "该版本只能写成接触保持加权 ACT 候选诊断。只有同时满足 `success`、`tcp_grasp_lift_success` 或严格抓取相关指标，才能写成稳定抓取/抬升；否则只能作为失败模式或负例证据。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    model_path = args.model or latest_model()
    model = load_model(model_path)
    rows = []
    rows.extend(base.run_split(args, model, "train_range", int(args.train_seed_start)))
    rows.extend(base.run_split(args, model, "heldout", int(args.heldout_seed_start)))
    summary_rows = base.aggregate(rows)

    base.write_csv(args.output_csv, rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "version": VERSION,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "model": model_path.relative_to(ROOT).as_posix(),
                "demo_run": DEMO_RUN.relative_to(ROOT).as_posix(),
                "task": args.task,
                "complexity": args.complexity,
                "summary": summary_rows,
                "rows": rows,
                "fixed_video": f"outputs/videos/{VERSION}_seed{FIXED_VIDEO_SEED}.mp4",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_md(args.output_md, args, model_path, model, rows, summary_rows)
    print(f"model: {model_path}", flush=True)
    print(f"rows: {len(rows)}", flush=True)
    print(f"output_csv: {args.output_csv}", flush=True)
    print(f"output_json: {args.output_json}", flush=True)
    print(f"output_md: {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
