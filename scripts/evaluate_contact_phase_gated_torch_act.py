from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "contact_phase_gated_torch_act_v1_candidate"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_torch_act_policy import configure_env, load_model, rollout_with_env  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the contact/phase-gated Torch ACT diagnostic candidate.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--task", default="place_blue_cube_blue_pad")
    parser.add_argument("--complexity", default="medium")
    parser.add_argument("--steps", type=int, default=2840)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--heldout-seed-start", type=int, default=100)
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / f"{VERSION}.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "contact_phase_gated_torch_act_report.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "contact_phase_gated_torch_act_report.md")
    return parser.parse_args()


def latest_model() -> Path:
    candidates = sorted((ROOT / "outputs" / "torch_act").glob(f"{VERSION}_*.pt"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no {VERSION} model found under outputs/torch_act")
    return candidates[-1]


def control_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        steps=int(args.steps),
        viewer=False,
        duration=0.0,
        speed=0.0,
        arm_kp=150.0,
        arm_force=100.0,
        gripper_kp=900.0,
        gripper_force=180.0,
        friction=3.0,
        clip_actions=True,
        action_alpha=0.12,
        max_arm_delta=0.006,
        max_gripper_delta=0.00025,
        replan_interval=4,
        temporal_ensemble=True,
        ensemble_decay=0.1,
        stop_on_unsafe=True,
        log_every=0,
        grasp_gate=True,
        close_phase=0.22,
        release_phase=0.78,
        near_threshold=0.11,
        release_distance=0.095,
        open_gripper=0.037,
        close_gripper=0.015,
    )


def run_split(args: argparse.Namespace, model: dict, split: str, seed_start: int) -> list[dict]:
    runner_args = control_args(args)
    rows = []
    for offset in range(int(args.episodes)):
        seed = int(seed_start) + offset
        env, obs = configure_env(runner_args, seed, str(args.task), str(args.complexity))
        summary = rollout_with_env(runner_args, model, env, obs, seed, str(args.task), str(args.complexity), viewer=None)
        row = {
            "version": VERSION,
            "split": split,
            "seed": seed,
            "task": str(args.task),
            "complexity": str(args.complexity),
            "success": bool(summary["success"]),
            "target_distance": float(summary["target_distance"]),
            "object_z": float(summary["object_z"]),
            "max_object_z": float(summary["max_object_z"]),
            "height_threshold_hit": bool(summary["height_threshold_hit"]),
            "grasp_success": bool(summary["grasp_success"]),
            "ever_grasp_success": bool(summary["ever_grasp_success"]),
            "ever_tcp_lift_success": bool(summary["ever_tcp_lift_success"]),
            "tcp_grasp_lift_success": bool(summary["tcp_grasp_lift_success"]),
            "strict_grasp_lift_success": bool(summary["strict_grasp_lift_success"]),
            "out_of_table": bool(summary["out_of_table"]),
            "steps_taken": int(summary["steps_taken"]),
            "stop_reason": summary["stop_reason"],
            "min_tcp_object_distance": summary["min_tcp_object_distance"],
            "min_tcp_object_distance_while_lifted": summary["min_tcp_object_distance_while_lifted"],
            "mean_action_norm": float(summary["mean_action_norm"]),
            "max_action_norm": float(summary["max_action_norm"]),
            "gate_open_steps": int(summary.get("gate_open_steps", 0)),
            "gate_closed_steps": int(summary.get("gate_closed_steps", 0)),
            "gate_policy_steps": int(summary.get("gate_policy_steps", 0)),
        }
        rows.append(row)
        print(
            f"{split} seed={seed} success={row['success']} tcp_lift={row['tcp_grasp_lift_success']} "
            f"strict={row['strict_grasp_lift_success']} max_object_z={row['max_object_z']:.4f} "
            f"target_distance={row['target_distance']:.4f}",
            flush=True,
        )
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row["split"]), []).append(row)

    output = []
    for split, items in groups.items():
        output.append(
            {
                "split": split,
                "episodes": len(items),
                "successes": sum(1 for item in items if item["success"]),
                "tcp_grasp_lift_successes": sum(1 for item in items if item["tcp_grasp_lift_success"]),
                "ever_grasp_successes": sum(1 for item in items if item["ever_grasp_success"]),
                "strict_grasp_lift_successes": sum(1 for item in items if item["strict_grasp_lift_success"]),
                "height_threshold_hits": sum(1 for item in items if item["height_threshold_hit"]),
                "out_of_table": sum(1 for item in items if item["out_of_table"]),
                "mean_target_distance": statistics.fmean(float(item["target_distance"]) for item in items),
                "mean_max_object_z": statistics.fmean(float(item["max_object_z"]) for item in items),
            }
        )
    return output


def q(path: str | Path) -> str:
    return f'"{path}"'


def ps_command(script: str, script_args: list[str | Path]) -> str:
    rendered = [q(PYTHON), q(ROOT / script)]
    for value in script_args:
        rendered.append(q(value) if isinstance(value, Path) else str(value))
    return "& " + " ".join(rendered)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def md_row(values: list[object]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"


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
            "--phase-loss-weights",
            '"grasp:5,lift:5,transfer:2,place_release:3"',
            "--model-prefix",
            VERSION,
        ],
    )
    eval_command = ps_command("scripts/evaluate_contact_phase_gated_torch_act.py", ["--model", model_path])
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
        "# Contact/Phase-gated Torch ACT 候选诊断报告",
        "",
        f"版本：`{VERSION}`",
        "",
        "定位：该版本把 phase-one-hot、grasp/lift/place_release 阶段加权、较高 gripper loss 和保守 grasp gate 组合到同一个 Torch ACT-style 候选中，用于诊断接触保持、夹爪闭合和动作块阶段切换是否仍是瓶颈。它不是完整官方 ACT，也不是 VLA 后训练。",
        "",
        "## 1. 模型与训练配置",
        "",
        f"- 模型：`{model_path.relative_to(ROOT).as_posix()}`",
        f"- trainable_params：`{metadata.get('trainable_params')}`",
        f"- train_time_seconds：`{float(metadata.get('train_time_seconds', 0.0)):.2f}`",
        f"- peak_vram_mb：`{float(metadata.get('peak_vram_mb', 0.0)):.2f}`",
        f"- device：`{metadata.get('device')}`",
        f"- phase_one_hot：`{metadata.get('phase_one_hot')}`",
        f"- phase_loss_weights：`{json.dumps(metadata.get('phase_loss_weights', {}), ensure_ascii=False)}`",
        f"- train_mse_norm：`{float(metadata.get('train_mse_norm', 0.0)):.8f}`",
        f"- val_mse_norm：`{float(metadata.get('val_mse_norm', 0.0)):.8f}`",
        "",
        "## 2. 汇总结果",
        "",
        md_row(["范围", "放置 success", "TCP 抬升", "标准曾经抓取", "严格抓取+高度", "高度达标", "出界", "平均目标距离", "平均最高高度"]),
        md_row(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for item in summary_rows:
        lines.append(
            md_row(
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
    lines.append(md_row(["范围", "seed", "success", "tcp_lift", "standard_ever_grasp", "strict", "max_object_z", "target_distance", "gate_closed_steps"]))
    lines.append(md_row(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
    for row in rows:
        lines.append(
            md_row(
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
            "重新导出固定视频命令：",
            "",
            "```powershell",
            export_command,
            "```",
            "",
            "## 5. 阶段结论",
            "",
            "如果该候选仍没有 held-out TCP 抬升和严格抓取成功，则说明当前 ACT-style 模型的主要短板不只是阶段标签或夹爪门控，而是接触几何、闭环视觉/触觉反馈和可泛化动作表示不足。论文中只能写成 ACT-style 诊断负例，不能写成完整 ACT 或稳定抓取成功。",
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
    rows.extend(run_split(args, model, "train_range", args.train_seed_start))
    rows.extend(run_split(args, model, "heldout", args.heldout_seed_start))
    summary_rows = aggregate(rows)

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
    write_csv(args.output_csv, rows)
    write_md(args.output_md, args, model_path, model, rows, summary_rows)
    print(f"json_path: {args.output_json}", flush=True)
    print(f"csv_path: {args.output_csv}", flush=True)
    print(f"md_path: {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
