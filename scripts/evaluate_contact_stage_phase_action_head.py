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
VERSION = "contact_stage_phase_action_head_v1_candidate"
DEMO_RUN = ROOT / "data" / "demos" / "contact_stage_demo_place_blue_cube_blue_pad_medium_v1"
FIXED_VIDEO_SEED = 101

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_phase_action_head import configure_env, load_model, rollout_with_env  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the phase action-head trained on contact-stage demos.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--task", default="place_blue_cube_blue_pad")
    parser.add_argument("--complexity", default="medium")
    parser.add_argument("--steps", type=int, default=2840)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--phase-mode", choices=("progress", "state", "hybrid"), default="progress")
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--heldout-seed-start", type=int, default=100)
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / f"{VERSION}.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "contact_stage_phase_action_head_report.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "contact_stage_phase_action_head_report.md")
    return parser.parse_args()


def latest_model() -> Path:
    candidates = sorted((ROOT / "outputs" / "phase_action_head").glob(f"{VERSION}_*.npz"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no {VERSION} model found under outputs/phase_action_head")
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
        stop_on_unsafe=True,
        log_every=0,
        phase_mode=str(args.phase_mode),
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
            "phase_mode": str(args.phase_mode),
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
            "phase_counts": json.dumps(summary["phase_counts"], ensure_ascii=False, sort_keys=True),
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
        "scripts/train_phase_action_head.py",
        [
            "--run-dir",
            DEMO_RUN,
            "--model-prefix",
            VERSION,
            "--hidden-sizes",
            "192,192",
            "--epochs",
            "10",
            "--batch-size",
            "1024",
            "--lr",
            "0.001",
            "--gripper-loss-weight",
            "8",
            "--seed",
            str(FIXED_VIDEO_SEED),
        ],
    )
    eval_command = ps_command("scripts/evaluate_contact_stage_phase_action_head.py", ["--model", model_path, "--phase-mode", args.phase_mode])
    viewer_command = ps_command(
        "scripts/run_phase_action_head.py",
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
            "--phase-mode",
            args.phase_mode,
            "--action-alpha",
            "0.12",
            "--max-arm-delta",
            "0.006",
            "--max-gripper-delta",
            "0.00025",
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
            "phase_action_head",
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
            "--phase-mode",
            args.phase_mode,
            "--action-alpha",
            "0.12",
            "--max-arm-delta",
            "0.006",
            "--max-gripper-delta",
            "0.00025",
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
    val_mse = float(metadata.get("val_mse", 0.0))

    lines = [
        "# Contact-stage Phase Action-Head 候选诊断报告",
        "",
        f"版本：`{VERSION}`",
        "",
        "定位：该版本使用 `contact_stage_demo_v1` 的 12 条成功 scripted contact-stage 示范训练分阶段 MLP action head。它是轻量 action-head / Adapter 后训练代理，不是 pretrained VLA、OpenVLA LoRA，也不是完整 ACT。",
        "",
        "## 1. 数据与模型",
        "",
        f"- 示范数据：`{DEMO_RUN.relative_to(ROOT).as_posix()}`",
        "- 示范采集：`contact_stage_demo_v1`，12/12 成功。",
        f"- 模型：`{model_path.relative_to(ROOT).as_posix()}`",
        f"- phase_mode：`{args.phase_mode}`",
        f"- trainable_params：`{metadata.get('trainable_params')}`",
        f"- samples：`{metadata.get('samples')}`",
        f"- feature_dim：`{metadata.get('feature_dim')}`",
        f"- action_dim：`{metadata.get('action_dim')}`",
        f"- train_mse：`{float(metadata.get('train_mse', 0.0)):.8f}`",
        f"- val_mse：`{val_mse:.8f}`",
        "",
        "说明：该模型离线训练误差较低，但验证误差很高，尤其说明 contact-stage 末段动作分布在少量示范下泛化不稳。",
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
    lines.append(md_row(["范围", "seed", "success", "tcp_lift", "standard_ever_grasp", "strict", "max_object_z", "target_distance"]))
    lines.append(md_row(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
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
            "该版本只能写成 contact-stage 成功示范训练的轻量 phase action-head 负例诊断。若闭环评测没有 `grasp_success`、`tcp_grasp_lift_success` 或 `strict_grasp_lift_success`，不能写成稳定抓取成功、真实 VLA 后训练成功、OpenVLA LoRA 成功或完整 ACT 成功。",
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
    rows.extend(run_split(args, model, "train_range", int(args.train_seed_start)))
    rows.extend(run_split(args, model, "heldout", int(args.heldout_seed_start)))
    summary_rows = aggregate(rows)

    write_csv(args.output_csv, rows)
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
                "phase_mode": args.phase_mode,
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
