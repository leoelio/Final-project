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
VERSION = "contact_aware_trajectory_knn_v1_candidate"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_trajectory_knn_policy import configure_env, load_model, rollout_with_env  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate relative-geometry/contact-aware trajectory-kNN candidate.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--task", default="place_blue_cube_blue_pad")
    parser.add_argument("--complexity", default="medium")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=2840)
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--heldout-seed-start", type=int, default=100)
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / f"{VERSION}.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "contact_aware_trajectory_knn_report.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "contact_aware_trajectory_knn_report.md")
    return parser.parse_args()


def latest_model() -> Path:
    candidates = sorted(
        (ROOT / "outputs" / "trajectory_knn_bc").glob("contact_aware_trajectory_knn_*.npz"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError("no contact-aware trajectory-kNN model found under outputs/trajectory_knn_bc")
    return candidates[-1]


def runner_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        steps=int(args.steps),
        k=3,
        phase_window=0.03,
        min_candidates=256,
        history_decay=0.25,
        viewer=False,
        duration=0.0,
        speed=0.0,
        arm_kp=150.0,
        arm_force=100.0,
        gripper_kp=800.0,
        gripper_force=140.0,
        friction=3.0,
        clip_actions=True,
        action_alpha=0.25,
        max_arm_delta=0.012,
        max_gripper_delta=0.0005,
        replan_interval=1,
        temporal_ensemble=True,
        ensemble_decay=0.1,
        stop_on_unsafe=True,
        log_every=0,
    )


def run_split(args: argparse.Namespace, model: dict, split: str, seed_start: int) -> list[dict]:
    control = runner_args(args)
    rows = []
    for offset in range(int(args.episodes)):
        seed = int(seed_start) + offset
        env, obs = configure_env(control, seed, str(args.task), str(args.complexity))
        summary = rollout_with_env(control, model, env, obs, seed, str(args.task), str(args.complexity), viewer=None)
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
            "tcp_grasp_lift_success": bool(summary["tcp_grasp_lift_success"]),
            "strict_grasp_lift_success": bool(summary["strict_grasp_lift_success"]),
            "out_of_table": bool(summary["out_of_table"]),
            "steps_taken": int(summary["steps_taken"]),
            "stop_reason": summary["stop_reason"],
            "min_tcp_object_distance": summary["min_tcp_object_distance"],
            "min_tcp_object_distance_while_lifted": summary["min_tcp_object_distance_while_lifted"],
            "mean_action_norm": float(summary["mean_action_norm"]),
            "max_action_norm": float(summary["max_action_norm"]),
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


def write_md(path: Path, args: argparse.Namespace, model_path: Path, rows: list[dict], summary_rows: list[dict]) -> None:
    train_command = ps_command(
        "scripts/train_trajectory_knn_bc.py",
        [
            "--run-dir",
            ROOT / "data" / "demos" / "place_blue_cube_blue_pad_medium_20260702_051752",
            "--horizon",
            "8",
            "--history",
            "8",
            "--sample-stride",
            "16",
            "--augment-relative",
            "--model-prefix",
            "contact_aware_trajectory_knn",
        ],
    )
    eval_command = ps_command("scripts/evaluate_contact_aware_trajectory_knn.py", ["--model", model_path])
    viewer_command = ps_command(
        "scripts/run_trajectory_knn_policy.py",
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
            "--k",
            "3",
            "--phase-window",
            "0.03",
            "--min-candidates",
            "256",
            "--history-decay",
            "0.25",
            "--action-alpha",
            "0.25",
            "--max-arm-delta",
            "0.012",
            "--max-gripper-delta",
            "0.0005",
            "--replan-interval",
            "1",
            "--temporal-ensemble",
            "--ensemble-decay",
            "0.1",
            "--stop-on-unsafe",
            "--log-every",
            "500",
        ],
    )
    export_command = ps_command(
        "scripts/export_video.py",
        [
            "--method",
            "contact_aware_trajectory_knn",
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
            "--log-every",
            "0",
        ],
    )
    video_path = ROOT / "outputs" / "videos" / f"{VERSION}_seed0.mp4"

    lines = [
        "# Contact-aware Trajectory-kNN 候选诊断报告",
        "",
        f"版本：`{VERSION}`",
        "",
        "定位：该版本复用 trajectory-kNN action chunk 框架，但训练时打开 `--augment-relative`，显式加入 object-to-TCP、object-to-target、TCP-to-target 相对几何特征。它是轻量 contact/TCP-aware 诊断候选，不是 VLA、不是完整 ACT。",
        "",
        "论文边界：该版本只能作为相对几何特征消融候选。即使出现放置成功，也必须同时报告 `grasp_success`、`max_object_z`、`tcp_grasp_lift_success` 和视频证据，不能写成稳定 learned grasp 或真实 VLA 后训练。",
        "",
        "## 1. 汇总结果",
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

    lines.extend(["", "## 2. 单次明细", ""])
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
            "## 3. 完整命令",
            "",
            "模型文件与固定视频：",
            "",
            "```text",
            model_path.relative_to(ROOT).as_posix(),
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
            "## 4. 阶段结论",
            "",
            "相对几何特征能让训练范围 seed0 出现可观察的 TCP 抬升和放置成功，但是否能泛化必须看 5+5 seeds。若 held-out 仍失败，则说明单纯 object/TCP 相对几何特征不足以解决泛化抓取，后续需要更强的视觉语言表征、接触监督或显式阶段子策略。",
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
    write_md(args.output_md, args, model_path, rows, summary_rows)
    print(f"json_path: {args.output_json}", flush=True)
    print(f"csv_path: {args.output_csv}", flush=True)
    print(f"md_path: {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
