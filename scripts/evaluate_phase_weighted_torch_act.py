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
VERSION = "phase_weighted_torch_act_v1_candidate"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_torch_act_policy import configure_env, load_model, rollout_with_env  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the phase-weighted Torch ACT candidate.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--task", default="place_blue_cube_blue_pad")
    parser.add_argument("--complexity", default="medium")
    parser.add_argument("--steps", type=int, default=2840)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--heldout-seed-start", type=int, default=100)
    parser.add_argument("--lift-threshold", type=float, default=0.085)
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / f"{VERSION}.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "phase_weighted_torch_act_report.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "phase_weighted_torch_act_report.md")
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
        gripper_kp=800.0,
        gripper_force=140.0,
        friction=3.0,
        clip_actions=True,
        action_alpha=0.25,
        max_arm_delta=0.012,
        max_gripper_delta=0.0005,
        replan_interval=4,
        temporal_ensemble=True,
        ensemble_decay=0.1,
        stop_on_unsafe=True,
        log_every=0,
        grasp_gate=False,
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
        max_object_z = float(summary.get("max_object_z", summary["object_z"]))
        ever_grasp_success = bool(summary.get("ever_grasp_success", False))
        row = {
            "version": VERSION,
            "split": split,
            "seed": seed,
            "task": str(args.task),
            "complexity": str(args.complexity),
            "success": bool(summary["success"]),
            "target_distance": float(summary["target_distance"]),
            "object_z": float(summary["object_z"]),
            "max_object_z": max_object_z,
            "height_threshold_hit": bool(max_object_z >= float(args.lift_threshold)),
            "grasp_success": bool(summary["grasp_success"]),
            "ever_grasp_success": ever_grasp_success,
            "strict_grasp_lift_success": bool(ever_grasp_success and max_object_z >= float(args.lift_threshold)),
            "out_of_table": bool(summary["out_of_table"]),
            "steps_taken": int(summary["steps_taken"]),
            "stop_reason": summary["stop_reason"],
            "mean_action_norm": float(summary["mean_action_norm"]),
            "max_action_norm": float(summary["max_action_norm"]),
        }
        rows.append(row)
        print(
            f"{split} seed={seed} success={row['success']} "
            f"ever_grasp={row['ever_grasp_success']} max_object_z={row['max_object_z']:.4f} "
            f"target_distance={row['target_distance']:.4f}",
            flush=True,
        )
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["split"], []).append(row)
    output = []
    for split, items in grouped.items():
        output.append(
            {
                "split": split,
                "episodes": len(items),
                "successes": sum(1 for item in items if item["success"]),
                "ever_grasp_successes": sum(1 for item in items if item["ever_grasp_success"]),
                "height_threshold_hits": sum(1 for item in items if item["height_threshold_hit"]),
                "strict_grasp_lift_successes": sum(1 for item in items if item["strict_grasp_lift_success"]),
                "mean_target_distance": statistics.fmean(float(item["target_distance"]) for item in items),
                "mean_max_object_z": statistics.fmean(float(item["max_object_z"]) for item in items),
            }
        )
    return output


def ps_command(script: str, script_args: list[str | Path], *, cuda: bool = False) -> str:
    lines = []
    if cuda:
        lines.append('$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"')
    rendered = [f'"{PYTHON}"', f'"{ROOT / script}"']
    for value in script_args:
        rendered.append(f'"{value}"' if isinstance(value, Path) else str(value))
    lines.append("& " + " ".join(rendered))
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def md_row(values: list[object]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"


def write_md(path: Path, args: argparse.Namespace, model_path: Path, metadata: dict, rows: list[dict], summary_rows: list[dict]) -> None:
    video_path = ROOT / "outputs" / "videos" / f"{VERSION}_seed0.mp4"
    video_metadata_path = video_path.with_suffix(".json")
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
            "6",
            "--batch-size",
            "256",
            "--lr",
            "0.0003",
            "--gripper-loss-weight",
            "4",
            "--phase-one-hot",
            "--phase-loss-weights",
            '"grasp:3,lift:3,place_release:2"',
            "--model-prefix",
            VERSION,
        ],
        cuda=True,
    )
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
            "0.02",
            "--action-alpha",
            "0.25",
            "--max-arm-delta",
            "0.012",
            "--max-gripper-delta",
            "0.0005",
            "--replan-interval",
            "4",
            "--temporal-ensemble",
            "--ensemble-decay",
            "0.1",
            "--stop-on-unsafe",
            "--log-every",
            "500",
        ],
        cuda=True,
    )
    eval_command = ps_command("scripts/evaluate_phase_weighted_torch_act.py", ["--model", model_path], cuda=True)

    lines = [
        "# Phase-weighted Torch ACT 候选诊断报告",
        "",
        f"版本：`{VERSION}`",
        "",
        "定位：该版本在 `phase_conditioned_torch_act_v1` 的基础上，只改变训练损失权重，对 `grasp/lift/place_release` 阶段样本加权。它是模型侧失败诊断候选，不登记为正式方法成功率。",
        "",
        "论文边界：当前结果不能写成完整官方 ACT、稳定抓取成功或真实 VLA 后训练；只有 `ever_grasp_success=True` 且 `max_object_z` 超过阈值时，才可写成严格抓取+抬升达标。",
        "",
        "## 1. 模型与训练配置",
        "",
        md_row(["项目", "值"]),
        md_row(["---", "---"]),
        md_row(["模型文件", f"`{model_path.relative_to(ROOT).as_posix()}`"]),
        md_row(["可训练参数", metadata.get("trainable_params", "")]),
        md_row(["训练时间秒", f"{float(metadata.get('train_time_seconds', 0.0)):.2f}"]),
        md_row(["峰值显存 MB", f"{float(metadata.get('peak_vram_mb', 0.0)):.2f}"]),
        md_row(["phase_loss_weights", json.dumps(metadata.get("phase_loss_weights", {}), ensure_ascii=False)]),
        md_row(["train_mse_norm", f"{float(metadata.get('train_mse_norm', 0.0)):.8f}"]),
        md_row(["val_mse_norm", f"{float(metadata.get('val_mse_norm', 0.0)):.8f}"]),
        "",
        "## 2. 闭环评测汇总",
        "",
        md_row(["范围", "success", "ever_grasp", "height_hit", "strict_grasp_lift", "mean_target_distance", "mean_max_object_z"]),
        md_row(["---", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for item in summary_rows:
        lines.append(
            md_row(
                [
                    item["split"],
                    f"{item['successes']}/{item['episodes']}",
                    f"{item['ever_grasp_successes']}/{item['episodes']}",
                    f"{item['height_threshold_hits']}/{item['episodes']}",
                    f"{item['strict_grasp_lift_successes']}/{item['episodes']}",
                    f"{float(item['mean_target_distance']):.4f}",
                    f"{float(item['mean_max_object_z']):.4f}",
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 3. 单次明细",
            "",
            md_row(["范围", "seed", "success", "ever_grasp", "max_object_z", "target_distance", "object_z"]),
            md_row(["---", "---:", "---:", "---:", "---:", "---:", "---:"]),
        ]
    )
    for row in rows:
        lines.append(
            md_row(
                [
                    row["split"],
                    row["seed"],
                    row["success"],
                    row["ever_grasp_success"],
                    f"{float(row['max_object_z']):.4f}",
                    f"{float(row['target_distance']):.4f}",
                    f"{float(row['object_z']):.4f}",
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
            video_metadata_path.relative_to(ROOT).as_posix(),
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
            "## 5. 阶段结论",
            "",
            "阶段加权降低了离线训练误差，但没有转化成闭环抓取或抬升成功；失败仍集中在接触保持、夹紧和 lift 阶段。这说明仅靠 phase one-hot 加阶段 loss 权重不足以解决当前 ACT-style baseline 的控制问题。",
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

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "version": VERSION,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "model_path": model_path.relative_to(ROOT).as_posix(),
                "fixed_video": f"outputs/videos/{VERSION}_seed0.mp4",
                "fixed_video_metadata": f"outputs/videos/{VERSION}_seed0.json",
                "metadata": model["metadata"],
                "rows": rows,
                "summary": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(args.output_csv, rows)
    write_md(args.output_md, args, model_path, model["metadata"], rows, summary_rows)
    print(f"json_path: {args.output_json}", flush=True)
    print(f"csv_path: {args.output_csv}", flush=True)
    print(f"md_path: {args.output_md}", flush=True)
    print(f"rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
