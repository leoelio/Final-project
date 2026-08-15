from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_trajectory_prior_residual_policy as runner  # noqa: E402
from trajectory_prior_residual_common import VERSION, load_residual_model  # noqa: E402


FIELDNAMES = [
    "version",
    "split",
    "seed",
    "task",
    "complexity",
    "success",
    "target_distance",
    "object_z",
    "max_object_z",
    "height_threshold_hit",
    "ever_grasp_success",
    "tcp_grasp_lift_success",
    "strict_grasp_lift_success",
    "continued_to_place",
    "out_of_table",
    "steps_taken",
    "stop_reason",
    "mean_residual_norm",
    "max_residual_norm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trajectory-prior residual BC and write a Chinese report.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--task", default="place_blue_cube_blue_pad")
    parser.add_argument("--complexity", default="medium")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--heldout-seed-start", type=int, default=100)
    parser.add_argument("--steps", type=int, default=2840)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--action-alpha", type=float, default=1.0)
    parser.add_argument("--max-arm-delta", type=float, default=0.02)
    parser.add_argument("--max-gripper-delta", type=float, default=0.0008)
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / f"{VERSION}.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "trajectory_prior_residual_bc_report.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "trajectory_prior_residual_bc_report.md")
    return parser.parse_args()


def runner_args(args: argparse.Namespace, model_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        model=model_path,
        task=str(args.task),
        complexity=str(args.complexity),
        seed=0,
        episodes=1,
        steps=int(args.steps),
        viewer=False,
        duration=0.0,
        speed=0.0,
        arm_kp=150.0,
        arm_force=100.0,
        gripper_kp=900.0,
        gripper_force=180.0,
        friction=3.0,
        approach_z=0.12,
        grasp_z=0.008,
        lift_z=0.18,
        residual_scale=float(args.residual_scale),
        clip_actions=True,
        action_alpha=float(args.action_alpha),
        max_arm_delta=float(args.max_arm_delta),
        max_gripper_delta=float(args.max_gripper_delta),
        require_lift_before_transfer=True,
        lift_threshold=0.085,
        tcp_lift_threshold=0.12,
        stop_on_unsafe=True,
        log_every=0,
    )


def flatten_row(split: str, summary: dict) -> dict:
    return {field: summary.get(field) for field in FIELDNAMES} | {
        "version": VERSION,
        "split": split,
        "seed": int(summary["seed"]),
    }


def run_split(args: argparse.Namespace, model: dict, model_path: Path, split: str, seed_start: int) -> list[dict]:
    base_args = runner_args(args, model_path)
    rows = []
    for offset in range(int(args.episodes)):
        seed = int(seed_start) + offset
        summary = runner.run_episode(base_args, model, seed, str(args.task), str(args.complexity), viewer=None)
        row = flatten_row(split, summary)
        rows.append(row)
        print(
            f"{split} seed={seed} success={row['success']} tcp_lift={row['tcp_grasp_lift_success']} "
            f"max_object_z={float(row['max_object_z']):.4f} target_distance={float(row['target_distance']):.4f}",
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
                "continued_to_place": sum(1 for item in items if item["continued_to_place"]),
                "out_of_table": sum(1 for item in items if item["out_of_table"]),
                "mean_target_distance": statistics.fmean(float(item["target_distance"]) for item in items),
                "mean_max_object_z": statistics.fmean(float(item["max_object_z"]) for item in items),
                "mean_residual_norm": statistics.fmean(float(item["mean_residual_norm"]) for item in items),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in FIELDNAMES} for row in rows)


def q(path: str | Path) -> str:
    return f'"{path}"'


def ps_command(script: str, args: list[str | Path]) -> str:
    rendered = [q(PYTHON), q(ROOT / script)]
    for value in args:
        rendered.append(q(value) if isinstance(value, Path) else str(value))
    return "& " + " ".join(rendered)


def md_row(values: list[object]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", "<br>") for value in values) + " |"


def write_md(path: Path, args: argparse.Namespace, model_path: Path, rows: list[dict], summary_rows: list[dict]) -> None:
    train_command = ps_command(
        "scripts/train_trajectory_prior_residual_bc.py",
        [
            "--run-dir",
            ROOT / "data" / "demos" / "contact_stage_demo_place_blue_cube_blue_pad_medium_v1",
            "--sample-stride",
            "4",
            "--ridge",
            "0.001",
            "--model-prefix",
            "trajectory_prior_residual_bc_v1_candidate",
        ],
    )
    eval_command = ps_command("scripts/evaluate_trajectory_prior_residual_bc.py", ["--model", model_path])
    viewer_command = ps_command(
        "scripts/run_trajectory_prior_residual_policy.py",
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
            "--residual-scale",
            str(args.residual_scale),
            "--action-alpha",
            str(args.action_alpha),
            "--max-arm-delta",
            str(args.max_arm_delta),
            "--max-gripper-delta",
            str(args.max_gripper_delta),
            "--log-every",
            "500",
        ],
    )

    lines = [
        "# Trajectory-prior Residual BC 候选实验报告",
        "",
        f"版本：`{VERSION}`",
        "",
        "定位：该版本属于 trajectory-conditioned BC / ACT 阶段的诊断候选。它把分阶段 scripted 轨迹作为强先验，再训练一个很小的 residual BC action head 去修正先验动作。它不是纯 BC、不是完整官方 ACT，也不是 VLA 后训练。",
        "",
        "研究意义：如果这个候选明显优于纯 ACT-style 模型，说明失败不只是网络容量问题，而是普通模仿学习缺少可靠接触阶段先验、夹爪闭环和阶段切换约束。",
        "",
        f"模型：`{model_path}`",
        "",
        "## 1. 汇总结论",
        "",
        md_row(["范围", "放置 success", "TCP 抬升", "标准曾经抓取", "严格抓取+高度", "进入放置阶段", "高度达标", "出界", "平均目标距离", "平均最高高度", "平均 residual norm"]),
        md_row(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
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
                    f"{item['continued_to_place']}/{item['episodes']}",
                    f"{item['height_threshold_hits']}/{item['episodes']}",
                    f"{item['out_of_table']}/{item['episodes']}",
                    f"{float(item['mean_target_distance']):.4f}",
                    f"{float(item['mean_max_object_z']):.4f}",
                    f"{float(item['mean_residual_norm']):.6f}",
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 2. 单次明细",
            "",
            md_row(["范围", "seed", "success", "TCP 抬升", "严格抓取+高度", "最高高度", "目标距离", "停止原因"]),
            md_row(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---"]),
        ]
    )
    for row in rows:
        lines.append(
            md_row(
                [
                    row["split"],
                    row["seed"],
                    row["success"],
                    row["tcp_grasp_lift_success"],
                    row["strict_grasp_lift_success"],
                    f"{float(row['max_object_z']):.4f}",
                    f"{float(row['target_distance']):.4f}",
                    row["stop_reason"],
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 3. 论文边界",
            "",
            "- 可以写：分阶段轨迹先验 + residual BC 显著提高了闭环抓取/放置上界，说明接触阶段结构约束很关键。",
            "- 可以写：它是对普通 trajectory-conditioned BC / ACT 的结构化诊断，不是纯学习策略。",
            "- 不能写：纯 ACT 已经稳定成功。",
            "- 不能写：VLA/LoRA/Adapter 已经完成真实机器人验证。",
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
            "慢速 MuJoCo viewer 命令：",
            "",
            "```powershell",
            viewer_command,
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    model_path = args.model or runner.latest_model()
    model = load_residual_model(model_path)
    rows = []
    rows.extend(run_split(args, model, model_path, "train_range", int(args.train_seed_start)))
    rows.extend(run_split(args, model, model_path, "heldout", int(args.heldout_seed_start)))
    summary_rows = aggregate(rows)
    payload = {
        "version": VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": str(model_path),
        "task": str(args.task),
        "complexity": str(args.complexity),
        "rows": rows,
        "summary": summary_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.output_csv, rows)
    write_md(args.output_md, args, model_path, rows, summary_rows)
    print(f"trajectory_prior_residual_json: {args.output_json}", flush=True)
    print(f"trajectory_prior_residual_csv: {args.output_csv}", flush=True)
    print(f"trajectory_prior_residual_md: {args.output_md}", flush=True)
    print(f"trajectory_prior_residual_rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
