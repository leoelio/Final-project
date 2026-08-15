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
VERSION = "contact_stage_subpolicy_v1_candidate"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_contact_stage_subpolicy as runner  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the contact-traced staged subpolicy diagnostic candidate.")
    parser.add_argument("--task", default="place_blue_cube_blue_pad")
    parser.add_argument("--complexity", default="medium")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--heldout-seed-start", type=int, default=100)
    parser.add_argument("--arm-kp", type=float, default=150.0)
    parser.add_argument("--arm-force", type=float, default=100.0)
    parser.add_argument("--gripper-kp", type=float, default=900.0)
    parser.add_argument("--gripper-force", type=float, default=180.0)
    parser.add_argument("--friction", type=float, default=3.0)
    parser.add_argument("--approach-z", type=float, default=0.12)
    parser.add_argument("--grasp-z", type=float, default=0.008)
    parser.add_argument("--lift-z", type=float, default=0.18)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--lift-threshold", type=float, default=0.085)
    parser.add_argument("--tcp-lift-threshold", type=float, default=0.12)
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / f"{VERSION}.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "contact_stage_subpolicy_report.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "contact_stage_subpolicy_report.md")
    return parser.parse_args()


def runner_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        task=str(args.task),
        complexity=str(args.complexity),
        seed=0,
        episodes=1,
        viewer=False,
        duration=0.0,
        speed=0.0,
        arm_kp=float(args.arm_kp),
        arm_force=float(args.arm_force),
        gripper_kp=float(args.gripper_kp),
        gripper_force=float(args.gripper_force),
        friction=float(args.friction),
        approach_z=float(args.approach_z),
        grasp_z=float(args.grasp_z),
        lift_z=float(args.lift_z),
        retries=int(args.retries),
        lift_threshold=float(args.lift_threshold),
        tcp_lift_threshold=float(args.tcp_lift_threshold),
        log_every=0,
    )


def flatten_row(split: str, summary: dict) -> dict:
    return {
        "version": VERSION,
        "split": split,
        "seed": int(summary["seed"]),
        "task": str(summary["task"]),
        "complexity": str(summary["complexity"]),
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
        "contact_count": float(summary["contact_count"]),
        "max_contact_count": float(summary["max_contact_count"]),
        "min_tcp_object_distance": summary["min_tcp_object_distance"],
        "min_tcp_object_distance_while_lifted": summary["min_tcp_object_distance_while_lifted"],
        "attempts": int(summary["attempts"]),
        "continued_to_place": bool(summary["continued_to_place"]),
        "steps_taken": int(summary["steps_taken"]),
        "first_lift_step": summary["first_lift_step"],
        "first_tcp_lift_step": summary["first_tcp_lift_step"],
        "stage_steps": json.dumps(summary["stage_steps"], ensure_ascii=False, sort_keys=True),
        "stage_min_tcp_distance": json.dumps(summary["stage_min_tcp_distance"], ensure_ascii=False, sort_keys=True),
        "stage_max_object_z": json.dumps(summary["stage_max_object_z"], ensure_ascii=False, sort_keys=True),
        "stage_max_contact_count": json.dumps(summary["stage_max_contact_count"], ensure_ascii=False, sort_keys=True),
    }


def run_split(args: argparse.Namespace, split: str, seed_start: int) -> list[dict]:
    base_args = runner_args(args)
    rows = []
    for offset in range(int(args.episodes)):
        seed = int(seed_start) + offset
        summary = runner.run_episode(base_args, seed, viewer=None)
        row = flatten_row(split, summary)
        rows.append(row)
        print(
            f"{split} seed={seed} success={row['success']} tcp_lift={row['tcp_grasp_lift_success']} "
            f"strict={row['strict_grasp_lift_success']} max_object_z={row['max_object_z']:.4f} "
            f"target_distance={row['target_distance']:.4f} attempts={row['attempts']}",
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
                "mean_attempts": statistics.fmean(float(item["attempts"]) for item in items),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def q(path: str | Path) -> str:
    return f'"{path}"'


def ps_command(script: str, args: list[str | Path]) -> str:
    rendered = [q(PYTHON), q(ROOT / script)]
    for value in args:
        rendered.append(q(value) if isinstance(value, Path) else str(value))
    return "& " + " ".join(rendered)


def md_row(values: list[object]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"


def write_md(path: Path, args: argparse.Namespace, rows: list[dict], summary_rows: list[dict]) -> None:
    viewer_command = ps_command(
        "scripts/run_contact_stage_subpolicy.py",
        [
            "--task",
            args.task,
            "--complexity",
            args.complexity,
            "--seed",
            "0",
            "--episodes",
            "1",
            "--viewer",
            "--duration",
            "60",
            "--speed",
            "0.05",
            "--arm-kp",
            str(args.arm_kp),
            "--arm-force",
            str(args.arm_force),
            "--gripper-kp",
            str(args.gripper_kp),
            "--gripper-force",
            str(args.gripper_force),
            "--friction",
            str(args.friction),
            "--retries",
            str(args.retries),
            "--log-every",
            "500",
        ],
    )
    eval_command = ps_command("scripts/evaluate_contact_stage_subpolicy.py", [])
    export_command = ps_command(
        "scripts/export_video.py",
        [
            "--method",
            "contact_stage_subpolicy",
            "--version",
            VERSION,
            "--task",
            args.task,
            "--complexity",
            args.complexity,
            "--seed",
            "0",
            "--steps",
            "2840",
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
            str(args.gripper_kp),
            "--gripper-force",
            str(args.gripper_force),
            "--friction",
            str(args.friction),
            "--log-every",
            "0",
        ],
    )

    lines = [
        "# Contact-stage Subpolicy 候选诊断报告",
        "",
        f"版本：`{VERSION}`",
        "",
        "定位：该版本是分阶段、带接触/抬升事件审计的 scripted subpolicy 上界候选，不是学习策略。它用于和线性 BC、MLP BC、trajectory-kNN、Torch ACT 等方法对比，说明当前环境中抓取、抬升、转移、释放本身可由分阶段控制完成，而纯模仿学习模型主要卡在接触保持和阶段切换。",
        "",
        "## 1. 汇总结论",
        "",
        md_row(["范围", "放置 success", "TCP 抬升", "曾经标准抓取", "严格抓取+高度", "进入放置阶段", "高度达标", "出界", "平均目标距离", "平均最高高度", "平均尝试次数"]),
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
                    f"{float(item['mean_attempts']):.2f}",
                ]
            )
        )

    lines.extend(["", "## 2. 单次明细", ""])
    lines.append(md_row(["范围", "seed", "success", "tcp_lift", "standard_ever_grasp", "strict", "continued_to_place", "max_object_z", "target_distance", "attempts"]))
    lines.append(md_row(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
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
                    row["continued_to_place"],
                    f"{float(row['max_object_z']):.4f}",
                    f"{float(row['target_distance']):.4f}",
                    row["attempts"],
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 3. 完整命令",
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
            "固定视频导出命令：",
            "",
            "```powershell",
            export_command,
            "```",
            "",
            "## 4. 论文边界",
            "",
            "该版本只能写成 scripted subpolicy / contact-stage 上界诊断，不能写成 BC、ACT、Diffusion Policy、VLA 或真实 WidowX 的学习策略成功。它的价值是给失败分析提供上界参照：如果分阶段控制能完成，而纯学习策略不能完成，则论文应重点讨论接触保持、夹爪闭环、阶段切换和视觉/触觉反馈不足。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = []
    rows.extend(run_split(args, "train_range", args.train_seed_start))
    rows.extend(run_split(args, "heldout", args.heldout_seed_start))
    summary_rows = aggregate(rows)

    payload = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": str(args.task),
        "complexity": str(args.complexity),
        "fixed_video": f"outputs/videos/{VERSION}_seed0.mp4",
        "rows": rows,
        "summary": summary_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.output_csv, rows)
    write_md(args.output_md, args, rows, summary_rows)
    print(f"json_path: {args.output_json}", flush=True)
    print(f"csv_path: {args.output_csv}", flush=True)
    print(f"md_path: {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
