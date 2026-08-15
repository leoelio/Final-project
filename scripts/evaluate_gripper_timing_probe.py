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
VERSION = "gripper_timing_contact_probe_v1_candidate"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_gripper_timing_probe as runner  # noqa: E402


FIELDNAMES = [
    "version",
    "variant",
    "split",
    "seed",
    "task",
    "complexity",
    "success",
    "target_distance",
    "object_z",
    "max_object_z",
    "height_threshold_hit",
    "grasp_success",
    "ever_standard_grasp_success",
    "tcp_grasp_lift_success",
    "strict_grasp_lift_success",
    "out_of_table",
    "contact_count",
    "max_contact_count",
    "min_tcp_object_distance",
    "min_tcp_object_distance_while_lifted",
    "min_ee_object_distance",
    "min_ee_object_distance_while_lifted",
    "attempts",
    "continued_to_place",
    "steps_taken",
    "first_lift_step",
    "first_tcp_lift_step",
    "first_standard_grasp_step",
    "stage_steps",
    "stage_min_tcp_distance",
    "stage_min_ee_distance",
    "stage_max_object_z",
    "stage_max_contact_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate gripper-close timing/contact-hold diagnostic variants.")
    parser.add_argument("--task", default="place_blue_cube_blue_pad")
    parser.add_argument("--complexity", default="medium")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--heldout-seed-start", type=int, default=100)
    parser.add_argument("--variants", default="baseline,tight_close_hold,early_close_hold")
    parser.add_argument("--arm-kp", type=float, default=150.0)
    parser.add_argument("--arm-force", type=float, default=100.0)
    parser.add_argument("--gripper-kp", type=float, default=900.0)
    parser.add_argument("--gripper-force", type=float, default=180.0)
    parser.add_argument("--friction", type=float, default=3.0)
    parser.add_argument("--approach-z", type=float, default=0.12)
    parser.add_argument("--grasp-z", type=float, default=0.008)
    parser.add_argument("--lift-z", type=float, default=0.18)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--lift-threshold", type=float, default=0.085)
    parser.add_argument("--tcp-lift-threshold", type=float, default=0.12)
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / f"{VERSION}.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "gripper_timing_contact_probe_report.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "gripper_timing_contact_probe_report.md")
    return parser.parse_args()


def variant_names(args: argparse.Namespace) -> list[str]:
    names = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    unknown = [name for name in names if name not in runner.VARIANTS]
    if unknown:
        raise ValueError(f"unknown timing variants: {unknown}")
    return names


def runner_args(args: argparse.Namespace, variant: str) -> argparse.Namespace:
    return argparse.Namespace(
        task=str(args.task),
        complexity=str(args.complexity),
        seed=0,
        episodes=1,
        variant=variant,
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


def flatten_row(summary: dict, split: str) -> dict:
    row = {field: summary.get(field) for field in FIELDNAMES}
    row["split"] = split
    for key in ("stage_steps", "stage_min_tcp_distance", "stage_min_ee_distance", "stage_max_object_z", "stage_max_contact_count"):
        row[key] = json.dumps(summary[key], ensure_ascii=False, sort_keys=True)
    return row


def run_split(args: argparse.Namespace, variant: str, split: str, seed_start: int) -> list[dict]:
    base_args = runner_args(args, variant)
    rows = []
    for offset in range(int(args.episodes)):
        seed = int(seed_start) + offset
        summary = runner.run_episode(base_args, seed, viewer=None)
        row = flatten_row(summary, split)
        rows.append(row)
        print(
            f"{variant} {split} seed={seed} success={row['success']} tcp_lift={row['tcp_grasp_lift_success']} "
            f"standard_ever={row['ever_standard_grasp_success']} strict={row['strict_grasp_lift_success']} "
            f"max_object_z={float(row['max_object_z']):.4f} min_tcp_lift={row['min_tcp_object_distance_while_lifted']} "
            f"min_ee_lift={row['min_ee_object_distance_while_lifted']}",
            flush=True,
        )
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    def mean_or_nan(values: list[float]) -> float:
        return statistics.fmean(values) if values else float("nan")

    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((str(row["variant"]), str(row["split"])), []).append(row)
    output = []
    for (variant, split), items in sorted(grouped.items()):
        output.append(
            {
                "variant": variant,
                "split": split,
                "episodes": len(items),
                "successes": sum(1 for item in items if item["success"] is True),
                "tcp_grasp_lift_successes": sum(1 for item in items if item["tcp_grasp_lift_success"] is True),
                "standard_ever_grasp_successes": sum(1 for item in items if item["ever_standard_grasp_success"] is True),
                "strict_grasp_lift_successes": sum(1 for item in items if item["strict_grasp_lift_success"] is True),
                "height_threshold_hits": sum(1 for item in items if item["height_threshold_hit"] is True),
                "continued_to_place": sum(1 for item in items if item["continued_to_place"] is True),
                "mean_target_distance": statistics.fmean(float(item["target_distance"]) for item in items),
                "mean_max_object_z": statistics.fmean(float(item["max_object_z"]) for item in items),
                "mean_min_tcp_distance_while_lifted": mean_or_nan(
                    [
                        float(item["min_tcp_object_distance_while_lifted"])
                        for item in items
                        if item["min_tcp_object_distance_while_lifted"] is not None
                    ]
                ),
                "mean_min_ee_distance_while_lifted": mean_or_nan(
                    [
                        float(item["min_ee_object_distance_while_lifted"])
                        for item in items
                        if item["min_ee_object_distance_while_lifted"] is not None
                    ]
                ),
            }
        )
    return output


def format_float(value: object, digits: int = 4) -> str:
    value = float(value)
    if value != value:
        return "无抬升"
    return f"{value:.{digits}f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def q(path: str | Path) -> str:
    return f'"{path}"'


def ps_command(script: str, values: list[str | Path]) -> str:
    rendered = ["&", q(PYTHON), q(ROOT / script)]
    for value in values:
        rendered.append(q(value) if isinstance(value, Path) else str(value))
    return " ".join(rendered)


def md_row(values: list[object]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", "<br>") for value in values) + " |"


def ratio(successes: int, total: int) -> str:
    return f"{successes}/{total}"


def write_md(path: Path, args: argparse.Namespace, rows: list[dict], summary_rows: list[dict]) -> None:
    viewer_command = ps_command(
        "scripts/run_gripper_timing_probe.py",
        [
            "--task",
            args.task,
            "--complexity",
            args.complexity,
            "--seed",
            "0",
            "--episodes",
            "1",
            "--variant",
            "tight_close_hold",
            "--viewer",
            "--duration",
            "60",
            "--speed",
            "0.05",
            "--gripper-kp",
            str(args.gripper_kp),
            "--gripper-force",
            str(args.gripper_force),
            "--friction",
            str(args.friction),
            "--log-every",
            "500",
        ],
    )
    eval_command = ps_command("scripts/evaluate_gripper_timing_probe.py", [])
    export_command = ps_command(
        "scripts/export_video.py",
        [
            "--method",
            "gripper_timing_contact_probe",
            "--version",
            VERSION,
            "--task",
            args.task,
            "--complexity",
            args.complexity,
            "--seed",
            "0",
            "--steps",
            "3560",
            "--variant",
            "tight_close_hold",
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
        "# Gripper Timing / Contact Probe 候选诊断报告",
        "",
        f"版本：`{VERSION}`",
        "",
        "定位：这是夹爪闭合时序和接触保持的 scripted diagnostic probe，不是学习策略，也不是正式 VLA/ACT 方法。它用于解释为什么 contact-stage 上界能放置成功但标准严格抓取仍为 0。",
        "",
        "## 1. 评测设置",
        "",
        f"- 任务：`{args.task}`，复杂度：`{args.complexity}`。",
        f"- 变体：`{', '.join(variant_names(args))}`。",
        f"- 每个变体 train-range `seed {args.train_seed_start}-{args.train_seed_start + args.episodes - 1}`，held-out `seed {args.heldout_seed_start}-{args.heldout_seed_start + args.episodes - 1}`。",
        "- 关键指标：同时记录标准 `grasp_success`/`strict_grasp_lift_success` 和 TCP 诊断口径 `tcp_grasp_lift_success`。",
        "",
        "## 2. 汇总结果",
        "",
        md_row(["变体", "范围", "放置 success", "TCP 抬升", "标准曾经抓取", "严格抓取+高度", "进入放置阶段", "平均目标距离", "平均最高高度", "lift 时 TCP 距离", "lift 时 EE 距离"]),
        md_row(["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for item in summary_rows:
        lines.append(
            md_row(
                [
                    item["variant"],
                    item["split"],
                    ratio(item["successes"], item["episodes"]),
                    ratio(item["tcp_grasp_lift_successes"], item["episodes"]),
                    ratio(item["standard_ever_grasp_successes"], item["episodes"]),
                    ratio(item["strict_grasp_lift_successes"], item["episodes"]),
                    ratio(item["continued_to_place"], item["episodes"]),
                    format_float(item["mean_target_distance"]),
                    format_float(item["mean_max_object_z"]),
                    format_float(item["mean_min_tcp_distance_while_lifted"]),
                    format_float(item["mean_min_ee_distance_while_lifted"]),
                ]
            )
        )

    lines.extend(["", "## 3. 单次明细", ""])
    lines.append(md_row(["变体", "范围", "seed", "success", "tcp_lift", "standard_ever", "strict", "max_object_z", "min_tcp_lift", "min_ee_lift"]))
    lines.append(md_row(["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
    for row in rows:
        lines.append(
            md_row(
                [
                    row["variant"],
                    row["split"],
                    row["seed"],
                    row["success"],
                    row["tcp_grasp_lift_success"],
                    row["ever_standard_grasp_success"],
                    row["strict_grasp_lift_success"],
                    f"{float(row['max_object_z']):.4f}",
                    "" if row["min_tcp_object_distance_while_lifted"] is None else f"{float(row['min_tcp_object_distance_while_lifted']):.4f}",
                    "" if row["min_ee_object_distance_while_lifted"] is None else f"{float(row['min_ee_object_distance_while_lifted']):.4f}",
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 4. 解释口径",
            "",
            "如果 `tcp_grasp_lift_success` 为真但 `ever_standard_grasp_success` 仍为假，说明 finger TCP 与物体在抬升过程中足够近，但环境当前标准 `grasp_success` 仍没有被触发。论文中必须把它写成指标口径/接触诊断差异，不能写成标准严格抓取成功。",
            "",
            "该诊断给下一轮训练的直接启发：不要只调目标距离或 viewer 速度，应把夹爪闭合时序、闭合后保持、lift 时物体随动、TCP/EE 距离差异和跨 seed 接触稳定性写入训练目标或偏好函数。",
            "",
            "## 5. 完整命令",
            "",
            "批量评测命令：",
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
            "## 6. 论文边界",
            "",
            "该版本只能作为 contact/gripper timing 候选诊断，不能写成 learned BC、ACT、Diffusion Policy、VLA、OpenVLA 后训练或真实 WidowX 成功。正式对比表仍以已有 25 个正式方法版本为准。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows: list[dict] = []
    for variant in variant_names(args):
        rows.extend(run_split(args, variant, "train_range", args.train_seed_start))
        rows.extend(run_split(args, variant, "heldout", args.heldout_seed_start))
    summary_rows = aggregate(rows)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "version": VERSION,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "task": args.task,
                "complexity": args.complexity,
                "fixed_video": f"outputs/videos/{VERSION}_seed0.mp4",
                "rows": rows,
                "summary": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(args.output_csv, rows)
    write_md(args.output_md, args, rows, summary_rows)
    print(f"json_path: {args.output_json}", flush=True)
    print(f"csv_path: {args.output_csv}", flush=True)
    print(f"md_path: {args.output_md}", flush=True)
    print(f"rows: {len(rows)}", flush=True)
    print(f"summary_rows: {len(summary_rows)}", flush=True)


if __name__ == "__main__":
    main()
