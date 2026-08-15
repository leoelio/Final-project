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
VERSION = "grasp_lift_subpolicy_probe_v1_candidate"

sys.path.insert(0, str(ROOT))

from widowx_env import TASKS, WidowXTabletopEnv  # noqa: E402
from widowx_env.scripted_expert import PickConfig, PickOnlyExpert, PickPlaceConfig, PickPlaceExpert  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate scripted grasp/lift upper-bound control probe.")
    parser.add_argument("--task", choices=sorted(TASKS), default="place_blue_cube_blue_pad")
    parser.add_argument("--complexity", choices=("easy", "medium", "hard", "language"), default="medium")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--heldout-seed-start", type=int, default=100)
    parser.add_argument("--arm-kp", type=float, default=150.0)
    parser.add_argument("--arm-force", type=float, default=100.0)
    parser.add_argument("--gripper-kp", type=float, default=800.0)
    parser.add_argument("--gripper-force", type=float, default=140.0)
    parser.add_argument("--friction", type=float, default=3.0)
    parser.add_argument("--approach-z", type=float, default=0.12)
    parser.add_argument("--grasp-z", type=float, default=0.008)
    parser.add_argument("--lift-z", type=float, default=0.18)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--lift-threshold", type=float, default=0.085)
    parser.add_argument("--tcp-lift-threshold", type=float, default=0.12)
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / f"{VERSION}.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "grasp_lift_subpolicy_probe_report.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "grasp_lift_subpolicy_probe_report.md")
    return parser.parse_args()


def configure_env(args: argparse.Namespace, seed: int) -> tuple[WidowXTabletopEnv, dict]:
    env = WidowXTabletopEnv(seed=seed)
    env.set_arm_actuator_strength(kp=float(args.arm_kp), force_limit=float(args.arm_force))
    env.set_gripper_actuator_strength(kp=float(args.gripper_kp), force_limit=float(args.gripper_force))
    env.set_grasp_contact_friction(sliding=float(args.friction))
    obs = env.reset(task=str(args.task), complexity=str(args.complexity), seed=seed)
    return env, obs


def make_expert(args: argparse.Namespace, env: WidowXTabletopEnv):
    if env.task.kind == "place" and env.task.target_geom:
        config = PickPlaceConfig(
            approach_z_offset=float(args.approach_z),
            grasp_z_offset=float(args.grasp_z),
            lift_z_offset=float(args.lift_z),
        )
        return PickPlaceExpert(env, config), "place"

    config = PickConfig(
        approach_z_offset=float(args.approach_z),
        grasp_z_offset=float(args.grasp_z),
        lift_z_offset=float(args.lift_z),
    )
    return PickOnlyExpert(env, config), "pick"


def empty_trace() -> dict:
    return {
        "steps_taken": 0,
        "max_object_z": 0.0,
        "max_contact_count": 0.0,
        "min_tcp_object_distance": None,
        "min_tcp_object_distance_while_lifted": None,
        "ever_grasp_success": False,
        "ever_tcp_lift_success": False,
        "first_grasp_step": None,
        "first_lift_step": None,
        "first_tcp_lift_step": None,
    }


def update_trace(trace: dict, env: WidowXTabletopEnv, lift_threshold: float, tcp_lift_threshold: float) -> None:
    metrics = env.metrics()
    object_position = env.object_position(env.episode_target_object)
    tcp_object_distance = float((sum((a - b) ** 2 for a, b in zip(env.tcp_position(), object_position))) ** 0.5)
    trace["steps_taken"] += 1
    trace["max_object_z"] = max(float(trace["max_object_z"]), float(metrics["object_z"]))
    trace["max_contact_count"] = max(float(trace["max_contact_count"]), float(metrics["contact_count"]))
    if trace["min_tcp_object_distance"] is None or tcp_object_distance < float(trace["min_tcp_object_distance"]):
        trace["min_tcp_object_distance"] = tcp_object_distance
    if bool(metrics["grasp_success"]) and trace["first_grasp_step"] is None:
        trace["first_grasp_step"] = int(trace["steps_taken"])
    if bool(metrics["grasp_success"]):
        trace["ever_grasp_success"] = True
    lifted = float(metrics["object_z"]) >= lift_threshold
    if lifted:
        if trace["first_lift_step"] is None:
            trace["first_lift_step"] = int(trace["steps_taken"])
        if trace["min_tcp_object_distance_while_lifted"] is None or tcp_object_distance < float(trace["min_tcp_object_distance_while_lifted"]):
            trace["min_tcp_object_distance_while_lifted"] = tcp_object_distance
        if tcp_object_distance < tcp_lift_threshold:
            trace["ever_tcp_lift_success"] = True
            if trace["first_tcp_lift_step"] is None:
                trace["first_tcp_lift_step"] = int(trace["steps_taken"])


def run_episode(args: argparse.Namespace, split: str, seed: int) -> dict:
    env, obs = configure_env(args, seed)
    expert, mode = make_expert(args, env)
    object_name = obs["target_object"]
    trace = empty_trace()
    total_attempts = max(0, int(args.retries)) + 1
    summary: dict[str, object] = {"success": False, "attempts": 0}

    for attempt in range(1, total_attempts + 1):
        if mode == "place":
            plan = expert.plan(object_name, env.task.target_geom)
        else:
            plan = expert.plan(object_name)

        summary = expert.execute(
            plan,
            record_step=lambda _action, current_env: update_trace(
                trace,
                current_env,
                float(args.lift_threshold),
                float(args.tcp_lift_threshold),
            ),
            speed=0.0,
        )
        summary["attempts"] = attempt
        if bool(summary["success"]):
            break

    metrics = env.metrics()
    max_object_z = float(trace["max_object_z"])
    height_threshold_hit = bool(max_object_z >= float(args.lift_threshold))
    row = {
        "version": VERSION,
        "split": split,
        "seed": int(seed),
        "task": str(args.task),
        "complexity": str(args.complexity),
        "mode": mode,
        "instruction": str(obs["instruction"]),
        "active_objects": ",".join(obs["active_objects"]),
        "target_object": str(object_name),
        "success": bool(summary["success"]),
        "target_distance": float(metrics["target_distance"]),
        "object_z": float(metrics["object_z"]),
        "max_object_z": max_object_z,
        "height_threshold_hit": height_threshold_hit,
        "grasp_success": bool(metrics["grasp_success"]),
        "ever_grasp_success": bool(trace["ever_grasp_success"]),
        "ever_tcp_lift_success": bool(trace["ever_tcp_lift_success"]),
        "strict_grasp_lift_success": bool(trace["ever_grasp_success"] and height_threshold_hit),
        "tcp_grasp_lift_success": bool(trace["ever_tcp_lift_success"] and height_threshold_hit),
        "out_of_table": bool(metrics["out_of_table"]),
        "contact_count": float(metrics["contact_count"]),
        "max_contact_count": float(trace["max_contact_count"]),
        "min_tcp_object_distance": trace["min_tcp_object_distance"],
        "min_tcp_object_distance_while_lifted": trace["min_tcp_object_distance_while_lifted"],
        "attempts": int(summary["attempts"]),
        "steps_taken": int(trace["steps_taken"]),
        "first_grasp_step": trace["first_grasp_step"],
        "first_lift_step": trace["first_lift_step"],
        "first_tcp_lift_step": trace["first_tcp_lift_step"],
    }
    print(
        f"{split} seed={seed} success={row['success']} ever_grasp={row['ever_grasp_success']} "
        f"tcp_lift={row['tcp_grasp_lift_success']} strict_grasp_lift={row['strict_grasp_lift_success']} max_object_z={row['max_object_z']:.4f} "
        f"target_distance={row['target_distance']:.4f} attempts={row['attempts']}",
        flush=True,
    )
    return row


def run_split(args: argparse.Namespace, split: str, seed_start: int) -> list[dict]:
    return [run_episode(args, split, int(seed_start) + offset) for offset in range(int(args.episodes))]


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["split"]), []).append(row)

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
                "tcp_grasp_lift_successes": sum(1 for item in items if item["tcp_grasp_lift_success"]),
                "mean_target_distance": statistics.fmean(float(item["target_distance"]) for item in items),
                "mean_max_object_z": statistics.fmean(float(item["max_object_z"]) for item in items),
                "mean_attempts": statistics.fmean(float(item["attempts"]) for item in items),
            }
        )
    return output


def ps_command(script: str, script_args: list[str | Path]) -> str:
    rendered = [f'"{PYTHON}"', f'"{ROOT / script}"']
    for value in script_args:
        rendered.append(f'"{value}"' if isinstance(value, Path) else str(value))
    return "& " + " ".join(rendered)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def md_row(values: list[object]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"


def write_md(path: Path, args: argparse.Namespace, rows: list[dict], summary_rows: list[dict]) -> None:
    video_path = ROOT / "outputs" / "videos" / f"{VERSION}_seed0.mp4"
    viewer_command = ps_command(
        "scripts/run_expert.py",
        [
            "--task",
            args.task,
            "--complexity",
            args.complexity,
            "--seed",
            "0",
            "--episodes",
            "1",
            "--min-success-rate",
            "0",
            "--viewer",
            "--duration",
            "60",
            "--speed",
            "0.02",
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
            "--approach-z",
            str(args.approach_z),
            "--grasp-z",
            str(args.grasp_z),
            "--lift-z",
            str(args.lift_z),
            "--retries",
            str(args.retries),
        ],
    )
    eval_command = ps_command("scripts/evaluate_grasp_lift_subpolicy_probe.py", [])
    export_command = ps_command(
        "scripts/export_video.py",
        [
            "--method",
            "grasp_lift_subpolicy_probe",
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
            "--log-every",
            "0",
        ],
    )

    lines = [
        "# Grasp/Lift 子策略上界诊断报告",
        "",
        f"版本：`{VERSION}`",
        "",
        "定位：该版本使用 scripted expert / IK 子策略，不是学习策略。它用于确认 MuJoCo 桌面任务和 WidowX 控制接口本身能否稳定完成接触、夹紧、抬升、移动和放置。",
        "",
        "论文边界：该结果只能作为控制上界与指标口径诊断，不能登记为 BC、ACT、Diffusion Policy 或 VLA 方法成功率；它也不能替代真实 WidowX 验证。",
        "",
        "## 1. 评测设置",
        "",
        f"- 任务：`{args.task}`，复杂度：`{args.complexity}`。",
        f"- 训练范围 seed：`{args.train_seed_start}-{args.train_seed_start + args.episodes - 1}`；留出范围 seed：`{args.heldout_seed_start}-{args.heldout_seed_start + args.episodes - 1}`。",
        f"- 抬升阈值：`max_object_z >= {args.lift_threshold}`。",
        f"- TCP 诊断阈值：抬升过程中 `tcp_object_distance < {args.tcp_lift_threshold}`。",
        f"- 重试次数：`{args.retries}`，夹爪参数：`kp={args.gripper_kp}`，`force={args.gripper_force}`，摩擦：`{args.friction}`。",
        "",
        "## 2. 汇总结果",
        "",
        md_row(["范围", "放置 success", "标准曾经抓取", "TCP 抬升", "高度达标", "标准严格抓取+高度", "平均目标距离", "平均最高高度", "平均尝试次数"]),
        md_row(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for item in summary_rows:
        lines.append(
            md_row(
                [
                    item["split"],
                    f"{item['successes']}/{item['episodes']}",
                    f"{item['ever_grasp_successes']}/{item['episodes']}",
                    f"{item['tcp_grasp_lift_successes']}/{item['episodes']}",
                    f"{item['height_threshold_hits']}/{item['episodes']}",
                    f"{item['strict_grasp_lift_successes']}/{item['episodes']}",
                    f"{float(item['mean_target_distance']):.4f}",
                    f"{float(item['mean_max_object_z']):.4f}",
                    f"{float(item['mean_attempts']):.2f}",
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 3. 单次明细",
            "",
            md_row(["范围", "seed", "success", "final_grasp", "ever_grasp", "tcp_lift", "strict_grasp_lift", "max_object_z", "target_distance", "min_tcp_lift_dist"]),
            md_row(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
        ]
    )
    for row in rows:
        lines.append(
            md_row(
                [
                    row["split"],
                    row["seed"],
                    row["success"],
                    row["grasp_success"],
                    row["ever_grasp_success"],
                    row["tcp_grasp_lift_success"],
                    row["strict_grasp_lift_success"],
                    f"{float(row['max_object_z']):.4f}",
                    f"{float(row['target_distance']):.4f}",
                    "" if row["min_tcp_object_distance_while_lifted"] is None else f"{float(row['min_tcp_object_distance_while_lifted']):.4f}",
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 4. 关键解释",
            "",
            "`final_grasp=False` 不等于没有抓起过。对于放置任务，物体被放到盘子后会离开夹爪，因此最终时刻 `grasp_success` 合理地变为 False。本报告同时保留标准 `ever_grasp_success` 口径和 TCP 诊断口径；论文主表仍以标准口径为准，诊断口径只用于解释控制过程。",
            "",
            "## 5. 完整命令",
            "",
            "固定视频证据：",
            "",
            "```text",
            video_path.relative_to(ROOT).as_posix(),
            video_path.with_suffix(".json").relative_to(ROOT).as_posix(),
            "```",
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
            "重新导出固定视频命令：",
            "",
            "```powershell",
            export_command,
            "```",
            "",
            "## 6. 阶段结论",
            "",
            "如果该上界诊断稳定达标，而 trajectory-conditioned BC / ACT 仍然失败，就说明当前主要瓶颈在学习策略的阶段表达、接触保持和夹爪控制，而不是 MuJoCo 环境完全不可抓取。",
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

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "version": VERSION,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "task": args.task,
                "complexity": args.complexity,
                "fixed_video": f"outputs/videos/{VERSION}_seed0.mp4",
                "lift_threshold": float(args.lift_threshold),
                "tcp_lift_threshold": float(args.tcp_lift_threshold),
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


if __name__ == "__main__":
    main()
