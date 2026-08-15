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

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_chunk_policy import configure_env as configure_chunk_env  # noqa: E402
from run_chunk_policy import load_model as load_chunk_model  # noqa: E402
from run_chunk_policy import rollout_with_env as rollout_chunk  # noqa: E402
from run_torch_act_policy import configure_env as configure_act_env  # noqa: E402
from run_torch_act_policy import load_model as load_act_model  # noqa: E402
from run_torch_act_policy import rollout_with_env as rollout_act  # noqa: E402


VERSION = "grasp_gated_trajectory_act_v1_candidate"


METHODS = [
    {
        "version": "grasp_gated_trajectory_conditioned_chunk_bc_v1_candidate",
        "base_version": "trajectory_conditioned_chunk_bc_v2",
        "method": "Trajectory-conditioned chunk BC + grasp gate",
        "export_method": "grasp_gated_trajectory_chunk_bc",
        "runner": "scripts/run_chunk_policy.py",
        "model": ROOT / "outputs" / "chunk_bc" / "trajectory_chunk_bc_20260720_043500.npz",
        "loader": load_chunk_model,
        "configure": configure_chunk_env,
        "rollout": rollout_chunk,
        "replan_interval": 1,
        "action_alpha": 0.12,
        "max_arm_delta": 0.006,
        "max_gripper_delta": 0.00025,
        "env_prefix": "",
    },
    {
        "version": "grasp_gated_torch_act_state_chunk_v1_candidate",
        "base_version": "torch_act_state_chunk_v1",
        "method": "Torch state ACT + grasp gate",
        "export_method": "grasp_gated_torch_act",
        "runner": "scripts/run_torch_act_policy.py",
        "model": ROOT / "outputs" / "torch_act" / "torch_act_state_chunk_20260720_055409.pt",
        "loader": load_act_model,
        "configure": configure_act_env,
        "rollout": rollout_act,
        "replan_interval": 4,
        "action_alpha": 0.12,
        "max_arm_delta": 0.006,
        "max_gripper_delta": 0.00025,
        "env_prefix": '$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"\n',
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate grasp-gated trajectory-conditioned BC / ACT candidates.")
    parser.add_argument("--task", default="place_blue_cube_blue_pad")
    parser.add_argument("--complexity", default="medium")
    parser.add_argument("--steps", type=int, default=2840)
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--heldout-seed-start", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--arm-kp", type=float, default=150.0)
    parser.add_argument("--arm-force", type=float, default=100.0)
    parser.add_argument("--gripper-kp", type=float, default=900.0)
    parser.add_argument("--gripper-force", type=float, default=180.0)
    parser.add_argument("--friction", type=float, default=3.0)
    parser.add_argument("--close-phase", type=float, default=0.22)
    parser.add_argument("--release-phase", type=float, default=0.78)
    parser.add_argument("--near-threshold", type=float, default=0.11)
    parser.add_argument("--release-distance", type=float, default=0.095)
    parser.add_argument("--open-gripper", type=float, default=0.037)
    parser.add_argument("--close-gripper", type=float, default=0.015)
    parser.add_argument("--lift-threshold", type=float, default=0.085)
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / f"{VERSION}.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "grasp_gated_trajectory_act_report.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "grasp_gated_trajectory_act_report.md")
    return parser.parse_args()


def control_args(args: argparse.Namespace, method: dict) -> SimpleNamespace:
    return SimpleNamespace(
        steps=int(args.steps),
        viewer=False,
        duration=0.0,
        speed=0.0,
        arm_kp=float(args.arm_kp),
        arm_force=float(args.arm_force),
        gripper_kp=float(args.gripper_kp),
        gripper_force=float(args.gripper_force),
        friction=float(args.friction),
        clip_actions=True,
        action_alpha=float(method["action_alpha"]),
        max_arm_delta=float(method["max_arm_delta"]),
        max_gripper_delta=float(method["max_gripper_delta"]),
        replan_interval=int(method["replan_interval"]),
        temporal_ensemble=True,
        ensemble_decay=0.1,
        stop_on_unsafe=True,
        log_every=0,
        grasp_gate=True,
        close_phase=float(args.close_phase),
        release_phase=float(args.release_phase),
        near_threshold=float(args.near_threshold),
        release_distance=float(args.release_distance),
        open_gripper=float(args.open_gripper),
        close_gripper=float(args.close_gripper),
    )


def run_one_split(args: argparse.Namespace, method: dict, split: str, seed_start: int) -> list[dict]:
    model = method["loader"](method["model"])
    runner_args = control_args(args, method)
    rows = []
    for offset in range(int(args.episodes)):
        seed = int(seed_start) + offset
        env, obs = method["configure"](runner_args, seed, str(args.task), str(args.complexity))
        summary = method["rollout"](runner_args, model, env, obs, seed, str(args.task), str(args.complexity), viewer=None)
        max_object_z = float(summary.get("max_object_z", summary["object_z"]))
        height_threshold_hit = bool(max_object_z >= float(args.lift_threshold))
        ever_grasp_success = bool(summary.get("ever_grasp_success", False))
        row = {
            "version": method["version"],
            "base_version": method["base_version"],
            "method": method["method"],
            "split": split,
            "seed": seed,
            "task": str(args.task),
            "complexity": str(args.complexity),
            "success": bool(summary["success"]),
            "target_distance": float(summary["target_distance"]),
            "object_z": float(summary["object_z"]),
            "max_object_z": max_object_z,
            "height_threshold_hit": height_threshold_hit,
            "grasp_success": bool(summary["grasp_success"]),
            "ever_grasp_success": ever_grasp_success,
            "strict_grasp_lift_success": bool(ever_grasp_success and height_threshold_hit),
            "out_of_table": bool(summary["out_of_table"]),
            "steps_taken": int(summary["steps_taken"]),
            "stop_reason": summary["stop_reason"],
            "mean_action_norm": float(summary["mean_action_norm"]),
            "max_action_norm": float(summary["max_action_norm"]),
            "gate_open_steps": int(summary.get("gate_open_steps", 0)),
            "gate_closed_steps": int(summary.get("gate_closed_steps", 0)),
            "gate_policy_steps": int(summary.get("gate_policy_steps", 0)),
        }
        rows.append(row)
        print(
            f"{row['version']} {split} seed={seed} "
            f"success={row['success']} ever_grasp={row['ever_grasp_success']} "
            f"strict_grasp_lift={row['strict_grasp_lift_success']} "
            f"max_object_z={row['max_object_z']:.4f} target_distance={row['target_distance']:.4f}",
            flush=True,
        )
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["version"], row["split"]), []).append(row)

    output = []
    for (version, split), items in sorted(groups.items()):
        output.append(
            {
                "version": version,
                "split": split,
                "episodes": len(items),
                "successes": sum(1 for item in items if item["success"]),
                "height_threshold_hits": sum(1 for item in items if item["height_threshold_hit"]),
                "ever_grasp_successes": sum(1 for item in items if item["ever_grasp_success"]),
                "strict_grasp_lift_successes": sum(1 for item in items if item["strict_grasp_lift_success"]),
                "mean_target_distance": statistics.fmean(float(item["target_distance"]) for item in items),
                "mean_max_object_z": statistics.fmean(float(item["max_object_z"]) for item in items),
                "stop_reasons": sorted({str(item["stop_reason"]) for item in items if item["stop_reason"]}),
            }
        )
    return output


def ps_command(method: dict, args: argparse.Namespace, seed: int) -> str:
    command = (
        f'& "{PYTHON}" "{ROOT / method["runner"]}" '
        f'--model "{method["model"]}" '
        f"--task {args.task} --complexity {args.complexity} --seed {seed} --episodes 1 --steps {args.steps} "
        f"--viewer --duration 60 --speed 0.05 --grasp-gate "
        f"--close-phase {args.close_phase} --release-phase {args.release_phase} "
        f"--near-threshold {args.near_threshold} --release-distance {args.release_distance} "
        f"--open-gripper {args.open_gripper} --close-gripper {args.close_gripper} "
        f"--action-alpha {method['action_alpha']} --max-arm-delta {method['max_arm_delta']} "
        f"--max-gripper-delta {method['max_gripper_delta']} --replan-interval {method['replan_interval']} "
        f"--temporal-ensemble --ensemble-decay 0.1 --gripper-kp {args.gripper_kp} "
        f"--gripper-force {args.gripper_force} --friction {args.friction} --stop-on-unsafe --log-every 500"
    )
    return f"{method['env_prefix']}{command}"


def video_path(method: dict, seed: int) -> Path:
    return ROOT / "outputs" / "videos" / f"{method['version']}_seed{seed}.mp4"


def export_command(method: dict, args: argparse.Namespace, seed: int) -> str:
    command = (
        f'& "{PYTHON}" "{ROOT / "scripts" / "export_video.py"}" '
        f"--method {method['export_method']} --task {args.task} --complexity {args.complexity} "
        f"--seed {seed} --steps {args.steps} --camera top_rgb --fps 24 --frame-stride 12 "
        f"--width 640 --height 480 --gripper-kp {args.gripper_kp} "
        f"--gripper-force {args.gripper_force} --friction {args.friction} --log-every 0"
    )
    return f"{method['env_prefix']}{command}"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def md_row(values: list[object]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"


def write_md(path: Path, args: argparse.Namespace, rows: list[dict], summary_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 抓取门控 Trajectory / ACT 诊断报告",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：在不重新训练模型的前提下，把同一组 trajectory-conditioned chunk BC 和 Torch ACT 模型接入保守抓取门控，判断失败是否主要来自夹爪闭环控制、接触保持和抬升阶段。",
        "",
        "论文边界：这是候选诊断实验，不是新的正式学习方法；只有 `ever_grasp_success=True` 且 `max_object_z` 超过阈值时，才记为严格抓取+高度达标。单独的 `success=True` 或单独的高度超过阈值都不能写成稳定抓取成功。",
        "",
        "## 1. 评测设置",
        "",
        f"- 任务：`{args.task}`，复杂度：`{args.complexity}`。",
        f"- 每个方法训练范围 `{args.train_seed_start}-{args.train_seed_start + args.episodes - 1}`，留出范围 `{args.heldout_seed_start}-{args.heldout_seed_start + args.episodes - 1}`。",
        f"- 控制层：`--grasp-gate`，`close_phase={args.close_phase}`，`release_phase={args.release_phase}`，`near_threshold={args.near_threshold}`。",
        f"- 抬升阈值：`max_object_z >= {args.lift_threshold}`。",
        "",
        "## 2. 汇总结果",
        "",
        md_row(["版本", "范围", "放置 success", "曾经抓取", "高度超过阈值", "严格抓取+高度", "平均目标距离", "平均最高高度", "停止原因"]),
        md_row(["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---"]),
    ]
    for item in summary_rows:
        lines.append(
            md_row(
                [
                    f"`{item['version']}`",
                    item["split"],
                    f"{item['successes']}/{item['episodes']}",
                    f"{item['ever_grasp_successes']}/{item['episodes']}",
                    f"{item['height_threshold_hits']}/{item['episodes']}",
                    f"{item['strict_grasp_lift_successes']}/{item['episodes']}",
                    f"{float(item['mean_target_distance']):.4f}",
                    f"{float(item['mean_max_object_z']):.4f}",
                    ", ".join(item["stop_reasons"]) if item["stop_reasons"] else "无",
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 3. 单次明细",
            "",
            md_row(["版本", "范围", "seed", "success", "ever_grasp", "strict_grasp_lift", "max_object_z", "target_distance", "gate_closed_steps"]),
            md_row(["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
        ]
    )
    for row in rows:
        lines.append(
            md_row(
                [
                    f"`{row['version']}`",
                    row["split"],
                    row["seed"],
                    row["success"],
                    row["ever_grasp_success"],
                    row["strict_grasp_lift_success"],
                    f"{float(row['max_object_z']):.4f}",
                    f"{float(row['target_distance']):.4f}",
                    row["gate_closed_steps"],
                ]
            )
        )

    lines.extend(["", "## 4. 固定视频证据", ""])
    lines.append(md_row(["版本", "seed", "视频文件", "元数据文件"]))
    lines.append(md_row(["---", "---:", "---", "---"]))
    for method in METHODS:
        path = video_path(method, args.train_seed_start)
        lines.append(
            md_row(
                [
                    f"`{method['version']}`",
                    args.train_seed_start,
                    f"`{path.relative_to(ROOT).as_posix()}`",
                    f"`{path.with_suffix('.json').relative_to(ROOT).as_posix()}`",
                ]
            )
        )

    lines.extend(["", "## 5. 完整 Viewer 命令", ""])
    for method in METHODS:
        lines.extend(
            [
                f"### `{method['version']}`",
                "",
                "```powershell",
                ps_command(method, args, args.train_seed_start),
                "```",
                "",
            ]
        )

    lines.extend(["## 6. 重新导出视频命令", ""])
    for method in METHODS:
        lines.extend([f"### `{method['version']}`", "", "```powershell", export_command(method, args, args.train_seed_start), "```", ""])

    lines.extend(["## 7. 重建评测命令", "", "```powershell", f'& "{PYTHON}" "{ROOT / "scripts" / "evaluate_grasp_gated_trajectory_act.py"}"', "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    all_rows = []
    for method in METHODS:
        all_rows.extend(run_one_split(args, method, "train_range", args.train_seed_start))
        all_rows.extend(run_one_split(args, method, "heldout", args.heldout_seed_start))
    summary_rows = aggregate(all_rows)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "version": VERSION,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "task": args.task,
                "complexity": args.complexity,
                "rows": all_rows,
                "summary": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(args.output_csv, all_rows)
    write_md(args.output_md, args, all_rows, summary_rows)
    print(f"json_path: {args.output_json}", flush=True)
    print(f"csv_path: {args.output_csv}", flush=True)
    print(f"md_path: {args.output_md}", flush=True)
    print(f"rows: {len(all_rows)}", flush=True)


if __name__ == "__main__":
    main()
