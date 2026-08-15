from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "control_safety_sweep_v1"
DEFAULT_TORCH_PACKAGE_DIR = Path("D:/vla_torch_cuda_pkgs")


METHODS = {
    "trajectory_conditioned_chunk_bc_v2": {
        "script": "scripts/run_chunk_policy.py",
        "model": "outputs/chunk_bc/trajectory_chunk_bc_20260720_043500.npz",
        "base_args": ["--replan-interval", "1", "--temporal-ensemble", "--ensemble-decay", "0.1"],
        "note": "8 帧历史状态条件 + 8 步动作块 MLP baseline。",
    },
    "trajectory_knn_chunk_bc_v1": {
        "script": "scripts/run_trajectory_knn_policy.py",
        "model": "outputs/trajectory_knn_bc/trajectory_knn_chunk_bc_20260720_053423.npz",
        "base_args": [
            "--k",
            "3",
            "--phase-window",
            "0.03",
            "--min-candidates",
            "256",
            "--history-decay",
            "0.25",
            "--replan-interval",
            "1",
            "--temporal-ensemble",
            "--ensemble-decay",
            "0.1",
        ],
        "note": "历史轨迹检索动作块；训练范围成功但留出泛化弱。",
    },
    "torch_act_state_chunk_v1": {
        "script": "scripts/run_torch_act_policy.py",
        "model": "outputs/torch_act/torch_act_state_chunk_20260720_055409.pt",
        "base_args": ["--replan-interval", "4", "--temporal-ensemble", "--ensemble-decay", "0.1"],
        "note": "state-only PyTorch Transformer ACT-style baseline。",
    },
}


PRESETS = {
    "current_slow": {
        "action_alpha": 0.25,
        "max_arm_delta": 0.012,
        "max_gripper_delta": 0.0005,
        "description": "当前慢速 viewer / 诊断默认限幅。",
    },
    "slower": {
        "action_alpha": 0.15,
        "max_arm_delta": 0.008,
        "max_gripper_delta": 0.00035,
        "description": "进一步减小动作平滑和手臂/夹爪增量。",
    },
    "very_slow": {
        "action_alpha": 0.08,
        "max_arm_delta": 0.005,
        "max_gripper_delta": 0.0002,
        "description": "极慢控制，检验是否只是控制太快导致失败。",
    },
}


FIELDNAMES = [
    "version",
    "method",
    "preset",
    "episodes",
    "successes",
    "success_rate",
    "mean_target_distance",
    "mean_ee_object_distance",
    "mean_object_z",
    "mean_contact_count",
    "mean_mean_action_norm",
    "max_action_norm",
    "grasp_successes",
    "out_of_table",
    "stop_reasons",
    "action_alpha",
    "max_arm_delta",
    "max_gripper_delta",
    "steps",
    "command",
    "interpretation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate control/action-limit presets for representative trajectory/ACT baselines.")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2840)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", default="place_blue_cube_blue_pad")
    parser.add_argument("--complexity", default="medium")
    parser.add_argument("--methods", default="trajectory_conditioned_chunk_bc_v2,trajectory_knn_chunk_bc_v1,torch_act_state_chunk_v1")
    parser.add_argument("--presets", default="current_slow,slower,very_slow")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / "control_safety_sweep_v1.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "control_safety_sweep.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "control_safety_sweep.md")
    return parser.parse_args()


def selected(value: str, available: dict[str, object]) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    missing = [name for name in names if name not in available]
    if missing:
        raise KeyError(f"unknown names: {missing}")
    return names


def build_command(method_name: str, preset_name: str, args: argparse.Namespace) -> list[str]:
    method = METHODS[method_name]
    preset = PRESETS[preset_name]
    return [
        str(PYTHON),
        str(ROOT / method["script"]),
        "--model",
        str(ROOT / method["model"]),
        "--task",
        args.task,
        "--complexity",
        args.complexity,
        "--seed",
        str(args.seed),
        "--episodes",
        str(args.episodes),
        "--steps",
        str(args.steps),
        "--no-viewer",
        "--action-alpha",
        str(preset["action_alpha"]),
        "--max-arm-delta",
        str(preset["max_arm_delta"]),
        "--max-gripper-delta",
        str(preset["max_gripper_delta"]),
        "--stop-on-unsafe",
        "--log-every",
        "0",
        *method["base_args"],
    ]


def parse_summaries(stdout: str) -> list[dict]:
    summaries = []
    for line in stdout.splitlines():
        if not line.startswith("episode_summary:"):
            continue
        payload = line.split("episode_summary:", 1)[1].strip()
        summaries.append(ast.literal_eval(payload))
    if not summaries:
        raise RuntimeError(f"no episode_summary lines found in output:\n{stdout[-1000:]}")
    return summaries


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def interpretation(successes: int, episodes: int, grasp_successes: int, out_of_table: int, preset_name: str) -> str:
    if successes == episodes and grasp_successes > 0:
        return "该控制档位能稳定完成并出现夹紧，可作为后续复测候选。"
    if successes > 0:
        return "该控制档位有少量成功，但仍需看批量复测和是否真正夹紧。"
    if out_of_table:
        return "该控制档位仍有出界风险，限幅没有解决安全问题。"
    if preset_name == "very_slow":
        return "极慢控制仍未成功，说明失败不能只归因于动作太快。"
    return "该控制档位未形成稳定成功，继续对比更慢档位。"


def run_one(method_name: str, preset_name: str, args: argparse.Namespace) -> dict[str, object]:
    command = build_command(method_name, preset_name, args)
    env = os.environ.copy()
    if "VLA_TORCH_PACKAGE_DIR" not in env and DEFAULT_TORCH_PACKAGE_DIR.exists():
        env["VLA_TORCH_PACKAGE_DIR"] = str(DEFAULT_TORCH_PACKAGE_DIR)
    result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True, errors="replace", env=env)
    summaries = parse_summaries(result.stdout)
    metrics = [summary.get("metrics", {}) for summary in summaries]
    successes = sum(1 for summary in summaries if summary.get("success"))
    grasp_successes = sum(1 for summary in summaries if summary.get("grasp_success"))
    out_of_table = sum(1 for summary in summaries if summary.get("out_of_table"))
    stop_reasons = sorted({str(summary.get("stop_reason")) for summary in summaries if summary.get("stop_reason")})
    preset = PRESETS[preset_name]
    return {
        "version": VERSION,
        "method": method_name,
        "preset": preset_name,
        "episodes": len(summaries),
        "successes": successes,
        "success_rate": successes / max(1, len(summaries)),
        "mean_target_distance": mean([float(summary["target_distance"]) for summary in summaries]),
        "mean_ee_object_distance": mean([float(metric.get("ee_object_distance", 0.0)) for metric in metrics]),
        "mean_object_z": mean([float(summary["object_z"]) for summary in summaries]),
        "mean_contact_count": mean([float(metric.get("contact_count", 0.0)) for metric in metrics]),
        "mean_mean_action_norm": mean([float(summary["mean_action_norm"]) for summary in summaries]),
        "max_action_norm": max(float(summary["max_action_norm"]) for summary in summaries),
        "grasp_successes": grasp_successes,
        "out_of_table": out_of_table,
        "stop_reasons": "；".join(stop_reasons) if stop_reasons else "无",
        "action_alpha": preset["action_alpha"],
        "max_arm_delta": preset["max_arm_delta"],
        "max_gripper_delta": preset["max_gripper_delta"],
        "steps": args.steps,
        "command": " ".join(f'"{item}"' if " " in item else item for item in command),
        "interpretation": interpretation(successes, len(summaries), grasp_successes, out_of_table, preset_name),
        "episode_summaries": summaries,
        "stderr_tail": result.stderr[-1000:],
        "preset_description": preset["description"],
        "method_note": METHODS[method_name]["note"],
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in FIELDNAMES})


def md_row(values: list[object]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", "<br>") for value in values) + " |"


def write_md(path: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best = sorted(rows, key=lambda row: (float(row["success_rate"]), -float(row["mean_target_distance"])), reverse=True)[0]
    lines = [
        "# 控制限幅扫表",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：量化检查 trajectory-conditioned BC / ACT 代表方法在更慢动作平滑和更小增量限制下是否明显改善。该扫表不替代主成功率评测，只回答“失败是否主要由动作太快导致”。",
        "",
        f"运行设置：task=`{args.task}`，complexity=`{args.complexity}`，seed 起点 `{args.seed}`，每格 `{args.episodes}` 个 episode，步数 `{args.steps}`，全部 `--no-viewer`。",
        "",
        "论文边界：如果 `very_slow` 仍不能稳定成功，只能写成控制限幅不是主要瓶颈之一；不能写成真实机器人控制已解决，也不能写成完整 ACT 或 VLA 策略成功。",
        "",
        "## 1. 总览",
        "",
        md_row(["项目", "值"]),
        md_row(["---", "---"]),
        md_row(["扫表格数", len(rows)]),
        md_row(["最佳格", f"{best['method']} / {best['preset']} / {best['successes']}/{best['episodes']}"]),
        "",
        "## 2. 结果表",
        "",
        md_row(["方法", "控制档", "成功率", "平均目标距离", "平均末端-物体距离", "平均物体高度", "平均接触数", "最大动作范数", "夹紧成功", "解释"]),
        md_row(["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---"]),
    ]
    for row in rows:
        lines.append(
            md_row(
                [
                    f"`{row['method']}`",
                    row["preset"],
                    f"{row['successes']}/{row['episodes']}",
                    f"{float(row['mean_target_distance']):.4f}",
                    f"{float(row['mean_ee_object_distance']):.4f}",
                    f"{float(row['mean_object_z']):.4f}",
                    f"{float(row['mean_contact_count']):.1f}",
                    f"{float(row['max_action_norm']):.4f}",
                    row["grasp_successes"],
                    row["interpretation"],
                ]
            )
        )

    lines.extend(["", "## 3. 控制档定义", ""])
    for name, preset in PRESETS.items():
        lines.append(f"- `{name}`：action_alpha={preset['action_alpha']}，max_arm_delta={preset['max_arm_delta']}，max_gripper_delta={preset['max_gripper_delta']}。{preset['description']}")

    lines.extend(
        [
            "",
            "## 4. 重建命令",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "evaluate_control_safety_sweep.py"}" --episodes {args.episodes} --steps {args.steps}',
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    method_names = selected(args.methods, METHODS)
    preset_names = selected(args.presets, PRESETS)
    rows = []
    for method_name in method_names:
        for preset_name in preset_names:
            print(f"running: method={method_name}, preset={preset_name}", flush=True)
            rows.append(run_one(method_name, preset_name, args))
    payload = {
        "version": VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "task": args.task,
        "complexity": args.complexity,
        "episodes_per_cell": args.episodes,
        "steps": args.steps,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows, args)
    print(f"control_safety_sweep_json: {args.output_json}", flush=True)
    print(f"control_safety_sweep_csv: {args.output_csv}", flush=True)
    print(f"control_safety_sweep_md: {args.output_md}", flush=True)
    print(f"control_safety_sweep_rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
