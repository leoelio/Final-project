from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "action_head_control_safety_sweep_v1"

sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_control_safety_sweep import PRESETS, mean, parse_summaries  # noqa: E402


METHODS = {
    "object_language_action_head_lite_v1": {
        "script": "scripts/run_object_action_head.py",
        "model": "outputs/object_action_head/object_action_head_lite_20260720_044703.npz",
        "note": "对象/语言结构特征 action-head proxy。",
    },
    "adapter_action_head_lite_v1": {
        "script": "scripts/run_peft_action_head.py",
        "model": "outputs/peft_action_head/adapter_action_head_lite_20260720_072914.npz",
        "note": "冻结 base action-head 后训练 Adapter residual。",
    },
    "lora_action_head_lite_v1": {
        "script": "scripts/run_peft_action_head.py",
        "model": "outputs/peft_action_head/lora_action_head_lite_20260720_072913.npz",
        "note": "冻结 base action-head 后训练 LoRA-style residual。",
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
    parser = argparse.ArgumentParser(description="Evaluate control/action-limit presets for action-head/PEFT proxy baselines.")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2840)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", default="place_blue_cube_blue_pad")
    parser.add_argument("--complexity", default="medium")
    parser.add_argument("--methods", default="object_language_action_head_lite_v1,adapter_action_head_lite_v1,lora_action_head_lite_v1")
    parser.add_argument("--presets", default="current_slow,slower,very_slow")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / "action_head_control_safety_sweep_v1.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "action_head_control_safety_sweep.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "action_head_control_safety_sweep.md")
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
    ]


def interpretation(successes: int, episodes: int, grasp_successes: int, preset_name: str) -> str:
    if successes == episodes and grasp_successes > 0:
        return "该控制档位能稳定完成并出现夹紧，可作为 action-head 后续复测候选。"
    if successes > 0:
        return "该控制档位有少量成功，但仍需看批量复测和是否真正夹紧。"
    if preset_name == "very_slow":
        return "极慢控制仍未成功，说明 action-head 失败不能只归因于动作太快。"
    return "该控制档位未形成稳定成功，继续对比更慢档位。"


def run_one(method_name: str, preset_name: str, args: argparse.Namespace) -> dict[str, object]:
    command = build_command(method_name, preset_name, args)
    result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True, errors="replace", env=os.environ.copy())
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
        "interpretation": interpretation(successes, len(summaries), grasp_successes, preset_name),
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
    best = sorted(rows, key=lambda row: (float(row["success_rate"]), -float(row["mean_target_distance"])), reverse=True)[0]
    lines = [
        "# Action-head 控制限幅扫表",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：量化检查 object action-head、Adapter action-head 和 LoRA-style action-head 在更慢动作平滑和更小增量限制下是否明显改善。该扫表不替代主成功率评测，只回答 action-head/PEFT proxy 的失败是否主要由动作太快导致。",
        "",
        f"运行设置：task=`{args.task}`，complexity=`{args.complexity}`，seed 起点 `{args.seed}`，每格 `{args.episodes}` 个 episode，步数 `{args.steps}`，全部 `--no-viewer`。",
        "",
        "论文边界：本表是本地 action-head/PEFT proxy 的控制扫表，不是真实 OpenVLA LoRA、RT-2 或真实机械臂验证。",
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
            f'& "{PYTHON}" "{ROOT / "scripts" / "evaluate_action_head_control_safety_sweep.py"}" --episodes {args.episodes} --steps {args.steps}',
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
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
    print(f"action_head_control_safety_sweep_json: {args.output_json}", flush=True)
    print(f"action_head_control_safety_sweep_csv: {args.output_csv}", flush=True)
    print(f"action_head_control_safety_sweep_md: {args.output_md}", flush=True)
    print(f"action_head_control_safety_sweep_rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
