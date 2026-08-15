from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "trajectory_phase_template_bc_v1_candidate"


FIELDNAMES = [
    "version",
    "split",
    "seed",
    "episodes",
    "successes",
    "success_rate",
    "grasp_successes",
    "mean_target_distance",
    "mean_ee_object_distance",
    "mean_object_z",
    "out_of_table",
    "command",
    "interpretation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trajectory phase template BC candidate.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=2840)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--heldout-seed", type=int, default=100)
    parser.add_argument("--task", default="place_blue_cube_blue_pad")
    parser.add_argument("--complexity", default="medium")
    parser.add_argument("--action-alpha", type=float, default=0.35)
    parser.add_argument("--max-arm-delta", type=float, default=0.018)
    parser.add_argument("--max-gripper-delta", type=float, default=0.0008)
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / "trajectory_phase_template_bc_v1.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "trajectory_phase_template_bc_report.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "trajectory_phase_template_bc_report.md")
    return parser.parse_args()


def latest_model() -> Path:
    candidates = sorted(
        (ROOT / "outputs" / "trajectory_phase_template_bc").glob("trajectory_phase_template_bc_*.npz"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError("no trajectory phase template BC model found")
    return candidates[-1]


def command_for(args: argparse.Namespace, model_path: Path, seed: int) -> list[str]:
    return [
        str(PYTHON),
        str(ROOT / "scripts" / "run_trajectory_phase_template_policy.py"),
        "--model",
        str(model_path),
        "--task",
        args.task,
        "--complexity",
        args.complexity,
        "--seed",
        str(seed),
        "--episodes",
        str(args.episodes),
        "--steps",
        str(args.steps),
        "--no-viewer",
        "--action-alpha",
        str(args.action_alpha),
        "--max-arm-delta",
        str(args.max_arm_delta),
        "--max-gripper-delta",
        str(args.max_gripper_delta),
        "--stop-on-unsafe",
        "--log-every",
        "0",
    ]


def parse_summaries(stdout: str) -> list[dict]:
    summaries = []
    for line in stdout.splitlines():
        if line.startswith("episode_summary:"):
            summaries.append(ast.literal_eval(line.split("episode_summary:", 1)[1].strip()))
    if not summaries:
        raise RuntimeError(f"no episode_summary lines found:\n{stdout[-1000:]}")
    return summaries


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def compact_command(command: list[str]) -> str:
    return " ".join(f'"{item}"' if " " in item else item for item in command)


def interpretation(successes: int, episodes: int, grasp_successes: int, out_of_table: int) -> str:
    if successes == episodes and grasp_successes == episodes:
        return "可作为稳定候选，但仍需 held-out 和语言任务复测。"
    if successes > 0 and grasp_successes == 0:
        return "有少量放置成功，但没有稳定抬升，更像模板推/碰到目标区，不应登记为可靠抓取策略。"
    if out_of_table:
        return "出现出界风险，当前模板不能作为正式 baseline。"
    return "未形成稳定成功，可作为 phase-template 反例和失败诊断。"


def run_split(args: argparse.Namespace, model_path: Path, split: str, seed: int) -> dict[str, object]:
    command = command_for(args, model_path, seed)
    result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True, errors="replace")
    summaries = parse_summaries(result.stdout)
    metrics = [item.get("metrics", {}) for item in summaries]
    successes = sum(1 for item in summaries if item.get("success"))
    grasp_successes = sum(1 for item in summaries if item.get("grasp_success"))
    out_of_table = sum(1 for item in summaries if item.get("out_of_table"))
    return {
        "version": VERSION,
        "split": split,
        "seed": seed,
        "episodes": len(summaries),
        "successes": successes,
        "success_rate": successes / max(1, len(summaries)),
        "grasp_successes": grasp_successes,
        "mean_target_distance": mean([float(item["target_distance"]) for item in summaries]),
        "mean_ee_object_distance": mean([float(metric.get("ee_object_distance", 0.0)) for metric in metrics]),
        "mean_object_z": mean([float(item["object_z"]) for item in summaries]),
        "out_of_table": out_of_table,
        "command": compact_command(command),
        "interpretation": interpretation(successes, len(summaries), grasp_successes, out_of_table),
        "episode_summaries": summaries,
        "stdout_tail": result.stdout[-1200:],
        "stderr_tail": result.stderr[-1200:],
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


def write_md(path: Path, rows: list[dict[str, object]], args: argparse.Namespace, model_path: Path) -> None:
    lines = [
        "# Trajectory Phase Template BC 候选实验",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：继续推进 trajectory-conditioned BC / ACT 阶段时，检查“显式轨迹相位模板 + 小型线性 action head”能否比普通动作块回归更稳定。",
        "",
        "方法定位：该方法按 phase 分成 128 个 bin，每个 bin 训练一个 ridge action head。输入只包含初始目标物体位置、目标区域位置、二者相对向量和 phase 特征，输出 7 维控制目标。它是轻量 trajectory-conditioned BC 候选，不是完整 ACT，也不是 VLA。",
        "",
        f"模型：`{model_path}`",
        "",
        "结论：当前候选不能登记为可靠 ACT baseline。它在 train-range 和 held-out 都只有少量成功，而且 `grasp_successes=0`，说明成功更可能来自推/碰到目标区域，而不是稳定抓取、抬升和放置。",
        "",
        "## 1. 评测结果",
        "",
        md_row(["split", "seed", "成功率", "grasp_successes", "平均目标距离", "平均末端-物体距离", "平均物体高度", "出界次数", "解释"]),
        md_row(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---"]),
    ]
    for row in rows:
        lines.append(
            md_row(
                [
                    row["split"],
                    row["seed"],
                    f"{row['successes']}/{row['episodes']}",
                    row["grasp_successes"],
                    f"{float(row['mean_target_distance']):.4f}",
                    f"{float(row['mean_ee_object_distance']):.4f}",
                    f"{float(row['mean_object_z']):.4f}",
                    row["out_of_table"],
                    row["interpretation"],
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 2. 论文可写边界",
            "",
            "- 可以写：phase-template 比普通 MLP action chunk 更有结构，但仍不能稳定完成抓取和抬升。",
            "- 可以写：少量 place 成功不能等同于抓取策略成功，必须同时检查 `grasp_successes`、物体高度和视频。",
            "- 不能写：该方法已经成为可靠 ACT baseline。",
            "- 不能写：该方法证明了 VLA 或 LoRA/Adapter 后训练有效。",
            "",
            "## 3. 重建命令",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "train_trajectory_phase_template_bc.py"}" --run-dir "{ROOT / "data" / "demos" / "place_blue_cube_blue_pad_medium_20260702_051752"}" --bins 128 --sample-stride 4 --ridge 0.001 --min-bin-samples 64 --feature-mode planned',
            f'& "{PYTHON}" "{ROOT / "scripts" / "evaluate_trajectory_phase_template_bc.py"}" --model "{model_path}" --episodes {args.episodes} --steps {args.steps}',
            "```",
            "",
            "## 4. Viewer 查看命令",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "run_trajectory_phase_template_policy.py"}" --model "{model_path}" --task {args.task} --complexity {args.complexity} --seed 1 --episodes 1 --steps {args.steps} --viewer --duration 60 --speed 0.05 --action-alpha {args.action_alpha} --max-arm-delta {args.max_arm_delta} --max-gripper-delta {args.max_gripper_delta} --stop-on-unsafe --log-every 500',
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    model_path = args.model or latest_model()
    rows = [
        run_split(args, model_path, "train_range", args.train_seed),
        run_split(args, model_path, "heldout", args.heldout_seed),
    ]
    payload = {
        "version": VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": str(model_path),
        "task": args.task,
        "complexity": args.complexity,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows, args, model_path)
    print(f"trajectory_phase_template_json: {args.output_json}", flush=True)
    print(f"trajectory_phase_template_csv: {args.output_csv}", flush=True)
    print(f"trajectory_phase_template_md: {args.output_md}", flush=True)
    print(f"trajectory_phase_template_rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
