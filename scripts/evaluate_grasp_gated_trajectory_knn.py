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
VERSION = "grasp_gated_trajectory_knn_v1_candidate"
MODEL = ROOT / "outputs" / "trajectory_knn_bc" / "trajectory_knn_chunk_bc_20260720_053423.npz"


FIELDNAMES = [
    "version",
    "split",
    "seed",
    "episodes",
    "successes",
    "success_rate",
    "grasp_successes",
    "out_of_table",
    "mean_target_distance",
    "mean_ee_object_distance",
    "mean_object_z",
    "mean_gate_closed_steps",
    "mean_gate_open_steps",
    "command",
    "interpretation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate grasp-gated trajectory-kNN diagnostic policy.")
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=2840)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--heldout-seed", type=int, default=100)
    parser.add_argument("--task", default="place_blue_cube_blue_pad")
    parser.add_argument("--complexity", default="medium")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / "grasp_gated_trajectory_knn_v1.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "grasp_gated_trajectory_knn_report.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "grasp_gated_trajectory_knn_report.md")
    return parser.parse_args()


def command_for(args: argparse.Namespace, seed: int) -> list[str]:
    return [
        str(PYTHON),
        str(ROOT / "scripts" / "run_grasp_gated_trajectory_knn_policy.py"),
        "--model",
        str(args.model),
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
        "--k",
        "3",
        "--phase-window",
        "0.03",
        "--min-candidates",
        "256",
        "--history-decay",
        "0.25",
        "--action-alpha",
        "0.85",
        "--max-arm-delta",
        "0.04",
        "--max-gripper-delta",
        "0.002",
        "--replan-interval",
        "1",
        "--temporal-ensemble",
        "--ensemble-decay",
        "0.1",
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


def interpretation(successes: int, episodes: int, grasp_successes: int, out_of_table: int, split: str) -> str:
    if grasp_successes > 0 and successes > 0:
        return "夹爪门控带来真实 grasp 样例，可作为后续复测候选。"
    if successes > 0 and grasp_successes == 0:
        return "有放置成功但没有抓取/抬升，说明门控没有解决真实 grasp。"
    if out_of_table:
        return "仍出现出界风险，夹爪门控可能放大轨迹偏差。"
    return f"{split} 未成功，夹爪门控不能单独解释或修复失败。"


def run_split(args: argparse.Namespace, split: str, seed: int) -> dict[str, object]:
    command = command_for(args, seed)
    result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True, errors="replace")
    summaries = parse_summaries(result.stdout)
    metrics = [summary.get("metrics", {}) for summary in summaries]
    successes = sum(1 for summary in summaries if summary.get("success"))
    grasp_successes = sum(1 for summary in summaries if summary.get("grasp_success"))
    out_of_table = sum(1 for summary in summaries if summary.get("out_of_table"))
    return {
        "version": VERSION,
        "split": split,
        "seed": seed,
        "episodes": len(summaries),
        "successes": successes,
        "success_rate": successes / max(1, len(summaries)),
        "grasp_successes": grasp_successes,
        "out_of_table": out_of_table,
        "mean_target_distance": mean([float(summary["target_distance"]) for summary in summaries]),
        "mean_ee_object_distance": mean([float(metric.get("ee_object_distance", 0.0)) for metric in metrics]),
        "mean_object_z": mean([float(summary["object_z"]) for summary in summaries]),
        "mean_gate_closed_steps": mean([float(summary["gate_closed_steps"]) for summary in summaries]),
        "mean_gate_open_steps": mean([float(summary["gate_open_steps"]) for summary in summaries]),
        "command": compact_command(command),
        "interpretation": interpretation(successes, len(summaries), grasp_successes, out_of_table, split),
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


def write_md(path: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    lines = [
        "# Grasp-gated Trajectory-kNN 候选实验",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：在 trajectory-conditioned BC / ACT 阶段中，检验“只给 trajectory-kNN 增加夹爪闭合/释放门控”是否能把放置成功转化为真实抓取、抬升和放置。",
        "",
        "方法定位：该方法复用 `trajectory_knn_chunk_bc_v1` 的历史轨迹检索和动作块，只在运行时加入 gripper gate。它是诊断型混合策略，不是纯 BC、不是完整 ACT，也不是 VLA。",
        "",
        "结论：当前候选不能登记为可靠 ACT baseline。train-range 有少量 place 成功，但 `grasp_successes=0`；held-out 失败且有出界样例。夹爪门控没有解决接触、对准和抬升问题。",
        "",
        "## 1. 评测结果",
        "",
        md_row(["split", "seed", "成功率", "grasp_successes", "出界", "平均目标距离", "平均末端-物体距离", "平均物体高度", "平均闭合步数", "解释"]),
        md_row(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---"]),
    ]
    for row in rows:
        lines.append(
            md_row(
                [
                    row["split"],
                    row["seed"],
                    f"{row['successes']}/{row['episodes']}",
                    row["grasp_successes"],
                    row["out_of_table"],
                    f"{float(row['mean_target_distance']):.4f}",
                    f"{float(row['mean_ee_object_distance']):.4f}",
                    f"{float(row['mean_object_z']):.4f}",
                    f"{float(row['mean_gate_closed_steps']):.1f}",
                    row["interpretation"],
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 2. 和原始 trajectory-kNN 的关系",
            "",
            "- 原始 `trajectory_knn_chunk_bc_v1` 在 train-range 是 5/5，但 held-out 是 0/5；它偏轨迹记忆。",
            "- 本候选强制夹爪阶段后 train-range 降到少量成功，且 `grasp_successes=0`，说明问题不是单独的夹爪开合命令。",
            "- 后续 ACT 改进应围绕接触前对准、夹爪闭合时机、物体是否随夹爪抬升，以及视觉/物体状态表征，而不是只加 gripper gate。",
            "",
            "## 3. 重建命令",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "evaluate_grasp_gated_trajectory_knn.py"}" --model "{args.model}" --episodes {args.episodes} --steps {args.steps}',
            "```",
            "",
            "## 4. Viewer 查看命令",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "run_grasp_gated_trajectory_knn_policy.py"}" --model "{args.model}" --task {args.task} --complexity {args.complexity} --seed 0 --episodes 1 --steps {args.steps} --viewer --duration 60 --speed 0.05 --k 3 --phase-window 0.03 --min-candidates 256 --history-decay 0.25 --action-alpha 0.85 --max-arm-delta 0.04 --max-gripper-delta 0.002 --replan-interval 1 --temporal-ensemble --ensemble-decay 0.1 --stop-on-unsafe --log-every 500',
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = [
        run_split(args, "train_range", args.train_seed),
        run_split(args, "heldout", args.heldout_seed),
    ]
    payload = {
        "version": VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": str(args.model),
        "task": args.task,
        "complexity": args.complexity,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows, args)
    print(f"grasp_gated_trajectory_knn_json: {args.output_json}", flush=True)
    print(f"grasp_gated_trajectory_knn_csv: {args.output_csv}", flush=True)
    print(f"grasp_gated_trajectory_knn_md: {args.output_md}", flush=True)
    print(f"grasp_gated_trajectory_knn_rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
