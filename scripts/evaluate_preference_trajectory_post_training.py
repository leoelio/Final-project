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
VERSION = "preference_trajectory_post_training_v1_candidate"
DEFAULT_MODEL = ROOT / "outputs" / "preference_post_training" / "preference_trajectory_post_training_20260720_000000.npz"


FIELDNAMES = [
    "version",
    "split",
    "seed",
    "episodes",
    "successes",
    "success_rate",
    "grasp_successes",
    "tcp_grasp_lift_successes",
    "strict_grasp_lift_successes",
    "height_threshold_hits",
    "out_of_table",
    "mean_target_distance",
    "mean_ee_object_distance",
    "mean_object_z",
    "mean_max_object_z",
    "mean_min_tcp_object_distance",
    "command",
    "interpretation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate preference-weighted trajectory post-training candidate.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=2840)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--heldout-seed", type=int, default=100)
    parser.add_argument("--task", default="place_blue_cube_blue_pad")
    parser.add_argument("--complexity", default="medium")
    parser.add_argument("--version", default=None)
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / "preference_trajectory_post_training_v1.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "preference_trajectory_post_training_report.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "preference_trajectory_post_training_report.md")
    return parser.parse_args()


def latest_model() -> Path:
    candidates = sorted(
        (ROOT / "outputs" / "preference_post_training").glob("preference_trajectory_post_training_*.npz"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"no preference post-training models found under {ROOT / 'outputs' / 'preference_post_training'}")
    return candidates[-1]


def load_metadata(model_path: Path) -> dict:
    import numpy as np

    with np.load(model_path) as data:
        return json.loads(data["metadata"].item())


def command_for(args: argparse.Namespace, seed: int, model_path: Path) -> list[str]:
    return [
        str(PYTHON),
        str(ROOT / "scripts" / "run_preference_trajectory_post_training_policy.py"),
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
        "--k",
        "3",
        "--phase-window",
        "0.03",
        "--min-candidates",
        "256",
        "--history-decay",
        "0.25",
        "--preference-power",
        "1.0",
        "--action-alpha",
        "0.85",
        "--max-arm-delta",
        "0.04",
        "--max-gripper-delta",
        "0.0015",
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


def interpretation(successes: int, episodes: int, grasp_successes: int, tcp_lift_successes: int, strict_successes: int, split: str) -> str:
    if successes > 0 and tcp_lift_successes > 0 and strict_successes == 0:
        return "有放置成功和 TCP 抬升迹象，但标准严格抓取仍未通过；只能作为后训练候选诊断。"
    if successes > 0 and grasp_successes > 0:
        return "偏好权重产生真实 grasp 样例，可作为后续复测候选。"
    if successes > 0 and grasp_successes == 0:
        return "有放置成功但没有抓取/抬升，说明偏好权重主要影响目标距离而非真实 grasp。"
    return f"{split} 未成功，当前偏好代理没有改善闭环抓取。"


def run_split(args: argparse.Namespace, split: str, seed: int, model_path: Path) -> dict[str, object]:
    command = command_for(args, seed, model_path)
    result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True, errors="replace")
    summaries = parse_summaries(result.stdout)
    metrics = [summary.get("metrics", {}) for summary in summaries]
    successes = sum(1 for summary in summaries if summary.get("success"))
    grasp_successes = sum(1 for summary in summaries if summary.get("grasp_success"))
    tcp_lift_successes = sum(1 for summary in summaries if summary.get("tcp_grasp_lift_success"))
    strict_successes = sum(1 for summary in summaries if summary.get("strict_grasp_lift_success"))
    height_hits = sum(1 for summary in summaries if summary.get("height_threshold_hit"))
    out_of_table = sum(1 for summary in summaries if summary.get("out_of_table"))
    return {
        "version": str(args.version),
        "split": split,
        "seed": seed,
        "episodes": len(summaries),
        "successes": successes,
        "success_rate": successes / max(1, len(summaries)),
        "grasp_successes": grasp_successes,
        "tcp_grasp_lift_successes": tcp_lift_successes,
        "strict_grasp_lift_successes": strict_successes,
        "height_threshold_hits": height_hits,
        "out_of_table": out_of_table,
        "mean_target_distance": mean([float(summary["target_distance"]) for summary in summaries]),
        "mean_ee_object_distance": mean([float(metric.get("ee_object_distance", 0.0)) for metric in metrics]),
        "mean_object_z": mean([float(summary["object_z"]) for summary in summaries]),
        "mean_max_object_z": mean([float(summary.get("max_object_z", summary["object_z"])) for summary in summaries]),
        "mean_min_tcp_object_distance": mean([float(summary.get("min_tcp_object_distance", 0.0) or 0.0) for summary in summaries]),
        "command": compact_command(command),
        "interpretation": interpretation(successes, len(summaries), grasp_successes, tcp_lift_successes, strict_successes, split),
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


def write_md(path: Path, rows: list[dict[str, object]], args: argparse.Namespace, model_path: Path, metadata: dict) -> None:
    preference_summary = metadata.get("preference_summary", {})
    lines = [
        "# Preference Trajectory Post-training 候选实验",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：在 trajectory-conditioned BC / ACT 阶段之后，测试一个不依赖外部 VLA/GPU 的轻量后训练候选：把成功/失败 attempt 和目标距离转成 trajectory-level preference weight，再用于 trajectory-kNN 的邻居投票。",
        "",
        "方法定位：这是偏好加权的轨迹记忆策略，不是在线 RL，不是真实人类偏好优化，也不是 OpenVLA 后训练。它只用于验证“trajectory-level preference/reward weighting 是否能改善当前闭环抓取”。",
        "",
        "## 1. 偏好来源与权重策略",
        "",
        f"- 偏好来源：{metadata.get('preference_source')}",
        f"- 权重策略：{metadata.get('preference_strategy')}",
        f"- episode 数：{preference_summary.get('episodes')}",
        f"- successful episodes：{preference_summary.get('successful_episodes')}",
        f"- attempts：{preference_summary.get('attempts')}",
        f"- preferred attempts：{preference_summary.get('preferred_attempts')}",
        f"- failed attempts：{preference_summary.get('failed_attempts')}",
        f"- 平均 attempt 目标距离：{float(preference_summary.get('mean_attempt_target_distance', 0.0)):.4f}",
        f"- 平均 preference weight：{float(preference_summary.get('mean_preference_weight', 0.0)):.4f}",
        "",
        "## 2. 闭环评测结果",
        "",
        md_row(["split", "seed", "成功率", "grasp_successes", "出界", "平均目标距离", "平均末端-物体距离", "平均物体高度", "解释"]),
        md_row(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---"]),
    ]
    lines.insert(3, f"version_id: `{args.version}`")
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
                    row["interpretation"],
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 3. 论文边界",
            "",
            "- 可以写成：基于示范 attempt 成败和目标距离的偏好加权轨迹后训练候选。若成功率不改善，它仍是一个负例。",
            "- 不能写成：在线 RL、真实 human preference optimization、OpenVLA LoRA/Adapter 或真实机器人后训练。",
            "- 若出现 `success=True` 但 `grasp_success=False`，必须同时报告抓取失败，不能只写放置成功。",
            "",
            "## 4. 重建命令",
            "",
            "训练：",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "train_preference_trajectory_post_training.py"}" --run-dir "{metadata.get("run_dir")}" --horizon {metadata.get("horizon")} --history {metadata.get("history")} --sample-stride {metadata.get("sample_stride")} --no-augment-relative',
            "```",
            "",
            "评测：",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "evaluate_preference_trajectory_post_training.py"}" --model "{model_path}" --episodes {args.episodes} --steps {args.steps}',
            "```",
            "",
            "Viewer 查看：",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "run_preference_trajectory_post_training_policy.py"}" --model "{model_path}" --task {args.task} --complexity {args.complexity} --seed 0 --episodes 1 --steps {args.steps} --viewer --duration 60 --speed 0.05 --k 3 --phase-window 0.03 --min-candidates 256 --history-decay 0.25 --preference-power 1.0 --action-alpha 0.85 --max-arm-delta 0.04 --max-gripper-delta 0.0015 --replan-interval 1 --temporal-ensemble --ensemble-decay 0.1 --stop-on-unsafe --log-every 500',
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    model_path = args.model or latest_model()
    metadata = load_metadata(model_path)
    args.version = args.version or str(metadata.get("version", VERSION))
    rows = [
        run_split(args, "train_range", args.train_seed, model_path),
        run_split(args, "heldout", args.heldout_seed, model_path),
    ]
    payload = {
        "version": str(args.version),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": str(model_path),
        "task": args.task,
        "complexity": args.complexity,
        "metadata": metadata,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows, args, model_path, metadata)
    print(f"preference_trajectory_post_training_json: {args.output_json}", flush=True)
    print(f"preference_trajectory_post_training_csv: {args.output_csv}", flush=True)
    print(f"preference_trajectory_post_training_md: {args.output_md}", flush=True)
    print(f"preference_trajectory_post_training_rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
