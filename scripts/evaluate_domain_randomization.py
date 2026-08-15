from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
CUDA_PACKAGE_DIR = Path("D:/vla_torch_cuda_pkgs")


DOMAIN_SPECS = {
    "nominal": {
        "description": "训练/常规评测域",
        "arm_kp": 150.0,
        "arm_force": 100.0,
        "gripper_kp": 800.0,
        "gripper_force": 140.0,
        "friction": 3.0,
    },
    "low_friction_soft_grip": {
        "description": "低摩擦、低夹爪力和较软机械臂，用于模拟接触不稳定",
        "arm_kp": 120.0,
        "arm_force": 80.0,
        "gripper_kp": 600.0,
        "gripper_force": 90.0,
        "friction": 1.2,
    },
    "high_friction_stiff_arm": {
        "description": "高摩擦、较硬机械臂和较强夹爪，用于模拟动力学偏差",
        "arm_kp": 180.0,
        "arm_force": 120.0,
        "gripper_kp": 900.0,
        "gripper_force": 160.0,
        "friction": 5.0,
    },
}


METHOD_SPECS = {
    "structured_waypoint_policy": {
        "version": "structured_waypoint_policy_v1",
        "script": "scripts/run_structured_waypoint_policy.py",
        "model": "outputs/structured_waypoint_policy/structured_waypoint_policy_20260720_065456.npz",
        "extra": [],
        "cuda": False,
        "purpose": "结构化强对照，检验任务在扰动域下是否仍可解",
    },
    "trajectory_knn_bc": {
        "version": "trajectory_knn_chunk_bc_v1",
        "script": "scripts/run_trajectory_knn_policy.py",
        "model": "outputs/trajectory_knn_bc/trajectory_knn_chunk_bc_20260720_053423.npz",
        "extra": [
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
            "0.0015",
            "--replan-interval",
            "1",
            "--temporal-ensemble",
            "--ensemble-decay",
            "0.1",
            "--stop-on-unsafe",
            "--log-every",
            "0",
        ],
        "cuda": False,
        "purpose": "轨迹记忆型 baseline，检验训练范围记忆对动力学扰动是否敏感",
    },
    "visual_act_cnn_cvae": {
        "version": "visual_act_cnn_cvae_v1",
        "script": "scripts/run_visual_act_cnn_cvae_policy.py",
        "model": "outputs/visual_act_cnn_cvae/visual_act_cnn_cvae_20260720_115104.pt",
        "extra": [
            "--action-alpha",
            "0.25",
            "--max-arm-delta",
            "0.012",
            "--max-gripper-delta",
            "0.0005",
            "--replan-interval",
            "4",
            "--temporal-ensemble",
            "--ensemble-decay",
            "0.1",
            "--stop-on-unsafe",
            "--log-every",
            "0",
        ],
        "cuda": True,
        "purpose": "小型 CNN 视觉 ACT-CVAE baseline，检验视觉 ACT-lite 在扰动域下的闭环稳定性",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate representative MuJoCo policies under simple domain randomization.")
    parser.add_argument("--methods", default="structured_waypoint_policy,trajectory_knn_bc,visual_act_cnn_cvae")
    parser.add_argument("--domains", default="nominal,low_friction_soft_grip,high_friction_stiff_arm")
    parser.add_argument("--task", default="place_blue_cube_blue_pad")
    parser.add_argument("--complexity", default="medium")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2840)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "domain_randomization_summary.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "domain_randomization_summary.md")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / "domain_randomization_eval_v1.json")
    return parser.parse_args()


def import_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def runtime_capabilities() -> dict[str, Any]:
    torch_available_without_path = import_available("torch")
    torch_available_with_path = False
    torch_version = ""
    cuda_available = False
    sys_path_added = False
    if CUDA_PACKAGE_DIR.exists() and str(CUDA_PACKAGE_DIR) not in sys.path:
        sys.path.insert(0, str(CUDA_PACKAGE_DIR))
        sys_path_added = True
    try:
        import torch

        torch_available_with_path = True
        torch_version = str(torch.__version__)
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        torch_available_with_path = False
    finally:
        if sys_path_added:
            try:
                sys.path.remove(str(CUDA_PACKAGE_DIR))
            except ValueError:
                pass
    return {
        "isaacsim": import_available("isaacsim"),
        "omni": import_available("omni"),
        "isaacgym": import_available("isaacgym"),
        "mujoco": import_available("mujoco"),
        "torch_without_extra_path": torch_available_without_path,
        "torch_with_vla_package_dir": torch_available_with_path,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
    }


def selected_names(value: str, available: dict[str, Any], label: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    missing = [name for name in names if name not in available]
    if missing:
        raise KeyError(f"unknown {label}: {missing}")
    return names


def command_for(method_name: str, domain_name: str, args: argparse.Namespace) -> list[str]:
    method = METHOD_SPECS[method_name]
    domain = DOMAIN_SPECS[domain_name]
    command = [
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
        "--duration",
        "60",
        "--speed",
        "0.05",
        "--arm-kp",
        str(domain["arm_kp"]),
        "--arm-force",
        str(domain["arm_force"]),
        "--gripper-kp",
        str(domain["gripper_kp"]),
        "--gripper-force",
        str(domain["gripper_force"]),
        "--friction",
        str(domain["friction"]),
    ]
    command.extend(method["extra"])
    return command


def parse_episode_summaries(stdout: str) -> list[dict[str, Any]]:
    summaries = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("episode_summary:"):
            continue
        raw = line.split("episode_summary:", 1)[1].strip()
        summaries.append(ast.literal_eval(raw))
    return summaries


def run_case(method_name: str, domain_name: str, args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, Any]]:
    method = METHOD_SPECS[method_name]
    domain = DOMAIN_SPECS[domain_name]
    command = command_for(method_name, domain_name, args)
    env = os.environ.copy()
    if method["cuda"]:
        env["VLA_TORCH_PACKAGE_DIR"] = str(CUDA_PACKAGE_DIR).replace("/", "\\")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=args.timeout_seconds,
    )
    summaries = parse_episode_summaries(completed.stdout)
    rows = []
    for summary in summaries:
        rows.append(
            {
                "version": "domain_randomization_eval_v1",
                "method_key": method_name,
                "method_version": method["version"],
                "domain": domain_name,
                "domain_description": domain["description"],
                "seed": str(summary.get("seed", "")),
                "task": str(summary.get("task", args.task)),
                "complexity": str(summary.get("complexity", args.complexity)),
                "success": str(bool(summary.get("success", False))),
                "target_distance": f"{float(summary.get('target_distance', 0.0)):.6f}",
                "object_z": f"{float(summary.get('object_z', 0.0)):.6f}",
                "grasp_success": str(bool(summary.get("grasp_success", False))),
                "out_of_table": str(bool(summary.get("out_of_table", False))),
                "steps_taken": str(int(summary.get("steps_taken", args.steps))),
                "arm_kp": str(domain["arm_kp"]),
                "arm_force": str(domain["arm_force"]),
                "gripper_kp": str(domain["gripper_kp"]),
                "gripper_force": str(domain["gripper_force"]),
                "friction": str(domain["friction"]),
                "purpose": method["purpose"],
            }
        )
    run_info = {
        "method": method_name,
        "domain": domain_name,
        "returncode": completed.returncode,
        "command": command,
        "stdout_tail": completed.stdout.splitlines()[-20:],
        "stderr_tail": completed.stderr.splitlines()[-20:],
        "parsed_episodes": len(summaries),
    }
    if completed.returncode != 0 or len(summaries) != args.episodes:
        run_info["warning"] = "subprocess returned nonzero or parsed episode count did not match requested episodes"
    return rows, run_info


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["method_key"], row["domain"]), []).append(row)
    summary_rows = []
    for (method_key, domain), items in sorted(grouped.items()):
        successes = sum(1 for item in items if item["success"] == "True")
        distances = [float(item["target_distance"]) for item in items]
        summary_rows.append(
            {
                "method_key": method_key,
                "method_version": items[0]["method_version"],
                "domain": domain,
                "episodes": str(len(items)),
                "success": f"{successes}/{len(items)}",
                "success_rate": f"{successes / max(1, len(items)):.3f}",
                "mean_target_distance": f"{sum(distances) / max(1, len(distances)):.6f}",
                "domain_description": items[0]["domain_description"],
            }
        )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"


def write_md(path: Path, rows: list[dict[str, str]], summary_rows: list[dict[str, str]], capabilities: dict[str, Any], commands: list[dict[str, Any]]) -> None:
    lines = [
        "# MuJoCo Domain Randomization 代理评测",
        "",
        "版本：`domain_randomization_eval_v1`",
        "",
        "用途：在 Isaac/Isaac Sim 尚不可用时，先用 MuJoCo 中的摩擦、执行器增益、执行器力限和夹爪力度扰动，做 sim-to-real gap 的前置鲁棒性评测。该报告不能写成高保真 Isaac domain randomization，也不能写成真实机械臂迁移验证。",
        "",
        "## 1. 本地能力检查",
        "",
        md_row(["能力", "状态"]),
        md_row(["---", "---"]),
    ]
    for key in ("isaacsim", "omni", "isaacgym", "mujoco", "torch_with_vla_package_dir", "cuda_available"):
        lines.append(md_row([key, str(capabilities.get(key))]))

    lines.extend(
        [
            "",
            "## 2. 扰动域",
            "",
            md_row(["域", "摩擦", "arm kp/force", "gripper kp/force", "说明"]),
            md_row(["---", "---:", "---", "---", "---"]),
        ]
    )
    for name, domain in DOMAIN_SPECS.items():
        lines.append(
            md_row(
                [
                    f"`{name}`",
                    domain["friction"],
                    f"{domain['arm_kp']} / {domain['arm_force']}",
                    f"{domain['gripper_kp']} / {domain['gripper_force']}",
                    domain["description"],
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 3. 汇总结果",
            "",
            md_row(["方法", "版本", "域", "成功率", "平均目标距离", "说明"]),
            md_row(["---", "---", "---", "---:", "---:", "---"]),
        ]
    )
    for row in summary_rows:
        lines.append(
            md_row(
                [
                    row["method_key"],
                    f"`{row['method_version']}`",
                    row["domain"],
                    row["success"],
                    row["mean_target_distance"],
                    row["domain_description"],
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 4. 逐 episode 结果",
            "",
            md_row(["方法", "域", "seed", "success", "target_distance", "object_z", "grasp", "out_of_table"]),
            md_row(["---", "---", "---:", "---", "---:", "---:", "---", "---"]),
        ]
    )
    for row in rows:
        lines.append(
            md_row(
                [
                    row["method_key"],
                    row["domain"],
                    row["seed"],
                    row["success"],
                    row["target_distance"],
                    row["object_z"],
                    row["grasp_success"],
                    row["out_of_table"],
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 5. 论文边界",
            "",
            "- 可以写：当前已建立 MuJoCo 层面的 domain randomization 代理评测，用于观察结构化控制、轨迹记忆 baseline 和视觉 ACT-lite 对简单动力学扰动的敏感性。",
            "- 不能写：Isaac domain randomization 已完成。",
            "- 不能写：真实 WidowX 或真实机械臂迁移成功/失败已经验证。",
            "- 后续若安装 Isaac，应保留本报告的字段：域参数、方法版本、success_rate、target_distance、视频证据和论文红线。",
            "",
            "## 6. 视频证据",
            "",
            "```text",
            "outputs/videos/domain_randomization_structured_low_friction_seed0.mp4",
            "outputs/videos/domain_randomization_trajectory_knn_low_friction_seed0.mp4",
            "outputs/videos/domain_randomization_visual_act_cnn_cvae_low_friction_seed0.mp4",
            "outputs/presentation_clips/06_domain_randomization_proxy.mp4",
            "```",
            "",
            "## 7. 重建命令",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "evaluate_domain_randomization.py"}"',
            "```",
            "",
            "## 8. 实际执行命令摘要",
            "",
        ]
    )
    for item in commands:
        lines.extend(
            [
                f"### `{item['method']}` / `{item['domain']}`",
                "",
                "```powershell",
                " ".join(f'"{part}"' if " " in str(part) else str(part) for part in item["command"]),
                "```",
                "",
            ]
        )

    lines.append(f"生成时间：{datetime.now().isoformat(timespec='seconds')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    method_names = selected_names(args.methods, METHOD_SPECS, "method")
    domain_names = selected_names(args.domains, DOMAIN_SPECS, "domain")
    capabilities = runtime_capabilities()
    rows: list[dict[str, str]] = []
    commands: list[dict[str, Any]] = []
    for domain_name in domain_names:
        for method_name in method_names:
            case_rows, run_info = run_case(method_name, domain_name, args)
            rows.extend(case_rows)
            commands.append(run_info)
            print(
                f"case: method={method_name} domain={domain_name} episodes={len(case_rows)} returncode={run_info['returncode']}",
                flush=True,
            )

    if not rows:
        raise RuntimeError("no domain-randomization rows were generated")
    summary_rows = summarize(rows)
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows, summary_rows, capabilities, commands)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "version": "domain_randomization_eval_v1",
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "capabilities": capabilities,
                "domain_specs": DOMAIN_SPECS,
                "method_specs": METHOD_SPECS,
                "rows": rows,
                "summary": summary_rows,
                "runs": commands,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"domain_randomization_csv: {args.output_csv}", flush=True)
    print(f"domain_randomization_md: {args.output_md}", flush=True)
    print(f"domain_randomization_json: {args.output_json}", flush=True)
    print(f"rows: {len(rows)}", flush=True)
    print(f"summary_rows: {len(summary_rows)}", flush=True)


if __name__ == "__main__":
    main()
