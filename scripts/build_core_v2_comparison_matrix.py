from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION = "core_v2_holdout_comparison_matrix_v1"

TASKS = [
    ("blue_to_blue", "蓝方块 -> 蓝盘", "主任务留出", "core_v2_holdout_blue_cube_blue_pad"),
    ("blue_to_red", "蓝方块 -> 红盘", "目标区域迁移", "core_v2_holdout_blue_cube_red_pad"),
    ("red_to_red", "红方块 -> 红盘", "目标物体迁移", "core_v2_holdout_red_cube_red_pad"),
    ("leftmost_cube", "最左方块 -> 碗", "空间关系/语言", "core_v2_holdout_leftmost_cube_to_bowl"),
]

METHODS = [
    ("expert", "Scripted expert oracle", "环境可行性上界"),
    ("structured_waypoint_policy", "Structured waypoint", "结构化控制上界"),
    ("linear_bc", "Linear BC", "普通模仿学习基线"),
    ("knn_bc", "kNN BC", "记忆检索基线"),
    ("trajectory_knn", "Trajectory-kNN", "轨迹条件化基线"),
    ("object_action_head", "Object-language action head", "多任务 action-head proxy"),
    ("trajectory_prior_residual", "Trajectory-prior residual", "结构化先验诊断候选"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_success(value: str) -> tuple[int, int]:
    left, right = value.split("/", 1)
    return int(left), int(right)


def md_row(values: list[object]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"


def source_prefix(method: str, base: str) -> str:
    return base if method != "trajectory_prior_residual" else base.replace("core_v2_holdout_", "core_v2_prior_holdout_")


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    labels = {key: (label, role) for key, label, role in METHODS}
    for task_key, task_label, task_role, base in TASKS:
        for method_key, _, _ in METHODS:
            source = source_prefix(method_key, base)
            csv_path = ROOT / "docs" / f"{source}.csv"
            json_path = ROOT / "outputs" / "evaluations" / f"{source}.json"
            summary = {item["method_key"]: item for item in read_csv(csv_path)}[method_key]
            details = read_json(json_path)["episodes_by_method"][method_key]
            failures = [
                f"seed{item['seed']} dist={float(item['target_distance']):.3f}"
                for item in details
                if not item["success"]
            ]
            rows.append(
                {
                    "任务key": task_key,
                    "任务": task_label,
                    "任务定位": task_role,
                    "方法key": method_key,
                    "方法": labels[method_key][0],
                    "方法定位": labels[method_key][1],
                    "版本": summary["version"],
                    "成功": summary["success"],
                    "成功率": summary["success_rate"],
                    "平均目标距离": summary["mean_target_distance"],
                    "主要失败": "；".join(failures) if failures else "无",
                    "证据CSV": csv_path.relative_to(ROOT).as_posix(),
                    "证据JSON": json_path.relative_to(ROOT).as_posix(),
                }
            )
    return rows


def main() -> None:
    rows = build_rows()
    method_summary = []
    for method_key, label, role in METHODS:
        selected = [row for row in rows if row["方法key"] == method_key]
        wins, episodes = zip(*(parse_success(row["成功"]) for row in selected))
        method_summary.append(
            {
                "方法": label,
                "方法key": method_key,
                "方法定位": role,
                "总成功": f"{sum(wins)}/{sum(episodes)}",
                "平均目标距离": f"{sum(float(row['平均目标距离']) for row in selected) / len(selected):.4f}",
            }
        )

    task_summary = []
    for task_key, label, role, _ in TASKS:
        selected = [row for row in rows if row["任务key"] == task_key]
        wins, episodes = zip(*(parse_success(row["成功"]) for row in selected))
        task_summary.append(
            {
                "任务": label,
                "任务key": task_key,
                "任务定位": role,
                "总成功": f"{sum(wins)}/{sum(episodes)}",
            }
        )

    csv_path = ROOT / "docs" / "core_v2_holdout_comparison_matrix.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "version": VERSION,
        "protocol": {
            "workspace_profile": "core_v2",
            "workspace": "x=0.23-0.45 m, y=-0.10-0.10 m, minimum distance=0.085 m",
            "gripper": "kp=1200, force=200, friction=5.0",
            "place_tcp_z": 0.041,
            "split": "每项任务 episode 0-19 训练，episode 20-24 留出；空间任务训练子集排除 seed 405 的单次失败。",
        },
        "rows": rows,
        "method_summary": method_summary,
        "task_summary": task_summary,
    }
    json_path = ROOT / "outputs" / "evaluations" / f"{VERSION}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Core V2 留出集对比矩阵",
        "",
        f"版本：`{VERSION}`",
        "",
        "## 协议",
        "",
        "- 物理协议：`core_v2` 可达工作区，抓夹 `kp=1200/force=200`、接触摩擦 `5.0`、放置 TCP 高度 `0.041 m`。",
        "- 任务：蓝方块到蓝盘、蓝方块到红盘、红方块到红盘、最左方块到碗。空间任务只在四种方块中选择目标，绿球是干扰物。",
        "- 数据划分：每项任务前 20 个 episode 训练，最后 5 个 episode 留出。空间训练集保留 19 条成功单次示范，并排除 seed 405 的失败轨迹。",
        "- 历史 `core_task_comparison_matrix_v1` 使用旧工作区和错误的 leftmost 目标选择，只作历史诊断，不再作为正式结论。",
        "",
        "## 方法汇总",
        "",
        md_row(["方法", "定位", "总落点成功（宽松）", "平均目标距离"]),
        md_row(["---", "---", "---:", "---:"]),
    ]
    lines.extend(md_row([item["方法"], item["方法定位"], item["总成功"], item["平均目标距离"]]) for item in method_summary)
    lines.extend(["", "## 逐任务结果", "", md_row(["任务", "方法", "落点成功（宽松）", "平均目标距离", "主要失败"]), md_row(["---", "---", "---:", "---:", "---"])])
    lines.extend(md_row([row["任务"], row["方法"], row["成功"], f"{float(row['平均目标距离']):.4f}", row["主要失败"]]) for row in rows)
    lines.extend([
        "",
        "## 可写结论",
        "",
        "- 本矩阵的成功列沿用历史终态落点口径：物体最终接近目标区域，并不单独证明抓取过程。它适合比较普通 BC 与检索基线的闭环落点失败。",
        "- 在可行的统一物理协议下，环境/结构化控制的宽松落点上界为 20/20；普通单任务 BC 和检索式 BC 在真正留出集上均为 0/20。",
        "- 多任务 object-language action head 的离线误差较低但闭环为 0/20，表明仅做单帧动作回归不足以处理接触阶段的分布漂移。",
        "- Trajectory-prior residual 为 20/20，但训练出的残差近似为零；成功主要来自显式规划先验，不能作为 VLA、纯 BC 或 LoRA 成功结果。",
        "- 严格抓放需同时验证抬升和 TCP 持续接近：`docs/core_v2_clip_semantic_waypoint_report.md` 对语义-结构化方法给出独立的 20/20 严格抓放证据；其他学习基线的严格口径以 `docs/strict_grasp_success_audit.md` 为准。",
        "",
        "## 不可写结论",
        "",
        "- 不能把 object-language action head 称为真实 VLM/VLA 后训练成功。",
        "- 不能把 trajectory-prior residual 称为端到端学习策略或真实机械臂结果。",
        "",
        "重建命令：",
        "",
        "```powershell",
        f'& "{ROOT / ".venv" / "Scripts" / "python.exe"}" "{ROOT / "scripts" / "build_core_v2_comparison_matrix.py"}"',
        "```",
    ])
    md_path = ROOT / "docs" / "core_v2_holdout_comparison_matrix.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"matrix_md: {md_path}", flush=True)
    print(f"matrix_csv: {csv_path}", flush=True)
    print(f"matrix_json: {json_path}", flush=True)
    print(f"matrix_rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
