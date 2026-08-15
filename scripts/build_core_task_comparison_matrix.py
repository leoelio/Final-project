from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION = "core_task_comparison_matrix_v1"


TASK_SPECS = [
    {
        "task_key": "blue_cube_blue_pad",
        "task": "place_blue_cube_blue_pad",
        "label": "蓝色立方体 -> 蓝色盘",
        "role": "主任务 / 训练相近任务",
        "csv": ROOT / "docs" / "core_task_blue_cube_blue_pad.csv",
        "json": ROOT / "outputs" / "evaluations" / "core_task_blue_cube_blue_pad.json",
    },
    {
        "task_key": "blue_cube_red_pad",
        "task": "place_blue_cube_red_pad",
        "label": "蓝色立方体 -> 红色盘",
        "role": "目标区域迁移",
        "csv": ROOT / "docs" / "core_task_blue_cube_red_pad.csv",
        "json": ROOT / "outputs" / "evaluations" / "core_task_blue_cube_red_pad.json",
    },
    {
        "task_key": "red_cube_red_pad",
        "task": "place_red_cube_red_pad",
        "label": "红色立方体 -> 红色盘",
        "role": "目标物体颜色迁移",
        "csv": ROOT / "docs" / "core_task_red_cube_red_pad.csv",
        "json": ROOT / "outputs" / "evaluations" / "core_task_red_cube_red_pad.json",
    },
    {
        "task_key": "leftmost_to_bowl",
        "task": "move_leftmost_to_bowl",
        "label": "最左物体 -> 碗",
        "role": "空间关系 / 语言任务",
        "csv": ROOT / "docs" / "core_task_leftmost_to_bowl.csv",
        "json": ROOT / "outputs" / "evaluations" / "core_task_leftmost_to_bowl.json",
    },
]

METHOD_ORDER = [
    "expert",
    "structured_waypoint_policy",
    "linear_bc",
    "knn_bc",
    "trajectory_knn",
    "object_action_head",
]

METHOD_LABELS = {
    "expert": "Scripted expert oracle",
    "structured_waypoint_policy": "Structured waypoint",
    "linear_bc": "Linear BC",
    "knn_bc": "kNN BC",
    "trajectory_knn": "Trajectory-kNN",
    "object_action_head": "Object-language action head",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact multi-task comparison matrix for core methods.")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "core_task_comparison_matrix.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "core_task_comparison_matrix.csv")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / "core_task_comparison_matrix_v1.json")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_success(value: str) -> tuple[int, int]:
    left, right = value.split("/", 1)
    return int(left), int(right)


def fmt_rate(success: int, total: int) -> str:
    return f"{success}/{total} ({success / max(1, total):.2f})"


def build_rows() -> tuple[list[dict[str, str]], dict[str, dict], dict[str, dict]]:
    rows: list[dict[str, str]] = []
    method_totals = {method: {"success": 0, "episodes": 0, "distance_sum": 0.0, "distance_count": 0} for method in METHOD_ORDER}
    task_totals = {spec["task_key"]: {"success": 0, "episodes": 0, "distance_sum": 0.0, "distance_count": 0} for spec in TASK_SPECS}

    for spec in TASK_SPECS:
        summary_rows = {row["method_key"]: row for row in read_csv(spec["csv"])}
        detail = read_json(spec["json"])
        episodes_by_method = detail.get("episodes_by_method", {})
        for method in METHOD_ORDER:
            if method not in summary_rows:
                raise RuntimeError(f"{spec['csv']} missing method {method}")
            summary = summary_rows[method]
            success, total = parse_success(summary["success"])
            mean_distance = float(summary["mean_target_distance"])
            failures = []
            for episode in episodes_by_method.get(method, []):
                if not episode.get("success"):
                    failures.append(f"seed{episode.get('seed')} dist={float(episode.get('target_distance', 0.0)):.3f}")
            rows.append(
                {
                    "版本": summary["version"],
                    "方法": METHOD_LABELS[method],
                    "方法key": method,
                    "阶段": summary["stage"],
                    "任务": spec["label"],
                    "任务key": spec["task_key"],
                    "任务定位": spec["role"],
                    "成功": summary["success"],
                    "成功率": f"{float(summary['success_rate']):.3f}",
                    "平均目标距离": f"{mean_distance:.4f}",
                    "seeds": summary["seeds"],
                    "主要失败seed": "；".join(failures[:3]) if failures else "无",
                    "证据CSV": spec["csv"].relative_to(ROOT).as_posix(),
                    "证据JSON": spec["json"].relative_to(ROOT).as_posix(),
                }
            )
            method_totals[method]["success"] += success
            method_totals[method]["episodes"] += total
            method_totals[method]["distance_sum"] += mean_distance
            method_totals[method]["distance_count"] += 1
            task_totals[spec["task_key"]]["success"] += success
            task_totals[spec["task_key"]]["episodes"] += total
            task_totals[spec["task_key"]]["distance_sum"] += mean_distance
            task_totals[spec["task_key"]]["distance_count"] += 1
    return rows, method_totals, task_totals


def md_row(items: list[object]) -> str:
    return "| " + " | ".join(str(item) for item in items) + " |"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(args: argparse.Namespace, rows: list[dict[str, str]], method_totals: dict[str, dict], task_totals: dict[str, dict]) -> None:
    method_summary = []
    for method in METHOD_ORDER:
        item = method_totals[method]
        method_summary.append(
            {
                "方法": METHOD_LABELS[method],
                "方法key": method,
                "总成功": fmt_rate(int(item["success"]), int(item["episodes"])),
                "平均任务距离": f"{item['distance_sum'] / max(1, item['distance_count']):.4f}",
            }
        )

    task_summary = []
    for spec in TASK_SPECS:
        item = task_totals[spec["task_key"]]
        task_summary.append(
            {
                "任务": spec["label"],
                "任务key": spec["task_key"],
                "任务定位": spec["role"],
                "全部方法总成功": fmt_rate(int(item["success"]), int(item["episodes"])),
                "平均方法距离": f"{item['distance_sum'] / max(1, item['distance_count']):.4f}",
            }
        )

    write_csv(args.output_csv, rows)
    task_specs_json = []
    for spec in TASK_SPECS:
        task_specs_json.append(
            {
                "task_key": spec["task_key"],
                "task": spec["task"],
                "label": spec["label"],
                "role": spec["role"],
                "csv": spec["csv"].relative_to(ROOT).as_posix(),
                "json": spec["json"].relative_to(ROOT).as_posix(),
            }
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "version": VERSION,
                "tasks": task_specs_json,
                "methods": METHOD_ORDER,
                "rows": rows,
                "method_summary": method_summary,
                "task_summary": task_summary,
                "policy": {
                    "keep_rule": "主文只保留代表性成功、代表性失败和关键边界；重复 0/3 失败只进附录或候选诊断索引。",
                    "video_rule": "每个阶段最多展示一个成功视频和一个有解释价值的失败视频；量化结论以 CSV/JSON 为主。",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# 核心多任务对比矩阵",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：回应“不能只看蓝色立方体放入一个目标”的问题，把主任务、目标区域迁移、物体颜色迁移和空间关系任务放到同一张表里。该矩阵只保留代表性方法，避免继续堆叠大量重复失败文件。",
        "",
        "## 1. 实验设计",
        "",
        "- 方法选择：oracle、structured waypoint、linear BC、kNN BC、trajectory-kNN、object-language action head。",
        "- 任务选择：蓝方块到蓝盘、蓝方块到红盘、红方块到红盘、最左物体到碗。",
        "- 指标优先级：成功率和平均目标距离优先；视频只作为定性片段，不替代数据表。",
        "- 文件控制：重复失败候选不进入主展示，只进入候选诊断或附录。",
        "",
        "## 2. 方法总览",
        "",
        md_row(["方法", "总成功", "平均任务距离", "主文定位"]),
        md_row(["---", "---:", "---:", "---"]),
    ]
    for item in method_summary:
        note = "可靠上界/环境可行性" if item["方法key"] in {"expert", "structured_waypoint_policy"} else "代表性学习基线"
        lines.append(md_row([item["方法"], item["总成功"], item["平均任务距离"], note]))

    lines.extend(
        [
            "",
            "## 3. 任务难度",
            "",
            md_row(["任务", "任务定位", "全部方法总成功", "平均方法距离", "解释"]),
            md_row(["---", "---", "---:", "---:", "---"]),
        ]
    )
    for item in task_summary:
        if item["任务key"] == "blue_cube_blue_pad":
            explanation = "训练相近任务，能区分 memory/waypoint 与弱 BC。"
        elif item["任务key"] == "red_cube_red_pad":
            explanation = "检验目标物体颜色迁移，oracle/structured 稳定，学习基线失败。"
        elif item["任务key"] == "blue_cube_red_pad":
            explanation = "检验目标区域迁移，当前环境和策略都更不稳定。"
        else:
            explanation = "检验空间关系和语言任务，当前只适合作为泛化压力测试。"
        lines.append(md_row([item["任务"], item["任务定位"], item["全部方法总成功"], item["平均方法距离"], explanation]))

    lines.extend(
        [
            "",
            "## 4. 明细矩阵",
            "",
            md_row(["方法", "任务", "成功", "成功率", "平均目标距离", "主要失败seed"]),
            md_row(["---", "---", "---:", "---:", "---:", "---"]),
        ]
    )
    for row in rows:
        lines.append(md_row([row["方法"], row["任务"], row["成功"], row["成功率"], row["平均目标距离"], row["主要失败seed"]]))

    lines.extend(
        [
            "",
            "## 5. 展示规则修正",
            "",
            "- 主展示不再按“所有模型都放一个视频”的方式堆叠，而是按任务难度和方法阶段组织。",
            "- 每个阶段只保留一个代表成功片段和一个代表失败片段；重复的 0/3 或 0/5 失败只保留在 CSV/JSON 和候选诊断索引。",
            "- 论文结论必须先引用本矩阵、`docs/evaluation_summary.csv`、`docs/language_generalization_summary.csv`、`docs/model_resource_summary.csv`，再播放视频。",
            "- `blue_cube_red_pad` 和 `leftmost_to_bowl` 中 oracle/structured 也不稳定，说明当前 MuJoCo 任务设计本身还需要继续调接触和释放阶段；不能把这些任务写成可靠成功任务。",
            "",
            "完整重建命令：",
            "",
            "```powershell",
            '& "C:\\Users\\Administrator\\Desktop\\final project-robot ic\\vla_robot_grasping\\.venv\\Scripts\\python.exe" "C:\\Users\\Administrator\\Desktop\\final project-robot ic\\vla_robot_grasping\\scripts\\build_core_task_comparison_matrix.py"',
            "```",
        ]
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows, method_totals, task_totals = build_rows()
    write_outputs(args, rows, method_totals, task_totals)
    print(f"core_task_comparison_md: {args.output_md}", flush=True)
    print(f"core_task_comparison_csv: {args.output_csv}", flush=True)
    print(f"core_task_comparison_json: {args.output_json}", flush=True)
    print(f"core_task_comparison_rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
