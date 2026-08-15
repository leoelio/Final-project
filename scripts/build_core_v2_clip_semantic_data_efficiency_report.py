from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION = "core_v2_clip_semantic_data_efficiency_v1"
TASK_LABELS = {
    "blue_to_blue": "蓝方块 -> 蓝盘",
    "blue_to_red": "蓝方块 -> 红盘",
    "red_to_red": "红方块 -> 红盘",
    "leftmost_cube": "最左方块 -> 碗",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Core V2 semantic-waypoint data-efficiency report.")
    parser.add_argument("--input-csv", type=Path, default=ROOT / "docs" / "core_v2_clip_semantic_data_efficiency.csv")
    parser.add_argument("--input-json", type=Path, default=ROOT / "outputs" / "evaluations" / f"{VERSION}.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "core_v2_clip_semantic_data_efficiency.md")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def parse_ratio(value: str) -> tuple[int, int]:
    left, right = value.split("/", 1)
    return int(left), int(right)


def md_row(values: list[object]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input_csv)
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    if payload.get("version") != VERSION:
        raise RuntimeError("unexpected evaluation version")

    budgets = sorted({int(row["demo_budget_per_task"]) for row in rows})
    if budgets != [5, 10, 20] or len(rows) != 12:
        raise RuntimeError("expected four tasks for each of the 5/10/20 budgets")

    aggregate: list[dict[str, object]] = []
    for budget in budgets:
        selected = [row for row in rows if int(row["demo_budget_per_task"]) == budget]
        successes, episodes = zip(*(parse_ratio(row["success"]) for row in selected))
        semantics, _ = zip(*(parse_ratio(row["semantic_correct"]) for row in selected))
        strict_grasps, _ = zip(*(parse_ratio(row["strict_grasp_success"]) for row in selected))
        aggregate.append(
            {
                "budget": budget,
                "samples": int(selected[0]["stored_samples"]),
                "success": f"{sum(successes)}/{sum(episodes)}",
                "semantic": f"{sum(semantics)}/{sum(episodes)}",
                "strict_grasp": f"{sum(strict_grasps)}/{sum(episodes)}",
                "distance": sum(float(row["mean_target_distance"]) for row in selected) / len(selected),
            }
        )

    model_paths = {int(row["demo_budget_per_task"]): row["model"].replace("\\", "/") for row in rows}
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    run_dirs = " ".join(
        f'"{ROOT / "data" / "demos" / name}"'
        for name in (
            "core_v2_place_blue_cube_blue_pad_medium_train20_v1",
            "core_v2_place_blue_cube_red_pad_medium_train20_v1",
            "core_v2_place_red_cube_red_pad_medium_train20_v1",
            "core_v2_move_leftmost_cube_to_bowl_language_train20_v1",
        )
    )
    lines = [
        "# Core V2 CLIP 语义-结构化执行数据效率报告",
        "",
        f"版本：`{VERSION}`",
        "",
        "## 实验设计",
        "",
        "- 方法固定为冻结 `openai/clip-vit-base-patch32` 的四类任务意图 adapter，加 MuJoCo 场景状态目标解析和 scripted waypoint expert。",
        "- 预算 `5/10/20` 指的是**每类任务**保留的示范上限；四类任务合计分别为 `20/40/79` 条样本。20 条预算时，空间任务原始成功示范只有 19 条。",
        "- 三个预算均在同一批从未参与训练的 seed 上评测：蓝到蓝 `20-24`、蓝到红 `120-124`、红到红 `220-224`、最左方块到碗 `420-424`。",
        "- 严格抓取定义：物体相对初始高度至少抬升 `0.06 m`，且在 TCP `0.06 m` 内累计至少 `50` 个仿真步；任务成功还要求最终目标距离小于 `0.065 m`。",
        "",
        "## 汇总结果",
        "",
        md_row(["每任务示范数", "总样本", "严格抓放成功", "语义正确", "严格抓取", "平均目标距离 (m)"]),
        md_row(["---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    lines.extend(
        md_row([
            item["budget"],
            item["samples"],
            item["success"],
            item["semantic"],
            item["strict_grasp"],
            f'{float(item["distance"]):.4f}',
        ])
        for item in aggregate
    )
    lines.extend([
        "",
        "完整的 12 行逐任务数值在 `docs/core_v2_clip_semantic_data_efficiency.csv`，逐 episode 记录在 `outputs/evaluations/core_v2_clip_semantic_data_efficiency_v1.json`。",
        "",
        "## 解释与边界",
        "",
        "- 本封闭四意图任务上，5 条每类示范已达到饱和，增加至 10 或 20 条没有改变这批固定留出 seed 的结果。这是一个应保留的阴性发现，不能把三条相同的成功视频重复包装成不同结论。",
        "- 该结果只能说明：在明确语言模板、固定物体集合和固定结构化执行器下，冻结 CLIP 的小型语义 adapter 对示范数量不敏感。它**不能**证明端到端 VLA 控制也只需要 5 条示范。",
        "- 对照 `core_v2_pretrained_vlm_action_head_v1`：同样冻结 CLIP、使用 79 条样本的连续动作头在四任务留出集为 `0/20`。因此这里的收益来自语义决策与接触控制解耦，不是连续动作回归被证明有效。",
        "",
        "## 复现命令",
        "",
        "```powershell",
        "$env:VLA_TORCH_PACKAGE_DIR=\"D:\\vla_torch_cuda_pkgs\"",
        f'& "{python}" "{ROOT / "scripts" / "train_clip_semantic_waypoint.py"}" --run-dirs {run_dirs} --output "{ROOT / "outputs" / "clip_semantic_waypoint"}" --model-prefix clip_semantic_waypoint_core_v2_5eps_v1 --workspace-profile core_v2 --epochs 200 --batch-size 32 --lr 0.02 --weight-decay 0.0001 --seed 0 --max-episodes-per-run 5',
        f'& "{python}" "{ROOT / "scripts" / "train_clip_semantic_waypoint.py"}" --run-dirs {run_dirs} --output "{ROOT / "outputs" / "clip_semantic_waypoint"}" --model-prefix clip_semantic_waypoint_core_v2_10eps_v1 --workspace-profile core_v2 --epochs 200 --batch-size 32 --lr 0.02 --weight-decay 0.0001 --seed 0 --max-episodes-per-run 10',
        "```",
        "",
        "```powershell",
        f'& "{python}" "{ROOT / "scripts" / "evaluate_clip_semantic_waypoint_data_efficiency.py"}" --model "5={ROOT / model_paths[5]}" --model "10={ROOT / model_paths[10]}" --model "20={ROOT / model_paths[20]}" --episodes 5 --workspace-profile core_v2 --image-size 224 --camera top_rgb --arm-kp 150 --arm-force 100 --gripper-kp 1200 --gripper-force 200 --friction 5 --place-tcp-z 0.041',
        f'& "{python}" "{ROOT / "scripts" / "build_core_v2_clip_semantic_data_efficiency_report.py"}"',
        "```",
        "",
        "## 交互式 viewer 命令",
        "",
        "```powershell",
        "$env:VLA_TORCH_PACKAGE_DIR=\"D:\\vla_torch_cuda_pkgs\"",
        f'& "{python}" "{ROOT / "scripts" / "run_clip_semantic_waypoint.py"}" --model "{ROOT / model_paths[5]}" --task move_leftmost_cube_to_bowl --complexity language --workspace-profile core_v2 --seed 420 --episodes 1 --viewer --duration 45 --speed 0.25 --image-size 224 --camera top_rgb --arm-kp 150 --arm-force 100 --gripper-kp 1200 --gripper-force 200 --friction 5 --place-tcp-z 0.041',
        "```",
    ])
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report_md: {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
