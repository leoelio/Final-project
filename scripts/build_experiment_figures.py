from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "outputs" / "figures"
DOC_PATH = ROOT / "docs" / "experiment_figures.md"


VERSION_LABELS = {
    "expert_scripted_v1": "Expert",
    "structured_waypoint_policy_v1": "Structured",
    "linear_bc_v1": "Linear BC",
    "knn_bc_v1": "kNN",
    "mlp_bc_v1": "MLP",
    "act_lite_chunk_bc_v1": "ACT-lite",
    "diffusion_policy_lite_v1": "Diffusion-lite",
    "torch_diffusion_policy_state_chunk_v1": "Torch Diffusion",
    "trajectory_conditioned_chunk_bc_v2": "Traj-Chunk",
    "trajectory_knn_chunk_bc_v1": "Traj-kNN",
    "torch_act_state_chunk_v1": "Torch ACT",
    "torch_act_state_chunk_cuda_v1": "Torch ACT CUDA",
    "phase_conditioned_torch_act_v1": "Phase ACT",
    "torch_act_cvae_state_chunk_v1": "ACT-CVAE",
    "visual_feature_act_lite_v1": "Visual ACT-lite",
    "object_language_action_head_lite_v1": "Obj-ActionHead",
    "reward_weighted_action_head_lite_v1": "RWR-ActionHead",
    "phase_conditioned_action_head_lite_v1": "Phase-Head",
    "adapter_action_head_lite_v1": "Adapter-Head",
    "lora_action_head_lite_v1": "LoRA-Head",
    "vision_language_action_head_lite_v1": "Vision-ActionHead",
    "clip_action_head_lite_v1": "CLIP-ActionHead",
    "multi_task_object_action_head_lite_v1": "MultiTask-ActionHead",
}

METHOD_COLORS = {
    "knn_bc": "#2673b8",
    "trajectory_knn": "#d15b2a",
    "object_action_head": "#6a8f32",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SVG figures from experiment CSV summaries.")
    parser.add_argument("--summary", type=Path, default=ROOT / "docs" / "evaluation_summary.csv")
    parser.add_argument("--language-summary", type=Path, default=ROOT / "docs" / "language_generalization_summary.csv")
    parser.add_argument("--resource-summary", type=Path, default=ROOT / "docs" / "model_resource_summary.csv")
    parser.add_argument("--data-efficiency", type=Path, default=ROOT / "docs" / "data_efficiency_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--doc", type=Path, default=DOC_PATH)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def rate(value: str) -> float | None:
    if not value or "/" not in value:
        return None
    left, right = value.split("/", 1)
    try:
        denom = float(right)
        return float(left) / denom if denom else None
    except ValueError:
        return None


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def label(version: str) -> str:
    return VERSION_LABELS.get(version, version.replace("_v1", "").replace("_", " "))


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,'Microsoft YaHei',sans-serif;fill:#24313f} .muted{fill:#637184} .axis{stroke:#8a96a8;stroke-width:1} .grid{stroke:#d8dee8;stroke-width:1} .title{font-size:22px;font-weight:700} .label{font-size:12px} .small{font-size:11px} .legend{font-size:12px}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def write_svg(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines + ["</svg>", ""]), encoding="utf-8")


def axis_y(lines: list[str], x: int, y: int, width: int, height: int) -> None:
    for tick in range(0, 6):
        value = tick / 5
        yy = y + height - value * height
        lines.append(f'<line class="grid" x1="{x}" y1="{yy:.1f}" x2="{x + width}" y2="{yy:.1f}"/>')
        lines.append(f'<text class="small muted" x="{x - 10}" y="{yy + 4:.1f}" text-anchor="end">{int(value * 100)}%</text>')
    lines.append(f'<line class="axis" x1="{x}" y1="{y}" x2="{x}" y2="{y + height}"/>')
    lines.append(f'<line class="axis" x1="{x}" y1="{y + height}" x2="{x + width}" y2="{y + height}"/>')


def build_main_success(rows: list[dict[str, str]], output: Path) -> None:
    data = []
    for row in rows:
        if row["version"] == "replay_demo_v1":
            continue
        train = rate(row["train_range_success"])
        held = rate(row["heldout_success"])
        if train is None or held is None:
            continue
        data.append((row["version"], train, held))

    margin_left = 70
    margin_top = 70
    chart_h = 300
    group_w = 82
    width = margin_left + max(1, len(data)) * group_w + 35
    height = 500
    lines = svg_header(width, height)
    lines.append('<text class="title" x="70" y="34">Main Task Closed-Loop Success</text>')
    lines.append('<text class="legend" x="70" y="56" fill="#2673b8">blue: train-range</text>')
    lines.append('<text class="legend" x="210" y="56" fill="#2f8f5b">green: held-out</text>')
    axis_y(lines, margin_left, margin_top, width - margin_left - 25, chart_h)
    for index, (version, train, held) in enumerate(data):
        x0 = margin_left + index * group_w + 18
        for offset, value, color in ((0, train, "#2673b8"), (24, held, "#2f8f5b")):
            bar_h = value * chart_h
            lines.append(f'<rect x="{x0 + offset}" y="{margin_top + chart_h - bar_h:.1f}" width="18" height="{bar_h:.1f}" fill="{color}"/>')
        lines.append(
            f'<text class="small" transform="translate({x0 + 14},{margin_top + chart_h + 18}) rotate(45)" '
            f'text-anchor="start">{esc(label(version))}</text>'
        )
    write_svg(output, lines)


def build_language_success(rows: list[dict[str, str]], output: Path) -> None:
    data = [(row["version"], float(row["success_rate"])) for row in rows]
    row_h = 32
    margin_left = 230
    margin_top = 65
    chart_w = 520
    width = 820
    height = margin_top + len(data) * row_h + 45
    lines = svg_header(width, height)
    lines.append('<text class="title" x="60" y="34">Language / Spatial Generalization Success</text>')
    for tick in range(0, 6):
        value = tick / 5
        xx = margin_left + value * chart_w
        lines.append(f'<line class="grid" x1="{xx:.1f}" y1="{margin_top - 18}" x2="{xx:.1f}" y2="{height - 38}"/>')
        lines.append(f'<text class="small muted" x="{xx:.1f}" y="{margin_top - 24}" text-anchor="middle">{int(value * 100)}%</text>')
    lines.append(f'<line class="axis" x1="{margin_left}" y1="{height - 38}" x2="{margin_left + chart_w}" y2="{height - 38}"/>')
    for index, (version, value) in enumerate(data):
        yy = margin_top + index * row_h
        color = "#2673b8" if value > 0 else "#a8b1bf"
        if "structured" in version:
            color = "#2f8f5b"
        lines.append(f'<text class="label" x="{margin_left - 12}" y="{yy + 17}" text-anchor="end">{esc(label(version))}</text>')
        lines.append(f'<rect x="{margin_left}" y="{yy}" width="{value * chart_w:.1f}" height="20" fill="{color}"/>')
        lines.append(f'<text class="small" x="{margin_left + value * chart_w + 8:.1f}" y="{yy + 15}">{value:.0%}</text>')
    write_svg(output, lines)


def build_resource_scatter(rows: list[dict[str, str]], output: Path) -> None:
    data = []
    for row in rows:
        if row["version"] == "replay_demo_v1":
            continue
        held = rate(row["heldout_success"])
        if held is None:
            continue
        params = int(float(row["trainable_params"] or 0))
        data.append((row["version"], params, held, row["stage"]))

    max_x = max(math.log10(params + 1) for _, params, _, _ in data) or 1.0
    width, height = 980, 560
    x0, y0, chart_w, chart_h = 90, 75, 740, 340
    lines = svg_header(width, height)
    lines.append('<text class="title" x="70" y="34">Trainable Parameters vs Held-Out Success</text>')
    for tick in range(0, 6):
        value = tick / 5
        yy = y0 + chart_h - value * chart_h
        lines.append(f'<line class="grid" x1="{x0}" y1="{yy:.1f}" x2="{x0 + chart_w}" y2="{yy:.1f}"/>')
        lines.append(f'<text class="small muted" x="{x0 - 10}" y="{yy + 4:.1f}" text-anchor="end">{int(value * 100)}%</text>')
    for tick in range(0, 6):
        value = tick / 5 * max_x
        xx = x0 + value / max_x * chart_w
        lines.append(f'<line class="grid" x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y0 + chart_h}"/>')
        lines.append(f'<text class="small muted" x="{xx:.1f}" y="{y0 + chart_h + 22}" text-anchor="middle">{value:.1f}</text>')
    lines.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + chart_h}"/>')
    lines.append(f'<line class="axis" x1="{x0}" y1="{y0 + chart_h}" x2="{x0 + chart_w}" y2="{y0 + chart_h}"/>')
    lines.append(f'<text class="label muted" x="{x0 + chart_w / 2:.1f}" y="{height - 65}" text-anchor="middle">log10(trainable parameters + 1)</text>')
    lines.append(f'<text class="label muted" transform="translate(26,{y0 + chart_h / 2:.1f}) rotate(-90)" text-anchor="middle">held-out success</text>')
    for index, (version, params, held, stage) in enumerate(data):
        xx = x0 + math.log10(params + 1) / max_x * chart_w
        yy = y0 + chart_h - held * chart_h
        color = "#2f8f5b" if "structured" in stage else "#2673b8"
        if "vla" in stage or "action_head" in stage:
            color = "#8c5fbf"
        if "baseline" in stage and "structured" not in stage:
            color = "#d15b2a"
        lines.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="6" fill="{color}" opacity="0.88"/>')
        dx = 8 if index % 2 == 0 else -8
        anchor = "start" if dx > 0 else "end"
        lines.append(f'<text class="small" x="{xx + dx:.1f}" y="{yy - 8:.1f}" text-anchor="{anchor}">{esc(label(version))}</text>')
    write_svg(output, lines)


def build_data_efficiency(rows: list[dict[str, str]], output: Path) -> None:
    budgets = sorted({int(row["demo_budget"]) for row in rows})
    by_series: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for row in rows:
        key = (row["method_key"], row["split"])
        by_series.setdefault(key, []).append((int(row["demo_budget"]), float(row["success_rate"])))

    width, height = 980, 520
    x0, y0, chart_w, chart_h = 90, 65, 730, 330
    lines = svg_header(width, height)
    lines.append('<text class="title" x="70" y="34">Data Efficiency Sweep</text>')
    axis_y(lines, x0, y0, chart_w, chart_h)
    for budget in budgets:
        xx = x0 + (budget - min(budgets)) / max(1, max(budgets) - min(budgets)) * chart_w
        lines.append(f'<line class="grid" x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y0 + chart_h}"/>')
        lines.append(f'<text class="small muted" x="{xx:.1f}" y="{y0 + chart_h + 22}" text-anchor="middle">{budget}</text>')
    lines.append(f'<text class="label muted" x="{x0 + chart_w / 2:.1f}" y="{height - 70}" text-anchor="middle">successful demonstrations used for fitting</text>')
    legend_x = x0 + chart_w + 30
    legend_y = y0 + 12
    for index, ((method, split), values) in enumerate(sorted(by_series.items())):
        values = sorted(values)
        color = METHOD_COLORS.get(method, "#637184")
        dash = ' stroke-dasharray="6 5"' if split == "heldout" else ""
        points = []
        for budget, value in values:
            xx = x0 + (budget - min(budgets)) / max(1, max(budgets) - min(budgets)) * chart_w
            yy = y0 + chart_h - value * chart_h
            points.append((xx, yy))
        point_text = " ".join(f"{xx:.1f},{yy:.1f}" for xx, yy in points)
        lines.append(f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="3"{dash}/>')
        for xx, yy in points:
            lines.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4" fill="{color}"/>')
        yy = legend_y + index * 20
        lines.append(f'<line x1="{legend_x}" y1="{yy}" x2="{legend_x + 28}" y2="{yy}" stroke="{color}" stroke-width="3"{dash}/>')
        lines.append(f'<text class="small" x="{legend_x + 36}" y="{yy + 4}">{esc(method)} / {esc(split)}</text>')
    write_svg(output, lines)


def write_doc(path: Path, figure_dir: Path) -> None:
    command = f'& "{ROOT / ".venv" / "Scripts" / "python.exe"}" "{ROOT / "scripts" / "build_experiment_figures.py"}"'
    lines = [
        "# 实验图表索引",
        "",
        "版本：`experiment_figures_v1`",
        "",
        "用途：从当前实验 CSV 自动生成论文/答辩可用的静态对比图。所有图表只读现有评测结果，不重新运行 MuJoCo。",
        "",
        "## 图表文件",
        "",
        "| 图表 | 文件 | 用途 |",
        "| --- | --- | --- |",
        "| 主任务成功率 | `outputs/figures/main_task_success.svg` | 对比 train-range 与 held-out 闭环成功率。 |",
        "| 语言泛化成功率 | `outputs/figures/language_success.svg` | 对比 language/spatial generalization。 |",
        "| 参数量与 held-out 成功率 | `outputs/figures/resource_vs_success.svg` | 说明轻量参数并不自动等于闭环泛化能力。 |",
        "| 数据效率曲线 | `outputs/figures/data_efficiency.svg` | 对比 10/25/50/92 条示范下的训练范围与 held-out 表现。 |",
        "",
        "## 重新生成命令",
        "",
        "```powershell",
        command,
        "```",
        "",
        "## 阶段性解读",
        "",
        "1. `Structured Waypoint Policy` 与 `Expert` 成功率高，说明环境和任务链路是可解的。",
        "2. 普通 learned baseline 在 held-out 与 language 任务上普遍失败，说明有限示范下自动学出阶段结构和接触控制仍然困难。",
        "3. `kNN` 和 `Trajectory-kNN` 在训练范围更像轨迹记忆，held-out 明显下降。",
        "4. 当前 action-head 与 vision-language proxy 仍不能代表真实 pretrained VLM/VLA，后续需要接入更强的预训练表征或结构化动作阶段建模。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    build_main_success(read_csv(args.summary), args.output_dir / "main_task_success.svg")
    build_language_success(read_csv(args.language_summary), args.output_dir / "language_success.svg")
    build_resource_scatter(read_csv(args.resource_summary), args.output_dir / "resource_vs_success.svg")
    build_data_efficiency(read_csv(args.data_efficiency), args.output_dir / "data_efficiency.svg")
    write_doc(args.doc, args.output_dir)
    print(f"figure_dir: {args.output_dir}", flush=True)
    print(f"doc_path: {args.doc}", flush=True)
    print("figures: main_task_success.svg, language_success.svg, resource_vs_success.svg, data_efficiency.svg", flush=True)


if __name__ == "__main__":
    main()
