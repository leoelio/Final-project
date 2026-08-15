from __future__ import annotations

import json
import math
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


# Mandatory editable-text settings from the publication-figure contract.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EVAL = ROOT / "outputs" / "evaluations"
PLATFORM = ROOT / "outputs" / "platform_research"
OUT = ROOT / "outputs" / "publication_figures" / "low_cost_widowx_v1"
MAIN = OUT / "main_figures"
EXTENDED = OUT / "extended_data_figures"
SOURCE = OUT / "source_data"
QA = OUT / "qa"

FINAL_WIDTH_MM = 180
WIDTH = 7.09  # 180 mm
COLORS = {
    "hero": "#0F4D92",
    "hero_light": "#8FB8D8",
    "reference": "#484878",
    "reference_mid": "#7884B4",
    "reference_soft": "#B4C0E4",
    "neutral": "#A8A8A8",
    "neutral_light": "#E2E2E2",
    "dark": "#303030",
    "gain": "#2E9E44",
    "loss": "#C4473A",
    "amber": "#D89025",
    "teal": "#3C8D91",
    "violet": "#8A6DB1",
}

TASK_LABELS = {
    "place_blue_cube_blue_pad": "Blue→blue",
    "place_blue_cube_red_pad": "Blue→red",
    "place_red_cube_red_pad": "Red→red",
    "move_leftmost_cube_to_bowl": "Leftmost→bowl",
    "blue_to_blue": "Blue→blue",
    "blue_to_red": "Blue→red",
    "red_to_red": "Red→red",
    "leftmost_cube": "Leftmost→bowl",
}

METHOD_DISPLAY = {
    "Trajectory-Conditioned Action-Chunk BC / ACT-lite": "Trajectory Action-Chunk BC",
    "Trajectory-Conditioned Action-Chunk BC / ACT-lite v2": "Trajectory Action-Chunk BC v2",
    "Trajectory-kNN Action-Chunk BC": "Trajectory-kNN Action-Chunk",
    "PyTorch State Transformer ACT": "State Transformer ACT",
    "PyTorch State Transformer ACT CUDA": "State ACT (CUDA)",
    "Phase-Conditioned PyTorch State ACT": "Phase-Conditioned ACT",
    "PyTorch State ACT-CVAE-lite": "State ACT-CVAE",
    "Visual-Feature ACT-lite": "Visual-Feature ACT",
    "Visual ACT-CNN-CVAE-lite": "Visual ACT-CNN-CVAE",
    "Object-Language Action Head-lite": "Object-Language Head",
    "Reward-Weighted Action Head-lite": "Reward-Weighted Head",
    "Phase-Conditioned Action Head-lite": "Phase-Conditioned Head",
    "Adapter Action Head-lite": "Adapter Action Head",
    "LoRA-style Action Head-lite": "LoRA-style Action Head",
    "Vision-Language Action Head-lite": "Vision-Language Head",
    "Frozen CLIP Action Head-lite": "Frozen CLIP Action Head",
    "Multi-task Object-Language Action Head-lite": "Multi-task Object-Language Head",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def prepare_dirs() -> None:
    for path in (MAIN, EXTENDED, SOURCE, QA):
        path.mkdir(parents=True, exist_ok=True)


def panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.03) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.7, length=3)


def set_parameter_log_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ticks = [1, 100, 10_000, 1_000_000]
    ax.set_xticks(ticks, ["1", "100", "10k", "1M"])
    ax.tick_params(axis="x", which="minor", labelbottom=False)


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.95)
    svg_path = directory / f"{stem}.svg"
    pdf_path = directory / f"{stem}.pdf"
    png_path = directory / f"{stem}.png"
    tiff_path = directory / f"{stem}.tiff"
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(
        tiff_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    paths = [svg_path, pdf_path, png_path, tiff_path]
    plt.close(fig)
    return paths


def write_source(df: pd.DataFrame, name: str) -> Path:
    path = SOURCE / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ratio(value: object) -> tuple[float, int, int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan, 0, 0
    match = re.search(r"(\d+)\s*/\s*(\d+)", str(value))
    if not match:
        return np.nan, 0, 0
    successes, total = int(match.group(1)), int(match.group(2))
    return (successes / total if total else np.nan), successes, total


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return centre - half, centre + half


def exact_binom_two_sided(a_only: int, b_only: int) -> float:
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def wrapped(labels: list[str], width: int = 22) -> list[str]:
    return ["\n".join(textwrap.wrap(str(label), width=width)) for label in labels]


def task_label(value: str) -> str:
    return TASK_LABELS.get(value, value.replace("_", " "))


def method_label(value: str) -> str:
    return METHOD_DISPLAY.get(value, value)


def figure_01_registered_landscape() -> list[Path]:
    df = pd.read_csv(DOCS / "final_method_version_index.csv")
    records = []
    for _, row in df.iterrows():
        train = ratio(row["主任务训练范围"])[0]
        held = ratio(row["主任务留出范围"])[0]
        language = ratio(row["语言/空间泛化"])[0]
        records.append(
            {
                "version": row["版本"],
                "method": row["方法"],
                "stage": row["阶段"],
                "train_rate": train,
                "heldout_rate": held,
                "language_rate": language,
                "trainable_params": pd.to_numeric(row["可训练参数"], errors="coerce"),
            }
        )
    plot_df = pd.DataFrame(records)
    write_source(plot_df, "fig01_registered_method_landscape.csv")

    fig = plt.figure(figsize=(WIDTH, 8.25))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.6, 1.25], hspace=0.42)
    ax = fig.add_subplot(gs[0])
    matrix = plot_df[["train_rate", "heldout_rate", "language_rate"]].to_numpy(float)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "registered_success", ["#F2F2F2", "#B9CBE1", COLORS["hero"]]
    )
    cmap.set_bad("#FFFFFF")
    im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", vmin=0, vmax=1, cmap=cmap)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = "—" if np.isnan(value) else f"{value * 100:.0f}"
            color = "white" if np.isfinite(value) and value >= 0.62 else COLORS["dark"]
            ax.text(j, i, text, ha="center", va="center", fontsize=5.8, color=color)
    ax.set_xticks(range(3), ["Train range", "Held-out", "Language/space"])
    ax.set_yticks(range(len(plot_df)), [method_label(v) for v in plot_df["method"]])
    ax.tick_params(length=0)
    ax.set_title("Registered outcomes under each method's original formal protocol", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.018)
    cbar.set_label("Success rate")
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["0", "0.5", "1.0"])
    panel_label(ax, "a", x=-0.25)

    ax2 = fig.add_subplot(gs[1])
    valid = plot_df[plot_df["heldout_rate"].notna() & plot_df["trainable_params"].notna()].copy()
    x = valid["trainable_params"].to_numpy(float) + 1
    y = valid["heldout_rate"].to_numpy(float)
    stage_codes = pd.Categorical(valid["stage"])
    stage_palette = plt.cm.Purples(np.linspace(0.30, 0.85, len(stage_codes.categories)))
    colors = [stage_palette[code] for code in stage_codes.codes]
    ax2.scatter(x, y, c=colors, s=28, edgecolor="white", linewidth=0.5, zorder=3)
    for _, row in valid.nlargest(4, "heldout_rate").iterrows():
        ax2.annotate(
            row["method"],
            (row["trainable_params"] + 1, row["heldout_rate"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=5.8,
        )
    set_parameter_log_axis(ax2)
    ax2.set_ylim(-0.04, 1.08)
    ax2.set_yticks([0, 0.5, 1.0])
    ax2.set_xlabel("Trainable parameters + 1 (log scale)")
    ax2.set_ylabel("Registered held-out rate")
    ax2.axhline(0, color=COLORS["neutral_light"], lw=0.8)
    clean_axes(ax2)
    panel_label(ax2, "b")
    return save_figure(fig, MAIN, "fig01_registered_method_search_landscape")


def figure_02_metric_boundary() -> list[Path]:
    df = pd.read_csv(DOCS / "strict_grasp_success_audit.csv")
    numeric = [
        "episodes",
        "loose_successes",
        "strict_grasp_successes",
        "loose_success_rate",
        "strict_grasp_success_rate",
        "mean_target_distance",
        "mean_object_z",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    write_source(df, "fig02_metric_boundary_audit_rows.csv")

    total_n = int(df["episodes"].sum())
    loose = int(df["loose_successes"].sum())
    strict = int(df["strict_grasp_successes"].sum())
    aggregate = pd.DataFrame(
        {
            "metric": ["Loose", "Strict grasp"],
            "successes": [loose, strict],
            "episodes": [total_n, total_n],
            "rate": [loose / total_n, strict / total_n],
        }
    )
    write_source(aggregate, "fig02_metric_boundary_aggregate.csv")

    fig = plt.figure(figsize=(WIDTH, 4.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 0.85, 1.25], height_ratios=[1, 1], hspace=0.42, wspace=0.48)
    ax = fig.add_subplot(gs[:, 0])
    for _, row in df.iterrows():
        color = COLORS["loss"] if row["loose_success_rate"] > row["strict_grasp_success_rate"] else COLORS["neutral"]
        ax.plot([0, 1], [row["loose_success_rate"], row["strict_grasp_success_rate"]], color=color, alpha=0.45, lw=0.9)
        ax.scatter([0, 1], [row["loose_success_rate"], row["strict_grasp_success_rate"]], color=color, s=8, zorder=3)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0, 1], ["Loose", "Strict grasp"])
    ax.set_ylabel("Success rate per audited row")
    clean_axes(ax)
    panel_label(ax, "a")

    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar([0, 1], aggregate["rate"], color=[COLORS["amber"], COLORS["hero"]], width=0.62)
    for bar, succ in zip(bars, [loose, strict]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.035, f"{succ}/{total_n}", ha="center", fontsize=6.5)
    ax2.set_xticks([0, 1], ["Loose", "Strict"])
    ax2.set_ylim(0, 0.40)
    ax2.set_ylabel("Episode-weighted rate")
    clean_axes(ax2)
    panel_label(ax2, "b")

    ax3 = fig.add_subplot(gs[1, 1:])
    sizes = 12 + 10 * df["episodes"].fillna(1).to_numpy(float)
    sc = ax3.scatter(
        df["mean_target_distance"] * 100,
        df["mean_object_z"] * 100,
        c=df["loose_success_rate"],
        cmap="Oranges",
        vmin=0,
        vmax=max(0.01, df["loose_success_rate"].max()),
        s=sizes,
        edgecolor="white",
        linewidth=0.5,
    )
    ax3.set_xlabel("Mean target distance (cm)")
    ax3.set_ylabel("Mean object height (cm)")
    cbar = fig.colorbar(sc, ax=ax3, fraction=0.04, pad=0.02)
    cbar.set_label("Loose rate")
    clean_axes(ax3)
    panel_label(ax3, "c", x=-0.09)
    return save_figure(fig, MAIN, "fig02_loose_vs_strict_metric_boundary")


def figure_03_semantic_adaptation() -> list[Path]:
    budget = pd.read_csv(DOCS / "core_v2_clip_semantic_data_efficiency.csv")
    budget["successes"] = budget["strict_grasp_success"].map(lambda x: ratio(x)[1])
    budget["episodes"] = budget["strict_grasp_success"].map(lambda x: ratio(x)[2])
    budget["mean_target_distance"] = pd.to_numeric(budget["mean_target_distance"], errors="coerce")
    budget_summary = (
        budget.groupby(["demo_budget_per_task", "stored_samples"], as_index=False)
        .agg(successes=("successes", "sum"), episodes=("episodes", "sum"), mean_target_distance=("mean_target_distance", "mean"))
    )
    budget_summary["rate"] = budget_summary["successes"] / budget_summary["episodes"]
    write_source(budget, "fig03_semantic_data_efficiency_rows.csv")

    ood = pd.read_csv(DOCS / "core_v2_clip_semantic_ood_generalization.csv")
    ood["semantic"] = bool_series(ood["semantic_correct"])
    ood["strict"] = bool_series(ood["strict_grasp_success"])
    ood_summary = (
        ood.groupby(["condition", "task_key"], as_index=False)
        .agg(episodes=("seed", "size"), semantic_correct=("semantic", "sum"), strict_success=("strict", "sum"))
    )
    ood_summary["semantic_rate"] = ood_summary["semantic_correct"] / ood_summary["episodes"]
    ood_summary["strict_rate"] = ood_summary["strict_success"] / ood_summary["episodes"]
    write_source(ood, "fig03_semantic_ood_rows.csv")

    heads = pd.read_csv(DOCS / "frozen_clip_semantic_adapter_same_protocol_comparison.csv")
    for col in ["trainable_params", "train_time_seconds", "episodes", "task_successes", "semantic_correct", "strict_grasp_successes"]:
        heads[col] = pd.to_numeric(heads[col], errors="coerce")
    heads["semantic_rate"] = heads["semantic_correct"] / heads["episodes"]
    write_source(heads, "fig03_semantic_head_comparison.csv")

    fig = plt.figure(figsize=(WIDTH, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.45, wspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    x = budget_summary["demo_budget_per_task"].to_numpy(float)
    y = budget_summary["rate"].to_numpy(float)
    lows, highs = zip(*(wilson(int(s), int(n)) for s, n in zip(budget_summary["successes"], budget_summary["episodes"])))
    yerr = np.vstack([y - np.asarray(lows), np.asarray(highs) - y])
    ax.errorbar(x, y, yerr=yerr, color=COLORS["hero"], marker="o", ms=4, lw=1.5, capsize=3)
    for xi, yi, n in zip(x, y, budget_summary["episodes"]):
        ax.text(xi, yi - 0.035, f"{int(n)}/{int(n)}", ha="center", va="top", fontsize=6)
    ax.set_ylim(0.72, 1.03)
    ax.set_xticks(x)
    ax.set_xlabel("Demonstrations per task")
    ax.set_ylabel("Canonical strict success")
    clean_axes(ax)
    panel_label(ax, "a")

    ax2 = fig.add_subplot(gs[:, 1])
    order = [(c, t) for c in ["paraphrase", "hard_distractors"] for t in ["blue_to_blue", "blue_to_red", "red_to_red", "leftmost_cube"]]
    ordered = ood_summary.set_index(["condition", "task_key"]).loc[order].reset_index()
    matrix = ordered[["semantic_rate", "strict_rate"]].to_numpy(float)
    im = ax2.imshow(matrix, vmin=0.45, vmax=1.0, cmap=mcolors.LinearSegmentedColormap.from_list("sem", ["#F4E6DF", "#9EC5DC", COLORS["hero"]]), aspect="auto")
    condition_labels = {"paraphrase": "Paraphrase", "hard_distractors": "Hard"}
    labels = [f"{condition_labels[row.condition]} · {task_label(row.task_key)}\n(n={int(row.episodes)})" for row in ordered.itertuples()]
    ax2.set_yticks(range(len(labels)), labels)
    ax2.set_xticks([0, 1], ["Intent", "Strict task"])
    ax2.tick_params(axis="y", labelsize=5.3, pad=1)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax2.text(j, i, f"{matrix[i, j] * 100:.0f}%", ha="center", va="center", color="white" if matrix[i, j] > 0.78 else COLORS["dark"], fontsize=6.2)
    ax2.tick_params(length=0)
    panel_label(ax2, "b", x=-0.31)

    ax3 = fig.add_subplot(gs[1, 0])
    conditions = ["paraphrase", "hard_distractors"]
    method_order = heads["version"].drop_duplicates().tolist()
    method_labels = ["Linear head", "Bottleneck adapter"]
    method_colors = [COLORS["hero"], COLORS["reference_mid"]]
    for method, label, color in zip(method_order, method_labels, method_colors):
        sub = heads[heads["version"] == method].set_index("condition").loc[conditions]
        ax3.plot([0, 1], sub["semantic_rate"], marker="o", lw=1.4, ms=4, color=color, label=label)
    ax3.set_xticks([0, 1], ["Paraphrase", "Hard distractor"])
    ax3.set_ylim(0.72, 1.02)
    ax3.set_ylabel("Semantic accuracy")
    ax3.legend(loc="lower right")
    clean_axes(ax3)
    panel_label(ax3, "c")
    return save_figure(fig, MAIN, "fig03_low_cost_semantic_adaptation")


def spatial_records() -> tuple[pd.DataFrame, pd.DataFrame]:
    configs = [
        ("Local pointer v1", EVAL / "clip_patch_pointer_core_v2_holdout_v1.json"),
        ("Local pointer v2", EVAL / "clip_patch_pointer_core_v2_v2_holdout.json"),
        ("Scaled frozen pointer", EVAL / "clip_patch_pointer_kaggle_v2_cpu_holdout.json"),
        ("Rank-4 visual LoRA", EVAL / "clip_lora_patch_pointer_kaggle_v1_holdout.json"),
    ]
    summaries = [{"method": "Frozen CLIP action head", "episodes": 20, "successes": 0, "mean_error_cm": np.nan}]
    rows: list[dict] = []
    for label, path in configs:
        data = read_json(path)
        frame = pd.DataFrame(data["rows"])
        frame["method"] = label
        frame["offline_source_error_m"] = pd.to_numeric(frame["offline_source_error_m"], errors="coerce")
        success = bool_series(frame["task_success"])
        summaries.append(
            {
                "method": label,
                "episodes": len(frame),
                "successes": int(success.sum()),
                "mean_error_cm": frame["offline_source_error_m"].mean() * 100,
            }
        )
        rows.extend(frame.to_dict("records"))
    rgb = pd.read_csv(DOCS / "clip_semantic_rgb_feedback_patch_pointer_holdout_v1.csv")
    rgb = rgb[rgb["mode"] == "rgb_open_loop"].copy()
    rgb["method"] = "Calibrated RGB geometry"
    rgb["offline_source_error_m"] = pd.to_numeric(rgb["final_source_position_error_m"], errors="coerce")
    summaries.append(
        {
            "method": "Calibrated RGB geometry",
            "episodes": len(rgb),
            "successes": int(bool_series(rgb["success"]).sum()),
            "mean_error_cm": rgb["offline_source_error_m"].mean() * 100,
        }
    )
    rows.extend(rgb.to_dict("records"))
    summary = pd.DataFrame(summaries)
    summary["success_rate"] = summary["successes"] / summary["episodes"]
    return summary, pd.DataFrame(rows)


def figure_04_spatial_allocation() -> list[Path]:
    summary, rows = spatial_records()
    write_source(summary, "fig04_spatial_method_summary.csv")
    write_source(rows, "fig04_spatial_episode_rows.csv")
    fig = plt.figure(figsize=(WIDTH, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.48, wspace=0.32)
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(summary))
    colors = [COLORS["reference_soft"]] * (len(summary) - 1) + [COLORS["hero"]]
    bars = ax.bar(x, summary["success_rate"], color=colors, edgecolor="white", width=0.72)
    for bar, row in zip(bars, summary.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.035, f"{row.successes}/{row.episodes}", ha="center", fontsize=6.3)
    ax.set_xticks(x, wrapped(summary["method"].tolist(), 18))
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Closed-loop task success")
    clean_axes(ax)
    panel_label(ax, "a")

    ax2 = fig.add_subplot(gs[1, 0])
    loc = summary[summary["mean_error_cm"].notna()].copy()
    y = np.arange(len(loc))
    ax2.barh(y, loc["mean_error_cm"], color=[COLORS["reference_soft"]] * (len(loc) - 1) + [COLORS["hero"]], height=0.62)
    ax2.axvline(3.0, color=COLORS["loss"], ls="--", lw=1, label="Promotion Gate (3 cm)")
    ax2.set_yticks(y, wrapped(loc["method"].tolist(), 17))
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean source error (cm)")
    ax2.legend(loc="lower right")
    clean_axes(ax2)
    panel_label(ax2, "b", x=-0.30)

    ax3 = fig.add_subplot(gs[1, 1])
    method_order = [m for m in summary["method"] if m != "Frozen CLIP action head"]
    for idx, method in enumerate(method_order):
        sub = pd.to_numeric(rows.loc[rows["method"] == method, "offline_source_error_m"], errors="coerce")
        sub = sub[sub.notna()] * 100
        jitter = np.linspace(-0.11, 0.11, len(sub)) if len(sub) > 1 else np.zeros(len(sub))
        color = COLORS["hero"] if method == "Calibrated RGB geometry" else COLORS["reference_mid"]
        ax3.scatter(np.full(len(sub), idx) + jitter, sub, s=10, color=color, alpha=0.65, edgecolor="none")
        ax3.plot([idx - 0.18, idx + 0.18], [sub.mean(), sub.mean()], color=COLORS["dark"], lw=1.2)
    ax3.axhline(3.0, color=COLORS["loss"], ls="--", lw=1)
    ax3.set_xticks(range(len(method_order)), wrapped(method_order, 12), rotation=25, ha="right", rotation_mode="anchor")
    ax3.set_ylabel("Episode source error (cm)")
    clean_axes(ax3)
    panel_label(ax3, "c")
    return save_figure(fig, MAIN, "fig04_spatial_grounding_allocation")


def rgb_overall(path: Path, label: str, cohort: str) -> dict:
    data = read_json(path)
    overall = data.get("overall")
    if overall is None:
        summary = pd.DataFrame(data["summary"])
        summary = summary[summary["mode"] == "rgb_visual_retry"]
        return {
            "version": label,
            "cohort": cohort,
            "episodes": int(summary["episodes"].sum()),
            "semantic": int(summary["semantic_correct"].sum()),
            "selection": int(summary["visual_selection_correct"].sum()),
            "first": int(summary["strict_grasp_success"].sum()),
            "final": int(summary["strict_grasp_success"].sum()),
            "recovery_triggered": int(summary["recovery_triggered"].sum()),
        }
    return {
        "version": label,
        "cohort": cohort,
        "episodes": int(overall["episodes"]),
        "semantic": int(overall["semantic_correct"]),
        "selection": int(overall["visual_selection_correct"]),
        "first": int(overall["first_attempt_success"]),
        "final": int(overall["task_success"]),
        "recovery_triggered": int(overall["recovery_triggered"]),
    }


def figure_05_rgb_development() -> list[Path]:
    records = [
        rgb_overall(EVAL / "rgb_grounding_refinement_v1_fixed20.json", "RGB refinement V1", "fixed-20"),
        rgb_overall(EVAL / "rgb_grounding_refinement_v2_fixed20.json", "RGB refinement V2", "fixed-20"),
        rgb_overall(EVAL / "rgb_object_identity_v2_fixed20.json", "Object identity V2", "fixed-20"),
        rgb_overall(EVAL / "rgb_grounding_refinement_v2_extended_v1.json", "RGB refinement V2", "extended-144"),
        rgb_overall(EVAL / "rgb_object_identity_v2_extended_v1.json", "Object identity V2", "extended-144"),
        rgb_overall(EVAL / "rgb_occlusion_recovery_v3_extended_v1.json", "Occlusion recovery V3", "extended-144"),
        rgb_overall(EVAL / "rgb_table_recovery_v4_extended_v1.json", "Table recovery V4", "extended-144"),
    ]
    df = pd.DataFrame(records)
    for col in ["semantic", "selection", "first", "final"]:
        df[f"{col}_rate"] = df[col] / df["episodes"]
    write_source(df, "fig05_rgb_development_summary.csv")

    fig = plt.figure(figsize=(WIDTH, 4.65))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.2], hspace=0.50, wspace=0.34)
    ax = fig.add_subplot(gs[0, :])
    fixed = df[df["cohort"] == "fixed-20"]
    x = np.arange(len(fixed))
    bars = ax.bar(x, fixed["final_rate"], color=[COLORS["reference_soft"], COLORS["hero_light"], COLORS["hero"]], width=0.64)
    for bar, row in zip(bars, fixed.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{row.final}/{row.episodes}", ha="center", fontsize=6.5)
    ax.set_xticks(x, fixed["version"])
    ax.set_ylim(0.80, 1.04)
    ax.set_ylabel("Strict success")
    ax.set_title("Fixed-20 diagnostic protocol", loc="left")
    clean_axes(ax)
    panel_label(ax, "a")

    ext = df[df["cohort"] == "extended-144"].reset_index(drop=True)
    ax2 = fig.add_subplot(gs[1, 0])
    y = np.arange(len(ext))
    ax2.barh(y + 0.16, ext["final_rate"], height=0.28, color=COLORS["hero"], label="Final")
    ax2.barh(y - 0.16, ext["first_rate"], height=0.28, color=COLORS["reference_soft"], label="First attempt")
    for yi, row in enumerate(ext.itertuples()):
        ax2.text(row.final_rate + 0.004, yi + 0.16, f"{row.final}/{row.episodes}", va="center", fontsize=5.8)
    ax2.set_yticks(y, wrapped(ext["version"].tolist(), 18))
    ax2.invert_yaxis()
    ax2.set_xlim(0.82, 1.01)
    ax2.set_xlabel("Success rate")
    ax2.legend(loc="lower right")
    clean_axes(ax2)
    panel_label(ax2, "b", x=-0.32)

    ax3 = fig.add_subplot(gs[1, 1])
    metrics = ["semantic_rate", "selection_rate", "first_rate", "final_rate"]
    matrix = ext[metrics].to_numpy(float)
    im = ax3.imshow(matrix, vmin=0.85, vmax=1.0, cmap=mcolors.LinearSegmentedColormap.from_list("rgbdev", ["#EFE8F2", "#7CAACB", COLORS["hero"]]), aspect="auto")
    ax3.set_yticks(range(len(ext)), [f"{v}\n(n={n})" for v, n in zip(ext["version"], ext["episodes"])])
    ax3.set_xticks(range(4), ["Intent", "Selection", "First", "Final"], rotation=25, ha="right", rotation_mode="anchor")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax3.text(j, i, f"{matrix[i, j] * 100:.1f}", ha="center", va="center", fontsize=5.8, color="white" if matrix[i, j] > 0.93 else COLORS["dark"])
    ax3.tick_params(length=0)
    panel_label(ax3, "c", x=-0.19)
    return save_figure(fig, MAIN, "fig05_rgb_frontend_development")


def figure_06_v4_repeatability() -> list[Path]:
    data = read_json(EVAL / "v4_independent_replication_v1.json")
    cohorts = pd.DataFrame(data["cohorts"])
    pooled = data["pooled_descriptive"]
    cohort_rows = []
    for row in cohorts.itertuples():
        cohort_rows.append(
            {
                "label": str(row.seed_range),
                "episodes": int(row.episodes),
                "successes": int(row.successes),
                "first": int(row.first),
                "semantic": int(row.semantic),
                "selection": int(row.selection),
                "ci_low": row.wilson95[0],
                "ci_high": row.wilson95[1],
            }
        )
    cohort_rows.append(
        {
            "label": "Pooled descriptive",
            "episodes": int(pooled["episodes"]),
            "successes": int(pooled["successes"]),
            "first": int(pooled["first_attempt_success"]),
            "semantic": int(pooled["semantic_correct"]),
            "selection": int(pooled["visual_selection_correct"]),
            "ci_low": pooled["wilson95"][0],
            "ci_high": pooled["wilson95"][1],
        }
    )
    summary = pd.DataFrame(cohort_rows)
    summary["success_rate"] = summary["successes"] / summary["episodes"]
    summary["first_rate"] = summary["first"] / summary["episodes"]
    write_source(summary, "fig06_v4_cohorts.csv")
    tasks = pd.DataFrame(
        [
            {"task": task_label(k), "episodes": v["episodes"], "successes": v["successes"]}
            for k, v in data["by_task"].items()
        ]
    )
    tasks["rate"] = tasks["successes"] / tasks["episodes"]
    write_source(tasks, "fig06_v4_task_breakdown.csv")

    fig = plt.figure(figsize=(WIDTH, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.22, 1], hspace=0.48, wspace=0.34)
    ax = fig.add_subplot(gs[:, 0])
    y = np.arange(len(summary))[::-1]
    rates = summary["success_rate"].to_numpy(float)
    lows = summary["ci_low"].to_numpy(float)
    highs = summary["ci_high"].to_numpy(float)
    colors = [COLORS["reference_mid"], COLORS["reference_mid"], COLORS["hero"]]
    for yi, rate, lo, hi, color in zip(y, rates, lows, highs, colors):
        ax.plot([lo, hi], [yi, yi], color=color, lw=2)
        ax.scatter(rate, yi, s=35, color=color, zorder=3, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y, [f"{r.label}\n{r.successes}/{r.episodes}" for r in summary.itertuples()])
    ax.set_xlim(0.86, 1.005)
    ax.axvline(pooled["success_rate"], color=COLORS["hero"], lw=0.8, ls=":")
    ax.set_xlabel("Strict success rate (Wilson 95% CI)")
    clean_axes(ax)
    panel_label(ax, "a", x=-0.29)

    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(summary))
    width = 0.34
    ax2.bar(x - width / 2, summary["first_rate"], width, color=COLORS["reference_soft"], label="First attempt")
    ax2.bar(x + width / 2, summary["success_rate"], width, color=[COLORS["reference_mid"], COLORS["reference_mid"], COLORS["hero"]], label="Final")
    ax2.set_xticks(x, ["4000–4011", "10000–10011", "Pooled"], rotation=20, ha="right", rotation_mode="anchor")
    ax2.set_ylim(0.84, 1.01)
    ax2.set_ylabel("Success rate")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=2)
    clean_axes(ax2)
    panel_label(ax2, "b")

    ax3 = fig.add_subplot(gs[1, 1])
    bars = ax3.bar(np.arange(len(tasks)), tasks["rate"], color=[COLORS["hero"], COLORS["hero_light"], COLORS["hero_light"], COLORS["hero_light"]], width=0.68)
    for bar, row in zip(bars, tasks.itertuples()):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004, f"{row.successes}/{row.episodes}", ha="center", fontsize=5.8)
    ax3.set_xticks(range(len(tasks)), wrapped(tasks["task"].tolist(), 10), rotation=18, ha="right", rotation_mode="anchor")
    ax3.set_ylim(0.90, 1.01)
    ax3.set_ylabel("Pooled strict success")
    clean_axes(ax3)
    panel_label(ax3, "c")
    return save_figure(fig, MAIN, "fig06_v4_repeatability")


def figure_07_recovery_pairs() -> list[Path]:
    v3 = read_json(EVAL / "rgb_occlusion_recovery_v3_extended_v1.json")
    v4 = read_json(EVAL / "rgb_table_recovery_v4_extended_v1.json")
    source = read_json(EVAL / "rgb_table_recovery_v4_source_control_v1.json")
    table_rows = pd.DataFrame(v4["rows"])
    source_rows = pd.DataFrame(source["rows"])
    keys = ["domain", "task", "seed"]
    merged = source_rows[keys + ["task_success"]].merge(
        table_rows[keys + ["task_success"]], on=keys, suffixes=("_source", "_table")
    )
    merged["source_success"] = bool_series(merged["task_success_source"])
    merged["table_success"] = bool_series(merged["task_success_table"])
    source_to_table_gain = int((~merged["source_success"] & merged["table_success"]).sum())
    source_to_table_loss = int((merged["source_success"] & ~merged["table_success"]).sum())

    comparisons = []
    for label, data in [("V3 bounded retry", v3), ("V4 table retry", v4)]:
        rows = pd.DataFrame(data["rows"])
        first = bool_series(rows["first_attempt_success"])
        final = bool_series(rows["task_success"])
        gain = int((~first & final).sum())
        loss = int((first & ~final).sum())
        comparisons.append(
            {
                "comparison": label,
                "episodes": len(rows),
                "first_success": int(first.sum()),
                "final_success": int(final.sum()),
                "gains": gain,
                "regressions": loss,
                "p_value": exact_binom_two_sided(gain, loss),
            }
        )
    comparisons.append(
        {
            "comparison": "V4 source→table policy",
            "episodes": len(merged),
            "first_success": int(merged["source_success"].sum()),
            "final_success": int(merged["table_success"].sum()),
            "gains": source_to_table_gain,
            "regressions": source_to_table_loss,
            "p_value": exact_binom_two_sided(source_to_table_gain, source_to_table_loss),
        }
    )
    df = pd.DataFrame(comparisons)
    write_source(df, "fig07_recovery_paired_comparisons.csv")
    write_source(merged, "fig07_v4_source_vs_table_pairs.csv")

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.75), gridspec_kw={"wspace": 0.42})
    ax = axes[0]
    x = np.arange(len(df))
    ax.plot(x, df["first_success"] / df["episodes"], marker="o", color=COLORS["reference_mid"], lw=1.3, label="Before")
    ax.plot(x, df["final_success"] / df["episodes"], marker="o", color=COLORS["hero"], lw=1.5, label="After")
    ax.set_xticks(x, wrapped(df["comparison"].tolist(), 12), rotation=20, ha="right", rotation_mode="anchor")
    ax.set_ylim(0.86, 0.96)
    ax.set_ylabel("Success rate")
    ax.legend(loc="lower right")
    clean_axes(ax)
    panel_label(ax, "a")

    ax2 = axes[1]
    width = 0.34
    ax2.bar(x - width / 2, df["gains"], width, color=COLORS["gain"], label="Gains")
    ax2.bar(x + width / 2, -df["regressions"], width, color=COLORS["loss"], label="Regressions")
    ax2.axhline(0, color=COLORS["dark"], lw=0.7)
    ax2.set_xticks(x, ["V3", "V4", "Source→table"])
    ax2.set_ylabel("Paired outcomes")
    ax2.legend(loc="upper right")
    clean_axes(ax2)
    panel_label(ax2, "b")

    ax3 = axes[2]
    p = df["p_value"].clip(lower=1e-6)
    bars = ax3.bar(x, -np.log10(p), color=[COLORS["reference_mid"], COLORS["hero"], COLORS["reference_soft"]])
    ax3.axhline(-math.log10(0.05), color=COLORS["loss"], ls="--", lw=1, label="Significance threshold (p=0.05)")
    for bar, value in zip(bars, df["p_value"]):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"p={value:.4f}", ha="center", fontsize=5.5)
    ax3.set_xticks(x, ["V3", "V4", "Source→table"], rotation=18, ha="right", rotation_mode="anchor")
    ax3.set_ylabel("−log10(exact p)")
    ax3.legend(loc="upper right")
    clean_axes(ax3)
    panel_label(ax3, "c")
    return save_figure(fig, MAIN, "fig07_bounded_visual_recovery")


def figure_08_contact_rejection() -> list[Path]:
    data = read_json(EVAL / "contact_phase_monitor_heldout_v1.json")
    closure = read_json(EVAL / "final_closure_audit_v1.json")["rejected_candidates"]
    variants = pd.DataFrame(
        [
            {"variant": key, **value}
            for key, value in data["by_variant"].items()
        ]
    )
    variants["success_rate"] = variants["task_success"] / variants["episodes"]
    write_source(variants, "fig08_contact_intervention_variants.csv")
    paired = pd.DataFrame([data["paired_v4_vs_monitor"]])
    write_source(paired, "fig08_contact_intervention_pairs.csv")
    counter = closure["same_state_early_deep_regrasp"]
    offline = closure["contact_monitor_early_regrasp"]

    fig = plt.figure(figsize=(WIDTH, 4.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.72, 1.2, 1], hspace=0.48, wspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    ax.bar([0], [offline["offline_balanced_accuracy"]], color=COLORS["reference_mid"], width=0.55)
    ax.text(0, offline["offline_balanced_accuracy"] + 0.025, f"{offline['offline_balanced_accuracy']:.4f}", ha="center", fontsize=6.5)
    ax.set_xticks([0], ["Offline\nmonitor"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Balanced accuracy")
    clean_axes(ax)
    panel_label(ax, "a", x=-0.30)

    ax2 = fig.add_subplot(gs[0, 1:])
    labels = ["V4 standard", "Fixed deep profile", "Monitor early regrasp"]
    colors = [COLORS["hero"], COLORS["teal"], COLORS["loss"]]
    x = np.arange(len(variants))
    rates = variants["success_rate"].to_numpy(float)
    yerr_low, yerr_high = [], []
    for row in variants.itertuples():
        lo, hi = wilson(int(row.task_success), int(row.episodes))
        yerr_low.append(row.success_rate - lo)
        yerr_high.append(hi - row.success_rate)
    bars = ax2.bar(x, rates, color=colors, width=0.62, yerr=np.vstack([yerr_low, yerr_high]), capsize=3, error_kw={"lw": 0.8})
    for bar, row in zip(bars, variants.itertuples()):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"{row.task_success}/{row.episodes}", ha="center", fontsize=6)
    ax2.set_xticks(x, wrapped(labels, 16))
    ax2.set_ylim(0.80, 1.02)
    ax2.set_ylabel("Held-out strict success")
    clean_axes(ax2)
    panel_label(ax2, "b")

    ax3 = fig.add_subplot(gs[1, 0:2])
    vals = [data["paired_v4_vs_monitor"]["improved"], -data["paired_v4_vs_monitor"]["regressed"]]
    bars = ax3.barh([0, 1], vals, color=[COLORS["gain"], COLORS["loss"]], height=0.55)
    ax3.axvline(0, color=COLORS["dark"], lw=0.8)
    ax3.set_yticks([0, 1], ["Monitor improves", "Monitor regresses"])
    ax3.set_xlabel(f"Paired episodes (exact p={data['paired_v4_vs_monitor']['exact_two_sided_p']:.6f})")
    for bar, value in zip(bars, vals):
        ax3.text(value + (0.3 if value >= 0 else -0.3), bar.get_y() + bar.get_height() / 2, str(abs(value)), va="center", ha="left" if value >= 0 else "right", fontsize=6.5)
    clean_axes(ax3)
    panel_label(ax3, "c", x=-0.18)

    ax4 = fig.add_subplot(gs[1, 2])
    cvals = [counter["continue_better"], counter["early_better"], counter["tie"]]
    cbars = ax4.bar([0, 1, 2], cvals, color=[COLORS["hero"], COLORS["loss"], COLORS["neutral"]], width=0.62)
    for bar, value in zip(cbars, cvals):
        ax4.text(bar.get_x() + bar.get_width() / 2, value + 0.8, str(value), ha="center", fontsize=6.5)
    ax4.set_xticks([0, 1, 2], ["Continue", "Early", "Tie"])
    ax4.set_ylabel("Same-state scenes")
    clean_axes(ax4)
    panel_label(ax4, "d")
    return save_figure(fig, MAIN, "fig08_contact_monitor_rejection")


def platform_runs() -> list[dict]:
    runs = []
    for path in sorted((PLATFORM / "adaptation").glob("*/paired_optimization.json")):
        data = read_json(path)
        if data.get("status") == "completed" and data.get("paired_summary"):
            data["_path"] = str(path.relative_to(ROOT))
            runs.append(data)
    return runs


def figure_09_platform_pilot() -> list[Path]:
    runs = platform_runs()
    preferred = next((r for r in runs if r["run_id"] == "opt-20260813-054052-a97020"), runs[-1])
    primary = pd.DataFrame(preferred["candidate_results"])
    primary_rows = []
    for candidate in preferred["candidate_results"]:
        for row in candidate["evaluation_rows"]:
            primary_rows.append({"method": candidate["label"], **row})
    episode = pd.DataFrame(primary_rows)
    write_source(episode, "fig09_platform_pilot_episode_rows.csv")

    repeats = []
    for run in runs:
        for candidate in run["candidate_results"]:
            repeats.append(
                {
                    "platform_run": run["run_id"],
                    "method": candidate["label"],
                    "successes": candidate["successes"],
                    "episodes": candidate["evaluation_episodes"],
                    "mean_target_error_mm": candidate["mean_target_error"] * 1000,
                    "trainable_params": candidate["trainable_params"],
                    "peak_rss_mb": candidate["peak_rss_mb"],
                    "elapsed_s": candidate["elapsed"],
                    "dataset_fingerprint": candidate["dataset_fingerprint"],
                }
            )
    repeat_df = pd.DataFrame(repeats)
    write_source(repeat_df, "fig09_platform_execution_repeats.csv")

    fig = plt.figure(figsize=(WIDTH, 5.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 0.75, 1], hspace=0.48, wspace=0.43)
    ax = fig.add_subplot(gs[0, 0])
    methods = episode["method"].drop_duplicates().tolist()
    pivot = episode.pivot(index="seed", columns="method", values="target_error") * 1000
    x = [0, 1]
    for seed, row in pivot.iterrows():
        ax.plot(x, [row[methods[0]], row[methods[1]]], color=COLORS["neutral"], lw=0.9, alpha=0.75)
        ax.scatter(x, [row[methods[0]], row[methods[1]]], color=[COLORS["reference_mid"], COLORS["hero"]], s=18, zorder=3)
    ax.set_xticks(x, ["LoRA-style\nresidual", "Registry RGB\nskill"])
    ax.set_ylabel("Target error (mm)")
    clean_axes(ax)
    panel_label(ax, "a")

    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar([0, 1], primary["successes"] / primary["evaluation_episodes"], color=[COLORS["reference_mid"], COLORS["hero"]], width=0.62)
    for bar, row in zip(bars, primary.itertuples()):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03, f"{row.successes}/{row.evaluation_episodes}", ha="center", fontsize=6)
    ax2.set_xticks([0, 1], ["LoRA", "RGB"])
    ax2.set_ylim(0, 1.12)
    ax2.set_ylabel("Success")
    ax2.text(0.5, 0.52, "exact p=0.25", transform=ax2.transAxes, ha="center", fontsize=6.2)
    clean_axes(ax2)
    panel_label(ax2, "b", x=-0.25)

    ax3 = fig.add_subplot(gs[0, 2])
    metrics = ["trainable_params", "peak_rss_mb", "elapsed"]
    display = ["Parameters", "Peak RSS\n(MB)", "Elapsed\n(s)"]
    values = primary[metrics].to_numpy(float).T
    for idx, (metric, label) in enumerate(zip(metrics, display)):
        vmax = max(values[idx].max(), 1)
        ax3.bar(idx - 0.18, values[idx, 0] / vmax, width=0.34, color=COLORS["reference_mid"])
        ax3.bar(idx + 0.18, values[idx, 1] / vmax, width=0.34, color=COLORS["hero"])
        ax3.text(idx - 0.18, values[idx, 0] / vmax + 0.035, f"{values[idx,0]:.1f}", ha="center", fontsize=5.3)
        ax3.text(idx + 0.18, values[idx, 1] / vmax + 0.035, f"{values[idx,1]:.1f}", ha="center", fontsize=5.3)
    ax3.set_xticks(range(3), display)
    ax3.set_ylabel("Within-metric normalised")
    ax3.set_ylim(0, 1.18)
    clean_axes(ax3)
    panel_label(ax3, "c")

    ax4 = fig.add_subplot(gs[1, :])
    method_colors = {methods[0]: COLORS["reference_mid"], methods[1]: COLORS["hero"]}
    for method in methods:
        sub = repeat_df[repeat_df["method"] == method].reset_index(drop=True)
        display = "LoRA-style residual" if "LoRA" in method else "Registry RGB skill"
        ax4.plot(range(1, len(sub) + 1), sub["peak_rss_mb"], marker="o", ms=3.5, lw=1.3, color=method_colors[method], label=display)
    ax4.set_xlabel("Repeated platform execution (same three evaluation seeds)")
    ax4.set_ylabel("Peak RSS (MB)")
    ax4.set_xticks(range(1, len(runs) + 1))
    ax4.legend(ncol=2, loc="upper center")
    clean_axes(ax4)
    panel_label(ax4, "d", x=-0.04)
    return save_figure(fig, MAIN, "fig09_platform_managed_paired_pilot")


def extended_01_data_efficiency() -> list[Path]:
    df = pd.read_csv(DOCS / "data_efficiency_summary.csv")
    df["demo_budget"] = pd.to_numeric(df["demo_budget"], errors="coerce")
    df["success_rate"] = pd.to_numeric(df["success_rate"], errors="coerce")
    df["mean_target_distance"] = pd.to_numeric(df["mean_target_distance"], errors="coerce")
    write_source(df, "edfig01_early_data_efficiency.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8), gridspec_kw={"wspace": 0.34})
    palette = {"knn_bc": COLORS["reference"], "trajectory_knn": COLORS["reference_mid"], "object_action_head": COLORS["violet"]}
    for split, ls in [("train_range", "-"), ("heldout", "--")]:
        for method, color in palette.items():
            sub = df[(df["split"] == split) & (df["method_key"] == method)].sort_values("demo_budget")
            axes[0].plot(sub["demo_budget"], sub["success_rate"], marker="o", ms=3.5, lw=1.2, ls=ls, color=color, label=f"{method.replace('_',' ')} · {split.replace('_',' ')}")
    axes[0].set_xlabel("Demonstrations")
    axes[0].set_ylabel("Success rate")
    axes[0].set_ylim(-0.03, 1.05)
    axes[0].legend(fontsize=5.3, ncol=2, loc="upper center")
    clean_axes(axes[0])
    panel_label(axes[0], "a")
    for method, color in palette.items():
        sub = df[(df["split"] == "heldout") & (df["method_key"] == method)].sort_values("demo_budget")
        axes[1].plot(sub["demo_budget"], sub["mean_target_distance"] * 100, marker="o", ms=3.5, lw=1.2, color=color, label=method.replace("_", " "))
    axes[1].set_xlabel("Demonstrations")
    axes[1].set_ylabel("Held-out target distance (cm)")
    clean_axes(axes[1])
    panel_label(axes[1], "b")
    return save_figure(fig, EXTENDED, "edfig01_early_demonstration_budget_sweep")


def extended_02_language_screen() -> list[Path]:
    df = pd.read_csv(DOCS / "language_generalization_summary.csv")
    df["success_rate"] = pd.to_numeric(df["success_rate"], errors="coerce")
    df["mean_target_distance"] = pd.to_numeric(df["mean_target_distance"], errors="coerce")
    write_source(df, "edfig02_language_generalisation.csv")
    df = df.sort_values(["success_rate", "mean_target_distance"], ascending=[True, False]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(WIDTH, 6.1))
    y = np.arange(len(df))
    colors = [COLORS["hero"] if rate > 0 else COLORS["reference_soft"] for rate in df["success_rate"]]
    ax.barh(y, df["success_rate"], color=colors, height=0.65)
    ax.set_yticks(y, wrapped(df["method_key"].str.replace("_", " ").tolist(), 24))
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Language/spatial task success rate")
    ax.set_title("Early screen; five seeds per registered method", loc="left")
    clean_axes(ax)
    panel_label(ax, "a", x=-0.28)
    return save_figure(fig, EXTENDED, "edfig02_early_language_generalisation_screen")


def extended_03_domain_randomisation() -> list[Path]:
    df = pd.read_csv(DOCS / "domain_randomization_summary.csv")
    df["success_bool"] = bool_series(df["success"])
    df["target_distance"] = pd.to_numeric(df["target_distance"], errors="coerce")
    summary = df.groupby(["method_key", "domain"], as_index=False).agg(episodes=("seed", "size"), successes=("success_bool", "sum"), mean_target_distance=("target_distance", "mean"))
    summary["success_rate"] = summary["successes"] / summary["episodes"]
    write_source(df, "edfig03_domain_randomisation_rows.csv")
    methods = summary["method_key"].drop_duplicates().tolist()
    domains = summary["domain"].drop_duplicates().tolist()
    rate = summary.pivot(index="method_key", columns="domain", values="success_rate").loc[methods, domains]
    dist = summary.pivot(index="method_key", columns="domain", values="mean_target_distance").loc[methods, domains] * 100
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0), gridspec_kw={"wspace": 0.42})
    im = axes[0].imshow(rate, vmin=0, vmax=1, cmap=mcolors.LinearSegmentedColormap.from_list("dom", ["#F3E1DD", "#C8D8E8", COLORS["hero"]]), aspect="auto")
    axes[0].set_xticks(range(len(domains)), wrapped([d.replace("_", " ") for d in domains], 12), rotation=25, ha="right", rotation_mode="anchor")
    axes[0].set_yticks(range(len(methods)), wrapped([m.replace("_", " ") for m in methods], 18))
    for i in range(len(methods)):
        for j in range(len(domains)):
            axes[0].text(j, i, f"{rate.iloc[i,j] * 100:.0f}%\n(n=2)", ha="center", va="center", fontsize=6, color="white" if rate.iloc[i,j] > 0.65 else COLORS["dark"])
    axes[0].tick_params(length=0)
    panel_label(axes[0], "a", x=-0.26)
    im2 = axes[1].imshow(dist, cmap="Purples", aspect="auto")
    axes[1].set_xticks(range(len(domains)), wrapped([d.replace("_", " ") for d in domains], 12), rotation=25, ha="right", rotation_mode="anchor")
    axes[1].set_yticks(range(len(methods)), wrapped([m.replace("_", " ") for m in methods], 18))
    for i in range(len(methods)):
        for j in range(len(domains)):
            axes[1].text(j, i, f"{dist.iloc[i,j]:.1f} cm", ha="center", va="center", fontsize=6, color="white" if dist.iloc[i,j] > dist.to_numpy().mean() else COLORS["dark"])
    axes[1].tick_params(length=0)
    panel_label(axes[1], "b", x=-0.26)
    return save_figure(fig, EXTENDED, "edfig03_mujoco_domain_randomisation_proxy")


def stage_family_figure(source_file: str, out_stem: str, source_name: str) -> list[Path]:
    df = pd.read_csv(DOCS / source_file)
    for target, source in [("train_rate", "主任务训练范围"), ("heldout_rate", "主任务留出范围"), ("language_rate", "语言/空间泛化")]:
        df[target] = df[source].map(lambda x: ratio(x)[0])
    df["params_numeric"] = pd.to_numeric(df["可训练参数"], errors="coerce")
    write_source(df, source_name)
    matrix = df[["train_rate", "heldout_rate", "language_rate"]].to_numpy(float)
    fig = plt.figure(figsize=(WIDTH, max(3.2, 0.42 * len(df) + 1.8)))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.65, 1], wspace=0.40)
    ax = fig.add_subplot(gs[0])
    cmap = mcolors.LinearSegmentedColormap.from_list("family", ["#F2F2F2", "#B4C8DD", COLORS["hero"]])
    cmap.set_bad("white")
    ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax.set_xticks([0, 1, 2], ["Train", "Held-out", "Language"])
    ax.set_yticks(range(len(df)), wrapped(df["方法"].tolist(), 22))
    for i in range(len(df)):
        for j in range(3):
            value = matrix[i, j]
            ax.text(j, i, "—" if np.isnan(value) else f"{value*100:.0f}", ha="center", va="center", fontsize=6, color="white" if np.isfinite(value) and value > 0.62 else COLORS["dark"])
    ax.tick_params(length=0)
    panel_label(ax, "a", x=-0.31)
    ax2 = fig.add_subplot(gs[1])
    params = df["params_numeric"].fillna(0).to_numpy(float)
    y = np.arange(len(df))
    ax2.barh(y, params + 1, color=COLORS["reference_mid"], height=0.62)
    set_parameter_log_axis(ax2)
    ax2.set_yticks(y, wrapped(df["方法"].tolist(), 18))
    ax2.invert_yaxis()
    ax2.set_xlabel("Trainable parameters + 1")
    clean_axes(ax2)
    panel_label(ax2, "b", x=-0.33)
    return save_figure(fig, EXTENDED, out_stem)


def extended_06_contact_diagnostics() -> list[Path]:
    files = [
        "contact_stage_subpolicy_report.csv",
        "contact_stage_phase_action_head_report.csv",
        "contact_stage_demo_torch_act_report.csv",
        "contact_phase_gated_torch_act_report.csv",
        "contact_hold_weighted_torch_act_report.csv",
        "contact_aware_phase_gated_torch_act_report.csv",
        "contact_aware_trajectory_knn_report.csv",
    ]
    summary = []
    for name in files:
        frame = pd.read_csv(DOCS / name)
        label = name.replace("_report.csv", "").replace("_", " ")
        strict = bool_series(frame["strict_grasp_lift_success"]).sum()
        tcp_col = "ever_tcp_lift_success" if "ever_tcp_lift_success" in frame.columns else "tcp_grasp_lift_success"
        tcp = bool_series(frame[tcp_col]).sum()
        summary.append(
            {
                "candidate": label,
                "episodes": len(frame),
                "strict_successes": int(strict),
                "tcp_lift_successes": int(tcp),
                "mean_min_tcp_distance_cm": pd.to_numeric(frame["min_tcp_object_distance"], errors="coerce").mean() * 100,
            }
        )
    summary_df = pd.DataFrame(summary)
    timing = pd.read_csv(DOCS / "gripper_timing_contact_probe_report.csv")
    timing["strict"] = bool_series(timing["strict_grasp_lift_success"])
    timing["tcp"] = bool_series(timing["tcp_grasp_lift_success"])
    timing_summary = timing.groupby("variant", as_index=False).agg(episodes=("seed", "size"), strict=("strict", "sum"), tcp=("tcp", "sum"), mean_max_object_z=("max_object_z", "mean"))
    safety = pd.concat(
        [
            pd.read_csv(DOCS / "control_safety_sweep.csv").assign(family="Trajectory chunk"),
            pd.read_csv(DOCS / "action_head_control_safety_sweep.csv").assign(family="Action head"),
        ],
        ignore_index=True,
    )
    write_source(summary_df, "edfig06_contact_candidate_summary.csv")
    write_source(timing, "edfig06_gripper_timing_rows.csv")
    write_source(safety, "edfig06_control_safety_sweeps.csv")
    fig = plt.figure(figsize=(WIDTH, 5.2))
    gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.38)
    ax = fig.add_subplot(gs[:, 0])
    y = np.arange(len(summary_df))
    ax.barh(y + 0.16, summary_df["tcp_lift_successes"] / summary_df["episodes"], height=0.28, color=COLORS["reference_mid"], label="TCP-lift evidence")
    ax.barh(y - 0.16, summary_df["strict_successes"] / summary_df["episodes"], height=0.28, color=COLORS["hero"], label="Strict grasp-lift")
    ax.set_yticks(y, wrapped(summary_df["candidate"].tolist(), 22))
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Episode rate")
    ax.legend(loc="lower right")
    clean_axes(ax)
    panel_label(ax, "a", x=-0.34)
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(timing_summary))
    ax2.bar(x - 0.16, timing_summary["tcp"] / timing_summary["episodes"], width=0.32, color=COLORS["reference_mid"], label="TCP lift")
    ax2.bar(x + 0.16, timing_summary["strict"] / timing_summary["episodes"], width=0.32, color=COLORS["hero"], label="Strict")
    ax2.set_xticks(x, wrapped(timing_summary["variant"].tolist(), 12), rotation=20, ha="right", rotation_mode="anchor")
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("Gripper-timing probe rate")
    ax2.legend(loc="upper right")
    clean_axes(ax2)
    panel_label(ax2, "b")
    ax3 = fig.add_subplot(gs[1, 1])
    for family, color in [("Trajectory chunk", COLORS["reference_mid"]), ("Action head", COLORS["violet"])]:
        sub = safety[safety["family"] == family]
        ax3.plot(pd.to_numeric(sub["max_arm_delta"], errors="coerce"), pd.to_numeric(sub["mean_target_distance"], errors="coerce") * 100, marker="o", ms=3.5, lw=1.2, color=color, label=family)
    ax3.set_xlabel("Maximum arm delta")
    ax3.set_ylabel("Mean target distance (cm)")
    ax3.legend(loc="upper right")
    clean_axes(ax3)
    panel_label(ax3, "c")
    return save_figure(fig, EXTENDED, "edfig06_contact_stage_candidate_diagnostics")


def extended_07_preference_ablation() -> list[Path]:
    df = pd.read_csv(DOCS / "preference_post_training_ablation_matrix.csv")
    columns = {
        "train_place": "训练范围放置",
        "held_place": "留出范围放置",
        "train_tcp": "训练范围TCP抬升",
        "held_tcp": "留出范围TCP抬升",
        "train_strict": "训练范围严格抓取",
        "held_strict": "留出范围严格抓取",
    }
    for target, source in columns.items():
        df[target] = df[source].map(lambda x: ratio(x)[0])
    write_source(df, "edfig07_preference_post_training_ablation.csv")
    matrix = df[list(columns)].to_numpy(float)
    fig, ax = plt.subplots(figsize=(WIDTH, 3.5))
    cmap = mcolors.LinearSegmentedColormap.from_list("pref", ["#F1E2DF", "#C7D5E6", COLORS["hero"]])
    cmap.set_bad("white")
    ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax.set_yticks(range(len(df)), wrapped(df["版本"].str.replace("_candidate", "").tolist(), 28))
    ax.set_xticks(range(len(columns)), ["Train\nplace", "Held\nplace", "Train\nTCP lift", "Held\nTCP lift", "Train\nstrict", "Held\nstrict"])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, "N/R" if np.isnan(value) else f"{value*100:.0f}%", ha="center", va="center", fontsize=6, color="white" if np.isfinite(value) and value > 0.62 else COLORS["dark"])
    ax.tick_params(length=0)
    panel_label(ax, "a", x=-0.25)
    return save_figure(fig, EXTENDED, "edfig07_preference_post_training_ablation")


def extended_08_rgb_localisation() -> list[Path]:
    records = []
    core = pd.read_csv(DOCS / "rgb_grounding_core_v2_v1.csv")
    for row in core.itertuples():
        records.append({"version": "RGB core calibration", "task": row.task, "error_m": row.position_error_m})
    for label, path in [
        ("Refinement V1", EVAL / "rgb_grounding_refinement_v1_fixed20.json"),
        ("Refinement V2", EVAL / "rgb_grounding_refinement_v2_fixed20.json"),
        ("Object identity V2", EVAL / "rgb_object_identity_v2_fixed20.json"),
        ("Table recovery V4", EVAL / "rgb_table_recovery_v4_extended_v1.json"),
    ]:
        data = read_json(path)
        rows = pd.DataFrame(data["rows"])
        if "mode" in rows.columns:
            rows = rows[rows["mode"] == "rgb_open_loop"]
            error_col = "initial_source_position_error_m" if "initial_source_position_error_m" in rows.columns else "final_source_position_error_m"
        else:
            error_col = "initial_source_position_error_m"
        for row in rows.itertuples():
            value = getattr(row, error_col)
            records.append({"version": label, "task": getattr(row, "task"), "error_m": value})
    df = pd.DataFrame(records)
    df["error_m"] = pd.to_numeric(df["error_m"], errors="coerce")
    df = df[df["error_m"].notna()].copy()
    write_source(df, "edfig08_rgb_localisation_rows.csv")
    versions = df["version"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0), gridspec_kw={"wspace": 0.38})
    for idx, version in enumerate(versions):
        sub = df[df["version"] == version]["error_m"] * 1000
        jitter = np.linspace(-0.10, 0.10, len(sub)) if len(sub) > 1 else np.zeros(len(sub))
        axes[0].scatter(np.full(len(sub), idx) + jitter, sub, s=8, alpha=0.45, color=COLORS["hero"] if version == "Table recovery V4" else COLORS["reference_mid"])
        axes[0].plot([idx - 0.18, idx + 0.18], [sub.median(), sub.median()], color=COLORS["dark"], lw=1.2)
    axes[0].set_xticks(range(len(versions)), wrapped(versions, 14), rotation=25, ha="right", rotation_mode="anchor")
    axes[0].set_ylabel("Source localisation error (mm)")
    clean_axes(axes[0])
    panel_label(axes[0], "a")
    tasks = df["task"].drop_duplicates().tolist()
    data = [df[df["task"] == task]["error_m"].to_numpy(float) * 1000 for task in tasks]
    bp = axes[1].boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["hero_light"])
        patch.set_edgecolor(COLORS["hero"])
    axes[1].set_xticks(range(1, len(tasks) + 1), wrapped([task_label(t) for t in tasks], 10), rotation=20, ha="right", rotation_mode="anchor")
    axes[1].set_ylabel("Error across available RGB stages (mm)")
    clean_axes(axes[1])
    panel_label(axes[1], "b")
    return save_figure(fig, EXTENDED, "edfig08_rgb_localisation_distributions")


def extended_09_strict_audit_rows() -> list[Path]:
    df = pd.read_csv(DOCS / "strict_grasp_success_audit.csv")
    for col in ["episodes", "loose_success_rate", "strict_grasp_success_rate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    write_source(df, "edfig09_full_strict_audit.csv")
    fig, ax = plt.subplots(figsize=(WIDTH, 8.0))
    y = np.arange(len(df))
    ax.barh(y + 0.15, df["loose_success_rate"], height=0.28, color=COLORS["amber"], label="Loose")
    ax.barh(y - 0.15, df["strict_grasp_success_rate"], height=0.28, color=COLORS["hero"], label="Strict grasp")
    labels = [f"{m} · {p} (n={int(n)})" for m, p, n in zip(df["method"], df["preset_or_seed"], df["episodes"])]
    ax.set_yticks(y, wrapped(labels, 42))
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Success rate")
    ax.legend(loc="lower right")
    clean_axes(ax)
    panel_label(ax, "a", x=-0.36)
    return save_figure(fig, EXTENDED, "edfig09_full_strict_grasp_audit")


def extended_10_platform_records() -> list[Path]:
    runs = platform_runs()
    repeat_rows = []
    for run in runs:
        for candidate in run["candidate_results"]:
            repeat_rows.append(
                {
                    "run": run["run_id"],
                    "method": candidate["label"],
                    "peak_rss_mb": candidate["peak_rss_mb"],
                    "elapsed_s": candidate["elapsed"],
                }
            )
    repeats = pd.DataFrame(repeat_rows)
    ledger_events = sum(1 for line in (PLATFORM / "experiment_ledger.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    study_rows = sum(1 for line in (PLATFORM / "study_registry.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    release_rows = sum(1 for line in (PLATFORM / "release_registry.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    records = pd.DataFrame({"record_type": ["Ledger events", "Study rows", "Release rows"], "count": [ledger_events, study_rows, release_rows]})
    write_source(repeats, "edfig10_platform_resource_repeats.csv")
    write_source(records, "edfig10_platform_record_counts.csv")
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.8), gridspec_kw={"wspace": 0.42})
    methods = repeats["method"].drop_duplicates().tolist()
    palette = [COLORS["reference_mid"], COLORS["hero"]]
    for method, color in zip(methods, palette):
        sub = repeats[repeats["method"] == method].reset_index(drop=True)
        axes[0].plot(range(1, len(sub) + 1), sub["peak_rss_mb"], marker="o", ms=3.5, lw=1.2, color=color, label=method)
        axes[1].plot(range(1, len(sub) + 1), sub["elapsed_s"], marker="o", ms=3.5, lw=1.2, color=color, label=method)
    axes[0].set_xlabel("Platform execution")
    axes[0].set_ylabel("Peak RSS (MB)")
    axes[0].legend(fontsize=5.5)
    clean_axes(axes[0])
    panel_label(axes[0], "a")
    axes[1].set_xlabel("Platform execution")
    axes[1].set_ylabel("Elapsed time (s)")
    clean_axes(axes[1])
    panel_label(axes[1], "b")
    bars = axes[2].bar(range(3), records["count"], color=[COLORS["reference_mid"], COLORS["hero_light"], COLORS["hero"]])
    for bar, value in zip(bars, records["count"]):
        axes[2].text(bar.get_x() + bar.get_width() / 2, value + max(records["count"]) * 0.02, str(value), ha="center", fontsize=6)
    axes[2].set_xticks(range(3), ["Ledger\nevents", "Study\nrows", "Release\nrows"])
    axes[2].set_ylabel("Record count")
    clean_axes(axes[2])
    panel_label(axes[2], "c")
    return save_figure(fig, EXTENDED, "edfig10_platform_execution_and_records")


def build_inventory(figure_outputs: list[Path]) -> None:
    mapping_rules = [
        (r"final_method|method_stage|evaluation_summary|task_bc|core_task|core_v2_(holdout|oracle|prior|clip_holdout)", "Fig. 1", "registered/derived method evidence"),
        (r"strict_grasp", "Fig. 2; Extended Data Fig. 9", "strict metric audit"),
        (r"core_v2_clip_semantic|kaggle_clip_semantic|frozen_clip_semantic", "Fig. 3", "semantic adaptation"),
        (r"patch_pointer|rgb_grounding_core", "Fig. 4", "spatial grounding"),
        (r"rgb_grounding_refinement|rgb_object_identity|rgb_occlusion|rgb_table_recovery", "Figs. 5–7; Extended Data Fig. 8", "RGB refinement/recovery"),
        (r"contact_phase_monitor|counterfactual_intervention", "Fig. 8", "contact-intervention audit"),
        (r"data_efficiency", "Extended Data Fig. 1; Fig. 3", "demonstration-budget experiment"),
        (r"language_generalization|independent_syntax|ood_", "Extended Data Fig. 2; Fig. 3", "language robustness"),
        (r"domain_randomization|low_friction_multitask|nominal_multitask", "Extended Data Fig. 3", "domain robustness"),
        (r"trajectory|torch_act|diffusion|phase_weighted", "Extended Data Fig. 4", "temporal-policy family"),
        (r"action_head|adapter|lora|peft", "Extended Data Fig. 5", "action-head/PEFT family"),
        (r"contact|grasp|gripper|control_safety", "Extended Data Fig. 6", "contact/timing/safety diagnostics"),
        (r"preference", "Extended Data Fig. 7", "preference post-training"),
    ]
    non_result = re.compile(r"(handoff|readiness|runbook|playbook|index|manifest|showcase|video|defense|registry|plan|qa|goal_completion|external_dependency|real_widowx|isaac|openvla|remote|intake)", re.I)
    inventory = []
    paths = list(DOCS.glob("*.csv")) + list(EVAL.glob("*.json"))
    for path in sorted(paths):
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        name = path.name.lower()
        figure = ""
        reason = ""
        for pattern, fig_id, why in mapping_rules:
            if re.search(pattern, name, re.I):
                figure, reason = fig_id, why
                break
        if figure:
            classification = "quantitative evidence mapped"
        elif non_result.search(name):
            classification = "metadata, readiness or planned work"
            reason = "not a completed quantitative experiment panel"
        else:
            classification = "derived/supporting evidence"
            reason = "covered through the registered method or diagnostic family summary"
            figure = "Fig. 1 or relevant Extended Data family"
        try:
            if path.suffix == ".csv":
                count = len(pd.read_csv(path))
            else:
                obj = read_json(path)
                count = len(obj.get("rows", [])) if isinstance(obj, dict) else len(obj)
        except Exception:
            count = np.nan
        inventory.append(
            {
                "source_path": rel,
                "record_count": count,
                "classification": classification,
                "figure_mapping": figure,
                "mapping_reason": reason,
            }
        )
    pd.DataFrame(inventory).to_csv(OUT / "data_inventory.csv", index=False, encoding="utf-8-sig")

    manifest = pd.DataFrame(
        [
            {
                "path": str(path.relative_to(OUT)).replace("\\", "/"),
                "format": path.suffix.lstrip("."),
                "bytes": path.stat().st_size,
            }
            for path in figure_outputs
        ]
    )
    manifest.to_csv(OUT / "figure_manifest.csv", index=False, encoding="utf-8-sig")


def build_contact_sheet(png_paths: list[Path]) -> Path:
    from PIL import Image, ImageDraw

    thumbs = []
    for path in png_paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((780, 560))
        canvas = Image.new("RGB", (820, 620), "white")
        canvas.paste(image, ((820 - image.width) // 2, 30))
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 590), path.stem, fill="black")
        thumbs.append(canvas)
    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 820, rows * 620), "white")
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 820, (idx // cols) * 620))
    out = QA / "figure_suite_contact_sheet.png"
    sheet.save(out, dpi=(150, 150))
    return out


def main() -> None:
    prepare_dirs()
    apply_style()
    outputs: list[Path] = []
    outputs += figure_01_registered_landscape()
    outputs += figure_02_metric_boundary()
    outputs += figure_03_semantic_adaptation()
    outputs += figure_04_spatial_allocation()
    outputs += figure_05_rgb_development()
    outputs += figure_06_v4_repeatability()
    outputs += figure_07_recovery_pairs()
    outputs += figure_08_contact_rejection()
    outputs += figure_09_platform_pilot()
    outputs += extended_01_data_efficiency()
    outputs += extended_02_language_screen()
    outputs += extended_03_domain_randomisation()
    outputs += stage_family_figure(
        "trajectory_act_stage_report.csv",
        "edfig04_trajectory_act_diffusion_candidates",
        "edfig04_trajectory_act_diffusion.csv",
    )
    outputs += stage_family_figure(
        "action_head_stage_report.csv",
        "edfig05_action_head_peft_candidates",
        "edfig05_action_head_peft.csv",
    )
    outputs += extended_06_contact_diagnostics()
    outputs += extended_07_preference_ablation()
    outputs += extended_08_rgb_localisation()
    outputs += extended_09_strict_audit_rows()
    outputs += extended_10_platform_records()
    build_inventory(outputs)
    png_paths = [p for p in outputs if p.suffix == ".png"]
    contact_sheet = build_contact_sheet(png_paths)
    print(f"Generated {len(outputs)} figure files across {len(png_paths)} figures.")
    print(f"Contact sheet: {contact_sheet}")


if __name__ == "__main__":
    main()
