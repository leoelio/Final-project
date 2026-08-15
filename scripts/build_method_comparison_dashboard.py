from __future__ import annotations

import argparse
import csv
from datetime import datetime
from html import escape
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_experiment_command_index import viewer_command  # noqa: E402


VERSION = "method_comparison_dashboard_v1"

STAGE_ORDER = [
    "scripted_oracle",
    "data_verification",
    "structured_control_baseline",
    "weak_bc_baseline",
    "non_neural_baseline",
    "neural_bc_baseline",
    "trajectory_conditioned_baseline",
    "trajectory_memory_baseline",
    "torch_act_baseline",
    "torch_act_cvae_baseline",
    "visual_feature_act_baseline",
    "visual_act_cnn_cvae_baseline",
    "diffusion_policy_baseline",
    "torch_diffusion_policy_baseline",
    "vla_action_head_proxy",
    "reward_weighted_bc_post_training",
    "phase_conditioned_action_head_proxy",
    "peft_action_head_proxy",
    "pretrained_vlm_action_head_proxy",
    "multi_task_action_head_proxy",
]

STAGE_LABELS = {
    "scripted_oracle": "脚本专家 / Oracle",
    "data_verification": "数据回放验证",
    "structured_control_baseline": "结构化强对照",
    "weak_bc_baseline": "Linear BC",
    "non_neural_baseline": "kNN BC",
    "neural_bc_baseline": "MLP BC",
    "trajectory_conditioned_baseline": "Trajectory / ACT-lite",
    "trajectory_memory_baseline": "Trajectory-kNN",
    "torch_act_baseline": "PyTorch ACT-style",
    "torch_act_cvae_baseline": "ACT-CVAE-lite",
    "visual_feature_act_baseline": "Visual-Feature ACT-lite",
    "visual_act_cnn_cvae_baseline": "Visual ACT-CNN-CVAE-lite",
    "diffusion_policy_baseline": "Diffusion Policy-lite",
    "torch_diffusion_policy_baseline": "PyTorch Diffusion Policy",
    "vla_action_head_proxy": "VLA Action-head 代理",
    "reward_weighted_bc_post_training": "Reward-weighted BC 代理",
    "phase_conditioned_action_head_proxy": "Phase Action-head",
    "peft_action_head_proxy": "Adapter / LoRA-style 代理",
    "pretrained_vlm_action_head_proxy": "Frozen CLIP / VLM 代理",
    "multi_task_action_head_proxy": "Multi-task Action-head",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese method comparison dashboard for thesis and defense.")
    parser.add_argument("--versions", type=Path, default=ROOT / "docs" / "experiment_versions.json")
    parser.add_argument("--method-gate", type=Path, default=ROOT / "docs" / "method_evidence_gate.csv")
    parser.add_argument("--resources", type=Path, default=ROOT / "docs" / "model_resource_summary.csv")
    parser.add_argument("--output-html", type=Path, default=ROOT / "docs" / "method_comparison_dashboard.html")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "method_comparison_dashboard.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "method_comparison_dashboard.csv")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_versions(path: Path) -> list[dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8-sig"))["methods"]


def by_key(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row[key]: row for row in rows if row.get(key)}


def parse_rate(value: str) -> float | None:
    if not value or value in {"未登记", "不适用", "not_applicable"}:
        return None
    if "/" not in value:
        return None
    left, right = value.split("/", 1)
    try:
        denom = float(right.split()[0].strip("()%"))
        return float(left) / denom if denom else None
    except ValueError:
        return None


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


def result_bucket(train: str, heldout: str, language: str) -> str:
    scores = [score for score in (parse_rate(train), parse_rate(heldout), parse_rate(language)) if score is not None]
    if not scores:
        return "not-applicable"
    if max(scores) >= 0.8:
        return "strong-oracle"
    if max(scores) > 0:
        return "partial"
    return "failed"


def stage_sort_key(stage: str) -> int:
    try:
        return STAGE_ORDER.index(stage)
    except ValueError:
        return len(STAGE_ORDER)


def num_text(value: str) -> str:
    if value in {"", None}:  # type: ignore[comparison-overlap]
        return "-"
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return str(value)


def web_path(path_text: str) -> str:
    path = Path(path_text.replace("\\", "/"))
    if path.is_absolute():
        try:
            path = path.relative_to(ROOT)
        except ValueError:
            return path.as_posix()
    return (Path("..") / path).as_posix()


def build_rows(versions: list[dict[str, str]], gate_rows: list[dict[str, str]], resource_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    gate = by_key(gate_rows, "版本")
    resources = by_key(resource_rows, "version")
    rows: list[dict[str, str]] = []
    for method in versions:
        version = method["version"]
        gate_row = gate.get(version, {})
        resource = resources.get(version, {})
        train = gate_row.get("主任务训练范围", method.get("train_range_success", ""))
        heldout = gate_row.get("主任务留出范围", method.get("heldout_success", ""))
        language = gate_row.get("语言/空间泛化", "未登记")
        command = viewer_command(method, task="place_blue_cube_blue_pad", complexity="medium", seed=0)
        row = {
            "version": version,
            "stage": method["stage"],
            "stage_label": STAGE_LABELS.get(method["stage"], method["stage"]),
            "method": method["method"],
            "artifact": method["artifact"],
            "train": train,
            "heldout": heldout,
            "language": language,
            "train_rate": pct(parse_rate(train)),
            "heldout_rate": pct(parse_rate(heldout)),
            "language_rate": pct(parse_rate(language)),
            "result_bucket": result_bucket(train, heldout, language),
            "trainable_params": gate_row.get("可训练参数", resource.get("trainable_params", "")),
            "artifact_size_mb": gate_row.get("模型大小MB", resource.get("artifact_size_mb", "")),
            "train_time_seconds": resource.get("train_time_seconds", ""),
            "peak_vram_mb": resource.get("peak_vram_mb", ""),
            "fixed_video": gate_row.get("固定视频", method.get("clip", "")),
            "note": method.get("note", ""),
            "paper_redline": gate_row.get("论文红线", method.get("note", "")),
            "viewer_command": command,
        }
        rows.append(row)
    return sorted(rows, key=lambda row: (stage_sort_key(row["stage"]), row["version"]))


def verify_rows(rows: list[dict[str, str]]) -> None:
    missing = []
    for row in rows:
        for key in ("artifact", "fixed_video"):
            path = ROOT / row[key]
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("\n".join(missing))


def bucket_label(bucket: str) -> str:
    return {
        "strong-oracle": "强对照 / Oracle",
        "partial": "部分成功",
        "failed": "闭环失败",
        "not-applicable": "不适用",
    }.get(bucket, bucket)


def table_row(row: dict[str, str]) -> str:
    search = " ".join(row[key] for key in ("version", "stage_label", "method", "note", "paper_redline")).lower()
    video = escape(web_path(row["fixed_video"]))
    artifact = escape(web_path(row["artifact"]))
    return f"""
<tr data-stage="{escape(row['stage'])}" data-bucket="{escape(row['result_bucket'])}" data-search="{escape(search)}">
  <td><code>{escape(row['version'])}</code></td>
  <td>{escape(row['stage_label'])}</td>
  <td>{escape(row['method'])}</td>
  <td>{escape(row['train'])}<span>{escape(row['train_rate'])}</span></td>
  <td>{escape(row['heldout'])}<span>{escape(row['heldout_rate'])}</span></td>
  <td>{escape(row['language'])}<span>{escape(row['language_rate'])}</span></td>
  <td class="num">{escape(num_text(row['trainable_params']))}</td>
  <td class="num">{escape(row['train_time_seconds'] or '-')}</td>
  <td class="num">{escape(row['peak_vram_mb'] or '-')}</td>
  <td><a href="{video}">视频</a><br><a href="{artifact}">artifact</a></td>
  <td><details><summary>说明</summary><p>{escape(row['note'])}</p><p><b>红线：</b>{escape(row['paper_redline'])}</p><pre>{escape(row['viewer_command'])}</pre></details></td>
</tr>
""".strip()


def build_html(rows: list[dict[str, str]]) -> str:
    stages = []
    for row in rows:
        if row["stage"] not in stages:
            stages.append(row["stage"])
    best_train = max((parse_rate(row["train"]) or 0.0 for row in rows), default=0.0)
    peft_rows = [row for row in rows if row["stage"] == "peft_action_head_proxy"]
    min_peft_params = min((int(float(row["trainable_params"])) for row in peft_rows if row["trainable_params"]), default=0)
    stage_options = "\n".join(
        f'<button type="button" data-filter="stage" data-value="{escape(stage)}">{escape(STAGE_LABELS.get(stage, stage))}</button>'
        for stage in stages
    )
    body_rows = "\n".join(table_row(row) for row in rows)
    generated_at = datetime.now().isoformat(timespec="seconds")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>方法评测比较看板</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fa;
      --panel: #fff;
      --text: #111827;
      --muted: #667085;
      --line: #d8dee8;
      --accent: #0f766e;
      --warn: #9a3412;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    header {{
      padding: 28px clamp(18px, 4vw, 56px) 18px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0 0 8px; font-size: clamp(26px, 3vw, 42px); letter-spacing: 0; }}
    .subtitle {{ max-width: 1120px; margin: 0; color: var(--muted); line-height: 1.7; }}
    .stats {{ display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 10px; max-width: 960px; margin-top: 18px; }}
    .stat {{ border: 1px solid var(--line); border-radius: 6px; background: #f9fafb; padding: 12px; }}
    .stat b {{ display: block; color: var(--accent); font-size: 23px; }}
    main {{ padding: 22px clamp(18px, 4vw, 56px) 44px; }}
    .toolbar {{ display: grid; gap: 12px; max-width: 1440px; margin: 0 auto 16px; }}
    input {{ width: 100%; min-height: 42px; border: 1px solid var(--line); border-radius: 6px; padding: 0 12px; font: inherit; }}
    .segmented {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    button {{ border: 1px solid var(--line); border-radius: 6px; background: #fff; padding: 8px 10px; cursor: pointer; }}
    button.active {{ border-color: var(--accent); color: var(--accent); font-weight: 700; }}
    .table-wrap {{ max-width: 1440px; margin: 0 auto; overflow: auto; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 1220px; }}
    th, td {{ padding: 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #f9fafb; z-index: 1; font-size: 13px; }}
    td span {{ display: block; color: var(--muted); font-size: 12px; margin-top: 3px; }}
    td.num {{ font-variant-numeric: tabular-nums; }}
    code, pre {{ font-family: Consolas, "Cascadia Mono", monospace; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #f2f4f7; border: 1px solid var(--line); border-radius: 6px; padding: 8px; font-size: 12px; }}
    summary {{ color: var(--warn); cursor: pointer; font-weight: 700; }}
    a {{ color: #175cd3; }}
    footer {{ padding: 0 clamp(18px, 4vw, 56px) 30px; color: var(--muted); }}
    @media (max-width: 800px) {{ .stats {{ grid-template-columns: repeat(2, minmax(140px, 1fr)); }} }}
  </style>
</head>
<body>
  <header>
    <h1>方法评测比较看板</h1>
    <p class="subtitle">版本：<code>{VERSION}</code>。本页把 25 个正式 MuJoCo 方法版本的阶段、成功率、语言/空间泛化、参数规模、训练资源、固定视频和慢速 viewer 命令放在同一个中文入口；不新增实验结果，所有结论仍以原始 CSV/JSON 和视频元数据为准。</p>
    <div class="stats">
      <div class="stat"><b>{len(rows)}</b><span>正式方法</span></div>
      <div class="stat"><b>{len(stages)}</b><span>阶段分组</span></div>
      <div class="stat"><b>{best_train:.0%}</b><span>最高 train-range</span></div>
      <div class="stat"><b>{min_peft_params:,}</b><span>最小 PEFT 参数</span></div>
    </div>
  </header>
  <main>
    <section class="toolbar">
      <input id="search" type="search" placeholder="搜索版本、阶段、方法、结论或论文红线">
      <div class="segmented" id="stageFilters">
        <button class="active" type="button" data-filter="stage" data-value="all">全部阶段</button>
        {stage_options}
      </div>
      <div class="segmented" id="bucketFilters">
        <button class="active" type="button" data-filter="bucket" data-value="all">全部结果</button>
        <button type="button" data-filter="bucket" data-value="strong-oracle">强对照 / Oracle</button>
        <button type="button" data-filter="bucket" data-value="partial">部分成功</button>
        <button type="button" data-filter="bucket" data-value="failed">闭环失败</button>
        <button type="button" data-filter="bucket" data-value="not-applicable">不适用</button>
      </div>
    </section>
    <section class="table-wrap">
      <table>
        <thead><tr><th>版本</th><th>阶段</th><th>方法</th><th>Train</th><th>Held-out</th><th>Language</th><th>参数</th><th>训练秒</th><th>显存 MB</th><th>证据</th><th>说明 / viewer</th></tr></thead>
        <tbody>{body_rows}</tbody>
      </table>
    </section>
  </main>
  <footer>生成时间：{escape(generated_at)}</footer>
  <script>
    const state = {{ stage: "all", bucket: "all", query: "" }};
    const rows = Array.from(document.querySelectorAll("tbody tr"));
    function applyFilters() {{
      const q = state.query.trim().toLowerCase();
      rows.forEach(row => {{
        const okStage = state.stage === "all" || row.dataset.stage === state.stage;
        const okBucket = state.bucket === "all" || row.dataset.bucket === state.bucket;
        const okQuery = !q || row.dataset.search.includes(q);
        row.style.display = okStage && okBucket && okQuery ? "" : "none";
      }});
    }}
    document.querySelectorAll("button[data-filter]").forEach(button => {{
      button.addEventListener("click", () => {{
        const filter = button.dataset.filter;
        state[filter] = button.dataset.value;
        button.parentElement.querySelectorAll("button").forEach(item => item.classList.remove("active"));
        button.classList.add("active");
        applyFilters();
      }});
    }});
    document.getElementById("search").addEventListener("input", event => {{
      state.query = event.target.value;
      applyFilters();
    }});
  </script>
</body>
</html>
"""


def build_md(rows: list[dict[str, str]]) -> str:
    lines = [
        "# 方法评测比较看板",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：把当前 25 个正式 MuJoCo 方法版本的阶段、成功率、语言/空间泛化、资源规模、固定视频和慢速 viewer 命令集中成中文比较入口。本文件不新增实验结果。",
        "",
        "## 打开 HTML 看板",
        "",
        "```powershell",
        f'Start-Process "{ROOT / "docs" / "method_comparison_dashboard.html"}"',
        "```",
        "",
        "## 重建命令",
        "",
        "```powershell",
        f'& "{ROOT / ".venv" / "Scripts" / "python.exe"}" "{ROOT / "scripts" / "build_method_comparison_dashboard.py"}"',
        "```",
        "",
        "## 总表",
        "",
        "| 版本 | 阶段 | Train | Held-out | Language | 参数 | 视频 | 论文红线 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['version']}`",
                    row["stage_label"],
                    row["train"],
                    row["heldout"],
                    row["language"],
                    num_text(row["trainable_params"]),
                    f"`{row['fixed_video']}`",
                    row["paper_redline"],
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## 论文口径",
        "",
        "- 该看板服务横向比较和答辩展示，不替代 `docs/evaluation_summary.csv`、`docs/language_generalization_summary.csv`、`docs/model_resource_summary.csv` 和视频元数据。",
        "- 当前完成的是 MuJoCo 实验包；真实 OpenVLA、Isaac 和真实 WidowX 验证仍必须保持未完成/后续阶段口径。",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
    ]
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    columns = [
        "version",
        "stage",
        "stage_label",
        "method",
        "artifact",
        "train",
        "heldout",
        "language",
        "train_rate",
        "heldout_rate",
        "language_rate",
        "result_bucket",
        "trainable_params",
        "artifact_size_mb",
        "train_time_seconds",
        "peak_vram_mb",
        "fixed_video",
        "note",
        "paper_redline",
        "viewer_command",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    methods = read_versions(args.versions)
    rows = build_rows(methods, read_csv(args.method_gate), read_csv(args.resources))
    verify_rows(rows)
    args.output_html.write_text(build_html(rows), encoding="utf-8")
    args.output_md.write_text(build_md(rows), encoding="utf-8")
    write_csv(args.output_csv, rows)
    print(f"method_comparison_dashboard_html: {args.output_html}", flush=True)
    print(f"method_comparison_dashboard_md: {args.output_md}", flush=True)
    print(f"method_comparison_dashboard_csv: {args.output_csv}", flush=True)
    print(f"method_comparison_rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
