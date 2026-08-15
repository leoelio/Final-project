from __future__ import annotations

import argparse
import csv
from datetime import datetime
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


STAGE_LABELS = {
    "scripted_oracle": "脚本专家 / Oracle",
    "data_verification": "数据回放验证",
    "structured_control_baseline": "结构化强对照",
    "weak_bc_baseline": "Linear BC",
    "non_neural_baseline": "kNN BC",
    "neural_bc_baseline": "MLP BC",
    "trajectory_conditioned_baseline": "Trajectory-conditioned BC / ACT-lite",
    "trajectory_memory_baseline": "Trajectory-kNN 轨迹记忆",
    "torch_act_baseline": "PyTorch ACT-style",
    "torch_act_cvae_baseline": "ACT-CVAE-lite",
    "visual_feature_act_baseline": "Visual-Feature ACT-lite",
    "visual_act_cnn_cvae_baseline": "Visual ACT-CNN-CVAE-lite",
    "diffusion_policy_baseline": "Diffusion Policy-lite",
    "torch_diffusion_policy_baseline": "PyTorch Diffusion Policy-lite",
    "vla_action_head_proxy": "VLA Action-head 代理",
    "reward_weighted_bc_post_training": "Reward-weighted BC 代理",
    "phase_conditioned_action_head_proxy": "Phase-conditioned Action-head",
    "peft_action_head_proxy": "Adapter / LoRA-style 代理",
    "pretrained_vlm_action_head_proxy": "Frozen CLIP / VLM 代理",
    "multi_task_action_head_proxy": "Multi-task Action-head",
    "language_oracle": "语言/空间 Oracle",
    "extra_video_evidence": "补充成功/失败片段",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an interactive local HTML gallery for video evidence.")
    parser.add_argument("--video-evidence", type=Path, default=ROOT / "docs" / "video_evidence_index.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "video_evidence_gallery.html")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def web_path(path_text: str) -> str:
    path = Path(path_text.replace("\\", "/"))
    if path.is_absolute():
        try:
            path = path.relative_to(ROOT)
        except ValueError:
            return path.as_posix()
    return ("../" / path).as_posix()


def slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in value.lower()).strip("-")


def result_class(result: str) -> str:
    if "success=True" in result:
        return "success"
    if "success=False" in result:
        return "failure"
    if "replay" in result:
        return "replay"
    return "neutral"


def ordered_unique(rows: list[dict[str, str]], key: str) -> list[str]:
    seen: list[str] = []
    for row in rows:
        value = row[key]
        if value not in seen:
            seen.append(value)
    return seen


def button_group(title: str, attr: str, values: list[str], labels: dict[str, str] | None = None) -> list[str]:
    labels = labels or {}
    lines = [f'<section class="filter-block"><h2>{escape(title)}</h2><div class="segmented" data-filter="{escape(attr)}">']
    lines.append('<button class="active" data-value="all" type="button">全部</button>')
    for value in values:
        lines.append(
            f'<button data-value="{escape(value)}" type="button">{escape(labels.get(value, value))}</button>'
        )
    lines.append("</div></section>")
    return lines


def metric_item(label: str, value: str) -> str:
    return f"<span><b>{escape(label)}</b>{escape(value or '-')}</span>"


def build_card(row: dict[str, str]) -> str:
    searchable = " ".join(
        row.get(key, "")
        for key in ("版本", "方法", "阶段", "任务", "复杂度", "指令", "活动物体", "证据用途", "论文红线")
    )
    result = row["结果"]
    stage_label = STAGE_LABELS.get(row["阶段"], row["阶段"])
    return f"""
<article class="video-card" data-type="{escape(row['视频类型'])}" data-stage="{escape(row['阶段'])}" data-result="{result_class(result)}" data-search="{escape(searchable.lower())}">
  <div class="video-shell">
    <video controls preload="metadata" src="{escape(web_path(row['视频文件']))}"></video>
  </div>
  <div class="card-body">
    <div class="card-meta">
      <span class="pill">{escape(row['视频类型'])}</span>
      <span class="pill {result_class(result)}">{escape(result)}</span>
    </div>
    <h3>{escape(row['版本'])}</h3>
    <p class="method">{escape(row['方法'])}</p>
    <p class="stage">{escape(stage_label)}</p>
    <div class="metrics">
      {metric_item('seed', row['seed'])}
      {metric_item('时长', row['时长'])}
      {metric_item('目标距离', row['目标距离'])}
      {metric_item('物体高度', row['物体高度'])}
    </div>
    <details>
      <summary>说明与边界</summary>
      <p><b>任务：</b>{escape(row['任务'])} / {escape(row['复杂度'])}</p>
      <p><b>指令：</b>{escape(row['指令'])}</p>
      <p><b>活动物体：</b>{escape(row['活动物体'])}</p>
      <p><b>证据用途：</b>{escape(row['证据用途'])}</p>
      <p><b>论文红线：</b>{escape(row['论文红线'])}</p>
      <p><a href="{escape(web_path(row['视频文件']))}">打开视频文件</a> · <a href="{escape(web_path(row['元数据文件']))}">查看 JSON 元数据</a></p>
    </details>
  </div>
</article>""".strip()


def build_html(rows: list[dict[str, str]]) -> str:
    types = ordered_unique(rows, "视频类型")
    stages = ordered_unique(rows, "阶段")
    result_values = ["success", "failure", "replay", "neutral"]
    result_labels = {"success": "成功", "failure": "失败", "replay": "回放", "neutral": "其他"}
    success_count = sum(1 for row in rows if result_class(row["结果"]) == "success")
    failure_count = sum(1 for row in rows if result_class(row["结果"]) == "failure")
    replay_count = sum(1 for row in rows if result_class(row["结果"]) == "replay")
    cards = "\n".join(build_card(row) for row in rows)
    stage_labels = {stage: STAGE_LABELS.get(stage, stage) for stage in stages}
    filter_lines = []
    filter_lines += button_group("视频类型", "type", types)
    filter_lines += button_group("阶段", "stage", stages, stage_labels)
    filter_lines += button_group("结果", "result", result_values, result_labels)
    filters = "\n".join(filter_lines)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>视频证据浏览页</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #15191f;
      --muted: #667085;
      --line: #d8dee8;
      --accent: #0f766e;
      --accent-2: #7c3aed;
      --success: #067647;
      --failure: #b42318;
      --replay: #175cd3;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      line-height: 1.5;
    }}
    header {{
      padding: 22px 28px 14px;
      border-bottom: 1px solid var(--line);
      background: #fff;
      position: sticky;
      top: 0;
      z-index: 10;
    }}
    .title-row {{
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 20px;
      flex-wrap: wrap;
    }}
    h1 {{
      margin: 0;
      font-size: 24px;
      letter-spacing: 0;
    }}
    .subtitle {{
      margin: 6px 0 0;
      color: var(--muted);
      max-width: 960px;
      font-size: 14px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(90px, 1fr));
      gap: 8px;
      min-width: 420px;
    }}
    .stat {{
      border: 1px solid var(--line);
      background: #fbfcfe;
      padding: 8px 10px;
      border-radius: 8px;
    }}
    .stat b {{ display: block; font-size: 18px; }}
    .stat span {{ color: var(--muted); font-size: 12px; }}
    main {{ padding: 18px 28px 32px; }}
    .controls {{
      display: grid;
      grid-template-columns: minmax(220px, 320px) 1fr;
      gap: 14px;
      align-items: start;
      margin-bottom: 18px;
    }}
    .search-panel, .filter-block {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .search-panel label, .filter-block h2 {{
      display: block;
      margin: 0 0 8px;
      font-size: 13px;
      font-weight: 700;
    }}
    input[type="search"] {{
      width: 100%;
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0 10px;
      font-size: 14px;
    }}
    .filter-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }}
    .segmented {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }}
    button {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      border-radius: 7px;
      padding: 6px 9px;
      cursor: pointer;
      font-size: 12px;
    }}
    button.active {{
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }}
    .result-line {{
      margin: 10px 0 16px;
      color: var(--muted);
      font-size: 13px;
    }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
      gap: 14px;
    }}
    .video-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      min-width: 0;
    }}
    .video-shell {{
      width: 100%;
      aspect-ratio: 4 / 3;
      background: #0b0f14;
    }}
    video {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }}
    .card-body {{ padding: 12px; }}
    .card-meta {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      margin-bottom: 8px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 3px 7px;
      border-radius: 999px;
      background: #eef2f6;
      color: #344054;
      font-size: 12px;
    }}
    .pill.success {{ background: #dcfae6; color: var(--success); }}
    .pill.failure {{ background: #fee4e2; color: var(--failure); }}
    .pill.replay {{ background: #dbeafe; color: var(--replay); }}
    h3 {{
      margin: 0;
      font-size: 15px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }}
    .method, .stage {{
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px;
      margin: 10px 0;
    }}
    .metrics span {{
      background: #f8fafc;
      border: 1px solid #edf1f7;
      border-radius: 7px;
      padding: 6px;
      min-width: 0;
      font-size: 12px;
      color: var(--muted);
    }}
    .metrics b {{
      display: block;
      color: var(--text);
      font-size: 11px;
      margin-bottom: 2px;
    }}
    details {{
      border-top: 1px solid var(--line);
      padding-top: 8px;
      font-size: 13px;
    }}
    summary {{ cursor: pointer; font-weight: 700; }}
    details p {{ margin: 7px 0; }}
    a {{ color: var(--accent-2); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .hidden {{ display: none !important; }}
    @media (max-width: 900px) {{
      header {{ position: static; padding: 18px; }}
      main {{ padding: 14px 18px 24px; }}
      .stats {{ min-width: 0; width: 100%; grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .controls {{ grid-template-columns: 1fr; }}
      .filter-grid {{ grid-template-columns: 1fr; }}
      .gallery {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="title-row">
      <div>
        <h1>视频证据浏览页</h1>
        <p class="subtitle">按视频类型、阶段和结果筛选当前 MuJoCo 实验视频。该页面用于论文和答辩检索片段，量化结论仍以 CSV 评测表和阶段报告为准。</p>
      </div>
      <div class="stats" aria-label="统计">
        <div class="stat"><b>{len(rows)}</b><span>视频证据</span></div>
        <div class="stat"><b>{success_count}</b><span>成功片段</span></div>
        <div class="stat"><b>{failure_count}</b><span>失败片段</span></div>
        <div class="stat"><b>{replay_count}</b><span>回放片段</span></div>
      </div>
    </div>
  </header>
  <main>
    <section class="controls" aria-label="筛选">
      <div class="search-panel">
        <label for="search">搜索版本、方法、任务或边界说明</label>
        <input id="search" type="search" placeholder="例如 ACT、domain、CLIP、失败、leftmost">
      </div>
      <div class="filter-grid">
        {filters}
      </div>
    </section>
    <p class="result-line"><span id="visibleCount">{len(rows)}</span> / {len(rows)} 条视频正在显示。生成时间：{escape(datetime.now().isoformat(timespec='seconds'))}</p>
    <section class="gallery" id="gallery">
      {cards}
    </section>
  </main>
  <script>
    const filters = {{ type: 'all', stage: 'all', result: 'all', search: '' }};
    const cards = Array.from(document.querySelectorAll('.video-card'));
    const visibleCount = document.getElementById('visibleCount');
    function applyFilters() {{
      let count = 0;
      for (const card of cards) {{
        const matchesType = filters.type === 'all' || card.dataset.type === filters.type;
        const matchesStage = filters.stage === 'all' || card.dataset.stage === filters.stage;
        const matchesResult = filters.result === 'all' || card.dataset.result === filters.result;
        const matchesSearch = !filters.search || card.dataset.search.includes(filters.search);
        const visible = matchesType && matchesStage && matchesResult && matchesSearch;
        card.classList.toggle('hidden', !visible);
        if (visible) count += 1;
      }}
      visibleCount.textContent = String(count);
    }}
    document.querySelectorAll('.segmented').forEach(group => {{
      group.addEventListener('click', event => {{
        const button = event.target.closest('button');
        if (!button) return;
        group.querySelectorAll('button').forEach(item => item.classList.remove('active'));
        button.classList.add('active');
        filters[group.dataset.filter] = button.dataset.value;
        applyFilters();
      }});
    }});
    document.getElementById('search').addEventListener('input', event => {{
      filters.search = event.target.value.trim().toLowerCase();
      applyFilters();
    }});
  </script>
</body>
</html>"""


def main() -> None:
    args = parse_args()
    rows = read_rows(args.video_evidence)
    if not rows:
        raise RuntimeError("video evidence CSV is empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(rows), encoding="utf-8")
    print(f"video_evidence_gallery: {args.output}", flush=True)
    print(f"video_cards: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
