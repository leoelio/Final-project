from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VERSION = "openvla_bridge_gallery_v1"
DEFAULT_SAMPLES = ROOT / "data" / "vla_bridge" / "openvla_dataset_bridge_v1" / "samples.jsonl"
DEFAULT_MANIFEST = ROOT / "data" / "vla_bridge" / "openvla_dataset_bridge_v1" / "manifest.json"
DEFAULT_FEASIBILITY = ROOT / "outputs" / "evaluations" / "openvla_feasibility_check_v1.json"
DEFAULT_OUTPUT = ROOT / "docs" / "openvla_bridge_gallery.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a visual HTML gallery for OpenVLA bridge samples.")
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--feasibility", type=Path, default=DEFAULT_FEASIBILITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def docs_rel(path_text: str) -> str:
    path = ROOT / path_text
    try:
        return "../" + path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def action_summary(action: list[float]) -> str:
    return ", ".join(f"{float(value):+.3f}" for value in action[:7])


def object_xy(sample: dict[str, Any]) -> str:
    target = sample.get("target_object")
    objects = sample.get("state", {}).get("objects", {})
    if not target or target not in objects:
        return "未记录"
    xyz = objects[target]
    return f"x={float(xyz[0]):.3f}, y={float(xyz[1]):.3f}, z={float(xyz[2]):.3f}"


def write_html(samples: list[dict[str, Any]], manifest: dict[str, Any], feasibility: dict[str, Any], output: Path) -> None:
    episodes = sorted({int(sample["episode_index"]) for sample in samples})
    feasibility_status = feasibility.get("feasibility", {}).get("status", "未记录")
    gpu = feasibility.get("feasibility", {}).get("checks", {}).get("gpu_memory_gb", "未记录")
    bridge_samples = manifest.get("samples_exported", len(samples))
    cards = []
    for sample in samples:
        episode = int(sample["episode_index"])
        step = int(sample["source_step"])
        image_src = docs_rel(str(sample["image"]))
        cards.append(
            f"""
        <article class="sample-card" data-episode="{episode}" data-task="{esc(sample.get('task', ''))}">
          <img src="{esc(image_src)}" alt="episode {episode} step {step} MuJoCo sample frame">
          <div class="sample-body">
            <div class="sample-head">
              <span>episode {episode}</span>
              <span>step {step}</span>
            </div>
            <p class="instruction">{esc(sample.get('instruction', ''))}</p>
            <dl>
              <div><dt>目标物</dt><dd>{esc(sample.get('target_object'))}</dd></div>
              <div><dt>目标位姿</dt><dd>{esc(object_xy(sample))}</dd></div>
              <div><dt>动作</dt><dd>{esc(action_summary(sample.get('action', [])))}</dd></div>
            </dl>
          </div>
        </article>"""
        )

    options = "\n".join(f'<option value="{episode}">episode {episode}</option>' for episode in episodes)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenVLA Bridge Gallery</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: Canvas;
      --fg: CanvasText;
      --muted: color-mix(in srgb, CanvasText 68%, Canvas);
      --line: color-mix(in srgb, CanvasText 20%, Canvas);
      --soft: color-mix(in srgb, CanvasText 7%, Canvas);
      --accent: color-mix(in srgb, Highlight 18%, Canvas);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--fg);
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1, h2 {{
      margin: 0;
      font-weight: 600;
      letter-spacing: 0;
    }}
    h1 {{ font-size: 28px; }}
    h2 {{ font-size: 18px; margin-top: 28px; }}
    p {{ line-height: 1.6; }}
    .muted {{ color: var(--muted); }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin: 18px 0 22px;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: var(--soft);
    }}
    .stat span {{
      display: block;
      color: var(--muted);
      font-size: 13px;
    }}
    .stat strong {{
      display: block;
      margin-top: 6px;
      font-size: 20px;
      font-weight: 600;
    }}
    .toolbar {{
      display: flex;
      align-items: end;
      gap: 12px;
      flex-wrap: wrap;
      padding: 12px 0;
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      margin-bottom: 18px;
    }}
    label {{
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 13px;
    }}
    select, button {{
      font: inherit;
      color: var(--fg);
      background: var(--bg);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
    }}
    button {{
      cursor: pointer;
    }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
      gap: 14px;
    }}
    .sample-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: var(--bg);
    }}
    .sample-card[hidden] {{ display: none; }}
    .sample-card img {{
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: cover;
      background: var(--soft);
    }}
    .sample-body {{ padding: 10px; }}
    .sample-head {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
    }}
    .instruction {{
      margin: 8px 0;
      font-weight: 600;
    }}
    dl {{
      display: grid;
      gap: 6px;
      margin: 0;
      font-size: 13px;
    }}
    dl div {{
      display: grid;
      grid-template-columns: 52px 1fr;
      gap: 8px;
    }}
    dt {{ color: var(--muted); }}
    dd {{
      margin: 0;
      overflow-wrap: anywhere;
      font-variant-numeric: tabular-nums;
    }}
    .boundary {{
      margin-top: 22px;
      border-left: 4px solid var(--line);
      padding-left: 12px;
      color: var(--muted);
    }}
    @media (max-width: 560px) {{
      main {{ padding: 16px; }}
      h1 {{ font-size: 22px; }}
      .gallery {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>OpenVLA 数据桥接浏览页</h1>
    <p class="muted">版本：{VERSION}。用于检查 MuJoCo demonstration 是否已经整理成 VLA 接入需要的图像、语言、状态和动作字段。</p>
    <section class="stats" aria-label="summary">
      <div class="stat"><span>样本数</span><strong>{len(samples)}</strong></div>
      <div class="stat"><span>episode</span><strong>{len(episodes)}</strong></div>
      <div class="stat"><span>图像尺寸</span><strong>{esc(manifest.get('image_size', ''))}</strong></div>
      <div class="stat"><span>GPU 显存</span><strong>{esc(gpu)}GB</strong></div>
    </section>
    <section class="toolbar" aria-label="filters">
      <label>episode
        <select id="episodeFilter">
          <option value="all">全部</option>
          {options}
        </select>
      </label>
      <button id="resetFilter" type="button">显示全部</button>
      <span id="visibleCount" class="muted">{len(samples)} / {len(samples)} samples</span>
    </section>
    <section class="gallery" id="gallery" aria-label="OpenVLA bridge samples">
      {''.join(cards)}
    </section>
    <section class="boundary">
      <p>本页不是策略评测结果。当前可行性检查结论：{esc(feasibility_status)}。真实 OpenVLA/机器人 VLA 训练需要在更大显存环境中另行登记版本。</p>
      <p>来源：{esc(manifest.get('jsonl_path', ''))}；报告：docs/openvla_dataset_bridge_report.md 与 docs/openvla_feasibility_report.md。</p>
    </section>
  </main>
  <script>
    const filter = document.getElementById('episodeFilter');
    const reset = document.getElementById('resetFilter');
    const cards = Array.from(document.querySelectorAll('.sample-card'));
    const count = document.getElementById('visibleCount');
    function applyFilter() {{
      const value = filter.value;
      let visible = 0;
      cards.forEach((card) => {{
        const show = value === 'all' || card.dataset.episode === value;
        card.hidden = !show;
        if (show) visible += 1;
      }});
      count.textContent = `${{visible}} / ${{cards.length}} samples`;
    }}
    filter.addEventListener('change', applyFilter);
    reset.addEventListener('click', () => {{
      filter.value = 'all';
      applyFilter();
    }});
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    samples = read_jsonl(args.samples)
    manifest = read_json(args.manifest)
    feasibility = read_json(args.feasibility)
    if not samples:
        raise ValueError(f"no samples found in {args.samples}")
    write_html(samples, manifest, feasibility, args.output)
    print(f"version: {VERSION}", flush=True)
    print(f"samples: {len(samples)}", flush=True)
    print(f"gallery: {args.output}", flush=True)


if __name__ == "__main__":
    main()
