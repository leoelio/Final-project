from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_defense_slide_outline import SLIDES, language_alias  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local HTML defense deck with figures and videos.")
    parser.add_argument("--versions", type=Path, default=ROOT / "docs" / "experiment_versions.json")
    parser.add_argument("--summary", type=Path, default=ROOT / "docs" / "evaluation_summary.csv")
    parser.add_argument("--language-summary", type=Path, default=ROOT / "docs" / "language_generalization_summary.csv")
    parser.add_argument("--resources", type=Path, default=ROOT / "docs" / "model_resource_summary.csv")
    parser.add_argument("--strict-grasp-json", type=Path, default=ROOT / "outputs" / "evaluations" / "strict_grasp_success_audit_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "defense_deck.html")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def by_version(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["version"]: row for row in rows if row.get("version")}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def rel(path: str) -> str:
    if not path:
        return ""
    return Path("..", path).as_posix()


def metric(row: dict | None, key: str, default: str = "-") -> str:
    if not row:
        return default
    value = row.get(key, "")
    return value if value not in ("", None) else default


def param_text(value: str) -> str:
    if value in ("", "-", None):
        return "-"
    try:
        return f"{int(float(value)):,}"
    except ValueError:
        return str(value)


def strict_counts(args: argparse.Namespace) -> tuple[str, str]:
    summary = read_json(args.strict_grasp_json).get("summary", {})
    episodes = summary.get("episodes", "?")
    loose = f"{summary.get('loose_successes', '?')}/{episodes}"
    strict = f"{summary.get('strict_grasp_successes', '?')}/{episodes}"
    return loose, strict


def method_rows(slide: dict, methods: dict[str, dict], summary: dict[str, dict], language: dict[str, dict], resources: dict[str, dict]) -> str:
    if not slide["versions"]:
        return ""
    rows = []
    for version in slide["versions"]:
        method = methods.get(version)
        if not method:
            continue
        main = summary.get(version, method)
        lang = language.get(language_alias(version), {})
        res = resources.get(version, {})
        rows.append(
            "<tr>"
            f"<td><code>{esc(version)}</code></td>"
            f"<td>{esc(method['method'])}</td>"
            f"<td>{esc(metric(main, 'train_range_success', method.get('train_range_success', '-')))}</td>"
            f"<td>{esc(metric(main, 'heldout_success', method.get('heldout_success', '-')))}</td>"
            f"<td>{esc(metric(lang, 'success'))}</td>"
            f"<td>{esc(param_text(metric(res, 'trainable_params', '0')))}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return (
        '<div class="tableWrap"><table>'
        "<thead><tr><th>版本</th><th>方法</th><th>Train</th><th>Held-out</th><th>Language</th><th>参数</th></tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def media_block(slide: dict) -> str:
    blocks = []
    if slide.get("figure"):
        blocks.append(
            '<figure class="mediaCard">'
            f'<img src="{esc(rel(slide["figure"]))}" alt="{esc(slide["title"])} 图表">'
            f'<figcaption>{esc(slide["figure"])}</figcaption>'
            "</figure>"
        )
    if slide.get("video"):
        blocks.append(
            '<figure class="mediaCard">'
            f'<video src="{esc(rel(slide["video"]))}" controls preload="metadata"></video>'
            f'<figcaption>{esc(slide["video"])}</figcaption>'
            "</figure>"
        )
    if not blocks:
        return '<div class="mediaPlaceholder">本页以讲解和指标表为主</div>'
    return "".join(blocks)


def link_block(slide: dict) -> str:
    links = slide.get("links", [])
    if not links:
        return ""
    cards = []
    for item in links:
        cards.append(
            '<a class="linkCard"'
            f' href="{esc(rel(item["path"]))}">'
            f'<strong>{esc(item["title"])}</strong>'
            f'<span><code>{esc(item["path"])}</code></span>'
            f'<p>{esc(item.get("note", ""))}</p>'
            "</a>"
        )
    return '<div class="linkGrid">' + "".join(cards) + "</div>"


def slide_html(slide: dict, index: int, total: int, methods: dict[str, dict], summary: dict[str, dict], language: dict[str, dict], resources: dict[str, dict]) -> str:
    table = method_rows(slide, methods, summary, language, resources)
    return f"""
    <section class="slide" data-slide="{index}">
      <div class="slideHeader">
        <span class="slideNo">{esc(slide['id'])} / {total:02d}</span>
        <h1>{esc(slide['title'])}</h1>
      </div>
      <div class="slideGrid">
        <main class="contentPanel">
          <p class="message">{esc(slide['message'])}</p>
          <div class="scriptBox">
            <h2>讲稿提示</h2>
            <p>{esc(slide['script'])}</p>
          </div>
          {link_block(slide)}
          {table}
        </main>
        <aside class="mediaPanel">
          {media_block(slide)}
        </aside>
      </div>
    </section>
    """


def verify_refs() -> None:
    missing = []
    for slide in SLIDES:
        for key in ("figure", "video"):
            path = slide.get(key)
            if path and not (ROOT / path).exists():
                missing.append(path)
        for link in slide.get("links", []):
            path = link.get("path")
            if path and not (ROOT / path).exists():
                missing.append(path)
    if missing:
        raise FileNotFoundError("\n".join(missing))


def write_html(args: argparse.Namespace) -> None:
    versions = read_json(args.versions)
    methods = {method["version"]: method for method in versions["methods"]}
    summary = by_version(read_csv(args.summary))
    language = by_version(read_csv(args.language_summary))
    resources = by_version(read_csv(args.resources))
    loose, strict = strict_counts(args)
    verify_refs()

    slides = "\n".join(
        slide_html(slide, index + 1, len(SLIDES), methods, summary, language, resources)
        for index, slide in enumerate(SLIDES)
    )
    nav_items = "\n".join(
        f'<button class="dot" data-target="{i}" title="{esc(slide["title"])}">{esc(slide["id"])}</button>'
        for i, slide in enumerate(SLIDES)
    )
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>轻量化 VLA 机械臂实验答辩 Deck</title>
  <style>
    :root {{
      --bg: #111318;
      --panel: #1b1f28;
      --panel2: #242a35;
      --text: #f1f4f8;
      --muted: #a8b0bd;
      --accent: #5bb7ff;
      --line: #343c4a;
      --good: #65d18f;
      --warn: #ffca5b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      overflow: hidden;
    }}
    .topbar {{
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      height: 58px;
      z-index: 10;
      display: flex;
      align-items: center;
      gap: 14px;
      padding: 0 18px;
      background: rgba(17,19,24,0.96);
      border-bottom: 1px solid var(--line);
    }}
    .brand {{
      font-size: 15px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .nav {{
      display: flex;
      gap: 6px;
      overflow-x: auto;
      flex: 1;
      padding-bottom: 2px;
    }}
    .dot, .navButton {{
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      height: 32px;
      min-width: 38px;
      padding: 0 10px;
      border-radius: 6px;
      cursor: pointer;
      font: inherit;
    }}
    .dot.active {{
      background: var(--accent);
      color: #05111d;
      border-color: var(--accent);
      font-weight: 700;
    }}
    .navButton {{
      min-width: 72px;
    }}
    .slides {{
      height: 100vh;
      padding-top: 58px;
    }}
    .slide {{
      display: none;
      width: 100vw;
      height: calc(100vh - 58px);
      padding: 28px 34px 34px;
    }}
    .slide.active {{ display: block; }}
    .slideHeader {{
      display: flex;
      align-items: center;
      gap: 18px;
      margin-bottom: 18px;
    }}
    .slideNo {{
      color: var(--accent);
      font-weight: 800;
      letter-spacing: 0;
      font-size: 20px;
      min-width: 78px;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(28px, 3vw, 46px);
      line-height: 1.1;
      letter-spacing: 0;
    }}
    .slideGrid {{
      display: grid;
      grid-template-columns: minmax(420px, 0.95fr) minmax(480px, 1.25fr);
      gap: 22px;
      height: calc(100% - 72px);
      min-height: 0;
    }}
    .contentPanel, .mediaPanel {{
      min-height: 0;
      overflow: auto;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
    }}
    .mediaPanel {{
      display: flex;
      flex-direction: column;
      gap: 14px;
      background: #0d0f14;
    }}
    .message {{
      margin: 0 0 18px;
      font-size: clamp(20px, 2vw, 30px);
      line-height: 1.35;
      font-weight: 700;
    }}
    .scriptBox {{
      border-left: 4px solid var(--accent);
      padding: 12px 14px;
      background: var(--panel2);
      border-radius: 6px;
      margin-bottom: 18px;
    }}
    .scriptBox h2 {{
      margin: 0 0 8px;
      font-size: 16px;
      color: var(--accent);
      letter-spacing: 0;
    }}
    .scriptBox p {{
      margin: 0;
      line-height: 1.55;
      color: var(--text);
      font-size: 16px;
    }}
    .tableWrap {{
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 6px;
    }}
    .linkGrid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
      margin-bottom: 18px;
    }}
    .linkCard {{
      display: block;
      border: 1px solid var(--line);
      background: #111722;
      color: var(--text);
      text-decoration: none;
      border-radius: 6px;
      padding: 12px;
    }}
    .linkCard:hover {{
      border-color: var(--accent);
    }}
    .linkCard strong {{
      display: block;
      margin-bottom: 4px;
      color: var(--accent);
    }}
    .linkCard span {{
      display: block;
      overflow-wrap: anywhere;
    }}
    .linkCard p {{
      margin: 8px 0 0;
      color: var(--muted);
      line-height: 1.45;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      min-width: 620px;
    }}
    th, td {{
      text-align: left;
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    th {{
      color: var(--accent);
      background: #151923;
      position: sticky;
      top: 0;
    }}
    code {{
      color: var(--good);
      font-family: Consolas, "Cascadia Mono", monospace;
      font-size: 0.92em;
    }}
    .mediaCard {{
      margin: 0;
      border: 1px solid var(--line);
      background: #080a0e;
      border-radius: 8px;
      padding: 10px;
    }}
    .mediaCard img, .mediaCard video {{
      display: block;
      width: 100%;
      max-height: calc((100vh - 170px) / 1.05);
      object-fit: contain;
      background: #05070a;
      border-radius: 4px;
    }}
    .mediaCard figcaption {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 8px;
      overflow-wrap: anywhere;
    }}
    .mediaPlaceholder {{
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 280px;
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 8px;
      font-size: 20px;
    }}
    .footerHint {{
      position: fixed;
      bottom: 10px;
      right: 18px;
      color: var(--muted);
      font-size: 13px;
      z-index: 9;
    }}
    @media (max-width: 960px) {{
      body {{ overflow: auto; }}
      .topbar {{ position: sticky; }}
      .slides {{ height: auto; padding-top: 0; }}
      .slide {{ height: auto; min-height: calc(100vh - 58px); padding: 18px; }}
      .slideGrid {{ grid-template-columns: 1fr; height: auto; }}
      .contentPanel, .mediaPanel {{ overflow: visible; }}
      .footerHint {{ display: none; }}
    }}
    @media print {{
      body {{ overflow: visible; background: white; color: black; }}
      .topbar, .footerHint {{ display: none; }}
      .slides {{ height: auto; padding: 0; }}
      .slide {{ display: block !important; height: auto; page-break-after: always; color: black; }}
      .contentPanel, .mediaPanel {{ border: 1px solid #999; background: white; color: black; }}
      .scriptBox {{ background: #f2f4f7; }}
      code {{ color: #064f8f; }}
    }}
  </style>
</head>
<body>
  <header class="topbar">
    <div class="brand">轻量化 VLA 机械臂实验答辩 Deck</div>
    <button class="navButton" id="prevBtn">上一页</button>
    <button class="navButton" id="nextBtn">下一页</button>
    <nav class="nav">{nav_items}</nav>
  </header>
  <div class="slides">
    {slides}
  </div>
  <div class="footerHint">← / → 翻页，点击页码跳转，视频可手动播放</div>
  <script>
    const slides = Array.from(document.querySelectorAll('.slide'));
    const dots = Array.from(document.querySelectorAll('.dot'));
    let current = 0;
    function show(index) {{
      current = Math.max(0, Math.min(slides.length - 1, index));
      slides.forEach((slide, i) => slide.classList.toggle('active', i === current));
      dots.forEach((dot, i) => dot.classList.toggle('active', i === current));
      document.querySelectorAll('video').forEach((video, i) => {{
        if (!slides[current].contains(video)) video.pause();
      }});
    }}
    document.getElementById('prevBtn').addEventListener('click', () => show(current - 1));
    document.getElementById('nextBtn').addEventListener('click', () => show(current + 1));
    dots.forEach((dot, i) => dot.addEventListener('click', () => show(i)));
    window.addEventListener('keydown', event => {{
      if (event.key === 'ArrowRight' || event.key === 'PageDown' || event.key === ' ') show(current + 1);
      if (event.key === 'ArrowLeft' || event.key === 'PageUp') show(current - 1);
      if (event.key === 'Home') show(0);
      if (event.key === 'End') show(slides.length - 1);
    }});
    show(0);
  </script>
</body>
</html>
"""
    html_text = html_text.replace("{STRICT_LOOSE}", loose).replace("{STRICT_SUCCESS}", strict)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    write_html(args)
    print(f"defense_deck_html: {args.output}", flush=True)
    print(f"slides: {len(SLIDES)}", flush=True)


if __name__ == "__main__":
    main()
