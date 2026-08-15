from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese stage showcase index with metrics, versions, commands and videos.")
    parser.add_argument("--versions", type=Path, default=ROOT / "docs" / "experiment_versions.json")
    parser.add_argument("--summary", type=Path, default=ROOT / "docs" / "evaluation_summary.csv")
    parser.add_argument("--language-summary", type=Path, default=ROOT / "docs" / "language_generalization_summary.csv")
    parser.add_argument("--resources", type=Path, default=ROOT / "docs" / "model_resource_summary.csv")
    parser.add_argument("--stage-evidence", type=Path, default=ROOT / "docs" / "stage_evidence_index.csv")
    parser.add_argument("--video-evidence", type=Path, default=ROOT / "docs" / "video_evidence_index.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "stage_showcase_index.md")
    parser.add_argument("--output-html", type=Path, default=ROOT / "docs" / "stage_showcase_index.html")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def by_version(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["version"]: row for row in rows if row.get("version")}


def split_items(text: str) -> list[str]:
    parts = str(text).replace("\n", "；").replace(";", "；").split("；")
    return [part.strip() for part in parts if part.strip()]


def md_escape(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def md_row(values: list[object]) -> str:
    return "| " + " | ".join(md_escape(value) for value in values) + " |"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def rel(path: str | Path, base: Path) -> str:
    target = ROOT / path if not Path(path).is_absolute() else Path(path)
    return Path(os.path.relpath(target.resolve(), base.resolve())).as_posix()


def existing_paths(items: list[str]) -> list[str]:
    return [item for item in items if item.startswith(("docs/", "outputs/", "data/")) and (ROOT / item).exists()]


def format_params(value: str) -> str:
    if value in ("", "-", None):
        return "-"
    try:
        return f"{int(float(value)):,}"
    except ValueError:
        return str(value)


def language_alias(version: str) -> str:
    if version == "expert_scripted_v1":
        return "expert_scripted_language_v1"
    return version


def method_metric_row(
    method: dict[str, str],
    summary: dict[str, dict[str, str]],
    language: dict[str, dict[str, str]],
    resources: dict[str, dict[str, str]],
) -> list[str]:
    version = method["version"]
    main = summary.get(version, method)
    lang = language.get(language_alias(version), {})
    res = resources.get(version, {})
    return [
        f"`{version}`",
        method["method"],
        method["stage"],
        main.get("train_range_success", method.get("train_range_success", "-")),
        main.get("heldout_success", method.get("heldout_success", "-")),
        lang.get("success", "not_applicable"),
        format_params(res.get("trainable_params", "0")),
        f"`{method.get('clip', '')}`",
    ]


def stage_versions(stage: dict[str, str], methods_by_version: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    versions: list[dict[str, str]] = []
    for item in split_items(stage.get("关键版本", "")):
        method = methods_by_version.get(item)
        if method:
            versions.append(method)
    return versions


def video_rows_for_stage(stage: dict[str, str], video_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    refs = set(split_items(stage.get("视频证据", "")))
    selected = []
    for row in video_rows:
        if row.get("视频文件") in refs:
            selected.append(row)
    return selected


def ps_command(script: str, args: list[str] | None = None, cuda: bool = False) -> str:
    parts = [f'& "{PYTHON}" "{ROOT / script}"']
    if args:
        parts.extend(str(arg) for arg in args)
    command = " ".join(parts)
    if cuda:
        return f'$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"; {command}'
    return command


def open_command(path: str) -> str:
    return f'Start-Process "{ROOT / path}"'


def write_md(
    path: Path,
    versions: dict,
    methods: list[dict[str, str]],
    stage_rows: list[dict[str, str]],
    video_rows: list[dict[str, str]],
    summary: dict[str, dict[str, str]],
    language: dict[str, dict[str, str]],
    resources: dict[str, dict[str, str]],
) -> None:
    methods_by_version = {method["version"]: method for method in methods}
    lines = [
        "# 阶段展示总索引",
        "",
        "版本：`stage_showcase_index_v1`",
        "",
        "用途：把当前实验的版本名称、阶段说明、评测比较、仿真视频片段展示和完整启动命令集中到一个中文入口，服务论文撰写、答辩演示和后续实验复现。",
        "",
        f"主任务：`{versions.get('task', '')}` / `{versions.get('complexity', '')}`。",
        "",
        f"覆盖：`{len(methods)}` 个正式方法版本，`{len(stage_rows)}` 个阶段，`{len(video_rows)}` 条视频证据。",
        "",
        "## 快速启动命令",
        "",
        "```powershell",
        open_command("docs/stage_showcase_index.html"),
        open_command("docs/experiment_dashboard.html"),
        open_command("docs/video_presentation_storyboard.html"),
        open_command("docs/video_evidence_gallery.html"),
        open_command("docs/defense_deck.html"),
        open_command("docs/openvla_bridge_gallery.html"),
        open_command("docs/external_dependency_readiness_audit.md"),
        ps_command("scripts/verify_experiment_artifacts.py", cuda=True),
        "```",
        "",
        "重新生成本索引：",
        "",
        "```powershell",
        ps_command("scripts/build_stage_showcase_index.py"),
        "```",
        "",
        "全部方法的 MuJoCo viewer 慢速启动命令见 `docs/reproducible_command_index.md`；统一使用 `--viewer --duration 60 --speed 0.05` 观察。",
        "",
        "## 阶段路线总览",
        "",
        md_row(["阶段", "名称", "覆盖", "关键版本", "展示入口"]),
        md_row(["---:", "---", "---", "---", "---"]),
    ]
    for stage in stage_rows:
        lines.append(
            md_row(
                [
                    stage["阶段编号"],
                    stage["阶段名称"],
                    stage["覆盖数量"],
                    stage["关键版本"],
                    stage["展示入口"],
                ]
            )
        )

    lines.extend(["", "## 分阶段讲解索引", ""])
    for stage in stage_rows:
        key_methods = stage_versions(stage, methods_by_version)
        stage_video_rows = video_rows_for_stage(stage, video_rows)
        report_paths = existing_paths(split_items(stage.get("阶段报告", "")))
        metric_paths = existing_paths(split_items(stage.get("量化证据", "")))
        video_paths = existing_paths(split_items(stage.get("视频证据", "")))
        display_paths = existing_paths(split_items(stage.get("展示入口", "")))
        lines.extend(
            [
                f"### 阶段 {stage['阶段编号']}：{stage['阶段名称']}",
                "",
                f"覆盖：{stage['覆盖数量']}",
                "",
                f"可写结论：{stage['论文可写结论']}",
                "",
                f"论文红线：{stage['论文红线']}",
                "",
                f"推荐讲解：{stage['推荐讲解']}",
                "",
                "阶段报告：",
                "",
            ]
        )
        lines.extend(f"- `{item}`" for item in report_paths)
        lines.extend(["", "量化证据：", ""])
        lines.extend(f"- `{item}`" for item in metric_paths)
        lines.extend(["", "仿真视频片段：", ""])
        if video_paths:
            lines.extend(f"- `{item}`" for item in video_paths)
        else:
            lines.append("- 本阶段主要引用评测表或全局视频证据页。")
        lines.extend(["", "展示入口：", ""])
        lines.extend(f"- `{item}`" for item in display_paths)
        if stage_video_rows:
            lines.extend(["", "视频证据摘要：", "", md_row(["视频类型", "版本", "结果", "视频文件", "证据用途"]), md_row(["---", "---", "---", "---", "---"])])
            for row in stage_video_rows[:8]:
                lines.append(md_row([row["视频类型"], f"`{row['版本']}`", row["结果"], f"`{row['视频文件']}`", row["证据用途"]]))
        if key_methods:
            lines.extend(
                [
                    "",
                    "关键版本与指标：",
                    "",
                    md_row(["版本", "方法", "阶段", "train", "held-out", "language", "参数", "固定视频"]),
                    md_row(["---", "---", "---", "---:", "---:", "---:", "---:", "---"]),
                ]
            )
            for method in key_methods:
                lines.append(md_row(method_metric_row(method, summary, language, resources)))
        lines.extend(["", "本阶段重建命令：", "", "```powershell", stage["重建命令"], "```", ""])

    lines.extend(
        [
            "## 全部正式方法版本",
            "",
            md_row(["版本", "方法", "阶段", "train", "held-out", "language", "参数", "固定视频"]),
            md_row(["---", "---", "---", "---:", "---:", "---:", "---:", "---"]),
        ]
    )
    for method in methods:
        lines.append(md_row(method_metric_row(method, summary, language, resources)))

    lines.extend(
        [
            "",
            "## OpenVLA 数据桥接边界",
            "",
            "- `openvla_dataset_bridge_v1`：已导出 MuJoCo 成功轨迹的 `image + instruction + state + action` 样本，浏览页是 `docs/openvla_bridge_gallery.html`。",
            "- `openvla_feasibility_check_v1`：已记录本机真实 OpenVLA/机器人 VLA LoRA 的训练限制，报告是 `docs/openvla_feasibility_report.md`。",
            "- `robot_vla_action_head_handoff_v1`：已定义真实 robot VLA action-head 在 48GB+ GPU 或云端运行时的输入契约、输出契约和入包门禁，报告是 `docs/robot_vla_action_head_handoff.md`。",
            "- `external_dependency_readiness_audit_v1`：已统一审计真实 robot VLA、Isaac 和真实 WidowX planned 版本的阻塞条件、回填工件和论文边界，报告是 `docs/external_dependency_readiness_audit.md`。",
            "- 下一阶段正式入包规则见 `docs/next_experiment_registry.md`，计划版本在真正运行前不能写成已完成实验。",
            "- 这些条目不是策略模型，不参与当前 25 个方法版本的成功率比较；`external_dependency_readiness_audit_v1` 也不是策略成功率结果，不能写成 OpenVLA LoRA、`robot_vla_action_head_lite_v1`、Isaac 或真实 WidowX 验证已经完成。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def link_cards(paths: list[str], base: Path) -> str:
    cards = []
    for item in paths:
        cards.append(f'<a class="link-card" href="{esc(rel(item, base))}"><code>{esc(item)}</code></a>')
    return "".join(cards)


def method_rows_html(
    methods: list[dict[str, str]],
    summary: dict[str, dict[str, str]],
    language: dict[str, dict[str, str]],
    resources: dict[str, dict[str, str]],
) -> str:
    rows = []
    for method in methods:
        values = method_metric_row(method, summary, language, resources)
        rows.append(
            "<tr>"
            f"<td>{esc(values[0])}</td>"
            f"<td>{esc(values[1])}</td>"
            f"<td>{esc(values[2])}</td>"
            f"<td>{esc(values[3])}</td>"
            f"<td>{esc(values[4])}</td>"
            f"<td>{esc(values[5])}</td>"
            f"<td>{esc(values[6])}</td>"
            f"<td><code>{esc(method.get('clip', ''))}</code></td>"
            "</tr>"
        )
    return "".join(rows)


def video_grid(paths: list[str], base: Path) -> str:
    videos = []
    for item in paths:
        if not item.endswith(".mp4"):
            continue
        if not (ROOT / item).exists():
            continue
        videos.append(
            '<figure class="video-card">'
            f'<video controls preload="metadata" src="{esc(rel(item, base))}"></video>'
            f"<figcaption><code>{esc(item)}</code></figcaption>"
            "</figure>"
        )
    if not videos:
        return '<p class="muted">本阶段主要引用评测表或全局视频证据页。</p>'
    return "".join(videos)


def write_html(
    path: Path,
    versions: dict,
    methods: list[dict[str, str]],
    stage_rows: list[dict[str, str]],
    video_rows: list[dict[str, str]],
    summary: dict[str, dict[str, str]],
    language: dict[str, dict[str, str]],
    resources: dict[str, dict[str, str]],
) -> None:
    methods_by_version = {method["version"]: method for method in methods}
    base = path.parent
    stage_cards = []
    for stage in stage_rows:
        key_methods = stage_versions(stage, methods_by_version)
        report_paths = existing_paths(split_items(stage.get("阶段报告", "")))
        metric_paths = existing_paths(split_items(stage.get("量化证据", "")))
        video_paths = existing_paths(split_items(stage.get("视频证据", "")))
        display_paths = existing_paths(split_items(stage.get("展示入口", "")))
        table = ""
        if key_methods:
            table = (
                '<div class="table-wrap"><table>'
                "<thead><tr><th>版本</th><th>方法</th><th>阶段</th><th>Train</th><th>Held-out</th><th>Language</th><th>参数</th><th>固定视频</th></tr></thead>"
                f"<tbody>{method_rows_html(key_methods, summary, language, resources)}</tbody></table></div>"
            )
        stage_cards.append(
            f"""
    <section class="stage-section" id="stage-{esc(stage['阶段编号'])}">
      <div class="stage-head">
        <span>阶段 {esc(stage['阶段编号'])}</span>
        <h2>{esc(stage['阶段名称'])}</h2>
        <p>{esc(stage['覆盖数量'])}</p>
      </div>
      <div class="stage-grid">
        <div>
          <h3>阶段说明</h3>
          <p><strong>可写结论：</strong>{esc(stage['论文可写结论'])}</p>
          <p><strong>论文红线：</strong>{esc(stage['论文红线'])}</p>
          <p><strong>推荐讲解：</strong>{esc(stage['推荐讲解'])}</p>
          <h3>报告与评测</h3>
          <div class="link-grid">{link_cards(report_paths + metric_paths + display_paths, base)}</div>
          <h3>关键版本与指标</h3>
          {table if table else '<p class="muted">本阶段引用全局展示、视频或评测表，不单独登记策略方法。</p>'}
        </div>
        <aside>
          <h3>仿真视频片段展示</h3>
          <div class="video-grid">{video_grid(video_paths, base)}</div>
        </aside>
      </div>
    </section>
            """
        )

    all_methods_table = method_rows_html(methods, summary, language, resources)
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>阶段展示总索引</title>
  <style>
    :root {{
      --ink: #20262e;
      --muted: #5b6470;
      --line: #d9dee7;
      --panel: #f6f8fb;
      --accent: #1c6da8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background: white;
      line-height: 1.55;
    }}
    header, main {{
      max-width: 1220px;
      margin: 0 auto;
      padding: 28px 24px;
    }}
    header {{
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 30px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0;
      font-size: 24px;
      letter-spacing: 0;
    }}
    h3 {{
      margin: 18px 0 8px;
      font-size: 16px;
      letter-spacing: 0;
    }}
    p {{
      color: var(--muted);
    }}
    code {{
      font-family: Consolas, "Cascadia Mono", monospace;
      font-size: 0.92em;
    }}
    .quick {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 10px;
      margin-top: 18px;
    }}
    .quick a, .link-card {{
      display: block;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      color: var(--ink);
      text-decoration: none;
      background: #fff;
      overflow-wrap: anywhere;
    }}
    .quick a:hover, .link-card:hover {{
      border-color: var(--accent);
    }}
    .stage-section {{
      border-top: 1px solid var(--line);
      padding: 30px 0;
    }}
    .stage-head span {{
      display: inline-block;
      color: var(--accent);
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .stage-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(360px, 0.95fr);
      gap: 20px;
      align-items: start;
    }}
    .link-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 8px;
    }}
    .video-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
    }}
    .video-card {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: var(--panel);
    }}
    .video-card video {{
      width: 100%;
      aspect-ratio: 4 / 3;
      background: #111827;
      border-radius: 6px;
    }}
    .video-card figcaption {{
      margin-top: 7px;
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      min-width: 860px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 9px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: var(--panel);
      color: #273443;
      position: sticky;
      top: 0;
    }}
    .muted {{
      color: var(--muted);
    }}
    .commands {{
      background: #10151d;
      color: #eef3f8;
      border-radius: 8px;
      padding: 14px;
      overflow: auto;
      font-size: 13px;
    }}
    @media (max-width: 900px) {{
      .stage-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>阶段展示总索引</h1>
    <p>版本：<code>stage_showcase_index_v1</code>。集中展示版本名称、阶段说明、评测比较、仿真视频片段展示和完整启动命令。</p>
    <p>主任务：<code>{esc(versions.get('task', ''))}</code> / <code>{esc(versions.get('complexity', ''))}</code>；覆盖 {len(methods)} 个正式方法版本、{len(stage_rows)} 个阶段、{len(video_rows)} 条视频证据。</p>
    <div class="quick">
      <a href="{esc(rel('docs/stage_showcase_index.md', base))}"><code>docs/stage_showcase_index.md</code></a>
      <a href="{esc(rel('docs/experiment_dashboard.html', base))}"><code>docs/experiment_dashboard.html</code></a>
      <a href="{esc(rel('docs/video_presentation_storyboard.html', base))}"><code>docs/video_presentation_storyboard.html</code></a>
      <a href="{esc(rel('docs/video_evidence_gallery.html', base))}"><code>docs/video_evidence_gallery.html</code></a>
      <a href="{esc(rel('docs/defense_deck.html', base))}"><code>docs/defense_deck.html</code></a>
      <a href="{esc(rel('docs/openvla_bridge_gallery.html', base))}"><code>docs/openvla_bridge_gallery.html</code></a>
      <a href="{esc(rel('docs/external_dependency_readiness_audit.md', base))}"><code>docs/external_dependency_readiness_audit.md</code></a>
      <a href="{esc(rel('docs/reproducible_command_index.md', base))}"><code>docs/reproducible_command_index.md</code></a>
    </div>
  </header>
  <main>
    <section class="stage-section">
      <div class="stage-head">
        <span>启动命令</span>
        <h2>快速打开与验证</h2>
      </div>
      <pre class="commands"><code>{esc(open_command('docs/stage_showcase_index.html'))}
{esc(open_command('docs/experiment_dashboard.html'))}
{esc(open_command('docs/video_presentation_storyboard.html'))}
{esc(open_command('docs/video_evidence_gallery.html'))}
{esc(open_command('docs/defense_deck.html'))}
{esc(open_command('docs/openvla_bridge_gallery.html'))}
{esc(open_command('docs/external_dependency_readiness_audit.md'))}
{esc(ps_command('scripts/verify_experiment_artifacts.py', cuda=True))}</code></pre>
      <p>全部方法的 MuJoCo viewer 慢速启动命令见 <code>docs/reproducible_command_index.md</code>；统一使用 <code>--viewer --duration 60 --speed 0.05</code>。</p>
    </section>
    {"".join(stage_cards)}
    <section class="stage-section">
      <div class="stage-head">
        <span>全部版本</span>
        <h2>25 个正式方法版本与指标</h2>
      </div>
      <div class="table-wrap"><table>
        <thead><tr><th>版本</th><th>方法</th><th>阶段</th><th>Train</th><th>Held-out</th><th>Language</th><th>参数</th><th>固定视频</th></tr></thead>
        <tbody>{all_methods_table}</tbody>
      </table></div>
    </section>
    <section class="stage-section">
      <div class="stage-head">
        <span>OpenVLA 边界</span>
        <h2>数据桥接不是策略成功率</h2>
      </div>
      <p><code>openvla_dataset_bridge_v1</code>、<code>openvla_feasibility_check_v1</code>、<code>robot_vla_action_head_handoff_v1</code> 和 <code>external_dependency_readiness_audit_v1</code> 是下一阶段入口，不参与当前 25 个方法版本的成功率比较；readiness audit 不是策略成功率结果。</p>
      <div class="link-grid">
        {link_cards(['docs/openvla_bridge_gallery.html', 'docs/openvla_dataset_bridge_report.md', 'docs/openvla_feasibility_report.md', 'docs/robot_vla_action_head_handoff.md', 'docs/external_dependency_readiness_audit.md', 'docs/next_phase_implementation.md', 'docs/next_experiment_registry.md'], base)}
      </div>
    </section>
  </main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    versions = read_json(args.versions)
    methods = versions["methods"]
    summary = by_version(read_csv(args.summary))
    language = by_version(read_csv(args.language_summary))
    resources = by_version(read_csv(args.resources))
    stage_rows = read_csv(args.stage_evidence)
    video_rows = read_csv(args.video_evidence)
    write_md(args.output_md, versions, methods, stage_rows, video_rows, summary, language, resources)
    write_html(args.output_html, versions, methods, stage_rows, video_rows, summary, language, resources)
    print(f"stage_showcase_md: {args.output_md}", flush=True)
    print(f"stage_showcase_html: {args.output_html}", flush=True)
    print(f"methods: {len(methods)}", flush=True)
    print(f"stages: {len(stage_rows)}", flush=True)
    print(f"video_evidence_rows: {len(video_rows)}", flush=True)


if __name__ == "__main__":
    main()
