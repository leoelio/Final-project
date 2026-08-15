from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
STORYBOARD_VERSION = "video_presentation_storyboard_v1"
STAGE_EVIDENCE_NUMBER_BY_PRESENTATION_KEY = {
    "01_task_data_oracle": "1",
    "02_basic_bc_baselines": "1",
    "03_trajectory_act_diffusion": "2",
    "04_action_head_peft_proxy": "3",
    "05_language_generalization": "4",
    "06_domain_randomization_proxy": "6",
    "07_candidate_diagnostics": "2",
}
PRESENTATION_STAGE_OVERRIDES = {
    "07_candidate_diagnostics": {
        "talk": "把这一段作为候选方法诊断总览：重点说明有些候选能把物体推到目标附近或出现局部 TCP 抬升，但严格抓取审计仍为 0，因此这些视频用于解释失败模式和后续改进方向。",
        "claim": "候选诊断显示控制门控、接触阶段、相对几何和偏好排序能改善局部指标，但当前不能证明稳定抓取。",
        "boundary": "候选诊断视频只证明失败模式和局部现象；不能写成完整 ACT、在线 RL、真实 VLA 后训练或真实 WidowX 成功。",
        "metric_refs": "docs/candidate_diagnostic_video_index.csv；docs/strict_grasp_success_audit.csv；docs/next_experiment_registry.csv",
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a stage-level video presentation storyboard.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs" / "presentation_clips" / "presentation_video_pack_manifest.json")
    parser.add_argument("--stage-evidence", type=Path, default=ROOT / "docs" / "stage_evidence_index.csv")
    parser.add_argument("--video-evidence", type=Path, default=ROOT / "docs" / "video_evidence_index.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "video_presentation_storyboard.md")
    parser.add_argument("--output-html", type=Path, default=ROOT / "docs" / "video_presentation_storyboard.html")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def md_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def md_row(values: list[object]) -> str:
    return "| " + " | ".join(md_escape(value) for value in values) + " |"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def rel(path: str | Path, base: Path) -> str:
    target = ROOT / path if not Path(path).is_absolute() else Path(path)
    return Path(os.path.relpath(target.resolve(), base.resolve())).as_posix()


def normalized(path: str) -> str:
    return path.replace("\\", "/")


def stage_number_from_key(key: str) -> str:
    prefix = key.split("_", 1)[0]
    return str(int(prefix)) if prefix.isdigit() else ""


def ps_command(script: str) -> str:
    return f'& "{PYTHON}" "{ROOT / script}"'


def stage_evidence_by_number(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["阶段编号"]: row for row in rows}


def video_rows_by_version(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["版本"]: row for row in rows}


def source_clip_summary(versions: list[str], video_by_version: dict[str, dict[str, str]]) -> str:
    parts = []
    for version in versions:
        row = video_by_version.get(version)
        if not row:
            parts.append(f"`{version}`")
            continue
        parts.append(f"`{version}`({row.get('结果', '-')}, {row.get('证据用途', '-')})")
    return "；".join(parts)


def build_stage_story_rows(manifest: dict, stage_rows: list[dict[str, str]], video_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    stage_by_number = stage_evidence_by_number(stage_rows)
    video_by_version = video_rows_by_version(video_rows)
    rows: list[dict[str, object]] = []
    offset = 0.0
    for index, stage in enumerate(manifest["stages"], start=1):
        number = stage_number_from_key(stage["key"]) or str(index)
        evidence_number = STAGE_EVIDENCE_NUMBER_BY_PRESENTATION_KEY.get(stage["key"], number)
        evidence = stage_by_number.get(evidence_number, {})
        override = PRESENTATION_STAGE_OVERRIDES.get(stage["key"], {})
        duration = float(stage["duration"])
        output = normalized(stage["output"])
        if not (ROOT / output).exists():
            raise FileNotFoundError(ROOT / output)
        rows.append(
            {
                "number": number,
                "evidence_number": evidence_number,
                "timeline": f"{offset:.0f}-{offset + duration:.0f}s",
                "title": stage["title"],
                "purpose": stage["purpose"],
                "output": output,
                "duration": duration,
                "clips": stage["clips"],
                "source_summary": source_clip_summary(stage["clips"], video_by_version),
                "talk": override.get("talk", evidence.get("推荐讲解", "按视频内容说明方法表现。")),
                "claim": override.get("claim", evidence.get("论文可写结论", stage["purpose"])),
                "boundary": override.get("boundary", evidence.get("论文红线", "展示视频不替代量化评测。")),
                "metric_refs": override.get("metric_refs", evidence.get("量化证据", "docs/evaluation_summary.csv")),
            }
        )
        offset += duration
    return rows


def write_md(path: Path, manifest: dict, rows: list[dict[str, object]]) -> None:
    master = manifest["master"]
    master_output = normalized(master["output"])
    master_duration = float(master.get("duration", sum(float(row["duration"]) for row in rows)))
    stage_count = len(rows)
    lines = [
        "# 视频展示讲稿与时间线",
        "",
        f"版本：`{STORYBOARD_VERSION}`",
        "",
        "用途：把当前 MuJoCo 仿真视频片段整理成可直接用于答辩、阶段汇报和论文展示的播放脚本。它不替代量化评测，所有成功率仍以 CSV 评测表为准。",
        "",
        "## 1. 总览 Reel 时间线",
        "",
        f"总览视频：`{master_output}`",
        "",
        md_row(["时间", "阶段短片", "展示重点", "讲稿提示", "论文红线"]),
        md_row(["---:", "---", "---", "---", "---"]),
    ]
    for row in rows:
        lines.append(md_row([row["timeline"], f"`{row['output']}`", row["purpose"], row["talk"], row["boundary"]]))

    lines.extend(
        [
            "",
            "## 2. 分阶段播放脚本",
            "",
        ]
    )
    for row in rows:
        lines.extend(
            [
                f"### {row['title']}",
                "",
                f"视频：`{row['output']}`",
                "",
                f"建议播放：`{row['timeline']}`，单阶段短片约 `{row['duration']:.1f}s`。",
                "",
                f"来源版本：{source_clip_summary(row['clips'], {}) if False else row['source_summary']}",
                "",
                f"可写结论：{row['claim']}",
                "",
                f"讲稿提示：{row['talk']}",
                "",
                f"论文红线：{row['boundary']}",
                "",
                f"量化证据：{row['metric_refs']}",
                "",
                f"阶段证据来源：`docs/stage_evidence_index.csv` 第 {row['evidence_number']} 阶段。",
                "",
            ]
        )

    lines.extend(
        [
            "## 3. 播放策略",
            "",
            f"1. 答辩开场先播放 `outputs/presentation_clips/00_defense_video_reel.mp4` 的 0-{master_duration:.0f} 秒，用它建立任务、方法和失败模式的整体视觉印象。",
            f"2. 被问到某一阶段时，切到对应 `01` 到 `{stage_count:02d}` 阶段短片，只解释该阶段的关键对照和失败模式。",
            "3. 若评委追问某个方法，打开 `docs/video_evidence_gallery.html` 或 `outputs/videos/*.mp4` 单独查看。",
            "4. 展示时同时说明量化表位置，避免只凭视频作结论。",
            "5. OpenVLA、Isaac、真实 WidowX 仍是后续阶段，不能用当前 MuJoCo 视频替代。",
            "",
            "## 4. 重建命令",
            "",
            "```powershell",
            ps_command("scripts/build_video_presentation_storyboard.py"),
            ps_command("scripts/verify_experiment_artifacts.py"),
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def stage_cards_html(rows: list[dict[str, object]], base: Path) -> str:
    cards = []
    for row in rows:
        cards.append(
            f"""
    <section class="stage-card">
      <div class="copy">
        <span>阶段 {esc(row['number'])} · {esc(row['timeline'])}</span>
        <h2>{esc(row['title'])}</h2>
        <p><strong>展示重点：</strong>{esc(row['purpose'])}</p>
        <p><strong>讲稿提示：</strong>{esc(row['talk'])}</p>
        <p><strong>可写结论：</strong>{esc(row['claim'])}</p>
        <p><strong>论文红线：</strong>{esc(row['boundary'])}</p>
        <p><strong>来源版本：</strong>{esc(row['source_summary'])}</p>
        <p><strong>量化证据：</strong>{esc(row['metric_refs'])}</p>
      </div>
      <figure>
        <video controls preload="metadata" src="{esc(rel(row['output'], base))}"></video>
        <figcaption><code>{esc(row['output'])}</code></figcaption>
      </figure>
    </section>
            """
        )
    return "".join(cards)


def write_html(path: Path, manifest: dict, rows: list[dict[str, object]]) -> None:
    base = path.parent
    master = manifest["master"]
    master_output = normalized(master["output"])
    master_duration = float(master.get("duration", sum(float(row["duration"]) for row in rows)))
    stage_count = len(rows)
    if not (ROOT / master_output).exists():
        raise FileNotFoundError(ROOT / master_output)
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>视频展示讲稿与时间线</title>
  <style>
    :root {{
      --ink: #20262e;
      --muted: #5d6875;
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
      margin: 0 0 10px;
      font-size: 22px;
      letter-spacing: 0;
    }}
    p {{
      color: var(--muted);
      margin: 8px 0;
    }}
    code {{
      font-family: Consolas, "Cascadia Mono", monospace;
      font-size: 0.92em;
    }}
    .quick {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 10px;
      margin-top: 16px;
    }}
    .quick a {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      color: var(--ink);
      text-decoration: none;
      overflow-wrap: anywhere;
    }}
    .quick a:hover {{
      border-color: var(--accent);
    }}
    .master {{
      display: grid;
      grid-template-columns: minmax(360px, 0.85fr) minmax(420px, 1.15fr);
      gap: 18px;
      align-items: start;
      border-bottom: 1px solid var(--line);
      padding: 26px 0;
    }}
    .stage-card {{
      display: grid;
      grid-template-columns: minmax(360px, 1fr) minmax(420px, 1fr);
      gap: 18px;
      align-items: start;
      border-bottom: 1px solid var(--line);
      padding: 26px 0;
    }}
    .stage-card span {{
      display: inline-block;
      color: var(--accent);
      font-weight: 700;
      margin-bottom: 6px;
    }}
    figure {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 10px;
    }}
    video {{
      width: 100%;
      aspect-ratio: 16 / 9;
      background: #111827;
      border-radius: 6px;
    }}
    figcaption {{
      margin-top: 8px;
      color: var(--muted);
      overflow-wrap: anywhere;
      font-size: 13px;
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
      .master, .stage-card {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>视频展示讲稿与时间线</h1>
    <p>版本：<code>{STORYBOARD_VERSION}</code>。把 {master_duration:.0f} 秒总览 reel 和 {stage_count} 个阶段短片组织成可直接展示的讲稿、证据和论文红线。</p>
    <div class="quick">
      <a href="{esc(rel('docs/video_presentation_storyboard.md', base))}"><code>docs/video_presentation_storyboard.md</code></a>
      <a href="{esc(rel('docs/presentation_video_pack.md', base))}"><code>docs/presentation_video_pack.md</code></a>
      <a href="{esc(rel('docs/video_evidence_gallery.html', base))}"><code>docs/video_evidence_gallery.html</code></a>
      <a href="{esc(rel('docs/stage_showcase_index.html', base))}"><code>docs/stage_showcase_index.html</code></a>
      <a href="{esc(rel('docs/defense_deck.html', base))}"><code>docs/defense_deck.html</code></a>
    </div>
  </header>
  <main>
    <section class="master">
      <div>
        <h2>总览 Reel 时间线</h2>
        <p>建议开场播放完整 0-{master_duration:.0f} 秒，用于建立任务、方法组、失败模式和阶段边界。后续根据提问切到单阶段短片。</p>
        <p>注意：视频是可视化证据，成功率和资源结论仍引用 CSV 和阶段报告。</p>
      </div>
      <figure>
        <video controls preload="metadata" src="{esc(rel(master_output, base))}"></video>
        <figcaption><code>{esc(master_output)}</code></figcaption>
      </figure>
    </section>
    {stage_cards_html(rows, base)}
    <section>
      <h2>重建命令</h2>
      <pre class="commands"><code>{esc(ps_command('scripts/build_video_presentation_storyboard.py'))}
{esc(ps_command('scripts/verify_experiment_artifacts.py'))}</code></pre>
    </section>
  </main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)
    stage_rows = read_csv(args.stage_evidence)
    video_rows = read_csv(args.video_evidence)
    rows = build_stage_story_rows(manifest, stage_rows, video_rows)
    write_md(args.output_md, manifest, rows)
    write_html(args.output_html, manifest, rows)
    print(f"video_presentation_storyboard_md: {args.output_md}", flush=True)
    print(f"video_presentation_storyboard_html: {args.output_html}", flush=True)
    print(f"stage_rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
