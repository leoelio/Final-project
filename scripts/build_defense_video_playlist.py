from __future__ import annotations

import argparse
import csv
from datetime import datetime
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
CANDIDATE_OVERVIEW_VIDEO = "outputs/presentation_clips/07_candidate_diagnostics.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese defense video playlist from claim and candidate evidence.")
    parser.add_argument("--claim-index", type=Path, default=ROOT / "docs" / "claim_video_playback_index.csv")
    parser.add_argument("--candidate-index", type=Path, default=ROOT / "docs" / "candidate_diagnostic_video_index.csv")
    parser.add_argument("--output-html", type=Path, default=ROOT / "docs" / "defense_video_playlist.html")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "defense_video_playlist.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "defense_video_playlist.csv")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def media_kind(path_text: str) -> str:
    suffix = Path(path_text).suffix.lower()
    if suffix in {".mp4", ".mov", ".avi", ".webm"}:
        return "video"
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        return "image"
    return "file"


def web_path(path_text: str) -> str:
    path = Path(path_text.replace("\\", "/"))
    if path.is_absolute():
        try:
            path = path.relative_to(ROOT)
        except ValueError:
            return path.as_posix()
    return (Path("..") / path).as_posix()


def normalize_claim(row: dict[str, str]) -> dict[str, str]:
    media = row["primary_video"]
    return {
        "section": "阶段 claim 播放顺序",
        "id": row["claim_id"],
        "title": row["claim_type"],
        "media": media,
        "media_kind": media_kind(media),
        "talk_prompt": row["talk_prompt"],
        "paper_redline": row["paper_redline"],
        "reference": row["quantitative_reference"],
        "open_command": row["playback_command"],
        "viewer_command": "",
        "export_command": "",
    }


def normalize_candidate(row: dict[str, str]) -> dict[str, str]:
    media = row["视频文件"]
    return {
        "section": "候选诊断负例播放顺序",
        "id": row["版本"],
        "title": row["方法定位"],
        "media": media,
        "media_kind": media_kind(media),
        "talk_prompt": row["实验结论"],
        "paper_redline": row["论文边界"],
        "reference": f"{row['报告文件']}；{row['元数据文件']}",
        "open_command": f'Start-Process "{ROOT / media}"',
        "viewer_command": row["完整viewer命令"],
        "export_command": row["重新导出视频命令"],
    }


def candidate_overview_entry(candidate_count: int) -> dict[str, str]:
    return {
        "section": "候选诊断负例播放顺序",
        "id": "candidate_diagnostic_montage_v1",
        "title": "候选诊断总览短片",
        "media": CANDIDATE_OVERVIEW_VIDEO,
        "media_kind": media_kind(CANDIDATE_OVERVIEW_VIDEO),
        "talk_prompt": f"先用 10 秒总览 {candidate_count} 个候选诊断视频，再按需要展开单个候选；重点说明目标距离成功、TCP 抬升和严格抓取成功必须分开报告。",
        "paper_redline": "候选诊断总览只用于解释失败模式和局部现象，不能写成完整 ACT、在线 RL、真实 VLA 后训练或真实 WidowX 成功。",
        "reference": "docs/candidate_diagnostic_video_index.csv；docs/strict_grasp_success_audit.csv；docs/video_presentation_storyboard.md",
        "open_command": f'Start-Process "{ROOT / CANDIDATE_OVERVIEW_VIDEO}"',
        "viewer_command": "",
        "export_command": "",
    }


def core_v2_ood_entries() -> list[dict[str, str]]:
    model = ROOT / "outputs" / "clip_semantic_waypoint" / "clip_semantic_waypoint_core_v2_v1_20260721_110325.npz"
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    common = "--workspace-profile core_v2 --image-size 224 --camera top_rgb --arm-kp 150 --arm-force 100 --gripper-kp 1200 --gripper-force 200 --friction 5 --place-tcp-z 0.041"
    return [
        {
            "section": "Core V2 OOD 对照播放顺序",
            "id": "core_v2_ood_hard_distractors_success_v1",
            "title": "7 物体干扰下的空间关系严格成功",
            "media": "outputs/videos/clip_semantic_ood_hard_leftmost_cube_seed1300.mp4",
            "media_kind": "video",
            "talk_prompt": "展示红蓝绿黄方块、圆柱和绿球同时出现时，策略仍按最左方块关系选择绿色方块并严格抓放入碗；对应 hard 干扰统计为 20/20。",
            "paper_redline": "该成功来自冻结 CLIP 意图 adapter 与 scripted waypoint expert 的分层结构，不能写成端到端 VLA、OpenVLA 或 LoRA 控制成功。",
            "reference": "docs/core_v2_clip_semantic_ood_generalization.md；docs/core_v2_clip_semantic_ood_generalization.csv；outputs/evaluations/core_v2_clip_semantic_ood_generalization_v1.json",
            "open_command": f'Start-Process "{ROOT / "outputs" / "videos" / "clip_semantic_ood_hard_leftmost_cube_seed1300.mp4"}',
            "viewer_command": f'$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"\n& "{python}" "{ROOT / "scripts" / "run_clip_semantic_waypoint.py"}" --model "{model}" --task move_leftmost_cube_to_bowl --complexity hard --seed 1300 --episodes 1 --viewer --duration 45 --speed 0.25 {common}',
            "export_command": f'$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"\n& "{python}" "{ROOT / "scripts" / "export_video.py"}" --method clip_semantic_waypoint --version clip_semantic_ood_hard_leftmost_cube --model "{model}" --task move_leftmost_cube_to_bowl --complexity hard --seed 1300 --width 640 --height 480 --fps 30 --frame-stride 8 --output "{ROOT / "outputs" / "videos" / "clip_semantic_ood_hard_leftmost_cube_seed1300.mp4"}" {common}',
        },
        {
            "section": "Core V2 OOD 对照播放顺序",
            "id": "core_v2_ood_paraphrase_failure_v1",
            "title": "未见同义表达的真实语义误判",
            "media": "outputs/videos/clip_semantic_ood_paraphrase_blue_to_red_seed700.mp4",
            "media_kind": "video",
            "talk_prompt": "展示指令 put the azure block on the red disk 被误判为红方块到红盘；蓝到红的未见改写总计仅 8/15，不隐藏语言泛化失败。",
            "paper_redline": "该失败说明当前四类 CLIP 意图 adapter 对词汇替换不稳健，不能宣传为通用自然语言理解或端到端 VLA 泛化。",
            "reference": "docs/core_v2_clip_semantic_ood_generalization.md；docs/core_v2_clip_semantic_ood_generalization.csv；outputs/evaluations/core_v2_clip_semantic_ood_generalization_v1.json",
            "open_command": f'Start-Process "{ROOT / "outputs" / "videos" / "clip_semantic_ood_paraphrase_blue_to_red_seed700.mp4"}',
            "viewer_command": f'$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"\n& "{python}" "{ROOT / "scripts" / "run_clip_semantic_waypoint.py"}" --model "{model}" --task place_blue_cube_red_pad --instruction "put the azure block on the red disk" --complexity medium --seed 700 --episodes 1 --viewer --duration 45 --speed 0.25 {common}',
            "export_command": f'$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"\n& "{python}" "{ROOT / "scripts" / "export_video.py"}" --method clip_semantic_waypoint --version clip_semantic_ood_paraphrase_blue_to_red --model "{model}" --task place_blue_cube_red_pad --instruction "put the azure block on the red disk" --complexity medium --seed 700 --width 640 --height 480 --fps 30 --frame-stride 8 --output "{ROOT / "outputs" / "videos" / "clip_semantic_ood_paraphrase_blue_to_red_seed700.mp4"}" {common}',
        },
    ]


def kaggle_remote_entries() -> list[dict[str, str]]:
    version = "kaggle_clip_semantic_adapter_core_v2_v1"
    model = ROOT / "outputs" / "clip_semantic_waypoint" / f"{version}_kernel_v3.npz"
    video = "outputs/videos/kaggle_clip_semantic_adapter_core_v2_v1_hard_leftmost_seed1900.mp4"
    common = "--workspace-profile core_v2 --camera top_rgb --arm-kp 150 --arm-force 100 --gripper-kp 1200 --gripper-force 200 --friction 5 --place-tcp-z 0.041"
    return [
        {
            "section": "Kaggle 远程补充实验",
            "id": version,
            "title": "Kaggle 冻结 CLIP 瓶颈适配器的 hard 空间任务闭环",
            "media": video,
            "media_kind": "video",
            "talk_prompt": "说明 Kaggle 只训练 1024->16->4 语义适配器，MuJoCo 由结构化 waypoint executor 执行。远程训练使用 79 条示范、63/16 分层训练验证，验证意图准确率 100%；本机同协议四任务 20/20 严格闭环成功。本次 Kaggle P100 与预装 PyTorch 不兼容，脚本实际 CPU fallback，不能说成 GPU 加速。",
            "paper_redline": "这是冻结 VLM 的高层语义决策后训练加结构化执行，不是端到端动作 VLA，不是 OpenVLA LoRA，也没有完成真实机械臂验证。",
            "reference": "docs/kaggle_clip_semantic_adapter_core_v2_v1_report.md；outputs/evaluations/kaggle_clip_semantic_adapter_core_v2_v1.json；outputs/kaggle_remote/kaggle_clip_semantic_adapter_core_v2_v1_kernel_v3/kaggle_clip_semantic_adapter_core_v2_v1_metrics.json",
            "open_command": f'Start-Process "{ROOT / video}"',
            "viewer_command": f'$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"\n& "{PYTHON}" "{ROOT / "scripts" / "run_clip_semantic_waypoint.py"}" --model "{model}" --task move_leftmost_cube_to_bowl --complexity hard --seed 1900 --episodes 1 --viewer --duration 60 --speed 0.05 {common}',
            "export_command": f'$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"\n& "{PYTHON}" "{ROOT / "scripts" / "export_video.py"}" --method clip_semantic_waypoint --version kaggle_clip_semantic_adapter_core_v2_v1_hard_leftmost --model "{model}" --task move_leftmost_cube_to_bowl --complexity hard --seed 1900 --width 640 --height 480 --fps 30 --frame-stride 8 --output "{ROOT / video}" {common}',
        },
        {
            "section": "Kaggle 远程补充实验",
            "id": "kaggle_clip_semantic_adapter_core_v2_ood_v1",
            "title": "Kaggle 冻结 CLIP 适配器的未见颜色同义词误判",
            "media": "outputs/videos/kaggle_clip_semantic_adapter_core_v2_ood_paraphrase_blue_to_red_seed700.mp4",
            "media_kind": "video",
            "talk_prompt": "先给出完整量化结果：未见英文改写 48/60、hard 多物体干扰 19/20。再播放这一条负例：指令要求 azure 蓝方块到红盘，模型却选择 red cube 到红盘。抓取动作发生不等于任务成功，必须同时报告 semantic_correct=False 和 task_success=False。",
            "paper_redline": "这是冻结 VLM 高层语义分类的失败证据，不是连续动作策略失败，也不能据此声称端到端 VLA、OpenVLA LoRA 或真实机械臂能力。",
            "reference": "docs/kaggle_clip_semantic_adapter_core_v2_v1_report.md；docs/kaggle_clip_semantic_adapter_core_v2_ood_v1_report.md；outputs/evaluations/kaggle_clip_semantic_adapter_core_v2_ood_v1.json",
            "open_command": f'Start-Process "{ROOT / "outputs" / "videos" / "kaggle_clip_semantic_adapter_core_v2_ood_paraphrase_blue_to_red_seed700.mp4"}',
            "viewer_command": f'$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"\n& "{PYTHON}" "{ROOT / "scripts" / "run_clip_semantic_waypoint.py"}" --model "{model}" --task place_blue_cube_red_pad --instruction "put the azure block on the red disk" --complexity medium --seed 700 --episodes 1 --viewer --duration 45 --speed 0.25 {common}',
            "export_command": f'$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"\n& "{PYTHON}" "{ROOT / "scripts" / "export_video.py"}" --method clip_semantic_waypoint --version kaggle_clip_semantic_adapter_core_v2_ood_paraphrase_blue_to_red_v1 --model "{model}" --task place_blue_cube_red_pad --instruction "put the azure block on the red disk" --complexity medium --seed 700 --width 640 --height 480 --fps 30 --frame-stride 8 --output "{ROOT / "outputs" / "videos" / "kaggle_clip_semantic_adapter_core_v2_ood_paraphrase_blue_to_red_seed700.mp4"}" {common}',
        },
    ]


def verify_media(entries: list[dict[str, str]]) -> None:
    missing = []
    for entry in entries:
        path = ROOT / entry["media"]
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("\n".join(missing))


def media_tag(entry: dict[str, str]) -> str:
    src = escape(web_path(entry["media"]))
    title = escape(entry["title"])
    if entry["media_kind"] == "video":
        return f'<video controls preload="metadata" src="{src}"></video>'
    if entry["media_kind"] == "image":
        return f'<img src="{src}" alt="{title}">'
    return f'<a href="{src}">打开文件</a>'


def card(entry: dict[str, str]) -> str:
    viewer = ""
    if entry["viewer_command"]:
        viewer = (
            "<details><summary>完整 viewer 命令</summary>"
            f"<pre>{escape(entry['viewer_command'])}</pre></details>"
        )
    export = ""
    if entry["export_command"]:
        export = (
            "<details><summary>重新导出视频命令</summary>"
            f"<pre>{escape(entry['export_command'])}</pre></details>"
        )
    return f"""
<article class="playlist-card" data-kind="{escape(entry['media_kind'])}">
  <div class="media-shell">{media_tag(entry)}</div>
  <div class="card-body">
    <p class="eyebrow">{escape(entry['section'])}</p>
    <h3><code>{escape(entry['id'])}</code> {escape(entry['title'])}</h3>
    <p><b>讲解提示：</b>{escape(entry['talk_prompt'])}</p>
    <p><b>论文红线：</b>{escape(entry['paper_redline'])}</p>
    <p><b>证据引用：</b>{escape(entry['reference'])}</p>
    <details open><summary>打开命令</summary><pre>{escape(entry['open_command'])}</pre></details>
    {viewer}
    {export}
  </div>
</article>
""".strip()


def build_html(entries: list[dict[str, str]]) -> str:
    claim_entries = [entry for entry in entries if entry["section"].startswith("阶段")]
    ood_entries = [entry for entry in entries if entry["section"].startswith("Core V2 OOD")]
    remote_entries = [entry for entry in entries if entry["section"].startswith("Kaggle 远程")]
    candidate_entries = [entry for entry in entries if entry["section"].startswith("候选")]
    video_count = sum(1 for entry in entries if entry["media_kind"] == "video")
    image_count = sum(1 for entry in entries if entry["media_kind"] == "image")
    claim_cards = "\n".join(card(entry) for entry in claim_entries)
    ood_cards = "\n".join(card(entry) for entry in ood_entries)
    remote_cards = "\n".join(card(entry) for entry in remote_entries)
    candidate_cards = "\n".join(card(entry) for entry in candidate_entries)
    generated_at = datetime.now().isoformat(timespec="seconds")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>答辩视频播放清单</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f6f8;
      --panel: #fff;
      --text: #17202a;
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
      background: #ffffff;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0 0 8px; font-size: clamp(26px, 3vw, 42px); letter-spacing: 0; }}
    .subtitle {{ max-width: 1080px; margin: 0; color: var(--muted); line-height: 1.7; }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
      max-width: 880px;
      margin-top: 18px;
    }}
    .stat {{
      border: 1px solid var(--line);
      background: #f9fafb;
      padding: 12px;
      border-radius: 6px;
    }}
    .stat b {{ display: block; font-size: 22px; color: var(--accent); }}
    main {{ padding: 28px clamp(18px, 4vw, 56px) 48px; }}
    section {{ max-width: 1320px; margin: 0 auto 30px; }}
    h2 {{ margin: 0 0 14px; font-size: 22px; letter-spacing: 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }}
    .playlist-card {{
      display: grid;
      grid-template-rows: auto 1fr;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      min-width: 0;
    }}
    .media-shell {{ background: #101828; aspect-ratio: 16 / 9; display: grid; place-items: center; }}
    video, img {{ width: 100%; height: 100%; object-fit: contain; display: block; }}
    .card-body {{ padding: 14px; min-width: 0; }}
    .eyebrow {{ margin: 0 0 6px; color: var(--accent); font-size: 13px; font-weight: 700; }}
    h3 {{ margin: 0 0 10px; font-size: 17px; line-height: 1.45; letter-spacing: 0; }}
    p {{ line-height: 1.65; margin: 8px 0; color: #344054; }}
    code {{ font-family: Consolas, "Cascadia Mono", monospace; font-size: 0.9em; }}
    details {{ margin-top: 10px; border-top: 1px solid var(--line); padding-top: 8px; }}
    summary {{ cursor: pointer; color: var(--warn); font-weight: 700; }}
    pre {{
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      background: #f2f4f7;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      line-height: 1.45;
      font-size: 12px;
    }}
    footer {{ color: var(--muted); padding: 0 clamp(18px, 4vw, 56px) 32px; }}
    @media (max-width: 720px) {{
      .stats {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>答辩视频播放清单</h1>
    <p class="subtitle">版本：<code>defense_video_playlist_v1</code>。按 claim 叙事顺序和候选诊断顺序集中播放当前 MuJoCo 实验包的视频证据；视频是定性展示，成功率、目标距离、资源和泛化结论仍以 CSV/JSON 评测表为准。</p>
    <div class="stats">
      <div class="stat"><b>{len(claim_entries)}</b><span>claim 入口</span></div>
      <div class="stat"><b>{len(ood_entries)}</b><span>OOD 对照</span></div>
      <div class="stat"><b>{len(remote_entries)}</b><span>Kaggle 补充</span></div>
      <div class="stat"><b>{len(candidate_entries)}</b><span>候选诊断</span></div>
      <div class="stat"><b>{video_count}</b><span>视频片段</span></div>
      <div class="stat"><b>{image_count}</b><span>图像证据</span></div>
    </div>
  </header>
  <main>
    <section>
      <h2>1. 阶段 claim 播放顺序</h2>
      <div class="grid">{claim_cards}</div>
    </section>
    <section>
      <h2>2. Core V2 OOD 对照播放顺序</h2>
      <div class="grid">{ood_cards}</div>
    </section>
    <section>
      <h2>3. Kaggle 远程补充实验</h2>
      <div class="grid">{remote_cards}</div>
    </section>
    <section>
      <h2>4. 候选诊断负例播放顺序</h2>
      <div class="grid">{candidate_cards}</div>
    </section>
  </main>
  <footer>生成时间：{escape(generated_at)}</footer>
</body>
</html>
"""


def build_md(entries: list[dict[str, str]]) -> str:
    lines = [
        "# 答辩视频播放清单",
        "",
        "版本：`defense_video_playlist_v1`",
        "",
        "用途：把 claim 级阶段短片和 trajectory/ACT 候选诊断片段集中成中文播放清单。视频只作为定性展示证据，不替代成功率、目标距离、资源消耗和语言/空间泛化表。",
        "",
        "## 1. 打开 HTML 播放页",
        "",
        "```powershell",
        f'Start-Process "{ROOT / "docs" / "defense_video_playlist.html"}"',
        "```",
        "",
        "## 2. 重建命令",
        "",
        "```powershell",
        f'& "{PYTHON}" "{ROOT / "scripts" / "build_defense_video_playlist.py"}"',
        "```",
        "",
    ]
    for section in ("阶段 claim 播放顺序", "Core V2 OOD 对照播放顺序", "Kaggle 远程补充实验", "候选诊断负例播放顺序"):
        lines += [f"## {section}", ""]
        for entry in [item for item in entries if item["section"] == section]:
            lines += [
                f"### `{entry['id']}` {entry['title']}",
                "",
                f"- 媒体文件：`{entry['media']}`",
                f"- 媒体类型：`{entry['media_kind']}`",
                f"- 证据引用：{entry['reference']}",
                f"- 讲解提示：{entry['talk_prompt']}",
                f"- 论文红线：{entry['paper_redline']}",
                "",
                "打开命令：",
                "",
                "```powershell",
                entry["open_command"],
                "```",
                "",
            ]
            if entry["viewer_command"]:
                lines += [
                    "完整 viewer 命令：",
                    "",
                    "```powershell",
                    entry["viewer_command"],
                    "```",
                    "",
                ]
            if entry["export_command"]:
                lines += [
                    "重新导出视频命令：",
                    "",
                    "```powershell",
                    entry["export_command"],
                    "```",
                    "",
                ]
    lines.append(f"生成时间：{datetime.now().isoformat(timespec='seconds')}")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, entries: list[dict[str, str]]) -> None:
    columns = [
        "section",
        "id",
        "title",
        "media",
        "media_kind",
        "talk_prompt",
        "paper_redline",
        "reference",
        "open_command",
        "viewer_command",
        "export_command",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(entries)


def main() -> None:
    args = parse_args()
    claim_rows = read_csv(args.claim_index)
    candidate_rows = read_csv(args.candidate_index)
    entries = (
        [normalize_claim(row) for row in claim_rows]
        + core_v2_ood_entries()
        + kaggle_remote_entries()
        + [candidate_overview_entry(len(candidate_rows))]
        + [normalize_candidate(row) for row in candidate_rows]
    )
    verify_media(entries)
    args.output_html.write_text(build_html(entries), encoding="utf-8")
    args.output_md.write_text(build_md(entries), encoding="utf-8")
    write_csv(args.output_csv, entries)
    print(f"defense_video_playlist_html: {args.output_html}", flush=True)
    print(f"defense_video_playlist_md: {args.output_md}", flush=True)
    print(f"defense_video_playlist_csv: {args.output_csv}", flush=True)
    print(f"playlist_entries: {len(entries)}", flush=True)


if __name__ == "__main__":
    main()
