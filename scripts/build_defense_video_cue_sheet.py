from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "defense_video_cue_sheet_v1"

FIELDNAMES = [
    "顺序",
    "分组",
    "cue_id",
    "标题",
    "媒体类型",
    "媒体文件",
    "建议起点秒",
    "建议终点秒",
    "时长秒",
    "打开命令",
    "备用viewer命令",
    "讲解提示",
    "证据引用",
    "论文红线",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese cue sheet for defense video playback.")
    parser.add_argument("--playlist", type=Path, default=ROOT / "docs" / "defense_video_playlist.csv")
    parser.add_argument("--quality", type=Path, default=ROOT / "docs" / "video_quality_audit.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "defense_video_cue_sheet.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "defense_video_cue_sheet.csv")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def norm_path(value: str) -> str:
    path = Path(value.replace("\\", "/"))
    if path.is_absolute():
        try:
            return path.relative_to(ROOT).as_posix()
        except ValueError:
            return path.as_posix()
    return path.as_posix()


def duration_lookup(rows: list[dict[str, str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for row in rows:
        media = norm_path(row["视频文件"])
        duration = row.get("ffprobe时长秒") or row.get("元数据时长秒") or ""
        if duration:
            lookup[media] = duration
    return lookup


def fmt_seconds(value: str) -> str:
    if not value:
        return "不适用"
    try:
        return f"{float(value):.1f}"
    except ValueError:
        return value


def probe_duration(media: str) -> str:
    path = ROOT / media
    if not path.exists():
        return ""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def cue_end(media_kind: str, media: str, durations: dict[str, str]) -> tuple[str, str]:
    if media_kind != "video":
        return "不适用", "不适用"
    media = norm_path(media)
    duration = durations.get(media, "") or probe_duration(media)
    if duration:
        durations[media] = duration
    if not duration:
        return "0.0", "全片"
    return "0.0", fmt_seconds(duration)


def build_rows(playlist: list[dict[str, str]], quality_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    durations = duration_lookup(quality_rows)
    rows: list[dict[str, str]] = []
    for index, row in enumerate(playlist, start=1):
        media = norm_path(row["media"])
        start, end = cue_end(row["media_kind"], media, durations)
        duration = durations.get(media, "") if row["media_kind"] == "video" else ""
        rows.append(
            {
                "顺序": str(index),
                "分组": row["section"],
                "cue_id": row["id"],
                "标题": row["title"],
                "媒体类型": row["media_kind"],
                "媒体文件": media,
                "建议起点秒": start,
                "建议终点秒": end,
                "时长秒": fmt_seconds(duration) if duration else "不适用",
                "打开命令": row["open_command"],
                "备用viewer命令": row["viewer_command"],
                "讲解提示": row["talk_prompt"],
                "证据引用": row["reference"],
                "论文红线": row["paper_redline"],
            }
        )
    return rows


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(value.replace("|", "\\|").replace("\n", "<br>") for value in values) + " |"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, str]]) -> None:
    video_rows = [row for row in rows if row["媒体类型"] == "video"]
    image_rows = [row for row in rows if row["媒体类型"] == "image"]
    lines = [
        "# 答辩视频 Cue Sheet",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：把答辩播放清单整理成现场可执行的 cue sheet，集中保留播放顺序、建议时间点、打开命令、备用 viewer 命令、讲解提示、证据引用和论文红线。它只组织已有视频和图像，不新增实验结论。",
        "",
        "## 1. 覆盖统计",
        "",
        md_row(["项目", "数量"]),
        md_row(["---", "---:"]),
        md_row(["cue 条目", str(len(rows))]),
        md_row(["视频条目", str(len(video_rows))]),
        md_row(["图像条目", str(len(image_rows))]),
        "",
        "## 2. 现场推荐",
        "",
        "1. 开场使用 `C08` 或 `candidate_diagnostic_montage_v1` 建立整体视觉印象。",
        "2. 正式讲解按 `C01` 到 `C10` 的 claim 顺序播放，再播放 Core V2 OOD 成功/失败对照；追问时再切到候选诊断条目。",
        "3. 每个 cue 的时间点默认从 0 秒开始；若是短片，建议播放全片。",
        "4. 视频只作为定性展示，不能替代成功率、目标距离、资源表、语言泛化表和严格抓取审计。",
        "5. 真实 OpenVLA、Isaac 和真实 WidowX 仍是后续阶段，不能用当前 MuJoCo 视频替代。",
        "",
        "## 3. Cue 总表",
        "",
        md_row(FIELDNAMES),
        md_row(["---"] * len(FIELDNAMES)),
    ]
    for row in rows:
        lines.append(md_row([row[field] for field in FIELDNAMES]))
    lines.extend(
        [
            "",
            "## 4. 重建命令",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_defense_video_cue_sheet.py"}"',
            "```",
            "",
            "## 5. 总体验证命令",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "verify_experiment_artifacts.py"}"',
            "```",
            "",
            f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = build_rows(read_csv(args.playlist), read_csv(args.quality))
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows)
    print(f"defense_video_cue_sheet_md: {args.output_md}", flush=True)
    print(f"defense_video_cue_sheet_csv: {args.output_csv}", flush=True)
    print(f"defense_video_cue_sheet_rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
