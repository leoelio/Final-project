from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "video_quality_audit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a playable-video quality audit from the video evidence index.")
    parser.add_argument("--video-evidence", type=Path, default=ROOT / "docs" / "video_evidence_index.csv")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "video_quality_audit.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "video_quality_audit.md")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def ffprobe_video(path: Path) -> dict[str, str]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"no video stream found: {path}")
    stream = {key: str(value) for key, value in streams[0].items()}
    if not stream.get("duration"):
        stream["duration"] = str(data.get("format", {}).get("duration", "0"))
    return stream


def parse_seconds(value: str) -> float:
    value = (value or "").strip().lower()
    if value.endswith("s"):
        value = value[:-1]
    return float(value) if value else 0.0


def parse_rate(value: str) -> float:
    if not value:
        return 0.0
    if "/" not in value:
        return float(value)
    numerator, denominator = value.split("/", 1)
    denominator_value = float(denominator)
    if denominator_value == 0:
        return 0.0
    return float(numerator) / denominator_value


def audit_row(row: dict[str, str]) -> dict[str, str]:
    video_path = ROOT / row["视频文件"]
    metadata_path = ROOT / row["元数据文件"]
    stream = ffprobe_video(video_path)

    width = int(float(stream.get("width", "0")))
    height = int(float(stream.get("height", "0")))
    fps = parse_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate", "0"))
    duration = float(stream.get("duration", "0") or 0)
    indexed_duration = parse_seconds(row.get("时长", ""))
    duration_delta = abs(duration - indexed_duration) if indexed_duration else 0.0
    frames = stream.get("nb_frames", "")

    metadata_ok = metadata_path.exists()
    resolution_ok = width >= 320 and height >= 240
    duration_ok = duration >= 1.0 and (not indexed_duration or duration_delta <= 0.75)
    playable_ok = width > 0 and height > 0 and duration > 0 and fps > 0
    status = "通过" if metadata_ok and resolution_ok and duration_ok and playable_ok else "需复查"
    if status == "通过":
        note = "ffprobe 可解码，元数据存在，时长和分辨率满足展示审计门槛。"
    else:
        note = "存在路径、元数据、时长、帧率或分辨率问题，需要重导出该片段。"

    return {
        "视频类型": row["视频类型"],
        "版本": row["版本"],
        "结果": row["结果"],
        "视频文件": row["视频文件"],
        "元数据文件": row["元数据文件"],
        "ffprobe可播放": "是" if playable_ok else "否",
        "元数据存在": "是" if metadata_ok else "否",
        "宽度": str(width),
        "高度": str(height),
        "fps": f"{fps:.2f}",
        "帧数": frames,
        "视频时长秒": f"{duration:.3f}",
        "索引时长秒": f"{indexed_duration:.3f}" if indexed_duration else "",
        "时长偏差秒": f"{duration_delta:.3f}" if indexed_duration else "",
        "时长通过": "是" if duration_ok else "否",
        "分辨率通过": "是" if resolution_ok else "否",
        "审计状态": status,
        "说明": note,
    }


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"


def count_by(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row[key]] = counts.get(row[key], 0) + 1
    return dict(sorted(counts.items()))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, str]], source: Path, output_csv: Path) -> None:
    passed = [row for row in rows if row["审计状态"] == "通过"]
    widths = [int(row["宽度"]) for row in rows]
    heights = [int(row["高度"]) for row in rows]
    durations = [float(row["视频时长秒"]) for row in rows]
    type_counts = count_by(rows, "视频类型")

    lines = [
        "# 视频质量审计",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：逐条检查 `docs/video_evidence_index.csv` 中登记的 MuJoCo 视频是否可播放、是否有对应元数据、时长是否和索引接近、分辨率是否满足展示要求。该文档只用于答辩与论文的视频证据管理，**不是成功率评测**，策略结论仍以 `docs/evaluation_summary.csv`、`docs/language_generalization_summary.csv` 等量化表为准。",
        "",
        "审计工具：`ffprobe`。",
        "",
        "## 1. 总览",
        "",
        md_row(["项目", "结果"]),
        md_row(["---", "---:"]),
        md_row(["来源索引", f"`{rel(source)}`"]),
        md_row(["输出 CSV", f"`{rel(output_csv)}`"]),
        md_row(["登记视频数", str(len(rows))]),
        md_row(["通过审计", str(len(passed))]),
        md_row(["需复查", str(len(rows) - len(passed))]),
        md_row(["分辨率范围", f"{min(widths)}x{min(heights)} 到 {max(widths)}x{max(heights)}"]),
        md_row(["时长范围", f"{min(durations):.2f}s 到 {max(durations):.2f}s"]),
        "",
        "## 2. 类型覆盖",
        "",
        md_row(["视频类型", "数量"]),
        md_row(["---", "---:"]),
    ]
    for video_type, count in type_counts.items():
        lines.append(md_row([video_type, str(count)]))

    lines.extend(
        [
            "",
            "## 3. 明细",
            "",
            md_row(["类型", "版本", "结果", "分辨率", "fps", "时长", "审计状态", "视频文件"]),
            md_row(["---", "---", "---", "---", "---:", "---:", "---", "---"]),
        ]
    )
    for row in rows:
        lines.append(
            md_row(
                [
                    row["视频类型"],
                    f"`{row['版本']}`",
                    row["结果"],
                    f"{row['宽度']}x{row['高度']}",
                    row["fps"],
                    f"{row['视频时长秒']}s",
                    row["审计状态"],
                    f"`{row['视频文件']}`",
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 4. 使用边界",
            "",
            "1. `审计状态=通过` 只说明视频文件可作为展示证据打开和播放。",
            "2. 该审计不判断策略是否成功，也不替代成功率、目标距离、语言泛化和资源消耗表。",
            "3. 若后续重新导出视频，必须先重建 `docs/video_evidence_index.csv`，再重建本审计。",
            "",
            "## 5. 重建命令",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_video_quality_audit.py"}"',
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_rows = read_csv(args.video_evidence)
    rows = [audit_row(row) for row in source_rows]
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows, args.video_evidence, args.output_csv)
    print(f"video_quality_audit_md: {args.output_md}", flush=True)
    print(f"video_quality_audit_csv: {args.output_csv}", flush=True)
    print(f"video_quality_audit_rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
