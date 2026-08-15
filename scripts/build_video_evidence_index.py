from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese video evidence index for registered MuJoCo rollouts.")
    parser.add_argument("--versions", type=Path, default=ROOT / "docs" / "experiment_versions.json")
    parser.add_argument("--language-summary", type=Path, default=ROOT / "docs" / "language_generalization_summary.csv")
    parser.add_argument("--method-audit", type=Path, default=ROOT / "docs" / "method_stage_audit.csv")
    parser.add_argument("--videos", type=Path, default=ROOT / "outputs" / "videos")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "video_evidence_index.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "video_evidence_index.csv")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def by_key(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row[key]: row for row in rows if row.get(key)}


def metadata_for(video_path: Path) -> dict:
    metadata_path = video_path.with_suffix(".json")
    if not metadata_path.exists():
        return {}
    return read_json(metadata_path)


def normalize_video_path(path_text: str, videos_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def metric(summary: dict, key: str) -> str:
    value = summary.get(key)
    if value is None:
        metrics = summary.get("metrics", {})
        value = metrics.get(key)
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def success_text(summary: dict, fallback: str = "") -> str:
    if "success" in summary:
        return "success=True" if bool(summary["success"]) else "success=False"
    if "steps_replayed" in summary:
        return f"replay {summary['steps_replayed']} steps"
    if fallback:
        return fallback
    return "not_applicable"


def evidence_role(video_type: str, success: str) -> str:
    if "replay" in success:
        return "数据可复现证据"
    if video_type == "候选诊断片段":
        return "候选方法诊断证据"
    if success == "success=True":
        return "成功样例"
    if success == "success=False":
        return "失败模式样例"
    if video_type == "阶段汇总视频":
        return "阶段讲解总览"
    return "定性展示证据"


def video_duration(metadata: dict) -> str:
    frames = metadata.get("frames")
    fps = metadata.get("fps")
    if not frames or not fps:
        return "-"
    try:
        return f"{float(frames) / float(fps):.1f}s"
    except (TypeError, ValueError, ZeroDivisionError):
        return "-"


def make_row(
    video_type: str,
    version: str,
    method: str,
    stage: str,
    video_path: Path,
    audit_row: dict[str, str] | None = None,
    fallback_success: str = "",
) -> dict[str, str]:
    metadata = metadata_for(video_path)
    summary = metadata.get("summary", {})
    success = success_text(summary, fallback=fallback_success)
    task = metadata.get("task", summary.get("task", "-"))
    complexity = metadata.get("complexity", summary.get("complexity", "-"))
    seed = metadata.get("seed", summary.get("seed", "-"))
    instruction = metadata.get("instruction", summary.get("instruction", "-"))
    active_objects = ", ".join(metadata.get("active_objects", summary.get("active_objects", []))) or "-"
    row = {
        "视频类型": video_type,
        "版本": version,
        "方法": method,
        "阶段": stage,
        "任务": str(task),
        "复杂度": str(complexity),
        "seed": str(seed),
        "指令": str(instruction),
        "活动物体": active_objects,
        "结果": success,
        "目标距离": metric(summary, "target_distance"),
        "末端到物体距离": metric(summary, "ee_object_distance"),
        "物体高度": metric(summary, "object_z"),
        "抓取标志": metric(summary, "grasp_success"),
        "帧数": str(metadata.get("frames", "-")),
        "fps": str(metadata.get("fps", "-")),
        "时长": video_duration(metadata),
        "视频文件": rel(video_path),
        "元数据文件": rel(video_path.with_suffix(".json")),
        "证据用途": evidence_role(video_type, success),
        "论文红线": (audit_row or {}).get("论文红线", "仅作定性视频证据，量化结论以 CSV 评测表为准"),
    }
    return row


def language_video(version: str, videos_dir: Path) -> Path | None:
    candidates = sorted(videos_dir.glob(f"{version}_language_seed*.mp4"))
    if candidates:
        return candidates[0]
    if version == "expert_scripted_language_v1":
        candidate = videos_dir / "expert_scripted_language_v1_seed200.mp4"
        return candidate if candidate.exists() else None
    return None


def build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    methods = read_json(args.versions)["methods"]
    audit_by_version = by_key(read_csv(args.method_audit), "版本")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    for method in methods:
        video_path = normalize_video_path(method["clip"], args.videos)
        audit_row = audit_by_version.get(method["version"])
        rows.append(
            make_row(
                "主任务固定片段",
                method["version"],
                method["method"],
                method["stage"],
                video_path,
                audit_row=audit_row,
                fallback_success=method.get("train_range_success", ""),
            )
        )
        seen.add(video_path.resolve().as_posix().lower())

    for language in read_csv(args.language_summary):
        version = language["version"]
        video_path = language_video(version, args.videos)
        if not video_path:
            continue
        key = video_path.resolve().as_posix().lower()
        if key in seen:
            continue
        audit_row = audit_by_version.get(version)
        rows.append(
            make_row(
                "语言/空间泛化片段",
                version,
                language.get("method_key", version),
                language.get("stage", "language_generalization"),
                video_path,
                audit_row=audit_row,
                fallback_success=language.get("success", ""),
            )
        )
        seen.add(key)

    for video_path in sorted(args.videos.glob("*.mp4")):
        stem = video_path.stem
        if "smoke" in stem:
            continue
        key = video_path.resolve().as_posix().lower()
        if key in seen:
            continue
        metadata = metadata_for(video_path)
        version = metadata.get("version", stem)
        is_candidate = "candidate" in version or "candidate" in stem
        rows.append(
            make_row(
                "候选诊断片段" if is_candidate else "补充片段",
                version,
                metadata.get("method", "-"),
                "candidate_method_diagnosis" if is_candidate else "extra_video_evidence",
                video_path,
                fallback_success=success_text(metadata.get("summary", {})),
            )
        )
        seen.add(key)

    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def count_by(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row[key]] = counts.get(row[key], 0) + 1
    return counts


def md_escape(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(md_escape(value) for value in values) + " |"


def write_md(path: Path, rows: list[dict[str, str]]) -> None:
    type_counts = count_by(rows, "视频类型")
    lines = [
        "# 视频证据索引",
        "",
        "版本：`video_evidence_index_v1`",
        "",
        "用途：集中整理当前所有正式 MuJoCo rollout 视频、语言/空间泛化视频和补充成功/失败片段。该文档服务论文说明、答辩演示和后续方法比较；量化结论仍以 CSV 评测表为准。",
        "",
        "## 1. 索引统计",
        "",
        md_row(["视频类型", "数量"]),
        md_row(["---", "---:"]),
    ]
    for video_type, count in type_counts.items():
        lines.append(md_row([video_type, str(count)]))

    lines.extend(
        [
            "",
            "## 2. 推荐展示入口",
            "",
            "```text",
            "docs/video_evidence_gallery.html",
            "docs/defense_deck.html",
            "docs/presentation_video_pack.md",
            "docs/video_presentation_storyboard.html",
            "outputs/presentation_clips/00_defense_video_reel.mp4",
            "outputs/showcase/all_registered_methods_grid.mp4",
            "outputs/showcase/language_generalization_grid.mp4",
            "```",
            "",
            "## 3. 视频明细",
            "",
            md_row(["类型", "版本", "阶段", "任务", "seed", "结果", "目标距离", "时长", "视频文件", "证据用途"]),
            md_row(["---", "---", "---", "---", "---:", "---", "---:", "---:", "---", "---"]),
        ]
    )
    for row in rows:
        lines.append(
            md_row(
                [
                    row["视频类型"],
                    f"`{row['版本']}`",
                    row["阶段"],
                    row["任务"],
                    row["seed"],
                    row["结果"],
                    row["目标距离"],
                    row["时长"],
                    f"`{row['视频文件']}`",
                    row["证据用途"],
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 4. 使用边界",
            "",
            "- 单方法 mp4 是定性证据，用于说明动作轨迹、接触状态、失败模式和成功样例。",
            "- 成功率、平均距离、参数量、训练时间和显存应引用 `docs/evaluation_summary.csv`、`docs/language_generalization_summary.csv`、`docs/model_resource_summary.csv` 和 `docs/stage_comparison_report.md`。",
            "- CLIP、Adapter、LoRA-style、reward-weighted 和 action-head 片段都只能写成本地 proxy，不能写成真实 OpenVLA/RT-2/机器人 VLA 后训练结果。",
            "- ACT-style 和 Diffusion-style 片段只能写成 state-only 或轻量代理 baseline，不能写成完整视觉 ACT 或完整视觉 Diffusion Policy。",
            "",
            "## 5. 重建命令",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_video_evidence_index.py"}"',
            "```",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows)
    print(f"video_evidence_md: {args.output_md}", flush=True)
    print(f"video_evidence_csv: {args.output_csv}", flush=True)
    print(f"video_rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
