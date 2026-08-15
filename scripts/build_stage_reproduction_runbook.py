from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "stage_reproduction_runbook_v1"


REPRESENTATIVE_VIEWERS = {
    "任务/数据/普通 BC": [
        "expert_scripted_v1",
        "structured_waypoint_policy_v1",
        "linear_bc_v1",
        "knn_bc_v1",
        "mlp_bc_v1",
    ],
    "Trajectory / ACT / Diffusion": [
        "trajectory_conditioned_chunk_bc_v2",
        "trajectory_knn_chunk_bc_v1",
        "torch_act_state_chunk_v1",
        "torch_act_cvae_state_chunk_v1",
        "visual_act_cnn_cvae_v1",
        "torch_diffusion_policy_state_chunk_v1",
    ],
    "Action-Head / PEFT / CLIP": [
        "object_language_action_head_lite_v1",
        "adapter_action_head_lite_v1",
        "lora_action_head_lite_v1",
        "clip_action_head_lite_v1",
        "multi_task_object_action_head_lite_v1",
    ],
    "语言/空间泛化": [
        "structured_waypoint_policy_v1",
        "torch_act_cvae_state_chunk_v1",
        "clip_action_head_lite_v1",
        "multi_task_object_action_head_lite_v1",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese stage-level reproduction runbook.")
    parser.add_argument("--stage-evidence", type=Path, default=ROOT / "docs" / "stage_evidence_index.csv")
    parser.add_argument("--task-bc-stage", type=Path, default=ROOT / "docs" / "task_bc_stage_report.csv")
    parser.add_argument("--trajectory-act-stage", type=Path, default=ROOT / "docs" / "trajectory_act_stage_report.csv")
    parser.add_argument("--action-head-stage", type=Path, default=ROOT / "docs" / "action_head_stage_report.csv")
    parser.add_argument("--video-quality-audit", type=Path, default=ROOT / "docs" / "video_quality_audit.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "stage_reproduction_runbook.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "stage_reproduction_runbook.csv")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def q(path: Path) -> str:
    return f'"{path}"'


def ps_command(script: str) -> str:
    return f'& "{PYTHON}" "{ROOT / script}"'


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", "<br>") for value in values) + " |"


def split_refs(value: str) -> list[str]:
    refs: list[str] = []
    for part in value.replace("；", "\n").replace("、", "\n").splitlines():
        item = part.strip().strip("`")
        if item:
            refs.append(item)
    return refs


def collect_methods(*groups: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    methods: dict[str, dict[str, str]] = {}
    for rows in groups:
        for row in rows:
            version = row.get("版本", "")
            if version:
                methods[version] = row
    return methods


def make_summary_rows(stage_rows: list[dict[str, str]], methods: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for stage in stage_rows:
        stage_name = stage["阶段名称"]
        versions = REPRESENTATIVE_VIEWERS.get(stage_name, [])
        viewer_count = sum(1 for version in versions if methods.get(version, {}).get("主任务viewer命令"))
        language_count = sum(1 for version in versions if methods.get(version, {}).get("语言viewer命令"))
        rows.append(
            {
                "阶段编号": stage["阶段编号"],
                "阶段名称": stage_name,
                "覆盖数量": stage["覆盖数量"],
                "关键版本": stage["关键版本"],
                "代表viewer版本": "；".join(versions) if versions else "见展示入口",
                "主任务viewer命令数": str(viewer_count),
                "语言viewer命令数": str(language_count),
                "量化证据": stage["量化证据"],
                "视频证据": stage["视频证据"],
                "展示入口": stage["展示入口"],
                "论文红线": stage["论文红线"],
                "重建命令": stage["重建命令"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_open_commands(lines: list[str], refs: list[str]) -> None:
    paths = [ROOT / ref for ref in refs if ref.startswith(("docs/", "outputs/"))]
    if not paths:
        return
    lines.extend(["", "打开相关文件：", "", "```powershell"])
    for path in paths[:6]:
        lines.append(f"Start-Process {q(path)}")
    lines.append("```")


def append_viewer_commands(lines: list[str], stage_name: str, methods: dict[str, dict[str, str]]) -> None:
    versions = REPRESENTATIVE_VIEWERS.get(stage_name, [])
    if not versions:
        return

    if stage_name == "语言/空间泛化":
        command_key = "语言viewer命令"
        title = "代表语言/空间泛化 viewer 命令"
    else:
        command_key = "主任务viewer命令"
        title = "代表主任务 viewer 命令"

    lines.extend(["", title, "", "说明：完整单方法 viewer 命令见 `docs/reproducible_command_index.md`。", ""])
    for version in versions:
        row = methods.get(version, {})
        command = row.get(command_key, "")
        if not command:
            continue
        method = row.get("方法", version)
        lines.extend([f"#### `{version}` / {method}", "", "```powershell", command, "```", ""])


def write_md(
    path: Path,
    stage_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    methods: dict[str, dict[str, str]],
    video_quality_rows: list[dict[str, str]],
) -> None:
    passed_videos = sum(1 for row in video_quality_rows if row.get("审计状态") == "通过")
    lines = [
        "# 阶段复现实验手册",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：把当前 MuJoCo 实验包按阶段整理成可操作的中文复现手册。它服务三个目标：保留各阶段版本名称、快速找到量化评测与视频证据、给出可直接复制的 viewer/重建命令。",
        "",
        "完整单方法命令仍以 `docs/reproducible_command_index.md` 为准；本手册提供阶段级入口和代表命令。所有 viewer 命令统一使用 `--viewer --duration 60 --speed 0.05` 以便观察夹爪接触、抬升和放置过程。",
        "",
        "## 0. 总入口",
        "",
        "```powershell",
        f"Start-Process {q(ROOT / 'docs' / 'stage_showcase_index.html')}",
        f"Start-Process {q(ROOT / 'docs' / 'video_evidence_gallery.html')}",
        f"Start-Process notepad.exe {q(ROOT / 'docs' / 'reproducible_command_index.md')}",
        f"Start-Process notepad.exe {q(ROOT / 'docs' / 'video_quality_audit.md')}",
        "```",
        "",
        "完整验证：",
        "",
        "```powershell",
        ps_command("scripts/verify_experiment_artifacts.py"),
        "```",
        "",
        "视频展示质量审计：",
        "",
        f"- 版本：`video_quality_audit_v1`",
        f"- 已通过视频：{passed_videos}/{len(video_quality_rows)}",
        "- 审计文件：`docs/video_quality_audit.md`、`docs/video_quality_audit.csv`",
        "- 边界：视频质量审计不是成功率评测。",
        "",
        "## 1. 阶段总览",
        "",
        md_row(["阶段", "覆盖", "关键版本", "量化证据", "视频证据", "展示入口"]),
        md_row(["---", "---", "---", "---", "---", "---"]),
    ]
    for row in summary_rows:
        lines.append(
            md_row(
                [
                    f"{row['阶段编号']}. {row['阶段名称']}",
                    row["覆盖数量"],
                    row["关键版本"],
                    row["量化证据"],
                    row["视频证据"],
                    row["展示入口"],
                ]
            )
        )

    lines.extend(["", "## 2. 分阶段复现", ""])
    for stage, summary in zip(stage_rows, summary_rows):
        stage_name = stage["阶段名称"]
        lines.extend(
            [
                f"### 阶段 {stage['阶段编号']}：{stage_name}",
                "",
                f"- 覆盖数量：{stage['覆盖数量']}",
                f"- 关键版本：{stage['关键版本']}",
                f"- 可写结论：{stage['论文可写结论']}",
                f"- 论文红线：{stage['论文红线']}",
                f"- 推荐讲解：{stage['推荐讲解']}",
                "",
                "量化证据：",
                "",
                "```text",
                stage["量化证据"],
                "```",
                "",
                "视频证据：",
                "",
                "```text",
                stage["视频证据"],
                "```",
                "",
                "重建或评测命令：",
                "",
                "```powershell",
                stage["重建命令"],
                "```",
            ]
        )
        append_open_commands(lines, split_refs(stage["展示入口"]) + split_refs(stage["视频证据"]))
        append_viewer_commands(lines, stage_name, methods)

    lines.extend(
        [
            "## 3. 使用边界",
            "",
            "1. 该手册是复现入口，不新增策略方法，也不改变任何评测结果。",
            "2. 展示视频用于说明现象，成功率、目标距离、语言泛化和资源规模仍以 CSV 评测表为准。",
            "3. 当前完成的是 MuJoCo 实验包；真实 OpenVLA/机器人 VLA、Isaac domain randomization 和真实 WidowX 验证仍在下一阶段。",
            "",
            "## 4. 重建命令",
            "",
            "```powershell",
            ps_command("scripts/build_stage_reproduction_runbook.py"),
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    stage_rows = read_csv(args.stage_evidence)
    task_rows = read_csv(args.task_bc_stage)
    trajectory_rows = read_csv(args.trajectory_act_stage)
    action_rows = read_csv(args.action_head_stage)
    video_quality_rows = read_csv(args.video_quality_audit)
    methods = collect_methods(task_rows, trajectory_rows, action_rows)
    summary_rows = make_summary_rows(stage_rows, methods)
    write_csv(args.output_csv, summary_rows)
    write_md(args.output_md, stage_rows, summary_rows, methods, video_quality_rows)
    print(f"stage_reproduction_runbook_md: {args.output_md}", flush=True)
    print(f"stage_reproduction_runbook_csv: {args.output_csv}", flush=True)
    print(f"stage_reproduction_rows: {len(summary_rows)}", flush=True)


if __name__ == "__main__":
    main()
