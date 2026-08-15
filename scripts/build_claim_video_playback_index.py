from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "claim_video_playback_index_v1"


TALK_PROMPTS = {
    "C01": "先证明环境和数据可复现，再说明 expert、structured waypoint 和 replay 不是 learned policy。",
    "C02": "播放普通 BC 失败和 kNN 轨迹记忆差异，强调简单 BC 对接触、抬升和放置不稳定。",
    "C03": "展示 trajectory-conditioned BC / ACT-style / Diffusion 对照，说明历史观测和动作块接口已建立，但不能写成完整官方 ACT。",
    "C04": "展示 action-head、Adapter、LoRA-style 和 CLIP proxy，强调资源可比但不是真实 pretrained VLA 后训练。",
    "C05": "展示 leftmost-to-bowl 语言/空间任务，强调规则或 frozen feature proxy 不能等同于真实 VLA 语言理解。",
    "C06": "图表为主，视频作为普通 BC 和 action-head 小数据行为的辅助定性证据。",
    "C07": "展示低摩擦和弱夹爪 domain randomization 代理，再打开 Isaac handoff，强调这是 MuJoCo 鲁棒性前置检查和后续 Isaac 回填入口。",
    "C08": "播放总览 reel 或宫格，说明视频是定性证据，不能替代成功率、距离、资源和泛化表。",
    "C09": "打开 OpenVLA bridge 预览图、gallery、handoff 门禁、remote run pack 和 result intake，强调数据桥接、远端运行契约、运行包和回填门禁都不是策略训练结果。",
    "C10": "播放总览 reel，展示方法、阶段、研究问题和 claim 的整体证据链，再打开真实 WidowX handoff 说明后续 trial 模板。",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a claim-to-video playback checklist.")
    parser.add_argument("--claim-evidence", type=Path, default=ROOT / "docs" / "claim_evidence_traceability.csv")
    parser.add_argument("--video-quality-audit", type=Path, default=ROOT / "docs" / "video_quality_audit.csv")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "claim_video_playback_index.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "claim_video_playback_index.md")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", "<br>") for value in values) + " |"


def split_items(value: str) -> list[str]:
    return [part.strip().strip("`") for part in value.replace("；", "\n").splitlines() if part.strip()]


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def ps_open_command(path_text: str, *, notepad: bool = False) -> str:
    path = ROOT / path_text.replace("/", "\\")
    if notepad:
        return f'Start-Process notepad.exe "{path}"'
    return f'Start-Process "{path}"'


def path_exists(path_text: str) -> bool:
    return (ROOT / path_text.replace("/", "\\")).exists()


def helper_commands(paths: list[str]) -> list[str]:
    commands = []
    for path_text in paths:
        suffix = Path(path_text).suffix.lower()
        commands.append(ps_open_command(path_text, notepad=suffix in {".md", ".csv", ".json", ".txt"}))
    return commands


def build_rows(claim_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for claim in claim_rows:
        videos = split_items(claim["video_evidence"])
        helpers = split_items(claim["display_entry"])
        primary = videos[0] if videos else ""
        missing = [path for path in videos + helpers if path.startswith(("docs/", "outputs/", "data/")) and not path_exists(path)]
        status = "可播放（有证据）" if primary and not missing else "需补播放证据"
        rows.append(
            {
                "claim_id": claim["claim_id"],
                "claim_type": claim["claim_type"],
                "primary_video": primary,
                "playback_command": ps_open_command(primary) if primary else "",
                "helper_commands": "；".join(helper_commands(helpers)),
                "quantitative_reference": claim["quantitative_evidence"],
                "talk_prompt": TALK_PROMPTS.get(claim["claim_id"], "按 claim 证据追踪矩阵逐条说明。"),
                "paper_redline": claim["paper_redline"],
                "evidence_status": status,
                "missing_evidence": "无" if not missing else "；".join(missing),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "# Claim 视频播放清单",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：把 `docs/claim_evidence_traceability.csv` 中的每条可写 claim 绑定到首选播放文件、辅助展示入口、讲稿提示、量化引用和论文红线。该清单服务答辩演示，不新增实验结果；视频是定性证据，不能替代成功率、目标距离、语言泛化、资源规模和 domain randomization 表。",
        "",
        "边界：当前完成的是 MuJoCo 实验包；真实 OpenVLA、Isaac 和真实 WidowX 仍必须作为后续阶段单独登记、评测和保存视频。",
        "",
        "## 1. 总览",
        "",
        md_row(["项目", "数量"]),
        md_row(["---", "---:"]),
        md_row(["claim 数", str(len(rows))]),
        md_row(["可播放", str(sum(1 for row in rows if row["evidence_status"] == "可播放（有证据）"))]),
        md_row(["需补播放证据", str(sum(1 for row in rows if row["evidence_status"] != "可播放（有证据）"))]),
        "",
        "## 2. 播放矩阵",
        "",
        md_row(["ID", "类型", "首选播放文件", "打开命令", "量化引用", "讲解提示", "论文红线", "状态"]),
        md_row(["---", "---", "---", "---", "---", "---", "---", "---"]),
    ]
    for row in rows:
        lines.append(
            md_row(
                [
                    row["claim_id"],
                    row["claim_type"],
                    row["primary_video"],
                    f"`{row['playback_command']}`",
                    row["quantitative_reference"],
                    row["talk_prompt"],
                    row["paper_redline"],
                    row["evidence_status"],
                ]
            )
        )

    lines.extend(["", "## 3. 分条播放命令", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row['claim_id']}：{row['claim_type']}",
                "",
                f"- 首选播放文件：`{row['primary_video']}`",
                f"- 量化引用：{row['quantitative_reference']}",
                f"- 讲解提示：{row['talk_prompt']}",
                f"- 论文红线：{row['paper_redline']}",
                "",
                "打开首选证据：",
                "",
                "```powershell",
                row["playback_command"],
                "```",
                "",
            ]
        )
        helpers = row["helper_commands"]
        if helpers:
            lines.extend(
                [
                    "打开辅助入口：",
                    "",
                    "```powershell",
                    *helpers.split("；"),
                    "```",
                    "",
                ]
            )

    lines.extend(
        [
            "## 4. 重建命令",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_claim_video_playback_index.py"}"',
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    claim_rows = read_csv(args.claim_evidence)
    if args.video_quality_audit.exists():
        read_csv(args.video_quality_audit)
    rows = build_rows(claim_rows)
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows)
    print(f"claim_video_playback_index_md: {rel(args.output_md)}", flush=True)
    print(f"claim_video_playback_index_csv: {rel(args.output_csv)}", flush=True)
    print(f"claim_video_playback_rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
