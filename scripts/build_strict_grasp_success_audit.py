from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION = "strict_grasp_success_audit_v1"
LIFT_Z_THRESHOLD = 0.085


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a strict grasp/lift success audit from completed rollout summaries.")
    parser.add_argument("--control-safety", type=Path, default=ROOT / "docs" / "control_safety_sweep.csv")
    parser.add_argument("--action-head-safety", type=Path, default=ROOT / "docs" / "action_head_control_safety_sweep.csv")
    parser.add_argument("--candidate-diagnostics", type=Path, default=ROOT / "docs" / "candidate_diagnostic_video_index.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "strict_grasp_success_audit.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "strict_grasp_success_audit.csv")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / "strict_grasp_success_audit_v1.json")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "version",
        "source_version",
        "method",
        "preset_or_seed",
        "episodes",
        "loose_successes",
        "loose_success_rate",
        "strict_grasp_successes",
        "strict_grasp_success_rate",
        "mean_target_distance",
        "mean_object_z",
        "grasp_successes",
        "diagnosis",
        "paper_boundary",
        "evidence_path",
        "reproduction_command",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_int(value: str) -> int:
    return int(float(value))


def as_float(value: str) -> float:
    return float(value)


def rate(successes: int, episodes: int) -> float:
    return successes / max(1, episodes)


def diagnosis(loose_successes: int, strict_successes: int, mean_object_z: float) -> str:
    if loose_successes > 0 and strict_successes == 0:
        return "放置距离指标有成功，但没有形成抓取/抬升；更像推放或碰撞到目标附近。"
    if strict_successes > 0:
        return "存在抓取/抬升成功样本，仍需结合目标距离和视频判断是否完成放置。"
    if mean_object_z < LIFT_Z_THRESHOLD:
        return "物体高度未超过抬升阈值，严格抓取口径判为失败。"
    return "未形成可证明的严格抓取成功。"


def paper_boundary(loose_successes: int, strict_successes: int) -> str:
    if loose_successes > 0 and strict_successes == 0:
        return "论文中必须同时报告 success 与 grasp_success/object_z，不能写成稳定抓取成功。"
    if strict_successes > 0:
        return "可以写成存在抓取/抬升样本，但仍不能省略批量成功率和失败模式。"
    return "只能写成该口径下未观察到可靠抓取。"


def sweep_rows(source_version: str, source_path: Path, rows: list[dict[str, str]]) -> list[dict[str, object]]:
    audited = []
    for row in rows:
        episodes = as_int(row["episodes"])
        loose_successes = as_int(row["successes"])
        strict_successes = as_int(row["grasp_successes"])
        mean_object_z = as_float(row["mean_object_z"])
        audited.append(
            {
                "version": VERSION,
                "source_version": source_version,
                "method": row["method"],
                "preset_or_seed": row["preset"],
                "episodes": episodes,
                "loose_successes": loose_successes,
                "loose_success_rate": f"{rate(loose_successes, episodes):.3f}",
                "strict_grasp_successes": strict_successes,
                "strict_grasp_success_rate": f"{rate(strict_successes, episodes):.3f}",
                "mean_target_distance": f"{as_float(row['mean_target_distance']):.4f}",
                "mean_object_z": f"{mean_object_z:.4f}",
                "grasp_successes": strict_successes,
                "diagnosis": diagnosis(loose_successes, strict_successes, mean_object_z),
                "paper_boundary": paper_boundary(loose_successes, strict_successes),
                "evidence_path": source_path.relative_to(ROOT).as_posix(),
                "reproduction_command": row["command"],
            }
        )
    return audited


def parse_bool_token(text: str, key: str) -> bool:
    for token in text.split(","):
        if "=" not in token:
            continue
        name, value = token.strip().split("=", 1)
        if name == key:
            return value == "True"
    return False


def candidate_rows(source_path: Path, rows: list[dict[str, str]]) -> list[dict[str, object]]:
    audited = []
    for row in rows:
        result_text = row["结果"]
        loose_successes = int(parse_bool_token(result_text, "success"))
        strict_successes = int(parse_bool_token(result_text, "grasp_success") and as_float(row["物体高度"]) > LIFT_Z_THRESHOLD)
        mean_object_z = as_float(row["物体高度"])
        audited.append(
            {
                "version": VERSION,
                "source_version": "candidate_diagnostic_video_index_v1",
                "method": row["版本"],
                "preset_or_seed": f"seed{row['seed']}",
                "episodes": 1,
                "loose_successes": loose_successes,
                "loose_success_rate": f"{float(loose_successes):.3f}",
                "strict_grasp_successes": strict_successes,
                "strict_grasp_success_rate": f"{float(strict_successes):.3f}",
                "mean_target_distance": f"{as_float(row['目标距离']):.4f}",
                "mean_object_z": f"{mean_object_z:.4f}",
                "grasp_successes": strict_successes,
                "diagnosis": diagnosis(loose_successes, strict_successes, mean_object_z),
                "paper_boundary": row["论文边界"],
                "evidence_path": row["视频文件"],
                "reproduction_command": row["完整viewer命令"],
            }
        )
    return audited


def md_row(values: list[object]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"


def write_md(path: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    lines = [
        "# 严格抓取成功口径审计",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：把已经完成的 trajectory/ACT、action-head/PEFT 控制限幅扫表和候选诊断视频统一放到一个更严格的成功口径下审计。该审计不改写已有 `success_rate`，而是在论文和答辩中补充 `grasp_success/object_z`，避免把“推到目标盘附近”误写成“稳定抓取并放置”。",
        "",
        "## 口径定义",
        "",
        "- 原始放置成功：沿用环境 `success`，对 place/push 任务主要表示目标物体接近目标区域。",
        f"- 严格抓取成功：沿用环境 `grasp_success`，即物体高度超过 `{LIFT_Z_THRESHOLD:.3f} m` 且靠近末端执行器。",
        "- 论文写法：若 `loose_successes > 0` 但 `strict_grasp_successes = 0`，只能写成“目标距离指标达标但未稳定抓取/抬升”。",
        "- 红线：不能写成稳定抓取成功，除非同时有批量 `success`、`grasp_success`、`object_z` 和视频证据支持。",
        "",
        "## 总结",
        "",
        f"- 审计行数：`{summary['rows']}`",
        f"- 原始放置成功合计：`{summary['loose_successes']}/{summary['episodes']}`",
        f"- 严格抓取成功合计：`{summary['strict_grasp_successes']}/{summary['episodes']}`",
        f"- 存在“放置成功但抓取失败”的行数：`{summary['loose_without_grasp_rows']}`",
        "",
        "结论：当前 learned trajectory/ACT、候选 trajectory 后训练和本地 action-head/PEFT proxy 的主要瓶颈不是 viewer 播放速度，而是接触、夹紧和抬升没有稳定闭环。后续若继续做 ACT 或 VLA 后训练，必须把 `grasp_success`、`object_z` 和视频诊断作为正式指标。",
        "",
        "## 审计表",
        "",
        md_row(["方法/版本", "来源", "preset/seed", "原始成功", "严格抓取成功", "平均目标距离", "平均物体高度", "诊断"]),
        md_row(["---", "---", "---", "---:", "---:", "---:", "---:", "---"]),
    ]
    for row in rows:
        lines.append(
            md_row(
                [
                    f"`{row['method']}`",
                    row["source_version"],
                    row["preset_or_seed"],
                    f"{row['loose_successes']}/{row['episodes']}",
                    f"{row['strict_grasp_successes']}/{row['episodes']}",
                    row["mean_target_distance"],
                    row["mean_object_z"],
                    row["diagnosis"],
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 复现命令",
            "",
            "重新生成本审计：",
            "",
            "```powershell",
            f'& "{ROOT / ".venv" / "Scripts" / "python.exe"}" "{ROOT / "scripts" / "build_strict_grasp_success_audit.py"}"',
            "```",
            "",
            "代表性 viewer 命令仍以各行 `reproduction_command` 和 `docs/stage_reproduction_runbook.md` 为准；观察时统一使用 `--viewer --duration 60 --speed 0.05`。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def build(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    rows.extend(sweep_rows("control_safety_sweep_v1", args.control_safety, read_csv(args.control_safety)))
    rows.extend(sweep_rows("action_head_control_safety_sweep_v1", args.action_head_safety, read_csv(args.action_head_safety)))
    rows.extend(candidate_rows(args.candidate_diagnostics, read_csv(args.candidate_diagnostics)))

    episodes = sum(int(row["episodes"]) for row in rows)
    loose_successes = sum(int(row["loose_successes"]) for row in rows)
    strict_successes = sum(int(row["strict_grasp_successes"]) for row in rows)
    loose_without_grasp_rows = sum(1 for row in rows if int(row["loose_successes"]) > 0 and int(row["strict_grasp_successes"]) == 0)
    summary = {
        "version": VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "lift_z_threshold": LIFT_Z_THRESHOLD,
        "rows": len(rows),
        "episodes": episodes,
        "loose_successes": loose_successes,
        "strict_grasp_successes": strict_successes,
        "loose_success_rate": rate(loose_successes, episodes),
        "strict_grasp_success_rate": rate(strict_successes, episodes),
        "loose_without_grasp_rows": loose_without_grasp_rows,
        "source_files": [
            args.control_safety.relative_to(ROOT).as_posix(),
            args.action_head_safety.relative_to(ROOT).as_posix(),
            args.candidate_diagnostics.relative_to(ROOT).as_posix(),
        ],
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    rows, summary = build(args)
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows, summary)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({"version": VERSION, "summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"strict_grasp_success_audit_md: {args.output_md}", flush=True)
    print(f"strict_grasp_success_audit_csv: {args.output_csv}", flush=True)
    print(f"strict_grasp_success_audit_json: {args.output_json}", flush=True)
    print(f"rows: {summary['rows']}", flush=True)
    print(f"loose_successes: {summary['loose_successes']}/{summary['episodes']}", flush=True)
    print(f"strict_grasp_successes: {summary['strict_grasp_successes']}/{summary['episodes']}", flush=True)


if __name__ == "__main__":
    main()
