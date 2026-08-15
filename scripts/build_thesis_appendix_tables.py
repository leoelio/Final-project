from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-ready appendix tables from current experiment artifacts.")
    parser.add_argument("--versions", type=Path, default=ROOT / "docs" / "experiment_versions.json")
    parser.add_argument("--evaluation", type=Path, default=ROOT / "docs" / "evaluation_summary.csv")
    parser.add_argument("--language", type=Path, default=ROOT / "docs" / "language_generalization_summary.csv")
    parser.add_argument("--resources", type=Path, default=ROOT / "docs" / "model_resource_summary.csv")
    parser.add_argument("--method-audit", type=Path, default=ROOT / "docs" / "method_stage_audit.csv")
    parser.add_argument("--domain-randomization", type=Path, default=ROOT / "docs" / "domain_randomization_summary.csv")
    parser.add_argument("--video-evidence", type=Path, default=ROOT / "docs" / "video_evidence_index.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "thesis_appendix_tables.md")
    parser.add_argument("--method-output-csv", type=Path, default=ROOT / "docs" / "thesis_method_comparison_table.csv")
    parser.add_argument("--domain-output-csv", type=Path, default=ROOT / "docs" / "thesis_domain_randomization_table.csv")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def by_version(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["version"]: row for row in rows if row.get("version")}


def by_chinese_version(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["版本"]: row for row in rows if row.get("版本")}


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pick(row: dict[str, str] | None, key: str, default: str = "") -> str:
    if not row:
        return default
    value = row.get(key, "")
    return value if value not in ("", None) else default


def rate_value(value: str) -> float:
    if not value or "/" not in value:
        return -1.0
    left, right = value.split("/", 1)
    try:
        total = float(right)
        return float(left) / total if total else -1.0
    except ValueError:
        return -1.0


def rate_text(value: str) -> str:
    parsed = rate_value(value)
    if parsed < 0:
        return value or "未记录"
    return f"{value} ({parsed:.0%})"


def method_group(stage: str) -> str:
    if stage in {"scripted_oracle", "data_verification", "structured_control_baseline"}:
        return "环境、示范与结构化强对照"
    if "bc" in stage or "trajectory" in stage:
        return "普通 BC / trajectory baseline"
    if "act" in stage or "diffusion" in stage:
        return "ACT / Diffusion baseline"
    if "action_head" in stage or "vlm" in stage or "peft" in stage or "reward" in stage or "multi_task" in stage:
        return "轻量 VLA / action-head proxy"
    return "其他"


def language_success_for(version: str, language_by_version: dict[str, dict[str, str]]) -> str:
    if version == "expert_scripted_v1":
        return pick(language_by_version.get("expert_scripted_language_v1"), "success", "未评测")
    return pick(language_by_version.get(version), "success", "未评测")


def build_method_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    versions = read_json(args.versions)
    evaluation = by_version(read_csv(args.evaluation))
    language = by_version(read_csv(args.language))
    resources = by_version(read_csv(args.resources))
    audit = by_chinese_version(read_csv(args.method_audit))
    video_rows = read_csv(args.video_evidence)
    video_counts: dict[str, int] = {}
    for row in video_rows:
        video_counts[row["版本"]] = video_counts.get(row["版本"], 0) + 1

    rows = []
    for method in versions["methods"]:
        version = method["version"]
        eval_row = evaluation.get(version, method)
        resource = resources.get(version)
        audit_row = audit.get(version)
        language_success = language_success_for(version, language)
        rows.append(
            {
                "版本": version,
                "阶段": method["stage"],
                "阶段分组": pick(audit_row, "阶段分组", method_group(method["stage"])),
                "方法": method["method"],
                "方法性质": pick(audit_row, "方法性质", ""),
                "训练方式": pick(audit_row, "训练方式", ""),
                "主任务训练范围": pick(eval_row, "train_range_success", method.get("train_range_success", "")),
                "主任务留出范围": pick(eval_row, "heldout_success", method.get("heldout_success", "")),
                "语言/空间泛化": language_success,
                "可训练参数": pick(resource, "trainable_params", "0"),
                "训练时间秒": pick(resource, "train_time_seconds", ""),
                "峰值显存MB": pick(resource, "peak_vram_mb", ""),
                "artifact": method["artifact"],
                "固定视频": method["clip"],
                "视频证据数量": str(video_counts.get(version, 0)),
                "论文可写结论": pick(audit_row, "论文可写", method.get("note", "")),
                "论文红线": pick(audit_row, "论文红线", "不能越过当前本地 MuJoCo proxy 的证据范围"),
            }
        )
    return rows


def build_domain_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_csv(args.domain_randomization)
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["method_key"], row["method_version"], row["domain"])
        grouped.setdefault(key, []).append(row)

    output = []
    for (method_key, method_version, domain), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][2])):
        success = sum(1 for item in items if item.get("success") == "True")
        total = len(items)
        mean_distance = sum(float(item["target_distance"]) for item in items) / total
        first = items[0]
        output.append(
            {
                "评测版本": "domain_randomization_eval_v1",
                "方法": method_key,
                "方法版本": method_version,
                "扰动域": domain,
                "成功率": f"{success}/{total}",
                "成功率数值": f"{success / total:.3f}",
                "平均目标距离": f"{mean_distance:.6f}",
                "摩擦": first["friction"],
                "arm_kp": first["arm_kp"],
                "arm_force": first["arm_force"],
                "gripper_kp": first["gripper_kp"],
                "gripper_force": first["gripper_force"],
                "论文红线": "MuJoCo 代理评测，不是 Isaac domain randomization 或真实机械臂迁移验证",
            }
        )
    return output


def md_escape(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(md_escape(value) for value in values) + " |"


def compact_method_table(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        md_row(["版本", "阶段分组", "train", "held-out", "language", "参数", "视频数", "论文红线"]),
        md_row(["---", "---", "---:", "---:", "---:", "---:", "---:", "---"]),
    ]
    for row in rows:
        lines.append(
            md_row(
                [
                    f"`{row['版本']}`",
                    row["阶段分组"],
                    rate_text(row["主任务训练范围"]),
                    rate_text(row["主任务留出范围"]),
                    rate_text(row["语言/空间泛化"]),
                    row["可训练参数"] or "0",
                    row["视频证据数量"],
                    row["论文红线"],
                ]
            )
        )
    return lines


def compact_domain_table(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        md_row(["方法版本", "扰动域", "成功率", "平均目标距离", "摩擦", "arm force", "gripper force", "边界"]),
        md_row(["---", "---", "---:", "---:", "---:", "---:", "---:", "---"]),
    ]
    for row in rows:
        lines.append(
            md_row(
                [
                    f"`{row['方法版本']}`",
                    row["扰动域"],
                    row["成功率"],
                    row["平均目标距离"],
                    row["摩擦"],
                    row["arm_force"],
                    row["gripper_force"],
                    row["论文红线"],
                ]
            )
        )
    return lines


def write_md(path: Path, method_rows: list[dict[str, str]], domain_rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    groups: dict[str, int] = {}
    for row in method_rows:
        groups[row["阶段分组"]] = groups.get(row["阶段分组"], 0) + 1

    video_total = sum(int(row["视频证据数量"]) for row in method_rows)
    lines = [
        "# 论文附录结果表",
        "",
        "版本：`thesis_appendix_tables_v1`",
        "",
        "用途：把当前正式方法版本、主任务结果、语言/空间泛化、资源统计、domain randomization 代理评测和视频证据统一整理成论文附录可引用的表格。该文件不新增实验结论，只聚合现有 CSV 和 JSON 证据。",
        "",
        "## 1. 覆盖统计",
        "",
        md_row(["项目", "数量"]),
        md_row(["---", "---:"]),
        md_row(["正式方法版本", str(len(method_rows))]),
        md_row(["阶段分组", str(len(groups))]),
        md_row(["主方法视频证据计数", str(video_total)]),
        md_row(["Domain randomization 汇总行", str(len(domain_rows))]),
        "",
        "阶段分组计数：",
        "",
        md_row(["阶段分组", "方法数"]),
        md_row(["---", "---:"]),
    ]
    for group, count in sorted(groups.items()):
        lines.append(md_row([group, str(count)]))

    lines.extend(
        [
            "",
            "## 2. 方法结果总表",
            "",
            *compact_method_table(method_rows),
            "",
            "## 3. Domain Randomization 代理汇总",
            "",
            *compact_domain_table(domain_rows),
            "",
            "## 4. 可引用 CSV",
            "",
            "```text",
            args.method_output_csv.relative_to(ROOT).as_posix(),
            args.domain_output_csv.relative_to(ROOT).as_posix(),
            "docs/video_evidence_gallery.html",
            "```",
            "",
            "## 5. 使用边界",
            "",
            "- 本表中的 `domain_randomization_eval_v1` 是 MuJoCo 代理评测，不能写成 Isaac 或真实机械臂迁移验证。",
            "- `clip_action_head_lite_v1`、`adapter_action_head_lite_v1`、`lora_action_head_lite_v1` 是本地 proxy，不能写成真实 OpenVLA/RT-2 后训练。",
            "- `torch_act_*` 和 `torch_diffusion_policy_state_chunk_v1` 是 state-only 或轻量 baseline，不能写成完整视觉 ACT / Diffusion Policy。",
            "- 视频证据只支持定性说明；成功率、目标距离、资源数值应以 CSV 表为准。",
            "",
            "## 6. 重建命令",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_thesis_appendix_tables.py"}"',
            "```",
            "",
            f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    method_rows = build_method_rows(args)
    domain_rows = build_domain_rows(args)
    method_fields = list(method_rows[0].keys())
    domain_fields = list(domain_rows[0].keys())
    write_csv(args.method_output_csv, method_rows, method_fields)
    write_csv(args.domain_output_csv, domain_rows, domain_fields)
    write_md(args.output_md, method_rows, domain_rows, args)
    print(f"thesis_appendix_tables_md: {args.output_md}", flush=True)
    print(f"thesis_method_comparison_csv: {args.method_output_csv}", flush=True)
    print(f"thesis_domain_randomization_csv: {args.domain_output_csv}", flush=True)
    print(f"method_rows: {len(method_rows)}", flush=True)
    print(f"domain_rows: {len(domain_rows)}", flush=True)


if __name__ == "__main__":
    main()
