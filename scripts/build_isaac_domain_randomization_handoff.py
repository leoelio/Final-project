from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VERSION = "isaac_domain_randomization_handoff_v1"
SOURCE_VERSION = "domain_randomization_eval_v1"
TARGET_VERSION = "isaac_domain_randomization_v1"


SCENE_CONTRACT = {
    "robot": "WidowX / Trossen WX250s MuJoCo 近似桌面机械臂",
    "task": "place_blue_cube_blue_pad",
    "complexity": "medium",
    "instruction": "place the blue cube on the blue pad",
    "target_object": "blue_cube",
    "target_geom": "target_blue_pad",
    "objects": [
        "red_cube",
        "blue_cube",
        "green_cube",
        "yellow_cube",
        "red_cylinder",
        "blue_cylinder",
        "green_ball",
        "target_red_pad",
        "target_blue_pad",
        "target_bowl",
    ],
}


REQUIRED_METRICS = [
    "success",
    "target_distance",
    "grasp_success",
    "object_z",
    "contact_count",
    "sim_to_sim_gap",
    "randomization_domain",
    "seed",
    "method_version",
]


REQUIRED_RETURN_FILES = [
    "outputs/evaluations/isaac_domain_randomization_v1.json",
    "docs/isaac_domain_randomization_summary.md",
    "docs/isaac_domain_randomization_summary.csv",
    "outputs/videos/isaac_domain_randomization_v1_seed0.mp4",
    "outputs/videos/isaac_domain_randomization_v1_seed0.json",
    "docs/isaac_domain_randomization_scene_notes.md",
]


PAPER_BOUNDARY = [
    "可以写成：已完成 Isaac domain randomization 的运行交接门禁。",
    "不能写成 Isaac domain randomization 已完成。",
    "不能写成真实 WidowX 或真实机械臂迁移成功/失败已经验证。",
    "只有 Isaac/Isaac Sim 场景真正运行、返回评测 JSON、CSV、报告和视频后，才能把 isaac_domain_randomization_v1 登记为完成实验。",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Isaac domain randomization handoff gate.")
    parser.add_argument("--source-json", type=Path, default=ROOT / "outputs" / "evaluations" / "domain_randomization_eval_v1.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "isaac_domain_randomization_handoff.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "isaac_domain_randomization_handoff.csv")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / f"{VERSION}.json")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def source_summary(source: dict[str, Any]) -> dict[str, Any]:
    rows = source.get("rows", [])
    domains = sorted({str(row.get("domain")) for row in rows if row.get("domain")})
    methods = sorted({str(row.get("method_version")) for row in rows if row.get("method_version")})
    seeds = sorted({str(row.get("seed")) for row in rows if row.get("seed")})
    return {
        "source_rows": len(rows),
        "source_summary_rows": len(source.get("summary", [])),
        "domains": domains,
        "methods": methods,
        "seeds": seeds,
    }


def build_rows(source: dict[str, Any], summary: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = [
        {
            "category": "source",
            "key": SOURCE_VERSION,
            "source": "outputs/evaluations/domain_randomization_eval_v1.json",
            "required_value": f"{summary['source_rows']} episode rows；{len(summary['domains'])} domains；{len(summary['methods'])} methods",
            "status": "ready",
            "note": "MuJoCo 代理结果可作为 Isaac 复现实验的参数和字段来源。",
        },
        {
            "category": "target",
            "key": TARGET_VERSION,
            "source": VERSION,
            "required_value": "Isaac/Isaac Sim 实际运行后才能登记为完成",
            "status": "waiting_external_runtime",
            "note": "当前本地能力检查中 Isaac/omni/isaacgym 不可用。",
        },
    ]

    for name, spec in source.get("domain_specs", {}).items():
        rows.append(
            {
                "category": "domain",
                "key": str(name),
                "source": SOURCE_VERSION,
                "required_value": (
                    f"arm_kp={spec.get('arm_kp')}；arm_force={spec.get('arm_force')}；"
                    f"gripper_kp={spec.get('gripper_kp')}；gripper_force={spec.get('gripper_force')}；"
                    f"friction={spec.get('friction')}"
                ),
                "status": "mapped",
                "note": str(spec.get("description", "")),
            }
        )

    for key, value in SCENE_CONTRACT.items():
        rows.append(
            {
                "category": "scene_contract",
                "key": key,
                "source": "assets/mujoco/tabletop_wx250s_scene.xml；widowx_env/tabletop_env.py",
                "required_value": "；".join(value) if isinstance(value, list) else str(value),
                "status": "mapped",
                "note": "Isaac 场景应保持同一桌面任务语义。",
            }
        )

    for metric in REQUIRED_METRICS:
        rows.append(
            {
                "category": "required_metric",
                "key": metric,
                "source": VERSION,
                "required_value": "required",
                "status": "required",
                "note": "用于和 MuJoCo 代理结果计算 sim-to-sim gap 或保持字段一致。",
            }
        )

    for file_path in REQUIRED_RETURN_FILES:
        rows.append(
            {
                "category": "required_return_file",
                "key": file_path,
                "source": TARGET_VERSION,
                "required_value": "must_exist_after_external_run",
                "status": "missing_until_isaac_run",
                "note": "远端或本机 Isaac 运行后回填。",
            }
        )

    for boundary in PAPER_BOUNDARY:
        rows.append(
            {
                "category": "paper_boundary",
                "key": boundary,
                "source": VERSION,
                "required_value": "must_follow",
                "status": "locked",
                "note": "论文、答辩和实验日志均需遵守。",
            }
        )

    return rows


def md_row(values: list[object]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in values) + " |"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["category", "key", "source", "required_value", "status", "note"])
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, source: dict[str, Any], summary: dict[str, Any], rows: list[dict[str, str]]) -> None:
    capabilities = source.get("capabilities", {})
    domain_specs = source.get("domain_specs", {})
    lines = [
        "# Isaac Domain Randomization 运行交接门禁",
        "",
        f"版本：`{VERSION}`",
        "",
        f"目标计划版本：`{TARGET_VERSION}`",
        "",
        f"来源版本：`{SOURCE_VERSION}`",
        "",
        "状态：`completed_prerequisite`。该文件只说明 Isaac/Isaac Sim 复现实验如何运行和回填结果，不是 Isaac 评测结果。",
        "",
        "## 1. 当前状态",
        "",
        md_row(["项目", "状态"]),
        md_row(["---", "---"]),
        md_row(["MuJoCo 可用", capabilities.get("mujoco")]),
        md_row(["Isaac Sim 可用", capabilities.get("isaacsim")]),
        md_row(["omni 可用", capabilities.get("omni")]),
        md_row(["isaacgym 可用", capabilities.get("isaacgym")]),
        md_row(["CUDA 可用", capabilities.get("cuda_available")]),
        "",
        f"MuJoCo 源结果覆盖 {summary['source_rows']} 条 episode、{len(summary['methods'])} 个方法版本、{len(summary['domains'])} 个扰动域。它可以作为 Isaac 复现实验的参数模板，但不能写成 Isaac domain randomization 已完成。",
        "",
        "## 2. 场景契约",
        "",
        md_row(["字段", "值"]),
        md_row(["---", "---"]),
    ]
    for key, value in SCENE_CONTRACT.items():
        lines.append(md_row([key, "；".join(value) if isinstance(value, list) else value]))

    lines.extend(
        [
            "",
            "## 3. 扰动域映射",
            "",
            md_row(["域", "摩擦", "arm kp/force", "gripper kp/force", "说明"]),
            md_row(["---", "---:", "---", "---", "---"]),
        ]
    )
    for name, spec in domain_specs.items():
        lines.append(
            md_row(
                [
                    f"`{name}`",
                    spec.get("friction"),
                    f"{spec.get('arm_kp')} / {spec.get('arm_force')}",
                    f"{spec.get('gripper_kp')} / {spec.get('gripper_force')}",
                    spec.get("description", ""),
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 4. Isaac 回填必须指标",
            "",
            "```text",
            *REQUIRED_METRICS,
            "```",
            "",
            "## 5. Isaac 回填必须文件",
            "",
            "```text",
            *REQUIRED_RETURN_FILES,
            "```",
            "",
            "## 6. 论文边界",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in PAPER_BOUNDARY)

    lines.extend(
        [
            "",
            "## 7. 检查表",
            "",
            md_row(["类别", "键", "来源", "要求/取值", "状态", "说明"]),
            md_row(["---", "---", "---", "---", "---", "---"]),
        ]
    )
    for row in rows:
        lines.append(md_row([row["category"], row["key"], row["source"], row["required_value"], row["status"], row["note"]]))

    lines.extend(
        [
            "",
            "## 8. 重建命令",
            "",
            "```powershell",
            f'& "{ROOT / ".venv" / "Scripts" / "python.exe"}" "{ROOT / "scripts" / "build_isaac_domain_randomization_handoff.py"}"',
            "```",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = read_json(args.source_json)
    if source.get("version") != SOURCE_VERSION:
        raise RuntimeError(f"unexpected source version: {source.get('version')}")
    summary = source_summary(source)
    rows = build_rows(source, summary)
    payload = {
        "version": VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed_prerequisite",
        "source_version": SOURCE_VERSION,
        "target_planned_version": TARGET_VERSION,
        "scene_contract": SCENE_CONTRACT,
        "source_summary": summary,
        "domain_specs": source.get("domain_specs", {}),
        "required_metrics": REQUIRED_METRICS,
        "required_return_files": REQUIRED_RETURN_FILES,
        "paper_boundary": PAPER_BOUNDARY,
        "rows": rows,
        "can_register_completed_isaac_method": False,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.output_csv, rows)
    write_md(args.output_md, source, summary, rows)
    print(f"isaac_domain_randomization_handoff_json: {args.output_json}", flush=True)
    print(f"isaac_domain_randomization_handoff_md: {args.output_md}", flush=True)
    print(f"isaac_domain_randomization_handoff_csv: {args.output_csv}", flush=True)
    print(f"rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
