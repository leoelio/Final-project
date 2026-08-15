from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "version_naming_and_gate_spec_v1"


FIELDNAMES = ["类别", "规则编号", "规则", "示例", "入包门禁"]


RULES = [
    {
        "类别": "正式方法版本",
        "规则编号": "N01",
        "规则": "正式方法版本使用 snake_case 描述方法结构、输入类型或轻量化方式，并以 `_v1`、`_v2` 递增。不要复用同一个版本名覆盖不同实验结果。",
        "示例": "trajectory_knn_chunk_bc_v1；torch_act_cvae_state_chunk_v1；adapter_action_head_lite_v1",
        "入包门禁": "必须进入 docs/experiment_versions.json、docs/evaluation_summary.csv、docs/model_resource_summary.csv、docs/video_evidence_index.csv，并通过 method_evidence_gate_v1。",
    },
    {
        "类别": "候选诊断版本",
        "规则编号": "N02",
        "规则": "未达到正式方法门槛、只用于解释失败模式或控制层诊断的版本，必须以 `_candidate` 结尾。",
        "示例": "grasp_gated_trajectory_knn_v1_candidate；preference_trajectory_post_training_v1_candidate",
        "入包门禁": "只能进入 candidate_diagnostic_video_index_v1、failure/diagnosis 报告和 readiness 审计；不能计入正式方法成功率。",
    },
    {
        "类别": "前置门禁版本",
        "规则编号": "N03",
        "规则": "数据桥接、可行性审计、handoff、remote run pack、result intake、readiness audit 不是策略方法，版本名应显式带 bridge、feasibility、handoff、pack、intake 或 audit。",
        "示例": "openvla_dataset_bridge_v1；robot_vla_remote_result_intake_v1；external_dependency_readiness_audit_v1",
        "入包门禁": "可以作为 supporting evidence；不能写成策略成功率结果，不能进入正式方法 25 个版本统计。",
    },
    {
        "类别": "planned 外部版本",
        "规则编号": "N04",
        "规则": "真实 Robot VLA、Isaac 和真实 WidowX 未来版本先登记为 planned 或 planned_external_dependency；没有真实回填前不能改成正式方法。",
        "示例": "robot_vla_action_head_lite_v1；isaac_domain_randomization_v1；real_widowx_validation_v1",
        "入包门禁": "必须先通过 external_dependency_readiness_audit_v1；formal_method_allowed_now 为 是 之前不能写入正式成功率表。",
    },
    {
        "类别": "视频命名",
        "规则编号": "N05",
        "规则": "主任务固定视频使用 `{version}_seed{seed}.mp4`；语言/空间泛化视频使用 `{version}_language_seed{seed}.mp4`；候选诊断保留 `_candidate`。",
        "示例": "trajectory_knn_chunk_bc_v1_seed0.mp4；clip_action_head_lite_v1_language_seed200.mp4",
        "入包门禁": "视频必须有同名 JSON 元数据，并在 docs/video_evidence_index.csv 中登记证据用途和论文红线。",
    },
    {
        "类别": "资源与评测",
        "规则编号": "N06",
        "规则": "每个正式方法版本必须同时有主任务评测、资源记录、固定视频和慢速 viewer 命令；失败方法也必须保留视频。",
        "示例": "linear_bc_v1 失败视频；torch_act_state_chunk_cuda_v1 资源记录",
        "入包门禁": "缺少任一项时不能称为最终正式方法；只能写成待补实验或候选诊断。",
    },
    {
        "类别": "阶段归属",
        "规则编号": "N07",
        "规则": "新增版本必须明确属于 8 个阶段之一：任务/数据/普通 BC、Trajectory/ACT/Diffusion、Action-Head/PEFT/CLIP、语言/空间泛化、数据效率、MuJoCo domain randomization、最终展示/答辩入口、外部依赖 readiness 门禁。",
        "示例": "robot_vla_lora_lite_v1 属于第 8 阶段直到真实回填完成",
        "入包门禁": "必须更新 stage_evidence_index_v1、stage_showcase_index_v1 或 next_experiment_registry_v1 中对应阶段说明。",
    },
    {
        "类别": "planned 到 formal 升级",
        "规则编号": "N08",
        "规则": "planned 版本升级为正式方法前，必须回填 artifact、train-range/held-out/language、资源、视频、失败模式、论文红线和慢速 viewer 命令。",
        "示例": "robot_vla_action_head_lite_v1 远端回填后才可成为正式方法",
        "入包门禁": "重建 final_method_version_index_v1、method_evidence_gate_v1、stage_showcase_index_v1、final_artifact_manifest_v1 并通过 verify_experiment_artifacts.py。",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chinese version naming and formal-method gate spec.")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "version_naming_and_gate_spec.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "version_naming_and_gate_spec.csv")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / f"{VERSION}.json")
    return parser.parse_args()


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(value.replace("|", "\\|").replace("\n", " ") for value in values) + " |"


def write_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(RULES)


def write_json(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rule_count": len(RULES),
        "rules": RULES,
        "formal_method_gate": [
            "docs/experiment_versions.json",
            "docs/evaluation_summary.csv",
            "docs/model_resource_summary.csv",
            "docs/video_evidence_index.csv",
            "docs/method_evidence_gate.md",
            "docs/final_method_version_index.md",
            "scripts/verify_experiment_artifacts.py",
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_md(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 版本命名与入包门禁规范",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：固定当前毕业设计实验包的版本命名、候选诊断、前置门禁、planned 外部版本和 planned→formal 升级规则。这个文件不新增实验结果，只保证后续新增 Robot VLA、Isaac 或真实 WidowX 实验时，版本名、评测表、视频证据和论文边界不会混乱。",
        "",
        "## 1. 核心原则",
        "",
        "- 正式方法版本必须有可追溯版本名、artifact、评测表、资源表、固定视频、viewer 命令和论文红线。",
        "- 候选诊断、前置门禁和 readiness audit 不能写成策略成功率结果。",
        "- planned 外部版本在真实回填前不能进入当前正式方法统计。",
        "- 失败视频和失败模式必须保留，不能只保留成功片段。",
        "",
        "## 2. 命名与入包规则",
        "",
        md_row(FIELDNAMES),
        md_row(["---", "---", "---", "---", "---"]),
    ]
    for row in RULES:
        lines.append(md_row([row[field] for field in FIELDNAMES]))

    lines.extend(
        [
            "",
            "## 3. planned 到 formal 的最短流程",
            "",
            "1. 在 `docs/next_experiment_registry.md` 中确认 planned 版本、前置条件和必须输出视频。",
            "2. 运行外部实验或远端训练，回填模型、评测 JSON/CSV、资源记录、主任务视频、语言视频和中文报告。",
            "3. 将版本写入 `docs/experiment_versions.json`，并补齐 `docs/evaluation_summary.csv`、`docs/model_resource_summary.csv`、`docs/video_evidence_index.csv`。",
            "4. 重建方法索引、阶段索引、manifest 和证据包。",
            "5. 运行完整验证；只有通过后，才可以写成正式方法版本。",
            "",
            "## 4. 重建命令",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_version_naming_and_gate_spec.py"}"',
            "```",
            "",
            "## 5. 完整验证命令",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "verify_experiment_artifacts.py"}"',
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    write_csv(args.output_csv)
    write_json(args.output_json)
    write_md(args.output_md)
    print(f"version_naming_spec_md: {args.output_md}", flush=True)
    print(f"version_naming_spec_csv: {args.output_csv}", flush=True)
    print(f"version_naming_spec_json: {args.output_json}", flush=True)
    print(f"version_naming_spec_rules: {len(RULES)}", flush=True)


if __name__ == "__main__":
    main()
