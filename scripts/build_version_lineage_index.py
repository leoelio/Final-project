from __future__ import annotations

import argparse
import csv
from datetime import datetime
from html import escape
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "version_lineage_index_v1"


STAGE_LABELS = {
    "scripted_oracle": "环境与示范",
    "structured_control_baseline": "结构化控制强基线",
    "data_verification": "数据验证",
    "weak_bc_baseline": "普通模仿学习",
    "non_neural_baseline": "普通模仿学习",
    "neural_bc_baseline": "普通模仿学习",
    "trajectory_conditioned_baseline": "动作块 / 轨迹条件",
    "trajectory_memory_baseline": "动作块 / 轨迹条件",
    "torch_act_baseline": "ACT-style",
    "torch_act_cvae_baseline": "ACT-style",
    "visual_feature_act_baseline": "ACT-style",
    "visual_act_cnn_cvae_baseline": "ACT-style",
    "diffusion_policy_baseline": "Diffusion Policy",
    "torch_diffusion_policy_baseline": "Diffusion Policy",
    "vla_action_head_proxy": "VLA/action-head 代理",
    "reward_weighted_bc_post_training": "轻量后训练代理",
    "phase_conditioned_action_head_proxy": "VLA/action-head 代理",
    "peft_action_head_proxy": "参数高效后训练代理",
    "pretrained_vlm_action_head_proxy": "VLM 表征代理",
    "multi_task_action_head_proxy": "多任务 action-head 代理",
}

PARENTS = {
    "expert_scripted_v1": "task_tabletop_widowx_mujoco_v1",
    "structured_waypoint_policy_v1": "expert_scripted_v1；task_tabletop_widowx_mujoco_v1",
    "replay_demo_v1": "demo_place_blue_cube_blue_pad_v1",
    "linear_bc_v1": "demo_place_blue_cube_blue_pad_v1",
    "knn_bc_v1": "demo_place_blue_cube_blue_pad_v1",
    "mlp_bc_v1": "demo_place_blue_cube_blue_pad_v1；linear_bc_v1",
    "act_lite_chunk_bc_v1": "demo_place_blue_cube_blue_pad_v1；linear_bc_v1；mlp_bc_v1",
    "diffusion_policy_lite_v1": "demo_place_blue_cube_blue_pad_v1；act_lite_chunk_bc_v1",
    "torch_diffusion_policy_state_chunk_v1": "diffusion_policy_lite_v1；demo_place_blue_cube_blue_pad_v1",
    "trajectory_conditioned_chunk_bc_v2": "act_lite_chunk_bc_v1；demo_place_blue_cube_blue_pad_v1",
    "trajectory_knn_chunk_bc_v1": "knn_bc_v1；trajectory_conditioned_chunk_bc_v2",
    "torch_act_state_chunk_v1": "trajectory_conditioned_chunk_bc_v2；demo_place_blue_cube_blue_pad_v1",
    "torch_act_state_chunk_cuda_v1": "torch_act_state_chunk_v1",
    "phase_conditioned_torch_act_v1": "torch_act_state_chunk_v1；trajectory_phase_labels_v1",
    "torch_act_cvae_state_chunk_v1": "torch_act_state_chunk_v1",
    "visual_feature_act_lite_v1": "torch_act_state_chunk_v1；rgb_feature_grid_v1",
    "visual_act_cnn_cvae_v1": "visual_feature_act_lite_v1；torch_act_cvae_state_chunk_v1",
    "object_language_action_head_lite_v1": "demo_place_blue_cube_blue_pad_v1；object_state_language_tokens_v1",
    "reward_weighted_action_head_lite_v1": "object_language_action_head_lite_v1；reward_weighted_bc_proxy_v1",
    "phase_conditioned_action_head_lite_v1": "object_language_action_head_lite_v1；trajectory_phase_labels_v1",
    "adapter_action_head_lite_v1": "object_language_action_head_lite_v1；adapter_proxy_v1",
    "lora_action_head_lite_v1": "object_language_action_head_lite_v1；lora_proxy_v1",
    "vision_language_action_head_lite_v1": "object_language_action_head_lite_v1；rgb_feature_grid_v1",
    "clip_action_head_lite_v1": "vision_language_action_head_lite_v1；frozen_clip_feature_proxy_v1",
    "multi_task_object_action_head_lite_v1": "object_language_action_head_lite_v1；language_generalization_task_set_v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese lineage index for all experiment versions.")
    parser.add_argument("--versions", type=Path, default=ROOT / "docs" / "experiment_versions.json")
    parser.add_argument("--final-method-index", type=Path, default=ROOT / "docs" / "final_method_version_index.csv")
    parser.add_argument("--next-registry", type=Path, default=ROOT / "docs" / "next_experiment_registry.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "version_lineage_index.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "version_lineage_index.csv")
    parser.add_argument("--output-html", type=Path, default=ROOT / "docs" / "version_lineage_index.html")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def split_items(value: str) -> list[str]:
    return [item.strip() for item in value.replace("；", "\n").splitlines() if item.strip()]


def result_text(row: dict[str, str]) -> str:
    parts = []
    if row.get("主任务训练范围"):
        parts.append(f"train={row['主任务训练范围']}")
    if row.get("主任务留出范围"):
        parts.append(f"held-out={row['主任务留出范围']}")
    if row.get("语言/空间泛化"):
        parts.append(f"language={row['语言/空间泛化']}")
    if row.get("可训练参数"):
        parts.append(f"params={row['可训练参数']}")
    return "；".join(parts) if parts else "不适用"


def lineage_level(status: str, stage: str) -> str:
    if status == "current_dataset":
        return "0. 数据与任务定义"
    if stage in {"scripted_oracle", "data_verification", "structured_control_baseline"}:
        return "1. 任务、示范与上界"
    if stage in {"weak_bc_baseline", "non_neural_baseline", "neural_bc_baseline"}:
        return "2. 普通模仿学习 baseline"
    if "trajectory" in stage or "act" in stage or "diffusion" in stage:
        return "3. Trajectory / ACT / Diffusion 对照"
    if "action_head" in stage or "vla" in stage or "vlm" in stage or "peft" in stage or "multi_task" in stage:
        return "4. Action-head / PEFT / VLM proxy"
    if status.startswith("planned"):
        return "7. 后续计划"
    return "5. 其他正式版本"


def row(
    version: str,
    level: str,
    status: str,
    category: str,
    parents: str,
    stage: str,
    method: str,
    artifact: str,
    result_or_gate: str,
    media: str,
    relation: str,
    redline: str,
) -> dict[str, str]:
    return {
        "版本": version,
        "谱系层级": level,
        "状态": status,
        "类别": category,
        "父级/依赖": parents,
        "阶段或登记阶段": stage,
        "方法或对象": method,
        "artifact或输出": artifact,
        "量化结果或成功门槛": result_or_gate,
        "首选视频或展示": media,
        "关系说明": relation,
        "论文边界": redline,
    }


def build_reverse_dependents(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    dependents: dict[str, list[str]] = {}
    for item in rows:
        version = item["version"]
        for parent in split_items(item["depends_on"]):
            dependents.setdefault(parent, []).append(version)
    return dependents


def build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    versions = read_json(args.versions)
    methods = versions["methods"]
    final_index = {item["版本"]: item for item in read_csv(args.final_method_index)}
    next_rows = read_csv(args.next_registry)
    next_dependents = build_reverse_dependents(next_rows)

    rows = [
        row(
            "demo_place_blue_cube_blue_pad_v1",
            "0. 数据与任务定义",
            "current_dataset",
            "demonstration_dataset",
            "task_tabletop_widowx_mujoco_v1",
            "data_collection",
            "MuJoCo scripted demonstrations",
            versions["dataset"]["path"],
            f"{versions['dataset']['episodes']} 条 episode，成功 {versions['dataset']['successes']}，成功率 {versions['dataset']['success_rate']:.2f}",
            "outputs/presentation_clips/01_task_data_oracle.mp4",
            "所有当前正式学习方法的主训练数据源；OpenVLA bridge 也从这里导出样本。",
            "只能写成 MuJoCo scripted demonstration 数据集，不能写成真实机械臂 demonstration。",
        )
    ]

    for method in methods:
        version = method["version"]
        final = final_index.get(version, {})
        stage = method["stage"]
        parents = PARENTS.get(version, "demo_place_blue_cube_blue_pad_v1")
        dependents = next_dependents.get(version, [])
        relation = f"正式 MuJoCo 方法版本；承接 {parents}"
        if dependents:
            relation += f"；后续关联 {'；'.join(dependents)}"
        rows.append(
            row(
                version,
                lineage_level("formal_current", stage),
                "formal_current",
                STAGE_LABELS.get(stage, stage),
                parents,
                stage,
                method["method"],
                method["artifact"],
                result_text(final),
                method["clip"],
                relation,
                final.get("论文边界", method.get("note", "")),
            )
        )

    for item in next_rows:
        status = item["status"]
        level = "6. 已完成前置/候选诊断" if status.startswith("completed") else "7. 后续计划/外部依赖"
        rows.append(
            row(
                item["version"],
                level,
                status,
                item["category"],
                item["depends_on"],
                item["stage_to_register"],
                item["method_name"],
                item["primary_artifact"],
                item["success_gate"],
                item["video_outputs"],
                "下一阶段注册表条目；只有满足 gate 并回填评测、资源和视频后，planned 才能进入正式方法包。",
                item["paper_boundary"],
            )
        )
    return rows


def should_exist(status: str, path_text: str) -> bool:
    if not path_text or path_text == "not_applicable":
        return False
    if "*" in path_text:
        return False
    if status.startswith("planned"):
        return False
    return True


def verify_rows(rows: list[dict[str, str]]) -> None:
    missing: list[str] = []
    for item in rows:
        for field in ("artifact或输出", "首选视频或展示"):
            for path_text in split_items(item[field]):
                if should_exist(item["状态"], path_text) and not (ROOT / path_text).exists():
                    missing.append(f"{item['版本']}: {path_text}")
    if missing:
        raise FileNotFoundError("\n".join(missing))


def web_path(path_text: str) -> str:
    path = Path(path_text.replace("\\", "/"))
    if path.is_absolute():
        try:
            path = path.relative_to(ROOT)
        except ValueError:
            return path.as_posix()
    return (Path("..") / path).as_posix()


def link_or_text(path_text: str) -> str:
    parts = []
    for item in split_items(path_text):
        if item == "not_applicable" or "*" in item or not (ROOT / item).exists():
            parts.append(escape(item))
        else:
            parts.append(f'<a href="{escape(web_path(item))}">{escape(item)}</a>')
    return "<br>".join(parts)


def build_html(rows: list[dict[str, str]]) -> str:
    generated_at = datetime.now().isoformat(timespec="seconds")
    grouped = {}
    for item in rows:
        grouped.setdefault(item["谱系层级"], []).append(item)
    sections = []
    for level, items in grouped.items():
        trs = []
        for item in items:
            trs.append(
                f"""
<tr>
  <td><code>{escape(item['版本'])}</code><span>{escape(item['状态'])}</span></td>
  <td>{escape(item['类别'])}</td>
  <td>{escape(item['父级/依赖'])}</td>
  <td>{escape(item['方法或对象'])}</td>
  <td>{link_or_text(item['artifact或输出'])}</td>
  <td>{escape(item['量化结果或成功门槛'])}</td>
  <td>{link_or_text(item['首选视频或展示'])}</td>
  <td>{escape(item['论文边界'])}</td>
</tr>
""".strip()
            )
        sections.append(
            f"""
<section>
  <h2>{escape(level)}</h2>
  <table>
    <thead><tr><th>版本</th><th>类别</th><th>父级/依赖</th><th>方法或对象</th><th>artifact</th><th>结果/Gate</th><th>视频/展示</th><th>论文边界</th></tr></thead>
    <tbody>{''.join(trs)}</tbody>
  </table>
</section>
""".strip()
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>实验版本谱系索引</title>
  <style>
    :root {{ color-scheme: light; --bg:#f5f7fa; --panel:#fff; --text:#17202a; --muted:#667085; --line:#d8dee8; --accent:#0f766e; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family:"Microsoft YaHei","Segoe UI",Arial,sans-serif; background:var(--bg); color:var(--text); }}
    header {{ padding:28px clamp(18px,4vw,56px) 18px; background:#fff; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 8px; font-size:clamp(26px,3vw,40px); letter-spacing:0; }}
    .subtitle {{ max-width:1100px; margin:0; color:var(--muted); line-height:1.7; }}
    .stats {{ display:grid; grid-template-columns:repeat(4,minmax(120px,1fr)); gap:10px; max-width:860px; margin-top:18px; }}
    .stat {{ border:1px solid var(--line); border-radius:6px; background:#f9fafb; padding:12px; }}
    .stat b {{ display:block; font-size:22px; color:var(--accent); }}
    main {{ padding:26px clamp(18px,4vw,56px) 48px; }}
    section {{ margin:0 0 28px; }}
    h2 {{ margin:0 0 12px; font-size:22px; letter-spacing:0; }}
    table {{ width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); font-size:13px; }}
    th,td {{ border:1px solid var(--line); padding:9px 10px; vertical-align:top; text-align:left; }}
    th {{ background:#eef2f6; }}
    td span {{ display:block; color:var(--muted); margin-top:4px; }}
    code {{ font-family:Consolas,"Cascadia Mono",monospace; font-size:.92em; }}
    a {{ color:var(--accent); word-break:break-all; }}
    footer {{ padding:0 clamp(18px,4vw,56px) 32px; color:var(--muted); }}
    @media (max-width: 820px) {{ .stats {{ grid-template-columns:repeat(2,minmax(120px,1fr)); }} table {{ font-size:12px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>实验版本谱系索引</h1>
    <p class="subtitle">版本：<code>{VERSION}</code>。本索引把数据集、25 个正式 MuJoCo 方法、已完成前置门禁、候选诊断负例和后续 OpenVLA/Isaac/真实 WidowX 计划版本放到同一张谱系表；它只组织已有证据，不新增实验结果。</p>
    <div class="stats">
      <div class="stat"><b>{len(rows)}</b><span>谱系条目</span></div>
      <div class="stat"><b>{sum(1 for item in rows if item['状态'] == 'formal_current')}</b><span>正式方法</span></div>
      <div class="stat"><b>{sum(1 for item in rows if item['状态'].startswith('completed'))}</b><span>前置/诊断</span></div>
      <div class="stat"><b>{sum(1 for item in rows if item['状态'].startswith('planned'))}</b><span>计划版本</span></div>
    </div>
  </header>
  <main>{''.join(sections)}</main>
  <footer>生成时间：{escape(generated_at)}</footer>
</body>
</html>
"""


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(value.replace("|", "\\|").replace("\n", " ") for value in values) + " |"


def build_mermaid(rows: list[dict[str, str]]) -> list[str]:
    selected = [
        ("demo_place_blue_cube_blue_pad_v1", "linear_bc_v1"),
        ("demo_place_blue_cube_blue_pad_v1", "knn_bc_v1"),
        ("demo_place_blue_cube_blue_pad_v1", "mlp_bc_v1"),
        ("knn_bc_v1", "trajectory_knn_chunk_bc_v1"),
        ("trajectory_conditioned_chunk_bc_v2", "torch_act_state_chunk_v1"),
        ("torch_act_state_chunk_v1", "torch_act_cvae_state_chunk_v1"),
        ("object_language_action_head_lite_v1", "adapter_action_head_lite_v1"),
        ("object_language_action_head_lite_v1", "lora_action_head_lite_v1"),
        ("clip_action_head_lite_v1", "robot_vla_action_head_lite_v1"),
        ("openvla_dataset_bridge_v1", "robot_vla_remote_run_pack_v1"),
        ("robot_vla_remote_run_pack_v1", "robot_vla_action_head_lite_v1"),
        ("isaac_domain_randomization_handoff_v1", "isaac_domain_randomization_v1"),
        ("real_widowx_validation_handoff_v1", "real_widowx_validation_v1"),
    ]
    known = {item["版本"] for item in rows}
    lines = ["```mermaid", "flowchart LR"]
    for parent, child in selected:
        if parent in known and child in known:
            lines.append(f'  {parent}["{parent}"] --> {child}["{child}"]')
    lines.append("```")
    return lines


def build_md(rows: list[dict[str, str]]) -> str:
    lines = [
        "# 实验版本谱系索引",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：把数据集、正式方法、候选诊断、前置门禁和后续计划版本放到一个中文谱系表中，避免后续写论文或答辩时混淆“正式结果”“候选负例”“前置门禁”和“计划版本”。",
        "",
        "## 1. 快速打开",
        "",
        "```powershell",
        f'& "{PYTHON}" "{ROOT / "scripts" / "showcase_launcher.py"}" --target lineage',
        "```",
        "",
        "## 2. 简化谱系图",
        "",
        *build_mermaid(rows),
        "",
        "## 3. 谱系总表",
        "",
        md_row(["版本", "谱系层级", "状态", "类别", "父级/依赖", "阶段或登记阶段", "结果/Gate", "首选展示", "论文边界"]),
        md_row(["---", "---", "---", "---", "---", "---", "---", "---", "---"]),
    ]
    for item in rows:
        lines.append(
            md_row(
                [
                    f"`{item['版本']}`",
                    item["谱系层级"],
                    item["状态"],
                    item["类别"],
                    item["父级/依赖"],
                    item["阶段或登记阶段"],
                    item["量化结果或成功门槛"],
                    f"`{item['首选视频或展示']}`",
                    item["论文边界"],
                ]
            )
        )
    lines.extend(
        [
            "",
            "## 4. 使用边界",
            "",
            "- `formal_current` 才是当前 MuJoCo 实验包正式方法版本。",
            "- `completed_prerequisite` 是数据桥接、运行包、handoff 或回填门禁，不是策略成功率结果。",
            "- `completed_diagnostic` 是候选负例或控制诊断，不改变正式方法成功率。",
            "- `planned` 和 `planned_external_dependency` 必须保持未完成状态，直到真实运行并回填评测、资源和视频证据。",
            "- 真实 OpenVLA、Isaac 和真实 WidowX 版本不能用当前 MuJoCo 视频替代。",
            "",
            "## 5. 重建命令",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_version_lineage_index.py"}"',
            "```",
            "",
            f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    verify_rows(rows)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_md(rows), encoding="utf-8")
    write_csv(args.output_csv, rows)
    args.output_html.write_text(build_html(rows), encoding="utf-8")
    print(f"version_lineage_index_md: {args.output_md}", flush=True)
    print(f"version_lineage_index_csv: {args.output_csv}", flush=True)
    print(f"version_lineage_index_html: {args.output_html}", flush=True)
    print(f"version_lineage_rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
