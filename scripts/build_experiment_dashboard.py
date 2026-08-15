from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


EXTRA_CLIPS = [
    {
        "version": "object_language_action_head_lite_v1_success_example",
        "method": "Object-Language Action Head-lite 成功样例",
        "clip": "outputs/videos/object_language_action_head_lite_v1_seed1_success_example.mp4",
        "note": "训练范围 seed1 局部成功，用于展示 action-head 代理基线并非完全无效。",
    },
    {
        "version": "expert_scripted_language_v1",
        "method": "语言任务 expert oracle",
        "clip": "outputs/videos/expert_scripted_language_v1_seed200.mp4",
        "note": "move_leftmost_to_bowl / language 的 oracle 参考。",
    },
    {
        "version": "structured_waypoint_policy_v1_language_eval",
        "method": "结构化 waypoint 语言成功样例",
        "clip": "outputs/videos/structured_waypoint_policy_v1_language_seed200.mp4",
        "note": "使用目标物和目标区域状态的结构化策略可完成 leftmost -> bowl，但不是 learned VLA。",
    },
    {
        "version": "object_language_action_head_lite_v1_language_eval",
        "method": "单任务 action-head 语言失败样例",
        "clip": "outputs/videos/object_language_action_head_lite_v1_language_seed200.mp4",
        "note": "单任务 action-head 不能迁移到 leftmost -> bowl。",
    },
    {
        "version": "reward_weighted_action_head_lite_v1_language_eval",
        "method": "Reward-weighted action-head 语言失败样例",
        "clip": "outputs/videos/reward_weighted_action_head_lite_v1_language_seed200.mp4",
        "note": "attempt 偏好和 dense shaping 加权后训练仍不能迁移到 leftmost -> bowl。",
    },
    {
        "version": "phase_conditioned_action_head_lite_v1_language_eval",
        "method": "阶段条件 action-head 语言失败样例",
        "clip": "outputs/videos/phase_conditioned_action_head_lite_v1_language_seed200.mp4",
        "note": "显式进度阶段拆分后，单任务 action-head 仍不能迁移到 leftmost -> bowl。",
    },
    {
        "version": "torch_act_cvae_state_chunk_v1_language_eval",
        "method": "ACT-CVAE-lite 语言失败样例",
        "clip": "outputs/videos/torch_act_cvae_state_chunk_v1_language_seed200.mp4",
        "note": "加入 CVAE latent 后，state-only ACT 仍不能迁移到 leftmost -> bowl。",
    },
    {
        "version": "torch_act_state_chunk_cuda_v1_language_eval",
        "method": "CUDA Torch ACT 语言失败样例",
        "clip": "outputs/videos/torch_act_state_chunk_cuda_v1_language_seed200.mp4",
        "note": "同结构 ACT 用 CUDA 训练后，language 任务仍失败；该版本主要用于资源对照。",
    },
    {
        "version": "torch_diffusion_policy_state_chunk_v1_language_eval",
        "method": "PyTorch state diffusion policy 语言失败样例",
        "clip": "outputs/videos/torch_diffusion_policy_state_chunk_v1_language_seed200.mp4",
        "note": "state-only PyTorch Diffusion Policy 动作块在 leftmost -> bowl 任务中仍失败；该版本不是视觉 Diffusion Policy。",
    },
    {
        "version": "visual_feature_act_lite_v1_language_eval",
        "method": "视觉特征 ACT-lite 语言失败样例",
        "clip": "outputs/videos/visual_feature_act_lite_v1_language_seed200.mp4",
        "note": "加入 MuJoCo RGB pooled features 后，ACT-lite 仍不能迁移到 leftmost -> bowl。",
    },
    {
        "version": "vision_language_action_head_lite_v1_language_eval",
        "method": "视觉-语言 action-head 代理语言失败样例",
        "clip": "outputs/videos/vision_language_action_head_lite_v1_language_seed200.mp4",
        "note": "冻结 RGB 视觉统计特征 + 语言 token 仍不能迁移到 leftmost -> bowl。",
    },
    {
        "version": "clip_action_head_lite_v1_language_eval",
        "method": "Frozen CLIP action-head 语言失败样例",
        "clip": "outputs/videos/clip_action_head_lite_v1_language_seed200.mp4",
        "note": "冻结 pretrained CLIP 图像/文本 encoder，只训练轻量 action head；language 任务仍失败。",
    },
    {
        "version": "adapter_action_head_lite_v1_language_eval",
        "method": "Adapter action-head 代理语言失败样例",
        "clip": "outputs/videos/adapter_action_head_lite_v1_language_seed200.mp4",
        "note": "冻结 object-language 主干，仅训练 Adapter 残差；language 任务仍失败。",
    },
    {
        "version": "lora_action_head_lite_v1_language_eval",
        "method": "LoRA-style action-head 代理语言失败样例",
        "clip": "outputs/videos/lora_action_head_lite_v1_language_seed200.mp4",
        "note": "冻结 object-language 主干，仅训练 LoRA-style 残差；language 任务仍失败。",
    },
    {
        "version": "multi_task_object_action_head_lite_v1_language_eval",
        "method": "多任务 action-head 语言失败样例",
        "clip": "outputs/videos/multi_task_object_action_head_lite_v1_language_seed400.mp4",
        "note": "naive 多任务 action-head 在训练附近语言 seed 上仍失败。",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local HTML dashboard for experiment results and rollout clips.")
    parser.add_argument("--versions", type=Path, default=ROOT / "docs" / "experiment_versions.json")
    parser.add_argument("--summary", type=Path, default=ROOT / "docs" / "evaluation_summary.csv")
    parser.add_argument("--language-summary", type=Path, default=ROOT / "docs" / "language_generalization_summary.csv")
    parser.add_argument("--resources", type=Path, default=ROOT / "docs" / "model_resource_summary.csv")
    parser.add_argument("--data-efficiency", type=Path, default=ROOT / "docs" / "data_efficiency_summary.csv")
    parser.add_argument("--strict-grasp-audit", type=Path, default=ROOT / "docs" / "strict_grasp_success_audit.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "experiment_dashboard.html")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: str | Path, base: Path) -> str:
    target = ROOT / path if not Path(path).is_absolute() else Path(path)
    return Path(os.path.relpath(target.resolve(), base.resolve())).as_posix()


def video_meta(clip: str) -> dict:
    path = ROOT / clip
    metadata = path.with_suffix(".json")
    if not metadata.exists():
        return {}
    return read_json(metadata)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def rate_cell(value: str) -> str:
    return f"<span class=\"metric\">{esc(value)}</span>"


def build_method_table(methods: list[dict], summary_rows: dict[str, dict]) -> str:
    rows = []
    for item in methods:
        row = summary_rows[item["version"]]
        rows.append(
            "<tr>"
            f"<td><code>{esc(item['version'])}</code></td>"
            f"<td>{esc(item['method'])}</td>"
            f"<td>{esc(item['stage'])}</td>"
            f"<td>{rate_cell(row['train_range_success'])}</td>"
            f"<td>{rate_cell(row['heldout_success'])}</td>"
            f"<td>{esc(item['note'])}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def build_language_table(rows: list[dict[str, str]]) -> str:
    html_rows = []
    for row in rows:
        html_rows.append(
            "<tr>"
            f"<td><code>{esc(row['version'])}</code></td>"
            f"<td>{esc(row['stage'])}</td>"
            f"<td>{rate_cell(row['success'])}</td>"
            f"<td>{float(row['mean_target_distance']):.4f}</td>"
            f"<td>{esc(row['seeds'])}</td>"
            "</tr>"
        )
    return "\n".join(html_rows)


def build_resource_table(rows: list[dict[str, str]]) -> str:
    html_rows = []
    for row in rows:
        history_horizon = "/".join(part for part in [row["history"], row["horizon"]] if part)
        html_rows.append(
            "<tr>"
            f"<td><code>{esc(row['version'])}</code></td>"
            f"<td>{esc(row['method'])}</td>"
            f"<td>{int(row['trainable_params']):,}</td>"
            f"<td>{esc(row['stored_samples'])}</td>"
            f"<td>{esc(row['feature_dim'])}</td>"
            f"<td>{esc(row['action_dim'])}</td>"
            f"<td>{esc(history_horizon)}</td>"
            f"<td>{float(row['artifact_size_mb']):.3f}</td>"
            f"<td>{esc(row['train_range_success'])} / {esc(row['heldout_success'])}</td>"
            "</tr>"
        )
    return "\n".join(html_rows)


def build_data_efficiency_table(rows: list[dict[str, str]]) -> str:
    html_rows = []
    for row in rows:
        html_rows.append(
            "<tr>"
            f"<td><code>{esc(row['method_key'])}</code></td>"
            f"<td>{esc(row['demo_budget'])}</td>"
            f"<td>{esc(row['split'])}</td>"
            f"<td>{rate_cell(row['success'])}</td>"
            f"<td>{float(row['mean_target_distance']):.4f}</td>"
            f"<td>{int(row['stored_samples']):,}</td>"
            "</tr>"
        )
    return "\n".join(html_rows)


def build_frozen_clip_adapter_comparison() -> str:
    path = ROOT / "docs" / "frozen_clip_semantic_adapter_same_protocol_comparison.csv"
    if not path.exists():
        return "<p>同协议对照尚未生成。</p>"
    rows = read_csv(path)
    html_rows = []
    for row in rows:
        condition = "未见改写" if row["condition"] == "paraphrase" else "hard 多物体干扰"
        task_success = f"{row['task_successes']}/{row['episodes']}"
        semantic_correct = f"{row['semantic_correct']}/{row['episodes']}"
        strict_grasp = f"{row['strict_grasp_successes']}/{row['episodes']}"
        html_rows.append(
            "<tr>"
            f"<td><code>{esc(row['version'])}</code></td>"
            f"<td>{esc(row['head'])}</td>"
            f"<td>{int(row['trainable_params']):,}</td>"
            f"<td>{float(row['train_time_seconds']):.2f}</td>"
            f"<td>{esc(condition)}</td>"
            f"<td>{rate_cell(task_success)}</td>"
            f"<td>{rate_cell(semantic_correct)}</td>"
            f"<td>{rate_cell(strict_grasp)}</td>"
            "</tr>"
        )
    return "\n".join(html_rows)


def build_strict_grasp_audit(rows: list[dict[str, str]]) -> tuple[str, dict[str, int]]:
    totals = {
        "rows": len(rows),
        "episodes": 0,
        "loose_successes": 0,
        "strict_successes": 0,
        "loose_without_grasp_rows": 0,
    }
    html_rows = []
    for row in rows:
        episodes = int(row["episodes"])
        loose = int(row["loose_successes"])
        strict = int(row["strict_grasp_successes"])
        totals["episodes"] += episodes
        totals["loose_successes"] += loose
        totals["strict_successes"] += strict
        totals["loose_without_grasp_rows"] += int(loose > 0 and strict == 0)
        html_rows.append(
            "<tr>"
            f"<td><code>{esc(row['method'])}</code></td>"
            f"<td>{esc(row['source_version'])}</td>"
            f"<td>{esc(row['preset_or_seed'])}</td>"
            f"<td>{loose}/{episodes}</td>"
            f"<td>{strict}/{episodes}</td>"
            f"<td>{float(row['mean_object_z']):.4f}</td>"
            f"<td>{esc(row['diagnosis'])}</td>"
            "</tr>"
        )
    return "\n".join(html_rows), totals


def build_video_card(item: dict, output_dir: Path) -> str:
    clip = item["clip"]
    meta = video_meta(clip)
    summary = meta.get("summary", {})
    success = summary.get("success", summary.get("steps_replayed", "unknown"))
    task = meta.get("task", summary.get("task", ""))
    seed = meta.get("seed", summary.get("seed", ""))
    video_src = rel(clip, output_dir)
    return (
        "<article class=\"clip\">"
        f"<video controls preload=\"metadata\" src=\"{esc(video_src)}\"></video>"
        f"<h3>{esc(item['method'])}</h3>"
        f"<p><code>{esc(item['version'])}</code></p>"
        f"<p>success: <strong>{esc(success)}</strong> · task: {esc(task)} · seed: {esc(seed)}</p>"
        f"<p>{esc(item.get('note', ''))}</p>"
        "</article>"
    )


def build_video_grid(methods: list[dict], output_dir: Path) -> str:
    clips = [
        {
            "version": item["version"],
            "method": item["method"],
            "clip": item["clip"],
            "note": item["note"],
        }
        for item in methods
    ]
    clips.extend(EXTRA_CLIPS)
    return "\n".join(build_video_card(item, output_dir) for item in clips)


def build_figure_grid(output_dir: Path) -> str:
    figures = [
        ("主任务成功率", "outputs/figures/main_task_success.svg"),
        ("语言/空间泛化成功率", "outputs/figures/language_success.svg"),
        ("参数量与 held-out 成功率", "outputs/figures/resource_vs_success.svg"),
        ("数据效率曲线", "outputs/figures/data_efficiency.svg"),
    ]
    cards = []
    for title, path in figures:
        if not (ROOT / path).exists():
            continue
        cards.append(
            "<article class=\"figure\">"
            f"<h3>{esc(title)}</h3>"
            f"<img src=\"{esc(rel(path, output_dir))}\" alt=\"{esc(title)}\">"
            "</article>"
        )
    return "\n".join(cards)


def build_reference_grid(output_dir: Path) -> str:
    links = [
        {
            "title": "阶段展示总索引",
            "path": "docs/stage_showcase_index.html",
            "version": "stage_showcase_index_v1",
            "note": "按阶段集中展示版本名称、评测比较、仿真视频片段和启动命令。",
        },
        {
            "title": "方法评测比较看板",
            "path": "docs/method_comparison_dashboard.html",
            "version": "method_comparison_dashboard_v1",
            "note": "按方法横向筛选阶段、成功率、语言泛化、参数、资源、固定视频和慢速 viewer 命令。",
        },
        {
            "title": "论文图表与视频证据索引",
            "path": "docs/thesis_visual_evidence_index.html",
            "version": "thesis_visual_evidence_index_v1",
            "note": "把论文图、附录表、答辩 HTML、阶段视频和候选诊断视频映射到中文图注、支撑结论和论文红线。",
        },
        {
            "title": "答辩追问 Q&A Playbook",
            "path": "docs/defense_qa_playbook.html",
            "version": "defense_qa_playbook_v1",
            "note": "把常见追问映射到推荐回答、首选证据、视频/图表和必须坚持的论文边界。",
        },
        {
            "title": "实验版本谱系索引",
            "path": "docs/version_lineage_index.html",
            "version": "version_lineage_index_v1",
            "note": "把数据集、25 个正式方法、候选诊断、前置门禁和后续计划版本放到同一张谱系表。",
        },
        {
            "title": "视频展示讲稿与时间线",
            "path": "docs/video_presentation_storyboard.html",
            "version": "video_presentation_storyboard_v1",
            "note": "60 秒总览 reel 和 6 个阶段短片的播放时间线、讲稿提示、证据引用和论文红线。",
        },
        {
            "title": "OpenVLA 数据桥接浏览页",
            "path": "docs/openvla_bridge_gallery.html",
            "version": "openvla_bridge_gallery_v1",
            "note": "查看 72 条 image + instruction + state + action 样本；本页不是策略评测结果。",
        },
        {
            "title": "OpenVLA 数据桥接报告",
            "path": "docs/openvla_dataset_bridge_report.md",
            "version": "openvla_dataset_bridge_v1",
            "note": "说明 MuJoCo 成功轨迹如何导出为后续 OpenVLA/机器人 VLA 微调数据格式。",
        },
        {
            "title": "OpenVLA 本地可行性检查",
            "path": "docs/openvla_feasibility_report.md",
            "version": "openvla_feasibility_check_v1",
            "note": "记录 RTX 3060 Laptop GPU / 6GB 环境下真实 OpenVLA LoRA 的本地训练边界。",
        },
        {
            "title": "Robot VLA action-head 交接门禁",
            "path": "docs/robot_vla_action_head_handoff.md",
            "version": "robot_vla_action_head_handoff_v1",
            "note": "定义真实 robot VLA action-head 在 48GB+ GPU 或云端运行时的输入契约、输出契约和入包门禁。",
        },
        {
            "title": "Robot VLA 远端运行包",
            "path": "docs/robot_vla_remote_run_pack.md",
            "version": "robot_vla_remote_run_pack_v1",
            "note": "打包 bridge 数据、远端命令模板、结果回填 schema 和 zip 归档；不是策略训练结果。",
        },
        {
            "title": "Robot VLA 远端结果回填门禁",
            "path": "docs/robot_vla_remote_result_intake.md",
            "version": "robot_vla_remote_result_intake_v1",
            "note": "检查远端返回的模型、feature cache、评测 JSON、视频和报告能否进入正式方法包。",
        },
        {
            "title": "Isaac domain randomization 交接门禁",
            "path": "docs/isaac_domain_randomization_handoff.md",
            "version": "isaac_domain_randomization_handoff_v1",
            "note": "固定 Isaac 复现实验的桌面场景契约、扰动域、回填指标、必须文件和论文红线；不是 Isaac 运行结果。",
        },
        {
            "title": "真实 WidowX 验证交接门禁",
            "path": "docs/real_widowx_validation_handoff.md",
            "version": "real_widowx_validation_handoff_v1",
            "note": "固定真实机械臂安全门禁、trial 字段、50 条记录模板、视频回填和论文红线；不是真实 trial 结果。",
        },
        {
            "title": "下一阶段实施方案",
            "path": "docs/next_phase_implementation.md",
            "version": "next_phase_implementation_v1",
            "note": "后续接入机器人预训练 VLA 表征、Isaac 和真实 WidowX 验证时从这里继续。",
        },
        {
            "title": "下一阶段实验注册表",
            "path": "docs/next_experiment_registry.md",
            "version": "next_experiment_registry_v1",
            "note": "计划版本进入正式实验包前必须补齐的版本名、评测表、资源表、视频和论文边界。",
        },
    ]
    cards = []
    for item in links:
        path = ROOT / item["path"]
        if not path.exists():
            continue
        cards.append(
            "<a class=\"linkCard\""
            f" href=\"{esc(rel(item['path'], output_dir))}\">"
            f"<h3>{esc(item['title'])}</h3>"
            f"<p><code>{esc(item['version'])}</code></p>"
            f"<p>{esc(item['note'])}</p>"
            "</a>"
        )
    return "\n".join(cards)


def main() -> None:
    args = parse_args()
    versions = read_json(args.versions)
    methods = versions["methods"]
    summary_rows = {row["version"]: row for row in read_csv(args.summary)}
    language_rows = read_csv(args.language_summary)
    resource_rows = read_csv(args.resources) if args.resources.exists() else []
    data_efficiency_rows = read_csv(args.data_efficiency) if args.data_efficiency.exists() else []
    strict_grasp_rows = read_csv(args.strict_grasp_audit) if args.strict_grasp_audit.exists() else []
    strict_grasp_table, strict_grasp_totals = build_strict_grasp_audit(strict_grasp_rows)
    output_dir = args.output.parent

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>轻量化 VLA 机械臂实验 Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1f2933;
      --muted: #5f6b7a;
      --line: #d7dde5;
      --panel: #f6f8fb;
      --accent: #1d6fb8;
    }}
    body {{
      margin: 0;
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background: #ffffff;
      line-height: 1.55;
    }}
    header, main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 24px;
    }}
    header {{
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 30px;
      letter-spacing: 0;
    }}
    h2 {{
      margin-top: 34px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--line);
      font-size: 22px;
      letter-spacing: 0;
    }}
    p {{
      color: var(--muted);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      margin: 14px 0 24px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 9px 10px;
      vertical-align: top;
      text-align: left;
    }}
    th {{
      background: var(--panel);
      color: #293544;
    }}
    code {{
      font-family: Consolas, "SFMono-Regular", monospace;
      font-size: 13px;
    }}
    .metric {{
      font-weight: 700;
      color: var(--accent);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    .clip {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
    }}
    .clip video {{
      width: 100%;
      aspect-ratio: 4 / 3;
      background: #111827;
      border-radius: 6px;
    }}
    .clip h3 {{
      margin: 10px 0 4px;
      font-size: 16px;
      letter-spacing: 0;
    }}
    .figure {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
    }}
    .figure h3 {{
      margin: 0 0 10px;
      font-size: 16px;
      letter-spacing: 0;
    }}
    .figure img {{
      display: block;
      width: 100%;
      background: #fff;
      border-radius: 4px;
    }}
    .linkCard {{
      display: block;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fff;
      color: var(--ink);
      text-decoration: none;
    }}
    .linkCard:hover {{
      border-color: var(--accent);
    }}
    .linkCard h3 {{
      margin: 0 0 6px;
      font-size: 16px;
      letter-spacing: 0;
    }}
    .linkCard p {{
      margin: 6px 0 0;
    }}
    .note {{
      background: var(--panel);
      border-left: 4px solid var(--accent);
      padding: 12px 14px;
      color: var(--ink);
    }}
  </style>
</head>
<body>
  <header>
    <h1>轻量化 VLA 机械臂实验 Dashboard</h1>
    <p>自动生成自 <code>docs/experiment_versions.json</code>、<code>docs/evaluation_summary.csv</code>、<code>docs/language_generalization_summary.csv</code> 和固定 MuJoCo 视频片段。</p>
    <p>中文阶段结果矩阵见 <a href="result_matrix.md"><code>docs/result_matrix.md</code></a>，方法横向比较见 <a href="method_comparison_dashboard.html"><code>docs/method_comparison_dashboard.html</code></a>，答辩叙事见 <a href="defense_storyboard.md"><code>docs/defense_storyboard.md</code></a>。</p>
  </header>
  <main>
    <section>
      <h2>阶段结论</h2>
      <p class="note">当前 MuJoCo 阶段证明了仿真、示范采集、回放、普通 baseline、PyTorch ACT、ACT-CVAE-lite、视觉特征 ACT-lite、action-head 代理基线和语言泛化评测链路。结果显示：简单 BC/ACT-lite/Diffusion-lite 不稳定，kNN 和 trajectory-kNN 更像轨迹记忆，state-only PyTorch ACT 仅局部成功，加入 CVAE latent 或 pooled RGB 视觉代理后仍未稳定抓取，符号 action-head 和阶段条件 action-head 有局部效果但泛化不足，naive 多任务 action-head 仍失败。</p>
    </section>

    <section>
      <h2>单任务闭环结果</h2>
      <table>
        <thead>
          <tr>
            <th>版本</th>
            <th>方法</th>
            <th>阶段</th>
            <th>训练范围</th>
            <th>留出范围</th>
            <th>说明</th>
          </tr>
        </thead>
        <tbody>
          {build_method_table(methods, summary_rows)}
        </tbody>
      </table>
    </section>

    <section>
      <h2>语言/空间泛化结果</h2>
      <table>
        <thead>
          <tr>
            <th>版本</th>
            <th>阶段</th>
            <th>成功率</th>
            <th>平均目标距离</th>
            <th>seeds</th>
          </tr>
        </thead>
        <tbody>
          {build_language_table(language_rows)}
        </tbody>
      </table>
    </section>

    <section>
      <h2>Kaggle 冻结 CLIP 适配器同协议对照</h2>
      <p class="note">两种模型均冻结相同 CLIP、使用相同结构化 waypoint executor，规范闭环均为 <strong>20/20</strong>。参数更多的 Kaggle 1024→16→4 瓶颈头在未见改写为 <strong>48/60</strong>，低于本地线性头的 <strong>51/60</strong>；hard 多物体干扰为 <strong>19/20</strong>，低于 <strong>20/20</strong>。这是一项远程轻量语义适配消融，不是端到端 VLA 或 OpenVLA LoRA。</p>
      <table>
        <thead>
          <tr>
            <th>版本</th>
            <th>头部</th>
            <th>可训练参数</th>
            <th>训练时间 (s)</th>
            <th>条件</th>
            <th>任务成功</th>
            <th>语义正确</th>
            <th>严格抓取</th>
          </tr>
        </thead>
        <tbody>
          {build_frozen_clip_adapter_comparison()}
        </tbody>
      </table>
      <p>完整对照见 <a href="frozen_clip_semantic_adapter_same_protocol_comparison.md"><code>同协议对照报告</code></a>；Kaggle 训练、规范闭环和 OOD 负例见 <a href="kaggle_clip_semantic_adapter_core_v2_v1_report.md"><code>Kaggle 报告</code></a> 与 <a href="defense_video_playlist.html"><code>视频播放列表</code></a>。</p>
    </section>

    <section>
      <h2>严格抓取成功口径审计</h2>
      <p class="note"><code>strict_grasp_success_audit_v1</code> 同时报告原始放置 <code>success</code>、<code>grasp_success</code> 和 <code>object_z</code>。当前原始放置成功合计 <strong>{strict_grasp_totals['loose_successes']}/{strict_grasp_totals['episodes']}</strong>，严格抓取成功合计 <strong>{strict_grasp_totals['strict_successes']}/{strict_grasp_totals['episodes']}</strong>；因此这些样例不能写成稳定抓取成功。</p>
      <table>
        <thead>
          <tr>
            <th>方法/版本</th>
            <th>来源</th>
            <th>preset/seed</th>
            <th>原始成功</th>
            <th>严格抓取成功</th>
            <th>object_z</th>
            <th>诊断</th>
          </tr>
        </thead>
        <tbody>
          {strict_grasp_table}
        </tbody>
      </table>
    </section>

    <section>
      <h2>模型资源对比</h2>
      <table>
        <thead>
          <tr>
            <th>版本</th>
            <th>方法</th>
            <th>可训练参数</th>
            <th>存储样本</th>
            <th>特征/观测维度</th>
            <th>动作维度</th>
            <th>历史/动作块</th>
            <th>模型大小 MB</th>
            <th>训练/留出成功率</th>
          </tr>
        </thead>
        <tbody>
          {build_resource_table(resource_rows)}
        </tbody>
      </table>
      <p>资源明细来源于 <code>docs/model_resource_summary.csv</code>，中文说明见 <code>docs/model_resource_summary.md</code>。</p>
    </section>

    <section>
      <h2>数据效率对比</h2>
      <table>
        <thead>
          <tr>
            <th>方法</th>
            <th>示范预算</th>
            <th>评测范围</th>
            <th>成功率</th>
            <th>平均目标距离</th>
            <th>存储样本</th>
          </tr>
        </thead>
        <tbody>
          {build_data_efficiency_table(data_efficiency_rows)}
        </tbody>
      </table>
      <p>数据效率结果来源于 <code>docs/data_efficiency_summary.csv</code>，中文说明见 <code>docs/data_efficiency_summary.md</code>。</p>
    </section>

    <section>
      <h2>实验图表</h2>
      <div class="grid">
        {build_figure_grid(output_dir)}
      </div>
      <p>静态图表由 <code>scripts/build_experiment_figures.py</code> 生成，索引见 <code>docs/experiment_figures.md</code>。</p>
    </section>

    <section>
      <h2>OpenVLA 数据桥接与下一阶段</h2>
      <p class="note">这里是从 MuJoCo 实验走向真实机器人 VLA 后训练和高保真仿真的入口：已完成 <code>openvla_dataset_bridge_v1</code> 数据格式桥接、<code>openvla_feasibility_check_v1</code> 本地可行性检查、<code>robot_vla_action_head_handoff_v1</code> 远端运行门禁、<code>robot_vla_remote_run_pack_v1</code> 远端运行包、<code>robot_vla_remote_result_intake_v1</code> 回填门禁、<code>isaac_domain_randomization_handoff_v1</code> Isaac 运行交接门禁和 <code>real_widowx_validation_handoff_v1</code> 真实机械臂验证交接门禁，但不能写成 OpenVLA LoRA、<code>robot_vla_action_head_lite_v1</code>、Isaac 运行结果或真实 WidowX 验证已经完成，也不参与当前 25 个策略方法的成功率比较。</p>
      <div class="grid">
        {build_reference_grid(output_dir)}
      </div>
    </section>

    <section>
      <h2>视频片段</h2>
      <div class="grid">
        {build_video_grid(methods, output_dir)}
      </div>
    </section>
  </main>
</body>
</html>
"""
    args.output.write_text(document, encoding="utf-8")
    print(f"dashboard_path: {args.output}", flush=True)
    print(f"methods: {len(methods)}", flush=True)
    print(f"language_rows: {len(language_rows)}", flush=True)


if __name__ == "__main__":
    main()
