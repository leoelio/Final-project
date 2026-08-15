from __future__ import annotations

import argparse
import csv
from datetime import datetime
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "thesis_visual_evidence_index_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese thesis/defense visual evidence index.")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "thesis_visual_evidence_index.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "thesis_visual_evidence_index.csv")
    parser.add_argument("--output-html", type=Path, default=ROOT / "docs" / "thesis_visual_evidence_index.html")
    return parser.parse_args()


def ps_open(path_text: str, *, notepad: bool = False) -> str:
    path = ROOT / path_text
    if notepad:
        return f'Start-Process notepad.exe "{path}"'
    return f'Start-Process "{path}"'


def ps_build(script: str) -> str:
    return f'& "{PYTHON}" "{ROOT / script}"'


def row(
    item_id: str,
    section: str,
    kind: str,
    title: str,
    evidence_path: str,
    supporting_sources: str,
    caption: str,
    supported_claim: str,
    paper_redline: str,
    open_command: str,
) -> dict[str, str]:
    return {
        "编号": item_id,
        "论文或答辩位置": section,
        "类型": kind,
        "建议标题": title,
        "证据文件": evidence_path,
        "配套证据": supporting_sources,
        "中文图注/表注/讲解": caption,
        "可支撑结论": supported_claim,
        "论文红线": paper_redline,
        "打开命令": open_command,
    }


def build_rows() -> list[dict[str, str]]:
    return [
        row(
            "F01",
            "第 4 章 主任务闭环结果",
            "论文图",
            "图 4-1 主任务 train-range 与 held-out 成功率",
            "outputs/figures/main_task_success.svg",
            "docs/evaluation_summary.csv；docs/method_evidence_gate.csv",
            "对比 expert、结构化控制、普通 BC、trajectory/ACT、Diffusion 和 action-head proxy 在训练范围与留出范围的闭环成功率。",
            "环境和任务可解，但多数 learned baseline 在留出范围缺乏稳定闭环能力。",
            "不能只用离线 loss 或单个视频替代闭环成功率；当前结果是 MuJoCo，不是真实 WidowX。",
            ps_open("outputs/figures/main_task_success.svg"),
        ),
        row(
            "F02",
            "第 4 章 语言/空间泛化",
            "论文图",
            "图 4-2 语言/空间泛化成功率",
            "outputs/figures/language_success.svg",
            "docs/language_generalization_summary.csv；docs/video_evidence_index.md",
            "展示 leftmost-to-bowl 等语言/空间任务下，不同策略对目标描述变化的闭环表现。",
            "当前语言/空间任务已经建立，规则或结构化策略可以作为 oracle，上层 learned proxy 尚未形成真实 VLA 语义泛化。",
            "规则解析、对象特征或 frozen CLIP proxy 不能写成完整 VLM/VLA 语言理解能力。",
            ps_open("outputs/figures/language_success.svg"),
        ),
        row(
            "F03",
            "第 4 章 资源与效果",
            "论文图",
            "图 4-3 参数量与 held-out 成功率",
            "outputs/figures/resource_vs_success.svg",
            "docs/model_resource_summary.csv；docs/method_comparison_dashboard.csv",
            "把可训练参数、模型大小和 held-out 成功率放在同一张图中，说明轻量参数并不自动带来闭环泛化。",
            "LoRA/Adapter/action-head proxy 的资源占用可对比，但当前仍是本地代理实验。",
            "不能把本地图中的 Adapter/LoRA-style proxy 写成真实 OpenVLA LoRA 或 OpenVLA-OFT 结果。",
            ps_open("outputs/figures/resource_vs_success.svg"),
        ),
        row(
            "F04",
            "第 4 章 数据效率",
            "论文图",
            "图 4-4 小规模 demonstration 数据效率曲线",
            "outputs/figures/data_efficiency.svg",
            "docs/data_efficiency_summary.csv；outputs/evaluations/data_efficiency_v2.json",
            "比较 10/25/50/92 条 scripted demonstration 条件下的训练范围和留出范围表现。",
            "小数据条件下，轨迹记忆类方法在训练范围更容易成功，留出范围仍明显不足。",
            "不能把 MuJoCo scripted demonstration 结论直接写成真实机械臂数据效率或真实 VLA 小样本优势。",
            ps_open("outputs/figures/data_efficiency.svg"),
        ),
        row(
            "T01",
            "附录 A 方法结果总表",
            "论文表",
            "表 A-1 正式方法版本结果、资源与视频证据",
            "docs/thesis_method_comparison_table.csv",
            "docs/thesis_appendix_tables.md；docs/method_comparison_dashboard.html",
            "汇总 25 个正式方法版本的阶段分组、主任务、语言泛化、参数规模、视频数和论文红线。",
            "可支撑普通 BC、trajectory-conditioned BC、ACT-style、Diffusion、action-head/PEFT proxy 的横向比较。",
            "表中 proxy、state-only、lite、candidate 的边界必须保留，不能改写为完整官方方法或真实机器人结果。",
            ps_open("docs/thesis_method_comparison_table.csv", notepad=True),
        ),
        row(
            "T02",
            "附录 A Domain randomization",
            "论文表",
            "表 A-2 MuJoCo domain randomization 代理评测",
            "docs/thesis_domain_randomization_table.csv",
            "docs/domain_randomization_summary.md；outputs/evaluations/domain_randomization_eval_v1.json",
            "汇总摩擦、执行器力限和夹爪力度扰动下的成功率与目标距离。",
            "MuJoCo domain randomization 代理可作为 Isaac 和真实机械臂前的鲁棒性前置检查。",
            "MuJoCo domain randomization 代理不能写成 Isaac domain randomization 已完成，也不能写成真实 WidowX 迁移验证。",
            ps_open("docs/thesis_domain_randomization_table.csv", notepad=True),
        ),
        row(
            "H01",
            "答辩现场 横向比较",
            "HTML 看板",
            "方法评测比较看板",
            "docs/method_comparison_dashboard.html",
            "docs/method_comparison_dashboard.md；docs/method_comparison_dashboard.csv",
            "method_comparison_dashboard_v1：按阶段、成功率、参数、训练资源、固定视频和慢速 viewer 命令筛选 25 个正式方法。",
            "用于现场解释不同阶段和不同方法之间的量化差异。",
            "当前完成的是 MuJoCo 实验包；真实 OpenVLA、Isaac 和真实 WidowX 仍是后续阶段。",
            ps_open("docs/method_comparison_dashboard.html"),
        ),
        row(
            "H02",
            "答辩现场 视频播放",
            "HTML 播放清单",
            "答辩视频播放清单",
            "docs/defense_video_playlist.html",
            "docs/defense_video_playlist.md；docs/defense_video_playlist.csv",
            "defense_video_playlist_v1：按 claim 顺序播放阶段短片，并附带候选诊断负例、讲稿提示、viewer 命令和导出命令。",
            "用于现场展示仿真视频片段和每个 claim 的证据入口。",
            "视频是定性展示证据，不替代成功率、目标距离、资源和泛化表。",
            ps_open("docs/defense_video_playlist.html"),
        ),
        row(
            "H03",
            "答辩现场 阶段讲解",
            "HTML 索引",
            "阶段展示总索引",
            "docs/stage_showcase_index.html",
            "docs/stage_showcase_index.md；docs/stage_reproduction_runbook.md",
            "按阶段列出目标、方法版本、视频证据、量化表和 viewer 命令。",
            "用于从任务/数据、普通 BC、trajectory/ACT、action-head、语言泛化和 domain randomization 逐段说明。",
            "阶段索引只组织证据，不新增实验结论。",
            ps_open("docs/stage_showcase_index.html"),
        ),
        row(
            "H04",
            "第 5 章 后续 VLA 桥接",
            "HTML/图像",
            "OpenVLA 数据桥接样本浏览",
            "docs/openvla_bridge_gallery.html",
            "data/vla_bridge/openvla_dataset_bridge_v1/preview_grid.png；docs/openvla_dataset_bridge_report.md",
            "展示 MuJoCo episode 已被整理成后续机器人 VLA/action-head 可读取的数据桥接样本。",
            "证明后续接入 OpenVLA/机器人 VLA 的数据格式和交接门禁已经准备好。",
            "bridge 样本不是 OpenVLA 训练结果，也不是真实机器人验证结果。",
            ps_open("docs/openvla_bridge_gallery.html"),
        ),
        row(
            "V00",
            "答辩开场",
            "视频",
            "总览视频",
            "outputs/presentation_clips/00_defense_video_reel.mp4",
            "docs/video_presentation_storyboard.html；docs/video_evidence_gallery.html",
            "用一个短片快速串起任务、baseline、trajectory/ACT、action-head、语言泛化和 domain randomization 代理。",
            "可作为答辩开场的定性总览。",
            "总览视频不能替代分项量化表。",
            ps_open("outputs/presentation_clips/00_defense_video_reel.mp4"),
        ),
        row(
            "V01",
            "答辩阶段 1",
            "视频",
            "任务与数据链路",
            "outputs/presentation_clips/01_task_data_oracle.mp4",
            "docs/task_bc_stage_report.md；docs/evaluation_summary.csv",
            "展示 expert、structured waypoint 和 replay，证明桌面抓取/放置链路可解且示范可复现。",
            "可支撑环境、任务、数据采集和 replay 验证已经闭环。",
            "expert、structured waypoint 和 replay 不能写成 learned VLA。",
            ps_open("outputs/presentation_clips/01_task_data_oracle.mp4"),
        ),
        row(
            "V02",
            "答辩阶段 2",
            "视频",
            "普通 BC 对照",
            "outputs/presentation_clips/02_basic_bc_baselines.mp4",
            "docs/task_bc_stage_report.md；docs/failure_mode_taxonomy.md",
            "展示 linear/MLP BC 失败和 kNN 轨迹记忆差异。",
            "可支撑普通简单 BC 在接触、夹紧、抬升和泛化上的不足。",
            "普通 BC 不能写成语言理解、任务泛化或 VLA 后训练结果。",
            ps_open("outputs/presentation_clips/02_basic_bc_baselines.mp4"),
        ),
        row(
            "V03",
            "答辩阶段 3",
            "视频",
            "Trajectory / ACT / Diffusion 对照",
            "outputs/presentation_clips/03_trajectory_act_diffusion.mp4",
            "docs/trajectory_act_stage_report.md；docs/trajectory_act_experiment_record.md；docs/trajectory_act_failure_diagnosis.md",
            "展示 trajectory-conditioned BC / ACT-style / Diffusion baseline，说明历史观测和动作块接口已经建立。",
            "可支撑 trajectory-conditioned BC / ACT 阶段完成，并有中文实验记录和失败诊断。",
            "当前是 state-only 或轻量本地 baseline，不能写成完整官方 ACT 或完整视觉 Diffusion Policy。",
            ps_open("outputs/presentation_clips/03_trajectory_act_diffusion.mp4"),
        ),
        row(
            "V04",
            "答辩阶段 4",
            "视频",
            "Action-head / PEFT proxy",
            "outputs/presentation_clips/04_action_head_peft_proxy.mp4",
            "docs/action_head_stage_report.md；docs/model_resource_summary.csv",
            "展示 action-head、Adapter、LoRA-style 和 CLIP proxy 的闭环表现与资源规模。",
            "可支撑轻量 action-head/PEFT proxy 的资源比较和失败边界。",
            "Adapter/LoRA-style 是本地 proxy，不是 OpenVLA LoRA；CLIP 不是机器人 VLA。",
            ps_open("outputs/presentation_clips/04_action_head_peft_proxy.mp4"),
        ),
        row(
            "V05",
            "答辩阶段 5",
            "视频",
            "语言/空间泛化",
            "outputs/presentation_clips/05_language_generalization.mp4",
            "docs/language_generalization_summary.csv；docs/failure_mode_taxonomy.md",
            "展示 leftmost-to-bowl 等语言/空间变化任务。",
            "可支撑语言泛化测试已经建立，当前 learned proxy 多数没有形成真实泛化。",
            "规则、对象特征、language token 或 frozen CLIP action head 不能写成完整 VLA 语义泛化。",
            ps_open("outputs/presentation_clips/05_language_generalization.mp4"),
        ),
        row(
            "V06",
            "答辩阶段 6",
            "视频",
            "MuJoCo domain randomization 代理",
            "outputs/presentation_clips/06_domain_randomization_proxy.mp4",
            "docs/domain_randomization_summary.md；docs/isaac_domain_randomization_handoff.md",
            "展示低摩擦、弱夹爪和执行器扰动下的代表行为。",
            "可支撑 MuJoCo domain randomization 代理评测已完成，Isaac 交接门禁已准备。",
            "不能写成 Isaac 已运行，也不能写成真实 WidowX sim-to-real 迁移成功或失败。",
            ps_open("outputs/presentation_clips/06_domain_randomization_proxy.mp4"),
        ),
        row(
            "C01",
            "答辩追问 候选负例",
            "诊断视频",
            "Trajectory phase template BC 候选",
            "outputs/videos/trajectory_phase_template_bc_v1_candidate_seed1.mp4",
            "docs/trajectory_phase_template_bc_report.md；outputs/evaluations/trajectory_phase_template_bc_v1.json",
            "展示显式 phase 模板仍不能稳定解决真实抓取和抬升。",
            "可用于解释为什么候选改进没有升级为正式可靠 baseline。",
            "grasp_successes=0 的候选负例不能写成稳定抓取成功。",
            ps_open("outputs/videos/trajectory_phase_template_bc_v1_candidate_seed1.mp4"),
        ),
        row(
            "C02",
            "答辩追问 候选负例",
            "诊断视频",
            "Grasp-gated trajectory-conditioned chunk BC 候选",
            "outputs/videos/grasp_gated_trajectory_conditioned_chunk_bc_v1_candidate_seed0.mp4",
            "docs/grasp_gated_trajectory_act_report.md；outputs/evaluations/grasp_gated_trajectory_act_v1_candidate.json",
            "展示加入保守夹爪门控和慢速动作后，trajectory-conditioned chunk BC 仍缺乏严格抓取成功。",
            "可支撑失败不是单纯 viewer 速度太快造成的。",
            "原始放置成功不能写成稳定抓取成功；必须同时报告 strict grasp 口径。",
            ps_open("outputs/videos/grasp_gated_trajectory_conditioned_chunk_bc_v1_candidate_seed0.mp4"),
        ),
        row(
            "C03",
            "答辩追问 候选负例",
            "诊断视频",
            "Grasp-gated Torch ACT 候选",
            "outputs/videos/grasp_gated_torch_act_state_chunk_v1_candidate_seed0.mp4",
            "docs/grasp_gated_trajectory_act_report.md；outputs/evaluations/grasp_gated_trajectory_act_v1_candidate.json",
            "展示 Torch ACT-style 在相同夹爪门控和慢速动作下仍不能稳定完成抓取/放置。",
            "可用于说明当前 ACT-style baseline 的主要瓶颈是接触和抓取阶段建模。",
            "不能写成完整官方 ACT，也不能把候选诊断写成正式成功结果。",
            ps_open("outputs/videos/grasp_gated_torch_act_state_chunk_v1_candidate_seed0.mp4"),
        ),
        row(
            "C04",
            "答辩追问 候选负例",
            "诊断视频",
            "Grasp-gated trajectory-kNN 候选",
            "outputs/videos/grasp_gated_trajectory_knn_v1_candidate_seed0.mp4",
            "docs/grasp_gated_trajectory_knn_report.md；outputs/evaluations/grasp_gated_trajectory_knn_v1.json",
            "展示单独夹爪门控不能把轨迹记忆转化为真实抓取策略。",
            "可支撑 trajectory-kNN 的训练范围成功更像记忆而不是泛化控制。",
            "训练范围成功不能写成策略泛化，grasp_successes=0 不能写成稳定抓取。",
            ps_open("outputs/videos/grasp_gated_trajectory_knn_v1_candidate_seed0.mp4"),
        ),
        row(
            "C05",
            "答辩追问 候选负例",
            "诊断视频",
            "Preference trajectory post-training 候选",
            "outputs/videos/preference_trajectory_post_training_v1_candidate_seed0.mp4",
            "docs/preference_trajectory_post_training_report.md；outputs/evaluations/preference_trajectory_post_training_v1.json",
            "展示偏好加权 trajectory 后训练主要改善训练范围轨迹复现，留出和严格抓取仍不足。",
            "可支撑轻量偏好/后训练候选已做诊断，但未形成正式稳定策略。",
            "不能写成在线 RL、真实 preference optimization 或真实抓取成功。",
            ps_open("outputs/videos/preference_trajectory_post_training_v1_candidate_seed0.mp4"),
        ),
        row(
            "C06",
            "答辩追问 上界诊断",
            "诊断视频",
            "Grasp/Lift 子策略上界诊断",
            "outputs/videos/grasp_lift_subpolicy_probe_v1_candidate_seed0.mp4",
            "docs/grasp_lift_subpolicy_probe_report.md；outputs/evaluations/grasp_lift_subpolicy_probe_v1_candidate.json",
            "展示 scripted expert / IK 控制上界能完成放置和 TCP 抬升，但标准 grasp_success 口径仍需单独报告。",
            "可用于解释 trajectory-conditioned BC / ACT 失败不是因为环境完全不可执行抓放流程。",
            "不能写成 learned BC/ACT/VLA baseline 成功，也不能替代真实 WidowX 验证。",
            ps_open("outputs/videos/grasp_lift_subpolicy_probe_v1_candidate_seed0.mp4"),
        ),
    ]


def split_sources(value: str) -> list[str]:
    return [part.strip() for part in value.replace("；", "\n").splitlines() if part.strip()]


def verify_rows(rows: list[dict[str, str]]) -> None:
    missing: list[str] = []
    for item in rows:
        for path_text in [item["证据文件"], *split_sources(item["配套证据"])]:
            path = ROOT / path_text
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("\n".join(missing))


def media_kind(path_text: str) -> str:
    suffix = Path(path_text).suffix.lower()
    if suffix in {".mp4", ".mov", ".webm", ".avi"}:
        return "video"
    if suffix in {".svg", ".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        return "image"
    if suffix in {".html", ".htm"}:
        return "html"
    return "file"


def web_path(path_text: str) -> str:
    path = Path(path_text.replace("\\", "/"))
    if path.is_absolute():
        try:
            path = path.relative_to(ROOT)
        except ValueError:
            return path.as_posix()
    return (Path("..") / path).as_posix()


def media_html(item: dict[str, str]) -> str:
    kind = media_kind(item["证据文件"])
    src = escape(web_path(item["证据文件"]))
    title = escape(item["建议标题"])
    if kind == "video":
        return f'<video controls preload="metadata" src="{src}"></video>'
    if kind == "image":
        return f'<img src="{src}" alt="{title}">'
    return f'<a class="file-link" href="{src}">打开证据文件</a>'


def support_links(value: str) -> str:
    links = []
    for path_text in split_sources(value):
        links.append(f'<a href="{escape(web_path(path_text))}">{escape(path_text)}</a>')
    return "<br>".join(links)


def build_html(rows: list[dict[str, str]]) -> str:
    generated_at = datetime.now().isoformat(timespec="seconds")
    cards = []
    for item in rows:
        cards.append(
            f"""
<article class="card" data-kind="{escape(item['类型'])}">
  <div class="preview">{media_html(item)}</div>
  <div class="body">
    <p class="eyebrow">{escape(item['编号'])} · {escape(item['论文或答辩位置'])} · {escape(item['类型'])}</p>
    <h2>{escape(item['建议标题'])}</h2>
    <p><b>图注/表注/讲解：</b>{escape(item['中文图注/表注/讲解'])}</p>
    <p><b>可支撑结论：</b>{escape(item['可支撑结论'])}</p>
    <p><b>论文红线：</b>{escape(item['论文红线'])}</p>
    <p><b>配套证据：</b><br>{support_links(item['配套证据'])}</p>
    <details><summary>打开命令</summary><pre>{escape(item['打开命令'])}</pre></details>
  </div>
</article>
""".strip()
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>论文图表与视频证据索引</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fa;
      --panel: #fff;
      --text: #17202a;
      --muted: #667085;
      --line: #d8dee8;
      --accent: #0f766e;
      --warn: #9a3412;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    header {{
      padding: 28px clamp(18px, 4vw, 56px) 18px;
      background: #fff;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0 0 8px; font-size: clamp(26px, 3vw, 40px); letter-spacing: 0; }}
    .subtitle {{ max-width: 1100px; margin: 0; color: var(--muted); line-height: 1.7; }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
      max-width: 860px;
      margin-top: 18px;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #f9fafb;
      padding: 12px;
    }}
    .stat b {{ display: block; font-size: 22px; color: var(--accent); }}
    main {{ padding: 26px clamp(18px, 4vw, 56px) 48px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(390px, 1fr)); gap: 16px; }}
    .card {{
      display: grid;
      grid-template-rows: auto 1fr;
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }}
    .preview {{
      aspect-ratio: 16 / 9;
      display: grid;
      place-items: center;
      background: #101828;
      padding: 8px;
    }}
    video, img {{ width: 100%; height: 100%; object-fit: contain; display: block; }}
    .file-link {{
      color: #fff;
      border: 1px solid #475467;
      padding: 10px 12px;
      border-radius: 6px;
      text-decoration: none;
    }}
    .body {{ padding: 14px; min-width: 0; }}
    .eyebrow {{ margin: 0 0 6px; color: var(--accent); font-weight: 700; font-size: 13px; }}
    h2 {{ margin: 0 0 10px; font-size: 18px; line-height: 1.45; letter-spacing: 0; }}
    p {{ margin: 8px 0; line-height: 1.68; color: #344054; }}
    a {{ color: #0f766e; word-break: break-all; }}
    details {{ margin-top: 10px; border-top: 1px solid var(--line); padding-top: 8px; }}
    summary {{ cursor: pointer; color: var(--warn); font-weight: 700; }}
    pre {{
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      background: #f2f4f7;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      line-height: 1.45;
      font-size: 12px;
    }}
    footer {{ padding: 0 clamp(18px, 4vw, 56px) 32px; color: var(--muted); }}
    @media (max-width: 720px) {{
      .stats {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>论文图表与视频证据索引</h1>
    <p class="subtitle">版本：<code>{VERSION}</code>。本索引用中文把论文图、附录表、答辩 HTML、阶段视频和候选诊断视频映射到可支撑结论与论文红线；它只组织现有证据，不新增实验结果。</p>
    <div class="stats">
      <div class="stat"><b>{len(rows)}</b><span>索引条目</span></div>
      <div class="stat"><b>{sum(1 for item in rows if item['类型'] == '视频')}</b><span>阶段视频</span></div>
      <div class="stat"><b>{sum(1 for item in rows if '图' in item['类型'])}</b><span>论文图</span></div>
      <div class="stat"><b>{sum(1 for item in rows if '诊断' in item['类型'])}</b><span>候选诊断</span></div>
    </div>
  </header>
  <main><div class="grid">{"".join(cards)}</div></main>
  <footer>生成时间：{escape(generated_at)}</footer>
</body>
</html>
"""


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(value.replace("|", "\\|") for value in values) + " |"


def build_md(rows: list[dict[str, str]]) -> str:
    lines = [
        "# 论文图表与视频证据索引",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：把当前 MuJoCo 实验包中的论文图、附录表、答辩 HTML、阶段视频和候选诊断视频集中成中文引用索引。该文件只组织已有证据，不新增实验结论。",
        "",
        "## 1. 打开 HTML 索引",
        "",
        "```powershell",
        ps_open("docs/thesis_visual_evidence_index.html"),
        "```",
        "",
        "## 2. 重建命令",
        "",
        "```powershell",
        ps_build("scripts/build_thesis_visual_evidence_index.py"),
        "```",
        "",
        "## 3. 引用索引",
        "",
        md_row(["编号", "位置", "类型", "建议标题", "证据文件", "图注/表注/讲解", "论文红线"]),
        md_row(["---", "---", "---", "---", "---", "---", "---"]),
    ]
    for item in rows:
        lines.append(
            md_row(
                [
                    item["编号"],
                    item["论文或答辩位置"],
                    item["类型"],
                    item["建议标题"],
                    f"`{item['证据文件']}`",
                    item["中文图注/表注/讲解"],
                    item["论文红线"],
                ]
            )
        )
    lines.extend(
        [
            "",
            "## 4. 使用边界",
            "",
            "- 视频证据只用于定性展示，不能替代成功率、目标距离、资源消耗和语言泛化 CSV/JSON。",
            "- `MuJoCo domain randomization 代理` 不能写成 Isaac domain randomization 已完成，也不能写成真实 WidowX sim-to-real 成功或失败。",
            "- `Adapter/LoRA-style proxy`、`CLIP action-head proxy` 不能写成真实 OpenVLA、RT-2 或 OpenVLA-OFT 训练结果。",
            "- `trajectory_act_experiment_record_v1` 和候选诊断视频可说明 trajectory-conditioned BC / ACT 阶段已经做了系统实验，但候选负例不能写成正式可靠策略。",
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
    rows = build_rows()
    verify_rows(rows)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_md(rows), encoding="utf-8")
    write_csv(args.output_csv, rows)
    args.output_html.write_text(build_html(rows), encoding="utf-8")
    print(f"thesis_visual_evidence_index_md: {args.output_md}", flush=True)
    print(f"thesis_visual_evidence_index_csv: {args.output_csv}", flush=True)
    print(f"thesis_visual_evidence_index_html: {args.output_html}", flush=True)
    print(f"thesis_visual_evidence_rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
