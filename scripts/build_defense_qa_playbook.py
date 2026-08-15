from __future__ import annotations

import argparse
import csv
from datetime import datetime
from html import escape
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "defense_qa_playbook_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese defense Q&A playbook with evidence links.")
    parser.add_argument("--strict-grasp-json", type=Path, default=ROOT / "outputs" / "evaluations" / "strict_grasp_success_audit_v1.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "defense_qa_playbook.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "defense_qa_playbook.csv")
    parser.add_argument("--output-html", type=Path, default=ROOT / "docs" / "defense_qa_playbook.html")
    return parser.parse_args()


def ps_open(path_text: str, *, notepad: bool = False) -> str:
    path = ROOT / path_text
    if notepad:
        return f'Start-Process notepad.exe "{path}"'
    return f'Start-Process "{path}"'


def ps_launcher(*args: str) -> str:
    rendered = " ".join(args)
    return f'& "{PYTHON}" "{ROOT / "scripts" / "showcase_launcher.py"}" {rendered}'


def ps_build() -> str:
    return f'& "{PYTHON}" "{ROOT / "scripts" / "build_defense_qa_playbook.py"}"'


def row(
    qid: str,
    topic: str,
    stage: str,
    question: str,
    answer: str,
    evidence: str,
    media: str,
    command: str,
    redline: str,
) -> dict[str, str]:
    return {
        "编号": qid,
        "追问主题": topic,
        "适用阶段": stage,
        "可能问题": question,
        "推荐回答": answer,
        "首选证据": evidence,
        "首选图表或视频": media,
        "现场打开命令": command,
        "必须坚持的边界": redline,
    }


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def strict_counts(summary: dict) -> tuple[str, str]:
    episodes = summary.get("episodes", "?")
    loose = f"{summary.get('loose_successes', '?')}/{episodes}"
    strict = f"{summary.get('strict_grasp_successes', '?')}/{episodes}"
    return loose, strict


def build_rows(strict_summary: dict) -> list[dict[str, str]]:
    loose, strict = strict_counts(strict_summary)
    return [
        row(
            "Q01",
            "任务设计",
            "阶段 1",
            "为什么用把蓝色方块放到蓝色盘子的任务，它是否太简单？",
            "这是初级可视化任务，但不是无意义玩具任务；它包含颜色目标、干扰物体、接触、夹紧、抬升、移动和放置，能区分 expert、普通 BC、trajectory/ACT、action-head proxy 和 domain randomization 行为。",
            "docs/task_bc_stage_report.md；docs/stage_showcase_index.md；docs/evaluation_summary.csv",
            "outputs/presentation_clips/01_task_data_oracle.mp4",
            ps_open("outputs/presentation_clips/01_task_data_oracle.mp4"),
            "不能把该任务写成真实生产场景，也不能只凭视频断言策略泛化。",
        ),
        row(
            "Q02",
            "数据可复现",
            "阶段 1",
            "你怎么证明采集的 demonstration 不是随机碰巧成功？",
            "我保存了 success、seed、任务、物体位置和动作轨迹，并用 replay_demo_v1 回放验证相同轨迹可以复现执行过程。",
            "docs/task_bc_stage_report.md；docs/method_evidence_gate.csv；data/demos/place_blue_cube_blue_pad_medium_20260702_051752",
            "outputs/videos/replay_demo_v1_seed0.mp4",
            ps_launcher("--target method:replay_demo_v1"),
            "Replay 只能证明数据可复现，不能写成 learned policy。",
        ),
        row(
            "Q03",
            "普通 BC 失败原因",
            "阶段 2",
            "为什么 linear/MLP BC 效果差，是不是只是速度太快或 viewer 参数不合适？",
            "我已经统一慢速 viewer，并额外做了控制限幅扫表和严格抓取审计；失败主要体现在接触、夹紧、抬升和时序阶段结构，而不是单纯播放速度。",
            "docs/task_bc_stage_report.md；docs/control_safety_sweep.md；docs/strict_grasp_success_audit.md",
            "outputs/presentation_clips/02_basic_bc_baselines.mp4",
            ps_open("outputs/presentation_clips/02_basic_bc_baselines.mp4"),
            "不能用离线 MSE 或短视频观感替代闭环成功率和 strict grasp 口径。",
        ),
        row(
            "Q04",
            "kNN 训练范围成功",
            "阶段 2",
            "kNN/trajectory-kNN 训练范围成功率高，是否说明普通方法已经足够？",
            "不是。kNN 类方法在训练范围更像轨迹记忆，held-out、语言/空间泛化和严格抓取口径都暴露出不足。",
            "docs/evaluation_summary.csv；docs/language_generalization_summary.csv；docs/method_comparison_dashboard.html",
            "outputs/videos/trajectory_knn_chunk_bc_v1_seed0.mp4",
            ps_launcher("--target method:trajectory_knn_chunk_bc_v1 --action viewer --dry-run"),
            "训练范围成功不能写成策略泛化，也不能写成真实 grasp 已解决。",
        ),
        row(
            "Q05",
            "ACT 定位",
            "阶段 3",
            "你这里的 ACT 是不是官方完整 ACT？",
            "不是。当前是本地轻量 ACT-style/state-only/ACT-CVAE/visual-feature baseline，用来做对照和失败诊断；它证明接口和评测链路建立了，但不能冒充完整官方视觉 ACT。",
            "docs/trajectory_act_stage_report.md；docs/trajectory_act_experiment_record.md；docs/trajectory_act_failure_diagnosis.md",
            "outputs/presentation_clips/03_trajectory_act_diffusion.mp4",
            ps_open("outputs/presentation_clips/03_trajectory_act_diffusion.mp4"),
            "不能写成完整官方 ACT、完整视觉 ACT 或真实机器人 ACT 复现。",
        ),
        row(
            "Q06",
            "Diffusion Policy 定位",
            "阶段 3",
            "Diffusion Policy baseline 为什么也失败？",
            "当前 diffusion 是 state/action-chunk 的轻量 baseline，数据规模小且接触阶段难；失败本身说明简单增加生成式动作模型并不能自动解决抓取和放置。",
            "docs/trajectory_act_stage_report.md；docs/model_resource_summary.csv；docs/failure_mode_taxonomy.md",
            "outputs/videos/torch_diffusion_policy_state_chunk_v1_seed0.mp4",
            ps_launcher("--target method:torch_diffusion_policy_state_chunk_v1"),
            "不能写成完整视觉 Diffusion Policy，也不能把失败视频当作所有 diffusion policy 的一般结论。",
        ),
        row(
            "Q07",
            "严格抓取口径",
            "阶段 3/4",
            "有些视频看起来放到盘子附近了，为什么你还说严格抓取成功是 0？",
            f"我把原始 success、grasp_success 和 object_z 分开审计；当前原始放置成功合计 {loose}，但严格抓取成功为 {strict}，所以不能把推到目标附近写成稳定抓取。",
            "docs/strict_grasp_success_audit.md；docs/trajectory_act_failure_diagnosis.md；docs/candidate_diagnostic_video_index.md",
            "outputs/videos/grasp_gated_torch_act_state_chunk_v1_candidate_seed0.mp4",
            ps_launcher("--target candidate:grasp_gated_torch_act_state_chunk_v1_candidate --action open-all"),
            "必须同时报告 success、grasp_success、object_z 和视频证据。",
        ),
        row(
            "Q08",
            "动作太快问题",
            "阶段 3/4",
            "你已经把动作放慢，为什么还是抓不住？",
            "我做了 grasp-gated、慢速动作、夹爪增强和控制限幅诊断；这些候选负例说明失败不是单纯速度问题，还涉及阶段结构、接触状态和训练分布。",
            "docs/control_safety_sweep.md；docs/grasp_gated_trajectory_act_report.md；docs/grasp_gated_trajectory_knn_report.md",
            "outputs/videos/grasp_gated_trajectory_conditioned_chunk_bc_v1_candidate_seed0.mp4",
            ps_launcher("--target candidate:grasp_gated_trajectory_conditioned_chunk_bc_v1_candidate --action viewer --dry-run"),
            "候选诊断不能登记为正式可靠 baseline；原始 success 不能替代 strict grasp。",
        ),
        row(
            "Q09",
            "Action-head/PEFT",
            "阶段 4",
            "Adapter/LoRA-style 是否已经证明轻量化 VLA 后训练有效？",
            "只能说本地 action-head/PEFT proxy 的资源对比链路已经建立；参数量和训练时间可比较，但当前没有证明真实 pretrained VLA 后训练有效。",
            "docs/action_head_stage_report.md；docs/model_resource_summary.csv；docs/method_comparison_dashboard.html",
            "outputs/presentation_clips/04_action_head_peft_proxy.mp4",
            ps_open("outputs/presentation_clips/04_action_head_peft_proxy.mp4"),
            "Adapter/LoRA-style 是本地 proxy，不是 OpenVLA LoRA；不能写成真实 pretrained VLA 后训练。",
        ),
        row(
            "Q10",
            "CLIP/VLM 表征",
            "阶段 4/5",
            "CLIP action head 是否说明模型理解语言了？",
            "不能这样说。CLIP/frozen feature 只作为本地 VLM 表征 proxy；语言/空间评测已经建立，但当前 learned proxy 多数为 0/5，不能等同于真实 VLA 语言理解。",
            "docs/language_generalization_summary.csv；docs/action_head_stage_report.md；docs/thesis_visual_evidence_index.html",
            "outputs/presentation_clips/05_language_generalization.mp4",
            ps_open("outputs/presentation_clips/05_language_generalization.mp4"),
            "规则解析、对象特征、语言 token 或 frozen CLIP 不能写成完整 VLM/VLA 语义泛化能力。",
        ),
        row(
            "Q11",
            "数据效率",
            "阶段 5",
            "小数据下哪个方法最好，能否证明轻量 VLA 更省数据？",
            "当前只完成 MuJoCo scripted demonstration 的数据效率扫表；可以比较 10/25/50/92 条示范下不同 baseline 的行为，但不能外推为真实机械臂或真实 VLA 小数据优势。",
            "docs/data_efficiency_summary.md；docs/data_efficiency_summary.csv；outputs/figures/data_efficiency.svg",
            "outputs/figures/data_efficiency.svg",
            ps_open("outputs/figures/data_efficiency.svg"),
            "不能把 MuJoCo scripted demonstration 结论直接写成真实机器人数据效率。",
        ),
        row(
            "Q12",
            "Domain Randomization",
            "阶段 6",
            "你是否已经完成 Isaac domain randomization 或 sim-to-real？",
            "没有。当前完成的是 MuJoCo 摩擦、力限和夹爪扰动代理评测，并建立了 Isaac 与真实 WidowX 回填门禁；Isaac 实际运行和真实机械臂 trial 仍是后续阶段。",
            "docs/domain_randomization_summary.md；docs/isaac_domain_randomization_handoff.md；docs/real_widowx_validation_handoff.md",
            "outputs/presentation_clips/06_domain_randomization_proxy.mp4",
            ps_open("outputs/presentation_clips/06_domain_randomization_proxy.mp4"),
            "MuJoCo domain randomization 不能写成 Isaac 已完成，也不能写成真实 WidowX 迁移验证。",
        ),
        row(
            "Q13",
            "OpenVLA 后续",
            "阶段 7",
            "你的课题叫 VLA，为什么现在还没有真实 OpenVLA 结果？",
            "毕业设计路线不是从零训练大 VLA；当前本机已经完成 MuJoCo 对照组、数据桥接、本地可行性检查、远端运行包和结果回填门禁。真实 OpenVLA/robot VLA 后训练需要 48GB+ GPU 或云端，后续按同一评测字段回填。",
            "docs/openvla_dataset_bridge_report.md；docs/openvla_feasibility_report.md；docs/robot_vla_remote_run_pack.md；docs/robot_vla_remote_result_intake.md",
            "docs/openvla_bridge_gallery.html",
            ps_open("docs/openvla_bridge_gallery.html"),
            "Bridge、feasibility、remote pack、intake 都不是策略效果；不能写成 OpenVLA LoRA 已完成。",
        ),
        row(
            "Q14",
            "复现与可视化",
            "追问入口",
            "现场如果想看某个方法，怎么打开？",
            "用 showcase_launcher 按 quick、claim、stage、method 或 candidate 打开固定视频、报告和慢速 MuJoCo viewer；viewer 默认 60 秒、0.05 倍速，便于观察抓取过程。",
            "docs/showcase_launcher_guide.md；docs/reproducible_command_index.md；docs/method_comparison_dashboard.html",
            "docs/method_comparison_dashboard.html",
            ps_launcher("--target comparison"),
            "现场 viewer 只用于可视化复现，不改变已登记成功率。",
        ),
    ]


def split_paths(value: str) -> list[str]:
    return [part.strip() for part in value.replace("；", "\n").splitlines() if part.strip()]


def verify_paths(rows: list[dict[str, str]]) -> None:
    missing: list[str] = []
    for item in rows:
        for path_text in [*split_paths(item["首选证据"]), item["首选图表或视频"]]:
            path = ROOT / path_text
            if not path.exists():
                missing.append(str(path))
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


def media_html(path_text: str) -> str:
    suffix = Path(path_text).suffix.lower()
    src = escape(web_path(path_text))
    if suffix in {".mp4", ".webm", ".mov", ".avi"}:
        return f'<video controls preload="metadata" src="{src}"></video>'
    if suffix in {".svg", ".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        return f'<img src="{src}" alt="">'
    return f'<a class="file-link" href="{src}">打开文件</a>'


def evidence_links(value: str) -> str:
    return "<br>".join(f'<a href="{escape(web_path(path))}">{escape(path)}</a>' for path in split_paths(value))


def build_html(rows: list[dict[str, str]]) -> str:
    cards = []
    for item in rows:
        cards.append(
            f"""
<article class="card">
  <div class="preview">{media_html(item['首选图表或视频'])}</div>
  <div class="body">
    <p class="eyebrow">{escape(item['编号'])} · {escape(item['适用阶段'])} · {escape(item['追问主题'])}</p>
    <h2>{escape(item['可能问题'])}</h2>
    <p><b>推荐回答：</b>{escape(item['推荐回答'])}</p>
    <p><b>首选证据：</b><br>{evidence_links(item['首选证据'])}</p>
    <p><b>必须坚持的边界：</b>{escape(item['必须坚持的边界'])}</p>
    <details><summary>现场打开命令</summary><pre>{escape(item['现场打开命令'])}</pre></details>
  </div>
</article>
""".strip()
        )
    generated_at = datetime.now().isoformat(timespec="seconds")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>答辩追问 Q&A Playbook</title>
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
    .subtitle {{ max-width: 1080px; margin: 0; color: var(--muted); line-height: 1.7; }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(3, minmax(120px, 1fr));
      gap: 10px;
      max-width: 700px;
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
      .stats {{ grid-template-columns: 1fr; }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>答辩追问 Q&A Playbook</h1>
    <p class="subtitle">版本：<code>{VERSION}</code>。本页把常见追问映射到推荐回答、首选证据、视频/图表和必须坚持的论文边界；它只组织已有证据，不新增实验结论。</p>
    <div class="stats">
      <div class="stat"><b>{len(rows)}</b><span>追问条目</span></div>
      <div class="stat"><b>{sum(1 for item in rows if item['首选图表或视频'].endswith('.mp4'))}</b><span>视频入口</span></div>
      <div class="stat"><b>{sum(1 for item in rows if '不能' in item['必须坚持的边界'])}</b><span>红线提醒</span></div>
    </div>
  </header>
  <main><div class="grid">{"".join(cards)}</div></main>
  <footer>生成时间：{escape(generated_at)}</footer>
</body>
</html>
"""


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(value.replace("|", "\\|").replace("\n", " ") for value in values) + " |"


def build_md(rows: list[dict[str, str]]) -> str:
    lines = [
        "# 答辩追问 Q&A Playbook",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：把常见答辩追问映射到推荐回答、首选证据、视频/图表和必须坚持的论文边界。该文件只组织现有证据，不新增实验结论。",
        "",
        "## 1. 打开 HTML",
        "",
        "```powershell",
        ps_open("docs/defense_qa_playbook.html"),
        "```",
        "",
        "## 2. 重建命令",
        "",
        "```powershell",
        ps_build(),
        "```",
        "",
        "## 3. 追问矩阵",
        "",
        md_row(["编号", "主题", "阶段", "可能问题", "推荐回答", "首选证据", "首选图表或视频", "现场打开命令", "必须坚持的边界"]),
        md_row(["---", "---", "---", "---", "---", "---", "---", "---", "---"]),
    ]
    for item in rows:
        lines.append(
            md_row(
                [
                    item["编号"],
                    item["追问主题"],
                    item["适用阶段"],
                    item["可能问题"],
                    item["推荐回答"],
                    item["首选证据"],
                    f"`{item['首选图表或视频']}`",
                    f"`{item['现场打开命令']}`",
                    item["必须坚持的边界"],
                ]
            )
        )
    lines.extend(
        [
            "",
            "## 4. 使用边界",
            "",
            "- 本 playbook 不是评测表；成功率、目标距离、资源消耗和语言泛化仍以 CSV/JSON 为准。",
            "- 现场回答时优先打开对应视频或图表，但必须同时引用量化证据。",
            "- 不能把 MuJoCo proxy、候选负例、OpenVLA bridge、Isaac handoff 或真实 WidowX handoff 写成已完成的真实 VLA/真实机器人结果。",
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
    strict_summary = read_json(args.strict_grasp_json).get("summary", {})
    rows = build_rows(strict_summary)
    verify_paths(rows)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_md(rows), encoding="utf-8")
    write_csv(args.output_csv, rows)
    args.output_html.write_text(build_html(rows), encoding="utf-8")
    print(f"defense_qa_playbook_md: {args.output_md}", flush=True)
    print(f"defense_qa_playbook_csv: {args.output_csv}", flush=True)
    print(f"defense_qa_playbook_html: {args.output_html}", flush=True)
    print(f"defense_qa_rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
