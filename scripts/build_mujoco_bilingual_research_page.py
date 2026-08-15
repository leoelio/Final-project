from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a bilingual one-page MuJoCo research summary from recorded results.")
    parser.add_argument("--output-html", type=Path, default=ROOT / "docs" / "mujoco_research_summary_zh_en.html")
    parser.add_argument("--output-json", type=Path, default=ROOT / "docs" / "mujoco_research_summary_zh_en.json")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_ratio(value: str) -> tuple[int, int]:
    numerator, denominator = value.split("/", maxsplit=1)
    return int(numerator), int(denominator)


def aggregate_core_method(rows: list[dict[str, str]], method_key: str) -> dict[str, float | int | str]:
    selected = [row for row in rows if row["方法key"] == method_key]
    wins, episodes = 0, 0
    for row in selected:
        hit, total = parse_ratio(row["成功"])
        wins += hit
        episodes += total
    return {
        "successes": wins,
        "episodes": episodes,
        "success_rate": wins / episodes,
        "mean_target_distance": sum(float(row["平均目标距离"]) for row in selected) / len(selected),
    }


def aggregate_ood(payload: dict, condition: str) -> dict[str, int | float]:
    rows = [row for row in payload["rows"] if row["condition"] == condition]
    successes = sum(int(row["task_success"]) for row in rows)
    semantics = sum(int(row["semantic_correct"]) for row in rows)
    grasps = sum(int(row["strict_grasp_success"]) for row in rows)
    return {
        "episodes": len(rows),
        "successes": successes,
        "semantic_correct": semantics,
        "strict_grasps": grasps,
        "success_rate": successes / len(rows),
    }


def aggregate_normalized_core() -> dict[str, float | int]:
    suffixes = ("blue_to_blue", "blue_to_red", "red_to_red", "leftmost_cube")
    payloads = [read_json(ROOT / "outputs" / "evaluations" / f"core_v2_clip_semantic_normalized_{suffix}.json") for suffix in suffixes]
    successes = sum(parse_ratio(str(payload["success"]))[0] for payload in payloads)
    episodes = sum(parse_ratio(str(payload["success"]))[1] for payload in payloads)
    strict_grasps = sum(parse_ratio(str(payload["strict_grasp_success"]))[0] for payload in payloads)
    semantic_correct = sum(parse_ratio(str(payload["semantic_correct"]))[0] for payload in payloads)
    return {
        "successes": successes,
        "episodes": episodes,
        "strict_grasps": strict_grasps,
        "semantic_correct": semantic_correct,
        "success_rate": successes / episodes,
        "mean_target_distance": sum(float(payload["mean_target_distance"]) for payload in payloads) / len(payloads),
    }


def build_data() -> dict:
    base = read_json(ROOT / "outputs" / "evaluations" / "core_v2_clip_semantic_waypoint_v1.json")
    direct_clip = read_json(ROOT / "outputs" / "evaluations" / "core_v2_pretrained_vlm_action_head_v1.json")
    ood_base = read_json(ROOT / "outputs" / "evaluations" / "core_v2_clip_semantic_ood_generalization_v1.json")
    ood_alias = read_json(ROOT / "outputs" / "evaluations" / "core_v2_clip_semantic_ood_alias_v1.json")
    ood_normalized = read_json(ROOT / "outputs" / "evaluations" / "core_v2_clip_semantic_ood_normalized_v1.json")
    independent_syntax = read_json(ROOT / "outputs" / "evaluations" / "core_v2_clip_semantic_independent_syntax_v1.json")
    contact_fusion = read_json(ROOT / "outputs" / "evaluations" / "clip_semantic_contact_fusion_v1.json")
    final_closure = read_json(ROOT / "outputs" / "evaluations" / "final_closure_audit_v1.json")
    core_rows = read_csv(ROOT / "docs" / "core_v2_holdout_comparison_matrix.csv")
    efficiency_rows = read_csv(ROOT / "docs" / "core_v2_clip_semantic_data_efficiency.csv")
    versions = read_json(ROOT / "docs" / "experiment_versions.json")
    video_rows = read_csv(ROOT / "docs" / "video_evidence_index.csv")

    normalized_core = aggregate_normalized_core()
    method_summaries = {
        "linear_bc": aggregate_core_method(core_rows, "linear_bc"),
        "trajectory_knn": aggregate_core_method(core_rows, "trajectory_knn"),
        "object_action_head": aggregate_core_method(core_rows, "object_action_head"),
    }
    direct_summary = direct_clip["summary"]
    data_efficiency = []
    for budget in (5, 10, 20):
        selected = [row for row in efficiency_rows if int(row["demo_budget_per_task"]) == budget]
        success = sum(parse_ratio(row["success"])[0] for row in selected)
        episodes = sum(parse_ratio(row["success"])[1] for row in selected)
        data_efficiency.append({"budget": budget, "samples": int(selected[0]["stored_samples"]), "successes": success, "episodes": episodes})

    metadata = base["model_metadata"]
    return {
        "version": "mujoco_research_summary_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "MuJoCo-only WidowX tabletop manipulation",
        "work": {
            "formal_methods": len(versions["methods"]),
            "video_evidence": len(video_rows),
            "successful_demos": int(metadata["samples"]),
            "language_validation_episodes": len(ood_normalized["rows"]) + int(independent_syntax["episodes"]),
        },
        "primary": {
            "version": "clip_semantic_waypoint_core_v2_normalized_v1",
            "base_model": "clip_semantic_waypoint_core_v2_v1",
            "canonical": normalized_core,
            "ood": {
                "base": {condition: aggregate_ood(ood_base, condition) for condition in ("paraphrase", "hard_distractors")},
                "alias_candidate": {condition: aggregate_ood(ood_alias, condition) for condition in ("paraphrase", "hard_distractors")},
                "normalized": {condition: aggregate_ood(ood_normalized, condition) for condition in ("paraphrase", "hard_distractors")},
            },
            "independent_syntax": {
                "successes": parse_ratio(str(independent_syntax["task_success"]))[0],
                "episodes": parse_ratio(str(independent_syntax["task_success"]))[1],
                "semantic_correct": parse_ratio(str(independent_syntax["semantic_correct"]))[0],
                "strict_grasps": parse_ratio(str(independent_syntax["strict_grasp_success"]))[0],
                "mean_target_distance": float(independent_syntax["mean_target_distance"]),
            },
            "resources": {
                "frozen_encoder_params": int(metadata["frozen_encoder_params"]),
                "trainable_params": 4100,
                "train_time_seconds": float(metadata["train_time_seconds"]),
                "peak_vram_mb": float(metadata["peak_vram_mb"]),
                "samples": int(metadata["samples"]),
                "validation_accuracy": float(metadata["val_accuracy"]),
            },
        },
        "canonical_methods": [
            {"id": "linear", "label": {"zh": "线性 BC", "en": "Linear BC"}, **method_summaries["linear_bc"]},
            {"id": "trajectory", "label": {"zh": "Trajectory-kNN", "en": "Trajectory-kNN"}, **method_summaries["trajectory_knn"]},
            {"id": "object_head", "label": {"zh": "对象-语言动作头", "en": "Object-language action head"}, **method_summaries["object_action_head"]},
            {
                "id": "continuous_clip",
                "label": {"zh": "冻结 CLIP 连续动作头", "en": "Frozen CLIP continuous action head"},
                "successes": parse_ratio(str(direct_summary["success"]))[0],
                "episodes": parse_ratio(str(direct_summary["success"]))[1],
                "success_rate": float(direct_summary["success_rate"]),
                "mean_target_distance": float(direct_summary["mean_target_distance"]),
            },
            {
                "id": "semantic_waypoint",
                "label": {"zh": "CLIP 语义 + 结构化执行", "en": "CLIP semantic + structured execution"},
                "successes": parse_ratio(str(base["summary"]["success"]))[0],
                "episodes": parse_ratio(str(base["summary"]["success"]))[1],
                "success_rate": 1.0,
                "mean_target_distance": float(base["summary"]["mean_target_distance"]),
            },
            {
                "id": "normalized",
                "label": {"zh": "采用：CLIP 语义 + 词表规范化 + 结构化执行", "en": "Adopted: CLIP semantic + normalization + structured execution"},
                **normalized_core,
            },
        ],
        "data_efficiency": data_efficiency,
        "contact_fusion": {
            "version": contact_fusion["version"],
            "stress": contact_fusion["low_friction_40_paired"],
            "nominal": contact_fusion["nominal_20_paired"],
        },
        "final_closure": final_closure,
        "videos": {
            "blue_blue": "../outputs/videos/clip_semantic_waypoint_core_v2_normalized_v1_blue_blue_seed3300.mp4",
            "red_red": "../outputs/videos/clip_semantic_waypoint_core_v2_normalized_v1_red_red_seed3300.mp4",
            "leftmost_bowl": "../outputs/videos/clip_semantic_waypoint_core_v2_normalized_v1_leftmost_bowl_seed3300.mp4",
            "success": "../outputs/videos/clip_semantic_waypoint_core_v2_normalized_v1_azure_red_seed700.mp4",
            "failure": "../outputs/videos/clip_core_v2_multitask_v1_leftmost_cube_seed420.mp4",
            "v4_recovery": "../videos/rgb_table_recovery_v4/seed4006_severe_leftmost_table_recovery_success.mp4",
        },
    }


def build_html(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MuJoCo Manipulation Research Summary</title>
  <style>
    :root { --ink:#1e2933; --muted:#5b6770; --line:#d6dde1; --bg:#f7f9fa; --paper:#ffffff; --teal:#087e8b; --green:#2a7f62; --amber:#b7791f; --red:#b84444; --blue:#356da8; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family:Arial,"Microsoft YaHei",sans-serif; line-height:1.55; }
    .shell { width:min(1180px,calc(100% - 32px)); margin:0 auto; }
    .topbar { display:flex; justify-content:space-between; gap:24px; align-items:flex-start; padding:34px 0 24px; border-bottom:1px solid var(--line); }
    .eyebrow { color:var(--teal); font-size:12px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; margin:0 0 8px; }
    h1,h2,h3,p { margin-top:0; }
    h1 { font-size:30px; line-height:1.18; margin-bottom:10px; max-width:850px; }
    h2 { font-size:20px; margin-bottom:12px; }
    h3 { font-size:15px; margin-bottom:6px; }
    .subtitle { max-width:850px; color:var(--muted); margin-bottom:0; }
    .lang-switch { display:flex; border:1px solid var(--line); border-radius:6px; overflow:hidden; flex:none; }
    .lang-switch button { border:0; background:var(--paper); padding:8px 12px; cursor:pointer; color:var(--ink); font-weight:700; }
    .lang-switch button[aria-pressed="true"] { background:var(--ink); color:var(--paper); }
    main { padding:26px 0 44px; }
    section { padding:25px 0; border-bottom:1px solid var(--line); }
    .metric-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:0; }
    .metric { background:var(--paper); border:1px solid var(--line); border-radius:7px; padding:14px; min-height:102px; }
    .metric dt { color:var(--muted); font-size:12px; font-weight:700; }
    .metric dd { margin:6px 0 0; font-size:25px; font-weight:700; }
    .metric small { color:var(--muted); display:block; font-size:12px; }
    .two-col { display:grid; grid-template-columns:1.1fr .9fr; gap:30px; }
    .adopt { border-left:4px solid var(--green); background:var(--paper); padding:16px; }
    .adopt strong { color:var(--green); }
    .chart-grid { display:grid; grid-template-columns:1fr 1fr; gap:28px; }
    .chart-title { display:flex; justify-content:space-between; align-items:baseline; gap:12px; margin-bottom:12px; }
    .chart-title p { color:var(--muted); font-size:13px; margin:0; }
    .bar-row { display:grid; grid-template-columns:minmax(110px,1.35fr) minmax(120px,3fr) 58px; gap:10px; align-items:center; margin:8px 0; font-size:13px; }
    .bar-label { font-weight:700; }
    .track { height:15px; background:#e8eef0; overflow:hidden; border-radius:3px; }
    .fill { height:100%; min-width:0; background:var(--blue); }
    .fill.primary { background:var(--green); }
    .fill.warning { background:var(--amber); }
    .value { text-align:right; font-variant-numeric:tabular-nums; }
    .legend { color:var(--muted); font-size:12px; margin:12px 0 0; }
    .table-wrap { overflow-x:auto; }
    table { border-collapse:collapse; width:100%; min-width:700px; background:var(--paper); }
    th,td { text-align:left; padding:10px; border-bottom:1px solid var(--line); vertical-align:top; font-size:13px; }
    th { background:#edf2f3; font-weight:700; }
    .ok { color:var(--green); font-weight:700; }
    .fail { color:var(--red); font-weight:700; }
    .timeline { list-style:none; margin:0; padding:0; display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }
    .timeline li { border-top:3px solid var(--teal); padding:11px 2px 0; }
    .timeline li.warning { border-top-color:var(--amber); }
    .timeline li.success { border-top-color:var(--green); }
    .timeline p { color:var(--muted); font-size:13px; margin-bottom:0; }
    .work-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }
    .work-item { border-top:2px solid var(--line); padding-top:10px; }
    .work-item strong { display:block; font-size:22px; }
    .work-item span { color:var(--muted); font-size:12px; }
    .video-grid { display:grid; grid-template-columns:1fr 1fr; gap:22px; }
    figure { margin:0; }
    video { width:100%; aspect-ratio:4/3; object-fit:cover; background:#101719; display:block; }
    figcaption { font-size:13px; color:var(--muted); margin-top:8px; }
    .issues { display:grid; grid-template-columns:1fr 1fr; gap:24px; }
    .issues ul { margin:0; padding-left:20px; }
    .issues li { margin:8px 0; }
    .note { color:var(--muted); font-size:12px; margin-bottom:0; }
    .gradient-heading { margin:18px 0 8px; font-size:13px; font-weight:700; }
    .task-compare { margin:10px 0; }
    .task-name { font-size:13px; font-weight:700; margin-bottom:3px; }
    .series-line { display:grid; grid-template-columns:68px minmax(96px,1fr) 40px; gap:8px; align-items:center; margin:3px 0; font-size:12px; }
    .series-name { color:var(--muted); }
    .series-track { height:9px; background:#e8eef0; overflow:hidden; border-radius:3px; }
    .series-fill { height:100%; background:var(--blue); }
    .series-fill.primary { background:var(--green); }
    .en { display:none; }
    body[data-lang="en"] .zh { display:none; }
    body[data-lang="en"] .en { display:inline; }
    body[data-lang="en"] .en.block { display:block; }
    .zh.block { display:block; }
    body[data-lang="en"] .zh.block { display:none; }
    @media (max-width:800px) { .topbar,.two-col,.chart-grid,.issues,.video-grid { display:block; } .lang-switch { margin-top:18px; } .metric-grid,.work-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } .chart-grid > * + *,.issues > * + *,.video-grid > * + * { margin-top:28px; } .timeline { grid-template-columns:1fr; } }
    @media (max-width:480px) { .shell { width:min(100% - 22px,1180px); } h1 { font-size:25px; } .metric-grid,.work-grid { grid-template-columns:1fr; } .topbar { padding-top:24px; } }
  </style>
</head>
<body data-lang="zh">
  <div class="shell">
    <header class="topbar">
      <div>
        <p class="eyebrow">MuJoCo-only · WidowX tabletop manipulation</p>
        <h1><span class="zh">轻量视觉-语言机械臂操作：研究结果总览</span><span class="en">Lightweight Vision-Language Manipulation: Research Results</span></h1>
        <p class="subtitle"><span class="zh">同一 MuJoCo 桌面环境下，对比普通模仿学习、连续动作头与“CLIP 语义决策 + 结构化接触执行”方案。页面同时保留成功、失败和被拒绝的改进候选。</span><span class="en">One MuJoCo tabletop environment compares imitation-learning baselines, continuous action heads, and CLIP semantic decision with structured contact execution. Successes, failures, and rejected improvements are all retained.</span></p>
      </div>
      <div class="lang-switch" aria-label="Language">
        <button id="zh-button" aria-pressed="true">中文</button><button id="en-button" aria-pressed="false">EN</button>
      </div>
    </header>
    <main>
      <section aria-label="Key findings">
        <dl class="metric-grid" id="metrics"></dl>
      </section>
      <section>
        <div class="two-col">
          <div>
            <h2><span class="zh">研究问题与可写结论</span><span class="en">Research Question and Defensible Conclusion</span></h2>
            <p class="zh">在小规模示范和有限算力下，语言表征能否帮助 WidowX 完成桌面搬运？答案是：<strong>可以完成任务级语义选择和可靠搬运，但当前可靠控制来自结构化接触执行，不来自端到端连续动作回归。</strong></p>
            <p class="en">With few demonstrations and limited compute, can language representation help a WidowX arm transport tabletop objects? <strong>Yes for task-level semantic selection and reliable transport, but the reliable contact control comes from the structured executor, not end-to-end continuous action regression.</strong></p>
          </div>
          <div class="adopt">
            <h3><span class="zh">最终采用方法</span><span class="en">Adopted Method</span></h3>
            <p class="zh"><strong>clip_semantic_waypoint_core_v2_normalized_v1</strong><br>冻结 CLIP 四类意图 adapter + 闭词表指令规范化 + MuJoCo structured waypoint 执行器。</p>
            <p class="en"><strong>clip_semantic_waypoint_core_v2_normalized_v1</strong><br>Frozen CLIP four-intent adapter + closed-vocabulary instruction normalization + MuJoCo structured waypoint executor.</p>
          </div>
        </div>
      </section>
      <section>
        <h2><span class="zh">最终闭环审计：V4 复核与候选拒绝</span><span class="en">Final Closure Audit: V4 Replication and Candidate Rejection</span></h2>
        <div class="adopt" id="final-closure-summary"></div>
        <p class="note"><span class="zh">该审计覆盖早期接触融合的探索性趋势：只有通过独立闭环与同状态反事实门槛的方案才能作为默认控制。详情见 <code>docs/final_closure_audit_v1.md</code>。</span><span class="en">This audit supersedes exploratory contact-fusion trends: only a method passing independent closed-loop and same-state counterfactual gates can become the default controller. See <code>docs/final_closure_audit_v1.md</code>.</span></p>
      </section>
      <section>
        <h2><span class="zh">标准搬运任务：学习控制与分层方案对比</span><span class="en">Canonical Transport: Learned Control vs. Hierarchical Policy</span></h2>
        <div class="chart-grid">
          <div><div class="chart-title"><h3><span class="zh">严格搬运成功率</span><span class="en">Strict Transport Success</span></h3><p>Core V2 · 20 episodes</p></div><div id="canonical-chart"></div></div>
          <div><div class="chart-title"><h3><span class="zh">平均最终目标距离</span><span class="en">Mean Final Target Distance</span></h3><p><span class="zh">越低越好</span><span class="en">Lower is better</span></p></div><div id="distance-chart"></div></div>
        </div>
        <p class="note"><span class="zh">严格搬运成功要求：语义意图正确、物体抬升至少 0.06 m 且与 TCP 持续接近、最终放入目标区域。连续 CLIP 动作头的 0/20 表明图文表征本身不足以解决抓取接触控制。</span><span class="en">Strict transport requires a correct intent, at least 0.06 m lift with sustained TCP proximity, and final placement. The continuous CLIP action head's 0/20 shows that image-text representation alone does not solve contact control.</span></p>
      </section>
      <section>
        <h2><span class="zh">语言与干扰验证：错误修复、独立句法与最终采用方案</span><span class="en">Language and Distractor Validation: Error Repair, Independent Syntax, and the Adopted Variant</span></h2>
        <div class="chart-grid">
          <div><div class="chart-title"><h3><span class="zh">原始改写集（事后修复验证）</span><span class="en">Original Paraphrase Suite (Post-hoc Repair)</span></h3><p>60 episodes</p></div><div id="paraphrase-chart"></div></div>
          <div><div class="chart-title"><h3><span class="zh">全物体干扰</span><span class="en">All-object Distractors</span></h3><p>20 episodes</p></div><div id="distractor-chart"></div></div>
        </div>
        <p class="note"><span class="zh">语义改写增强候选使用 316 个图像-指令样本，但在同一测试协议上退化为 48/60 与 16/20，故不采用。词表规范化是在查看原改写错误后加入的，因此原改写集的 60/60 是修复验证，不是独立 OOD 泛化结论。</span><span class="en">The semantic-augmentation candidate used 316 image-instruction samples but regressed to 48/60 and 16/20 under the same protocol, so it was rejected. Vocabulary normalization was added after reviewing the original paraphrase errors, so its 60/60 is a repair check, not an independent OOD claim.</span></p>
        <div class="adopt" id="independent-syntax" style="margin-top:18px"></div>
      </section>
      <section>
        <h2><span class="zh">研究迭代与工作量</span><span class="en">Research Iterations and Work Performed</span></h2>
        <ol class="timeline">
          <li><h3><span class="zh">1. 环境与数据闭环</span><span class="en">1. Environment and Data Loop</span></h3><p><span class="zh">WidowX、彩色对象、目标盘/碗、随机 seed、示范保存与可复现回放。</span><span class="en">WidowX, colored objects, pads/bowl, randomized seeds, saved demonstrations, and reproducible replay.</span></p></li>
          <li><h3><span class="zh">2. 普通基线</span><span class="en">2. Conventional Baselines</span></h3><p><span class="zh">线性/MLP/kNN BC、动作块、ACT-style、Diffusion 和检索控制均被纳入对照。</span><span class="en">Linear/MLP/kNN BC, action chunks, ACT-style, diffusion, and retrieval control were retained as comparators.</span></p></li>
          <li><h3><span class="zh">3. 接触失败诊断</span><span class="en">3. Contact Failure Diagnosis</span></h3><p><span class="zh">记录严格抓取、目标距离、夹爪时序、失控与出界，避免仅以“接近目标”判成功。</span><span class="en">Strict grasp, target distance, gripper timing, instability, and out-of-table events were recorded instead of treating proximity as success.</span></p></li>
          <li><h3><span class="zh">4. 语义-控制解耦</span><span class="en">4. Semantic-Control Decoupling</span></h3><p><span class="zh">冻结 CLIP 只做四类任务意图；structured waypoint 专门处理接触、抬升和释放。</span><span class="en">Frozen CLIP selects one of four task intents; structured waypoint handles contact, lift, and release.</span></p></li>
          <li class="warning"><h3><span class="zh">5. 负例保留</span><span class="en">5. Negative Result Retained</span></h3><p><span class="zh">训练时语义改写增强没有提升 OOD，反而退化，保留为被拒绝候选。</span><span class="en">Training-time semantic paraphrase augmentation did not improve OOD and regressed, so it is kept as a rejected candidate.</span></p></li>
          <li class="success"><h3><span class="zh">6. 最终补强</span><span class="en">6. Final Strengthening</span></h3><p><span class="zh">闭词表别名规范化修复颜色、形状和目标区域同义词混淆，并在原协议达到 60/60。</span><span class="en">Closed-vocabulary alias normalization fixes color, shape, and target-region synonym ambiguity and reaches 60/60 on the original protocol.</span></p></li>
        </ol>
        <div class="work-grid" id="work-grid" style="margin-top:24px"></div>
      </section>
      <section>
        <h2><span class="zh">数据效率与资源口径</span><span class="en">Data Efficiency and Resource Accounting</span></h2>
        <div class="chart-grid"><div><div class="chart-title"><h3><span class="zh">每类示范预算</span><span class="en">Demonstration Budget per Intent</span></h3><p><span class="zh">固定留出集</span><span class="en">Fixed holdout</span></p></div><div id="efficiency-chart"></div></div><div class="table-wrap"><table id="resource-table"></table></div></div>
      </section>
      <section>
        <h2><span class="zh">视频证据</span><span class="en">Video Evidence</span></h2>
        <div class="video-grid">
          <figure><video controls preload="metadata"><source id="blue-blue-video" type="video/mp4"></video><figcaption><span class="zh">蓝色立方体放入蓝色盘：标准指令，`seed=3300`。</span><span class="en">Blue cube to blue pad: canonical instruction, `seed=3300`.</span></figcaption></figure>
          <figure><video controls preload="metadata"><source id="red-red-video" type="video/mp4"></video><figcaption><span class="zh">红色立方体放入红色盘：标准指令，`seed=3300`。</span><span class="en">Red cube to red pad: canonical instruction, `seed=3300`.</span></figcaption></figure>
          <figure><video controls preload="metadata"><source id="leftmost-bowl-video" type="video/mp4"></video><figcaption><span class="zh">最左立方体放入碗：空间关系任务，`seed=3300`。</span><span class="en">Leftmost cube to bowl: spatial-relation task, `seed=3300`.</span></figcaption></figure>
          <figure><video controls preload="metadata"><source id="success-video" type="video/mp4"></video><figcaption><span class="zh">采用版本：`put the azure block on the red disk` 被规范化为训练词表后，正确抓取蓝方块并放入红盘。</span><span class="en">Adopted variant: after normalizing “put the azure block on the red disk,” the blue cube is grasped and placed on the red pad.</span></figcaption></figure>
          <figure><video controls preload="metadata"><source id="failure-video" type="video/mp4"></video><figcaption><span class="zh">对照失败：冻结 CLIP 连续动作头在空间任务中未完成有效抓取，作为 0/20 的可视化证据。</span><span class="en">Baseline failure: the frozen CLIP continuous action head does not complete a valid grasp in the spatial task, matching its 0/20 result.</span></figcaption></figure>
          <figure><video controls preload="metadata"><source id="v4-recovery-video" type="video/mp4"></video><figcaption><span class="zh">V4 桌面范围恢复：严重接触条件下，方块离开源区后由有界桌面 RGB 重定位并在第二次放置成功。单例用于复核，结论以 144 条和 288 条 aggregate 为准。</span><span class="en">V4 table-range recovery: under severe contact, the cube leaves the source workspace, is relocalized in the bounded tabletop, and succeeds on the second placement. This example is for review; the 144- and 288-episode aggregates support the conclusion.</span></figcaption></figure>
        </div>
      </section>
      <section>
        <h2><span class="zh">未解决问题与下一步优化目标</span><span class="en">Open Problems and Next Optimization Targets</span></h2>
        <div class="issues"><div><h3><span class="zh">当前边界</span><span class="en">Current Boundaries</span></h3><ul><li><span class="zh">最终成功来自分层策略；它不是端到端 VLA，也不能证明连续学习控制已解决抓取。</span><span class="en">Final success is hierarchical, not end-to-end VLA; it does not prove learned continuous control has solved grasping.</span></li><li><span class="zh">词表规范化是桌面任务的手工闭词表，超出颜色/形状/目标区域/左右关系时需重新设计。</span><span class="en">Vocabulary normalization is a manual closed lexicon for tabletop colors, shapes, regions, and left/right relations.</span></li><li><span class="zh">所有结论均限于 MuJoCo 的四类任务与固定评测协议，不外推到真实机器人。</span><span class="en">All conclusions are limited to four MuJoCo tasks and the fixed protocol; no real-robot claim is made.</span></li></ul></div><div><h3><span class="zh">下一步目标</span><span class="en">Next Targets</span></h3><ul><li><span class="zh">在保持语义选择模块不变时，学习可闭环修正的 grasp/lift 子策略，并与 waypoint 上界对比。</span><span class="en">Keep semantic selection fixed while learning a closed-loop grasp/lift subpolicy and compare it with the waypoint upper bound.</span></li><li><span class="zh">构造词表之外的独立同义词、组合指令和更强外观扰动，避免只优化当前别名表。</span><span class="en">Create independent synonym, compositional-instruction, and appearance-shift tests beyond the current alias table.</span></li><li><span class="zh">扩大对象形状、遮挡和接触条件，记录按失败类型分层的成功率与置信区间。</span><span class="en">Expand shapes, occlusions, and contact conditions; report failure-stratified success rates and confidence intervals.</span></li></ul></div></div>
      </section>
      <section>
        <h2><span class="zh">方法版本与量化比较</span><span class="en">Method Versions and Quantitative Comparison</span></h2>
        <div class="table-wrap"><table id="method-table"></table></div>
        <p class="note"><span class="zh">数据来源：Core V2 留出矩阵、冻结 CLIP 连续动作头报告、CLIP 语义-结构化执行评测、OOD 复测、数据效率表和视频索引。生成版本：<code>mujoco_research_summary_v1</code>。</span><span class="en">Sources: Core V2 holdout matrix, frozen CLIP continuous-head report, CLIP semantic-waypoint evaluation, OOD reruns, data-efficiency table, and video index. Generated version: <code>mujoco_research_summary_v1</code>.</span></p>
      </section>
      <section>
        <h2><span class="zh">历史探索：语义决策 + MuJoCo 接触反馈恢复（未采用）</span><span class="en">Historical Exploration: Semantic Decision + MuJoCo Contact-feedback Recovery (Not Adopted)</span></h2>
        <div class="chart-grid">
          <div><div class="chart-title"><h3><span class="zh">低摩擦多任务压力测试</span><span class="en">Low-friction Multitask Stress Test</span></h3><p>40 paired episodes</p></div><div id="contact-fusion-chart"></div></div>
          <div class="adopt" id="contact-fusion-summary"></div>
        </div>
        <p class="note"><span class="zh">该融合器复用冻结 CLIP 意图 adapter，不新增训练参数；早期低摩擦趋势保留为历史证据，但后续接触监测器的独立闭环回退和同状态提前重抓反事实均未通过门槛，因此任何提前重抓策略都不作为默认方案。</span><span class="en">This fusion executor reused the frozen CLIP intent adapter with no new trainable parameters. Its early low-friction trend remains historical evidence, but later monitor regressions and same-state early-regrasp counterfactuals failed the gate, so no early-regrasp policy is the default.</span></p>
      </section>
    </main>
  </div>
  <script>
    const data = __DATA__;
    let lang = "zh";
    const text = (pair) => pair[lang];
    const ratio = (row) => `${row.successes}/${row.episodes}`;
    const percent = (value) => `${Math.round(value * 100)}%`;
    function bar(container, rows, accessor, emphasisId) {
      container.innerHTML = rows.map((row) => {
        const value = accessor(row); const cls = row.id === emphasisId ? "primary" : row.id.includes("alias") ? "warning" : "";
        return `<div class="bar-row"><div class="bar-label">${text(row.label)}</div><div class="track"><div class="fill ${cls}" style="width:${Math.max(0, Math.min(100, value * 100))}%"></div></div><div class="value">${percent(value)}</div></div>`;
      }).join("");
    }
    function render() {
      const primary = data.primary;
      const metricLabels = lang === "zh" ? ["标准任务严格成功", "独立句法（闭词表）", "全物体干扰成功", "连续 CLIP 动作头"] : ["Canonical strict success", "Independent syntax (closed vocabulary)", "All-object distractors", "Continuous CLIP action head"];
      const metricNotes = lang === "zh" ? ["4 项任务，共 20 episode", "8 条新句子，共 40 episode", "20 个 episode，采用版本", "同样冻结 CLIP，20 episode"] : ["4 tasks, 20 episodes", "8 new sentences, 40 episodes", "20 episodes, adopted variant", "same frozen CLIP, 20 episodes"];
      const metrics = [
        `${primary.canonical.successes}/${primary.canonical.episodes}`,
        `${primary.independent_syntax.successes}/${primary.independent_syntax.episodes}`,
        `${primary.ood.normalized.hard_distractors.successes}/${primary.ood.normalized.hard_distractors.episodes}`,
        "0/20",
      ];
      document.getElementById("metrics").innerHTML = metrics.map((value, index) => `<div class="metric"><dt>${metricLabels[index]}</dt><dd>${value}</dd><small>${metricNotes[index]}</small></div>`).join("");
      const closure = data.final_closure;
      const pooled = closure.v4_replication.pooled_descriptive;
      const monitor = closure.rejected_candidates.contact_monitor_early_regrasp;
      const counterfactual = closure.rejected_candidates.same_state_early_deep_regrasp;
      document.getElementById("final-closure-summary").innerHTML = lang === "zh" ? `<h3>当前默认：V4 冻结 CLIP 意图 + RGB 几何 + 结构化执行</h3><p>两批互不重叠 seed 的描述性复核为 <strong>${pooled.successes}/${pooled.episodes} (${(pooled.success_rate * 100).toFixed(1)}%)</strong>，语义与对象选择均为 ${pooled.semantic_correct}/${pooled.episodes}。接触监测器提前重抓为 ${monitor.candidate_success[0]}/${monitor.candidate_success[1]}，低于 V4 的 ${monitor.v4_success[0]}/${monitor.v4_success[1]}，且出现 ${monitor.paired_regressed} 条回退；同状态反事实中提前深抓取独有收益为 ${counterfactual.early_better}/${counterfactual.scenes}。因此两个候选均被拒绝，不训练新的选择器。</p>` : `<h3>Current Default: V4 Frozen CLIP Intent + RGB Geometry + Structured Execution</h3><p>Two disjoint seed cohorts give a descriptive replication of <strong>${pooled.successes}/${pooled.episodes} (${(pooled.success_rate * 100).toFixed(1)}%)</strong>, with semantic and object selection both ${pooled.semantic_correct}/${pooled.episodes}. Monitor-triggered early regrasp reaches ${monitor.candidate_success[0]}/${monitor.candidate_success[1]}, below V4 at ${monitor.v4_success[0]}/${monitor.v4_success[1]}, with ${monitor.paired_regressed} regressions; same-state counterfactual early deep regrasp has ${counterfactual.early_better}/${counterfactual.scenes} exclusive gains. Both candidates are rejected and no new selector is trained.</p>`;
      const canonicalGateRows = data.canonical_methods.filter((row) => ["continuous_clip", "semantic_waypoint", "normalized"].includes(row.id));
      const physicalStress = data.contact_fusion.stress;
      const taskLabels = {
        place_blue_cube_blue_pad:{zh:"蓝方块 -> 蓝盘",en:"Blue cube -> blue pad"},
        place_blue_cube_red_pad:{zh:"蓝方块 -> 红盘",en:"Blue cube -> red pad"},
        place_red_cube_red_pad:{zh:"红方块 -> 红盘",en:"Red cube -> red pad"},
        move_leftmost_cube_to_bowl:{zh:"最左立方体 -> 碗",en:"Leftmost cube -> bowl"},
      };
      bar(document.getElementById("canonical-chart"), canonicalGateRows, (row) => row.success_rate, "normalized");
      const gradientTitle = lang === "zh" ? "低摩擦压力下的分任务渐变（每任务 10 个配对 seed）" : "Per-task gradient under low friction (10 paired seeds per task)";
      const gradientRows = Object.entries(physicalStress.by_task).map(([task, row]) => `<div class="task-compare"><div class="task-name">${text(taskLabels[task])}</div><div class="series-line"><div class="series-name">${lang === "zh" ? "标准" : "Standard"}</div><div class="series-track"><div class="series-fill" style="width:${row.standard_successes / row.episodes * 100}%"></div></div><div class="value">${row.standard_success}</div></div><div class="series-line"><div class="series-name">${lang === "zh" ? "融合" : "Fusion"}</div><div class="series-track"><div class="series-fill primary" style="width:${row.fusion_successes / row.episodes * 100}%"></div></div><div class="value">${row.fusion_success}</div></div></div>`).join("");
      document.getElementById("canonical-chart").insertAdjacentHTML("beforeend", `<div class="gradient-heading">${gradientTitle}</div>${gradientRows}`);
      const distances = data.canonical_methods.map((row) => ({ ...row, rate: Math.max(0, 1 - row.mean_target_distance / 0.35) }));
      document.getElementById("distance-chart").innerHTML = distances.map((row) => `<div class="bar-row"><div class="bar-label">${text(row.label)}</div><div class="track"><div class="fill ${row.id === "normalized" ? "primary" : ""}" style="width:${row.rate * 100}%"></div></div><div class="value">${row.mean_target_distance.toFixed(3)} m</div></div>`).join("");
      const oodRows = [
        { id:"base", label:{zh:"原始 CLIP 语义头",en:"Base CLIP semantic head"}, ...primary.ood.base },
        { id:"alias_candidate", label:{zh:"改写增强候选（拒绝）",en:"Augmentation candidate (rejected)"}, ...primary.ood.alias_candidate },
        { id:"normalized", label:{zh:"采用：词表规范化",en:"Adopted: vocabulary normalization"}, ...primary.ood.normalized },
      ];
      bar(document.getElementById("paraphrase-chart"), oodRows.map((row) => ({...row, ...row.paraphrase})), (row) => row.success_rate, "normalized");
      bar(document.getElementById("distractor-chart"), oodRows.map((row) => ({...row, ...row.hard_distractors})), (row) => row.success_rate, "normalized");
      const syntax = primary.independent_syntax;
      document.getElementById("independent-syntax").innerHTML = lang === "zh" ? `<h3>独立句法验证（固定闭词表）</h3><p><strong>${syntax.successes}/${syntax.episodes}</strong> 任务成功，${syntax.semantic_correct}/${syntax.episodes} 语义正确，${syntax.strict_grasps}/${syntax.episodes} 严格抓取；8 条新完整句子、4 个任务、每条 5 个新 seed。它验证的是词表内的新句式，不验证开放词汇。</p>` : `<h3>Independent Syntax Check (Fixed Closed Vocabulary)</h3><p><strong>${syntax.successes}/${syntax.episodes}</strong> task success, ${syntax.semantic_correct}/${syntax.episodes} semantic correctness, and ${syntax.strict_grasps}/${syntax.episodes} strict grasps across 8 new full sentences, 4 tasks, and 5 new seeds per sentence. It validates new syntax within the fixed vocabulary, not open-vocabulary language.</p>`;
      const fusion = data.contact_fusion;
      const stress = fusion.stress;
      const fusionRows = [
        { label:{zh:"标准语义执行器",en:"Standard semantic executor"}, rate:stress.standard_successes / stress.episodes, value:stress.standard_success },
        { label:{zh:"语义 + 接触反馈融合",en:"Semantic + contact-feedback fusion"}, rate:stress.fusion_successes / stress.episodes, value:stress.fusion_success, primary:true },
      ];
      document.getElementById("contact-fusion-chart").innerHTML = fusionRows.map((row) => `<div class="bar-row"><div class="bar-label">${text(row.label)}</div><div class="track"><div class="fill ${row.primary ? "primary" : ""}" style="width:${row.rate * 100}%"></div></div><div class="value">${row.value}</div></div>`).join("");
      document.getElementById("contact-fusion-summary").innerHTML = lang === "zh" ? `<h3>配对结果与边界</h3><p><strong>${stress.standard_success} -> ${stress.fusion_success}</strong>（${stress.delta_success_points > 0 ? "+" : ""}${stress.delta_success_points.toFixed(1)} pp），全样本平均目标距离 <strong>${stress.standard_mean_target_distance.toFixed(4)} m -> ${stress.fusion_mean_target_distance.toFixed(4)} m</strong>。失败转成功 ${stress.improved} 条，成功转失败 ${stress.regressed} 条，恢复分支触发 ${stress.regrasp_attempts} 次；融合专用最终运输保持代理为 ${stress.fusion_sustained_transport_successes}/${stress.episodes}，因此恢复分支的独立收益尚未证明。常规域配对检查为 ${fusion.nominal.standard_success} / ${fusion.nominal.fusion_success}。精确 McNemar 双侧 p=${stress.exact_mcnemar_p_value.toFixed(4)}，结果显示改善趋势但样本量尚不足以宣称统计显著。</p>` : `<h3>Paired Result and Boundary</h3><p><strong>${stress.standard_success} -> ${stress.fusion_success}</strong> (${stress.delta_success_points > 0 ? "+" : ""}${stress.delta_success_points.toFixed(1)} pp), with all-episode mean target distance <strong>${stress.standard_mean_target_distance.toFixed(4)} m -> ${stress.fusion_mean_target_distance.toFixed(4)} m</strong>. ${stress.improved} failures became successes, ${stress.regressed} successes regressed, and the recovery branch triggered ${stress.regrasp_attempts} times; the fusion-only final transport-hold proxy is ${stress.fusion_sustained_transport_successes}/${stress.episodes}, so the recovery branch's independent benefit is not yet established. The nominal paired check was ${fusion.nominal.standard_success} / ${fusion.nominal.fusion_success}. Exact two-sided McNemar p=${stress.exact_mcnemar_p_value.toFixed(4)}: improvement trend, but insufficient sample size for a significance claim.</p>`;
      const work = [[data.work.formal_methods,lang === "zh" ? "正式方法版本" : "formal method versions"],[data.work.successful_demos,lang === "zh" ? "成功示范" : "successful demonstrations"],[data.work.language_validation_episodes,lang === "zh" ? "语言验证 episode" : "language-validation episodes"],[data.work.video_evidence,lang === "zh" ? "视频证据条目" : "video evidence entries"]];
      document.getElementById("work-grid").innerHTML = work.map(([count,label]) => `<div class="work-item"><strong>${count}</strong><span>${label}</span></div>`).join("");
      document.getElementById("efficiency-chart").innerHTML = data.data_efficiency.map((row) => `<div class="bar-row"><div class="bar-label">${lang === "zh" ? `每类 ${row.budget} 条` : `${row.budget} per intent`}</div><div class="track"><div class="fill primary" style="width:${row.successes / row.episodes * 100}%"></div></div><div class="value">${row.successes}/${row.episodes}</div></div>`).join("");
      const r = primary.resources;
      const rows = lang === "zh" ? [["冻结 CLIP 编码器",r.frozen_encoder_params.toLocaleString()],["可训练语义头",r.trainable_params.toLocaleString()],["成功示范",r.samples],["训练时间",`${r.train_time_seconds.toFixed(2)} s`],["峰值显存",`${r.peak_vram_mb.toFixed(1)} MB`],["验证意图准确率",percent(r.validation_accuracy)]] : [["Frozen CLIP encoder",r.frozen_encoder_params.toLocaleString()],["Trainable semantic head",r.trainable_params.toLocaleString()],["Successful demonstrations",r.samples],["Training time",`${r.train_time_seconds.toFixed(2)} s`],["Peak VRAM",`${r.peak_vram_mb.toFixed(1)} MB`],["Validation intent accuracy",percent(r.validation_accuracy)]];
      document.getElementById("resource-table").innerHTML = `<thead><tr><th>${lang === "zh" ? "参数" : "Measure"}</th><th>${lang === "zh" ? "数值" : "Value"}</th></tr></thead><tbody>${rows.map((row) => `<tr><td>${row[0]}</td><td>${row[1]}</td></tr>`).join("")}</tbody>`;
      const headers = lang === "zh" ? ["方法", "控制形式", "标准任务", "平均目标距离", "结论"] : ["Method", "Control form", "Canonical", "Mean target distance", "Finding"];
      const forms = lang === "zh" ? ["直接动作回归", "轨迹检索", "对象-语言动作回归", "冻结 CLIP + 连续动作回归", "冻结 CLIP 意图 + 结构化接触", "同左 + 闭词表规范化"] : ["Direct action regression", "Trajectory retrieval", "Object-language regression", "Frozen CLIP + continuous action regression", "Frozen CLIP intent + structured contact", "Previous row + closed-vocabulary normalization"];
      const findings = lang === "zh" ? ["未形成有效抓取", "未泛化到留出状态", "语义/控制耦合仍失败", "视觉语言表征不等于接触控制", "分层解耦后稳定搬运", "采用：修复同义词 OOD，仍非端到端 VLA"] : ["No valid grasp", "No holdout generalization", "Semantic-control coupling still fails", "Vision-language features are not contact control", "Stable transport after hierarchical decoupling", "Adopted: fixes synonym OOD; still not end-to-end VLA"];
      document.getElementById("method-table").innerHTML = `<thead><tr>${headers.map((head) => `<th>${head}</th>`).join("")}</tr></thead><tbody>${data.canonical_methods.map((row,index) => `<tr><td>${text(row.label)}</td><td>${forms[index]}</td><td class="${row.success_rate === 1 ? "ok" : "fail"}">${ratio(row)} (${percent(row.success_rate)})</td><td>${row.mean_target_distance.toFixed(4)} m</td><td>${findings[index]}</td></tr>`).join("")}</tbody>`;
      document.getElementById("blue-blue-video").src = data.videos.blue_blue;
      document.getElementById("red-red-video").src = data.videos.red_red;
      document.getElementById("leftmost-bowl-video").src = data.videos.leftmost_bowl;
      document.getElementById("success-video").src = data.videos.success;
      document.getElementById("failure-video").src = data.videos.failure;
      document.getElementById("v4-recovery-video").src = data.videos.v4_recovery;
    }
    function setLanguage(value) { lang = value; document.body.dataset.lang = value; document.documentElement.lang = value === "zh" ? "zh-CN" : "en"; document.getElementById("zh-button").setAttribute("aria-pressed", String(value === "zh")); document.getElementById("en-button").setAttribute("aria-pressed", String(value === "en")); render(); }
    document.getElementById("zh-button").addEventListener("click", () => setLanguage("zh"));
    document.getElementById("en-button").addEventListener("click", () => setLanguage("en"));
    setLanguage(location.hash === "#en" ? "en" : "zh");
  </script>
</body>
</html>'''
    return page.replace("__DATA__", payload)


def main() -> None:
    args = parse_args()
    data = build_data()
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_html.write_text(build_html(data), encoding="utf-8")
    print(f"summary_json: {args.output_json}")
    print(f"summary_html: {args.output_html}")
    print(f"primary_canonical: {data['primary']['canonical']['successes']}/{data['primary']['canonical']['episodes']}")
    print(f"normalized_original_repair: {data['primary']['ood']['normalized']['paraphrase']['successes']}/{data['primary']['ood']['normalized']['paraphrase']['episodes']}")
    print(f"independent_closed_vocab_syntax: {data['primary']['independent_syntax']['successes']}/{data['primary']['independent_syntax']['episodes']}")


if __name__ == "__main__":
    main()
