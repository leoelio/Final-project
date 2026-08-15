from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def ratio(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({numerator / denominator * 100:.1f}%)"


def build_payload() -> dict:
    v4 = read_json(ROOT / "outputs" / "evaluations" / "v4_independent_replication_v1.json")
    monitor = read_json(ROOT / "outputs" / "evaluations" / "contact_phase_monitor_heldout_v1.json")
    counterfactual = read_json(ROOT / "outputs" / "evaluations" / "counterfactual_intervention_pilot_v1.json")

    pooled = v4["pooled_descriptive"]
    v4_standard = monitor["by_variant"]["v4_standard"]
    monitor_candidate = monitor["by_variant"]["monitor_early_deep"]
    paired = monitor["paired_v4_vs_monitor"]
    labels = counterfactual["labels"]

    require(v4["version"] == "v4_independent_replication_v1", "unexpected V4 replication version")
    require((pooled["successes"], pooled["episodes"], pooled["semantic_correct"], pooled["visual_selection_correct"]) == (278, 288, 288, 288), "unexpected V4 replication totals")
    require(v4_standard["task_success"] == 143 and v4_standard["episodes"] == 144, "unexpected held-out V4 result")
    require(monitor_candidate["task_success"] == 127 and monitor_candidate["episodes"] == 144, "unexpected monitor result")
    require((paired["improved"], paired["regressed"]) == (1, 17), "unexpected monitor paired outcome")
    require(labels.get("early_better", 0) == 0 and labels.get("continue_better") == 47 and counterfactual["training_allowed"] is False, "unexpected counterfactual gate")

    evidence_paths = {
        "V4 独立复核": "docs/v4_independent_replication_v1.md",
        "接触监测器独立闭环": "docs/contact_phase_monitor_heldout_v1_analysis.md",
        "同状态反事实审计": "docs/counterfactual_intervention_pilot_v1_audit.md",
        "技术闭环说明": "docs/final_research_closed_loop_v2.md",
        "V4 桌面重定位成功视频": "videos/rgb_table_recovery_v4/seed4006_severe_leftmost_table_recovery_success.mp4",
        "监测器提前重抓成功视频": "outputs/videos/contact_phase_monitor_v1/seed10001_severe_leftmost_early_retry_success.mp4",
        "监测器误触发回退视频": "outputs/videos/contact_phase_monitor_v1/seed10000_mild_leftmost_false_trigger_failure.mp4",
    }
    missing = [path for path in evidence_paths.values() if not (ROOT / path).exists()]
    require(not missing, f"missing final closure evidence: {missing}")

    return {
        "version": "final_closure_audit_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "MuJoCo-only WidowX tabletop manipulation. No Isaac or real-robot claim.",
        "recommended_system": {
            "name": "V4 frozen CLIP intent + RGB geometry + structured execution",
            "components": [
                "frozen CLIP closed-set task intent",
                "top-view RGB object grounding with workspace and shape constraints",
                "structured PickPlaceExpert execution",
                "at most one bounded-table RGB relocalization and standard retry",
            ],
            "not_claimed": [
                "end-to-end VLA",
                "OpenVLA LoRA fine-tuning",
                "real-robot transfer",
            ],
        },
        "v4_replication": {
            "cohorts": v4["cohorts"],
            "pooled_descriptive": pooled,
            "interpretation": "Two disjoint seed cohorts reproduce the same frozen V4 configuration. The pooled value is descriptive replication evidence, not a causal comparison.",
        },
        "rejected_candidates": {
            "contact_monitor_early_regrasp": {
                "offline_balanced_accuracy": 0.9196,
                "v4_success": [v4_standard["task_success"], v4_standard["episodes"]],
                "candidate_success": [monitor_candidate["task_success"], monitor_candidate["episodes"]],
                "paired_improved": paired["improved"],
                "paired_regressed": paired["regressed"],
                "exact_two_sided_p": paired["exact_two_sided_p"],
                "decision": "rejected: offline separability did not translate into safe closed-loop intervention",
            },
            "same_state_early_deep_regrasp": {
                "scenes": counterfactual["scenes"],
                "continue_better": labels["continue_better"],
                "early_better": labels.get("early_better", 0),
                "tie": labels.get("tie", 0),
                "minimum_exclusive_support": counterfactual["minimum_exclusive_support"],
                "training_allowed": counterfactual["training_allowed"],
                "decision": counterfactual["decision"],
            },
        },
        "evidence_paths": evidence_paths,
        "closure": {
            "scope_closed": True,
            "recommended_default": "V4 remains the default reproducible MuJoCo policy.",
            "training_decision": "Do not train another early-regrasp selector from the current action space.",
            "remaining_boundaries": [
                "Results are limited to the four MuJoCo tasks and three recorded contact domains.",
                "The CLIP component provides closed-set intent classification; RGB geometry and structured execution provide spatial grounding and control.",
                "Future VLA, Isaac, or real WidowX experiments require separate evidence and cannot inherit these success rates.",
            ],
        },
    }


def write_markdown(path: Path, payload: dict) -> None:
    replication = payload["v4_replication"]
    pooled = replication["pooled_descriptive"]
    monitor = payload["rejected_candidates"]["contact_monitor_early_regrasp"]
    counterfactual = payload["rejected_candidates"]["same_state_early_deep_regrasp"]
    command_root = 'C:\\Users\\Administrator\\Desktop\\final project-robot ic\\vla_robot_grasping'

    lines = [
        "# MuJoCo 最终闭环审计 V1",
        "",
        "用途：作为 2026-07-29 之后的最终结论入口。它以独立 V4 复核、接触监测器闭环回退和同状态反事实审计覆盖早期接触融合的探索性趋势，不删除历史结果，但不再把它们作为默认方案依据。",
        "",
        "## 1. 范围与最终推荐",
        "",
        "本审计只覆盖 MuJoCo WidowX 桌面任务，不包含 Isaac 或真实机械臂。当前默认方案为：冻结 CLIP 闭词表任务意图 + 顶视 RGB 几何定位 + 结构化 PickPlaceExpert 执行 + 最多一次有界桌面 RGB 重定位重试。",
        "",
        "它不是端到端 VLA、OpenVLA LoRA 微调或真实机械臂迁移。运行时不用 MuJoCo 物体真值选择目标、决定重试或规划动作；真值只用于离线评分。",
        "",
        "## 2. V4 独立复核",
        "",
        "| 独立批次 | seed | 严格成功 | 语义正确 | 初始对象选择正确 | 首轮成功 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for cohort in replication["cohorts"]:
        lines.append(
            f"| {cohort['name']} | {cohort['seed_range']} | {ratio(cohort['successes'], cohort['episodes'])} | {cohort['semantic']}/{cohort['episodes']} | {cohort['selection']}/{cohort['episodes']} | {cohort['first']}/{cohort['episodes']} |"
        )
    lines.extend(
        [
            "",
            f"两批互不重叠 seed 的描述性合并结果为严格成功 **{ratio(pooled['successes'], pooled['episodes'])}**，Wilson 95% CI `{pooled['wilson95'][0]:.3f}-{pooled['wilson95'][1]:.3f}`；语义与初始对象选择均为 `{pooled['semantic_correct']}/{pooled['episodes']}`。该结果说明 V4 在独立场景范围内可重复，不构成与其他方法的因果比较。",
            "",
            "## 3. 学习候选的拒绝证据",
            "",
            "| 候选 | 离线/先导结果 | 独立或同状态结果 | 决策 |",
            "| --- | --- | --- | --- |",
            f"| 冻结 CLIP + 本体量接触监测器 | seed-disjoint 平衡准确率 `{monitor['offline_balanced_accuracy']:.4f}` | V4 `{monitor['v4_success'][0]}/{monitor['v4_success'][1]}`，提前重抓 `{monitor['candidate_success'][0]}/{monitor['candidate_success'][1]}`；改进 `{monitor['paired_improved']}`、回退 `{monitor['paired_regressed']}`，精确双侧 `p={monitor['exact_two_sided_p']:.6f}` | 拒绝：离线可分性没有变成安全闭环触发。 |",
            f"| 同状态提前深抓取 | 48 个 lift_post 分叉状态 | 继续 V4 更优 `{counterfactual['continue_better']}`，提前重抓更优 `{counterfactual['early_better']}`，平局 `{counterfactual['tie']}`；双向独有收益门槛 `{counterfactual['minimum_exclusive_support']}` | 不训练选择器，不扩大同类数据。 |",
            "",
            "这两项负结果回答了一个关键问题：不能因为监测头有较高离线准确率，或少数视频中提前重抓成功，就把它接入默认控制。闭环回退和同状态反事实优先于单例视频。",
            "",
            "## 4. 视频与证据入口",
            "",
            "- `videos/rgb_table_recovery_v4/seed4006_severe_leftmost_table_recovery_success.mp4`：V4 在严重接触条件下的有界桌面 RGB 重定位成功。",
            "- `outputs/videos/contact_phase_monitor_v1/seed10001_severe_leftmost_early_retry_success.mp4`：被拒绝监测器的成功单例，保留以避免只呈现失败。",
            "- `outputs/videos/contact_phase_monitor_v1/seed10000_mild_leftmost_false_trigger_failure.mp4`：同类候选的误触发回退单例，解释拒绝原因。",
            "- `docs/v4_independent_replication_v1.md`、`docs/contact_phase_monitor_heldout_v1_analysis.md`、`docs/counterfactual_intervention_pilot_v1_audit.md`：对应的 aggregate 统计、协议和完整命令。",
            "",
            "## 5. 可复现命令",
            "",
            "```powershell",
            f"cd \"{command_root}\"",
            "$env:VLA_TORCH_PACKAGE_DIR='D:\\vla_torch_cuda_pkgs'",
            "",
            "# 重新生成本审计，并运行项目级一致性检查",
            "& .\\.venv\\Scripts\\python.exe .\\scripts\\build_final_closure_audit.py",
            "& .\\.venv\\Scripts\\python.exe .\\scripts\\verify_experiment_artifacts.py",
            "",
            "# 交互式复核 V4 桌面范围恢复（打开 MuJoCo viewer）",
            "& .\\.venv\\Scripts\\python.exe .\\scripts\\run_clip_semantic_rgb_feedback.py `",
            "  --model .\\outputs\\clip_semantic_waypoint\\clip_semantic_waypoint_core_v2_v1_20260721_110325.npz `",
            "  --task move_leftmost_cube_to_bowl --complexity language --workspace-profile core_v2 `",
            "  --seed 4006 --feedback-attempts 1 --recovery-search table --viewer --duration 35 --speed 0.18 `",
            "  --arm-kp 105 --arm-force 70 --gripper-kp 550 --gripper-force 75 --friction 0.8",
            "```",
            "",
            "## 6. 最终边界",
            "",
            "- 当前结论仅限四个 MuJoCo 任务和三档已记录接触域。",
            "- CLIP 仅承担闭词表意图识别；对象的具体位置由顶视 RGB 几何获得，机械臂轨迹由结构化执行器生成。",
            "- 真实 VLA、Isaac 或真实 WidowX 若继续开展，必须以新数据、新评测和新视频单独建档，不能继承本页成功率。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    payload = build_payload()
    json_path = ROOT / "outputs" / "evaluations" / "final_closure_audit_v1.json"
    markdown_path = ROOT / "docs" / "final_closure_audit_v1.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(markdown_path, payload)
    print(f"wrote: {json_path}")
    print(f"wrote: {markdown_path}")


if __name__ == "__main__":
    main()
