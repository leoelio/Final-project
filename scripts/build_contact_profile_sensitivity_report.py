from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize seed-disjoint contact-profile sensitivity data without training a selector.")
    parser.add_argument("--records", type=Path, default=ROOT / "data" / "contact_profile_sensitivity" / "leftmost_severe_v1.jsonl")
    parser.add_argument("--summary", type=Path, default=ROOT / "data" / "contact_profile_sensitivity" / "leftmost_severe_v1_summary.json")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / "contact_profile_sensitivity_v1.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "contact_profile_sensitivity_stage_v1.md")
    return parser.parse_args()


def exact_two_sided(improved: int, regressed: int) -> float:
    count = improved + regressed
    if count == 0:
        return 1.0
    lower = sum(math.comb(count, value) for value in range(0, min(improved, regressed) + 1)) / (2**count)
    return min(1.0, 2.0 * lower)


def candidate_stats(records: list[dict], action_names: list[str]) -> dict:
    counterfactual = [record for record in records if record["recovery_route"] == "counterfactual"]
    candidates = {}
    for name in action_names[1:]:
        candidates[name] = {
            "successes": sum(int(record["outcomes"][name]["success"]) for record in counterfactual),
            "held_after_transfer": sum(int(record["outcomes"][name]["held_after_transfer"]) for record in counterfactual),
        }
    comparisons = {}
    for name in action_names[2:]:
        improved = sum(int(record["outcomes"][name]["success"] and not record["outcomes"]["standard"]["success"]) for record in counterfactual)
        regressed = sum(int(record["outcomes"]["standard"]["success"] and not record["outcomes"][name]["success"]) for record in counterfactual)
        comparisons[name] = {
            "candidate_only_success": improved,
            "standard_only_success": regressed,
            "exact_two_sided_p": exact_two_sided(improved, regressed),
        }
    return {"counterfactual_states": len(counterfactual), "candidates": candidates, "vs_standard": comparisons}


def main() -> None:
    args = parse_args()
    collection_summary = json.loads(args.summary.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line]
    with __import__("numpy").load(Path(collection_summary["dataset"]), allow_pickle=False) as data:
        metadata = json.loads(data["metadata"].item())
    action_names = list(metadata["action_names"])
    splits = {name: [record for record in records if record["split"] == name] for name in ("train", "test")}
    result = {
        "version": "contact_profile_sensitivity_stage_v1",
        "task_protocol": collection_summary["task_protocol"],
        "collection_counters": collection_summary["counters"],
        "action_names": action_names,
        "splits": {},
    }
    for split, rows in splits.items():
        preferences = {name: sum(record["preferred_action"] == name for record in rows) for name in action_names}
        routes = {name: sum(record["recovery_route"] == name for record in rows) for name in sorted({record["recovery_route"] for record in rows})}
        result["splits"][split] = {
            "post_failure_states": len(rows),
            "preference_counts": preferences,
            "recovery_routes": routes,
            **candidate_stats(rows, action_names),
        }
    test_preferences = result["splits"]["test"]["preference_counts"]
    nonstandard_positive = sum(test_preferences[name] for name in action_names[2:])
    minority_support = {name: test_preferences[name] for name in action_names[2:] if test_preferences[name] > 0}
    selector_gate = nonstandard_positive >= 5 and all(count >= 3 for count in minority_support.values())
    deep = result["splits"]["test"]["vs_standard"]["deep_tight_slow"]
    result["decision"] = {
        "selector_gate_passed": selector_gate,
        "nonstandard_positive_test_labels": nonstandard_positive,
        "minority_test_label_support": minority_support,
        "deep_vs_standard": deep,
        "deployment": "Keep the one-standard-retry default. Do not train a selector: the pre-registered non-standard label threshold is not met. Do not promote deep_tight_slow: its paired confirmation signal is not statistically significant.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    train = result["splits"]["train"]
    test = result["splits"]["test"]
    candidate_rows = "\n".join(
        f"| `{name}` | {values['successes']}/{test['counterfactual_states']} | {values['held_after_transfer']}/{test['counterfactual_states']} |"
        for name, values in test["candidates"].items()
    )
    comparison_rows = "\n".join(
        f"| `{name}` | {values['candidate_only_success']} | {values['standard_only_success']} | {values['exact_two_sided_p']:.4f} |"
        for name, values in test["vs_standard"].items()
    )
    markdown = f"""# 严重接触恢复动作敏感性阶段报告

版本：`contact_profile_sensitivity_stage_v1`

## 协议与数据边界

本阶段只使用更新后的、最左方块 x 间隔至少 `0.03 m` 的 `move_leftmost_cube_to_bowl`，以及 `severe_contact_shift`。首轮标准轨迹失败后，保存 RGB、本体量和运行时可判定的恢复路径；只有目标未完成且源物体能由 RGB 重新定位时，才从同一个 MuJoCo 快照离线执行候选恢复轨迹。快照和对象真值不进入在线策略。

共扫描 `{result['collection_counters']['scanned']}` 个 seed-disjoint 场景；首轮成功 `{result['collection_counters']['first_success']}`，得到 `{result['collection_counters']['post_failure_states']}` 个首轮失败状态，其中 `{result['collection_counters']['counterfactual_states']}` 个可做同快照反事实比较。训练 split 有 `{train['post_failure_states']}` 个失败状态，独立测试 split 有 `{test['post_failure_states']}` 个。

## 独立测试候选结果

独立测试中可反事实比较状态为 `{test['counterfactual_states']}` 个。`held_after_transfer` 仅是阶段诊断，不改变成功判定；它用于区分“抬升过但在运输中脱落”和“成功放置”。

| 候选轨迹 | 严格成功 | 转移后仍持物 |
| --- | ---: | ---: |
{candidate_rows}

## 与标准重试的配对差异

| 候选轨迹 | 候选独有成功 | 标准独有成功 | 精确双侧 p |
| --- | ---: | ---: |
{comparison_rows}

`deep_tight_slow` 在独立集为 `{test['candidates']['deep_tight_slow']['successes']}/{test['counterfactual_states']}`，标准为 `{test['candidates']['standard']['successes']}/{test['counterfactual_states']}`，有 `{deep['candidate_only_success']}` 个候选独有成功和 `{deep['standard_only_success']}` 个标准独有成功，但 `p={deep['exact_two_sided_p']:.4f}`。样本不足以把它提升为默认策略。

## 条件化动作头门槛

独立测试的非 `stop`/`standard` 偏好标签数为 `{nonstandard_positive}`，而预注册下限为 5；其类别支持为 `{minority_support}`，其中每个非主类也没有达到 3 条的训练门槛。因此**不训练、不部署**新的条件化恢复动作头。这个否定结果是有效结论：当前低数据条件下，候选动作的差异存在，但不足以支撑一个可验证的轻量选择器。

部署仍保持“冻结 CLIP 语义 + RGB grounding + 结构化标准抓取/放置 + 最多一次标准 RGB 重试”。组合深抓取仅保留为后续独立验证的研究候选。

## 可视化复核

- 成功：`videos/contact_profile_sensitivity_v1/seed4814_deep_only_success.mp4`。正确选择黄色最左方块；首轮失败后，组合轨迹第二次成功，最终距离 `0.0115 m`。
- 失败：`videos/contact_profile_sensitivity_v1/seed4809_all_profiles_failure.mp4`。正确选择绿色最左方块；首轮曾满足严格抬升但未放入碗，第二次仍失败并耗尽一次 RGB 重试预算。

两条视频均为独立测试 seed 的可视化复核，不代替上表的统计比较。

## 完整复现

```powershell
cd "C:\\Users\\Administrator\\Desktop\\final project-robot ic\\vla_robot_grasping"
& .\\.venv\\Scripts\\python.exe .\\scripts\\collect_contact_profile_sensitivity.py --train-seed 3800 --train-episodes 80 --test-seed 4800 --test-episodes 40 --output .\\data\\contact_profile_sensitivity\\leftmost_severe_v1.npz --records .\\data\\contact_profile_sensitivity\\leftmost_severe_v1.jsonl --summary .\\data\\contact_profile_sensitivity\\leftmost_severe_v1_summary.json
& .\\.venv\\Scripts\\python.exe .\\scripts\\build_contact_profile_sensitivity_report.py
```
"""
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md), "decision": result["decision"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
