from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a descriptive two-cohort V4 replication report.")
    parser.add_argument("--cohort-a", type=Path, required=True)
    parser.add_argument("--cohort-b", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser.parse_args()


def wilson(successes: int, episodes: int) -> list[float]:
    if not episodes:
        return [0.0, 0.0]
    z = 1.96
    rate = successes / episodes
    den = 1 + z * z / episodes
    center = (rate + z * z / (2 * episodes)) / den
    delta = z * sqrt(rate * (1 - rate) / episodes + z * z / (4 * episodes * episodes)) / den
    return [center - delta, center + delta]


def cohort_a(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"]
    return {
        "name": data["version"],
        "seed_range": data["seed_range"],
        "rows": rows,
        "successes": sum(row["task_success"] for row in rows),
        "semantic": sum(row["semantic_correct"] for row in rows),
        "selection": sum(row["visual_selection_correct"] for row in rows),
        "first": sum(row["first_attempt_success"] for row in rows),
    }


def cohort_b(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in data["rows"] if row["variant"] == "v4_standard"]
    return {
        "name": data["version"] + " / v4_standard",
        "seed_range": data["seed_range"],
        "rows": rows,
        "successes": sum(row["task_success"] for row in rows),
        "semantic": sum(row["semantic_correct"] for row in rows),
        "selection": sum(row["visual_selection_correct"] for row in rows),
        "first": sum(row["first_attempt_success"] for row in rows),
    }


def task_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    tasks = sorted({row["task"] for row in rows})
    return {task: {"episodes": sum(row["task"] == task for row in rows), "successes": sum(row["task"] == task and row["task_success"] for row in rows)} for task in tasks}


def main() -> None:
    args = parse_args()
    first = cohort_a(args.cohort_a)
    second = cohort_b(args.cohort_b)
    if set(row["seed"] for row in first["rows"]) & set(row["seed"] for row in second["rows"]):
        raise RuntimeError("cohorts must use disjoint scene seeds")
    cohorts = [first, second]
    episodes = sum(len(item["rows"]) for item in cohorts)
    successes = sum(item["successes"] for item in cohorts)
    all_rows = first["rows"] + second["rows"]
    report = {
        "version": "v4_independent_replication_v1",
        "cohorts": [{key: value for key, value in item.items() if key != "rows"} | {"episodes": len(item["rows"]), "wilson95": wilson(item["successes"], len(item["rows"]))} for item in cohorts],
        "pooled_descriptive": {
            "episodes": episodes,
            "successes": successes,
            "success_rate": successes / episodes,
            "wilson95": wilson(successes, episodes),
            "semantic_correct": sum(row["semantic_correct"] for row in all_rows),
            "visual_selection_correct": sum(row["visual_selection_correct"] for row in all_rows),
            "first_attempt_success": sum(row["first_attempt_success"] for row in all_rows),
        },
        "by_task": task_counts(all_rows),
        "interpretation_boundary": "The cohorts have disjoint seeds and identical V4 policy configuration. The pooled value is a descriptive replication summary, not a paired causal comparison against another method.",
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# V4 有界桌面 RGB 恢复：独立复核汇总",
        "",
        "## 边界",
        "",
        "两批使用完全不重叠的 scene seed，且均运行冻结 CLIP 语义、RGB 初始定位、一次有界桌面重定位和标准结构化重试。合并统计只用于描述重复稳定性，不是与另一方法的配对因果比较。",
        "",
        "| 批次 | seed | 成功 | 语义正确 | 对象选择正确 | 首轮成功 | Wilson 95% CI |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in report["cohorts"]:
        lo, hi = item["wilson95"]
        lines.append(f"| {item['name']} | {item['seed_range']} | {item['successes']}/{item['episodes']} | {item['semantic']}/{item['episodes']} | {item['selection']}/{item['episodes']} | {item['first']}/{item['episodes']} | [{lo:.3f}, {hi:.3f}] |")
    pooled = report["pooled_descriptive"]
    lo, hi = pooled["wilson95"]
    lines.extend([
        "",
        f"合并描述性结果：严格任务成功 `{pooled['successes']}/{pooled['episodes']}` ({pooled['success_rate']:.1%})，Wilson 95% CI `[{lo:.3f}, {hi:.3f}]`；语义 `{pooled['semantic_correct']}/{pooled['episodes']}`，初始对象选择 `{pooled['visual_selection_correct']}/{pooled['episodes']}`。",
        "",
        "| 任务 | 成功 |",
        "| --- | ---: |",
    ])
    lines.extend(f"| {task} | {item['successes']}/{item['episodes']} |" for task, item in report["by_task"].items())
    lines.extend([
        "",
        "该复核支持 V4 作为当前默认可复现方案。它不证明任何学习候选优于 V4；接触监测器和提前深抓取选择器均已在独立闭环或反事实门槛中被拒绝。",
    ])
    args.markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
