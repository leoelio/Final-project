from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize validated contact-stage collection data.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser.parse_args()


def read_rows(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if line]


def counts(rows: list[dict]) -> dict[str, int]:
    return {"trials": len(rows), "successes": sum(row["task_success"] for row in rows), "failures": sum(not row["task_success"] for row in rows)}


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    validation = json.loads((run_dir / "validation.json").read_text(encoding="utf-8"))
    if not validation["quota_met"]:
        raise RuntimeError("the collection gate has not been met")
    rows = read_rows(run_dir)
    tasks = sorted({row["task"] for row in rows})
    profiles = ("nominal", "stress")
    by_task = {task: counts([row for row in rows if row["task"] == task]) for task in tasks}
    by_profile = {profile: counts([row for row in rows if row["profile"] == profile]) for profile in profiles}
    failure_stages = Counter(row["failure_stage"] for row in rows if not row["task_success"])
    summary = {
        "version": "contact_phase_pairs_v3_summary",
        "dataset": str(run_dir),
        "collection_gate": validation,
        "by_task": by_task,
        "by_profile": by_profile,
        "failure_stages": dict(sorted(failure_stages.items())),
        "interpretation_boundary": "This is a paired data-collection audit. It is not an online policy evaluation and it does not establish a policy improvement.",
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 接触阶段配对数据集 V3：采集审计",
        "",
        "## 用途边界",
        "",
        "本报告只证明训练数据的配对结构、阶段记录和成功/失败配额均已达到准入条件；它不是策略在线评测，不能据此宣称策略性能提升。MuJoCo 物体真值仅用于离线标签，后续策略输入仅限 RGB、机械臂本体状态与动作历史。",
        "",
        "## 总体配额",
        "",
        "| 项目 | 数值 |",
        "| --- | ---: |",
        f"| 轨迹数 | {validation['trials']} |",
        f"| 配对 scene 数 | {validation['pairs']} |",
        f"| 成功 | {validation['successes']} |",
        f"| 失败 | {validation['failures']} |",
        f"| 准入门槛（成功/失败） | {validation['minimum_successes']} / {validation['minimum_failures']} |",
        f"| 配额通过 | {'是' if validation['quota_met'] else '否'} |",
        "",
        "## 按任务覆盖",
        "",
        "| 任务 | 轨迹 | 成功 | 失败 |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines.extend(f"| {task} | {item['trials']} | {item['successes']} | {item['failures']} |" for task, item in by_task.items())
    lines.extend([
        "",
        "## 接触条件与配对转变",
        "",
        "| 条件 | 轨迹 | 成功 | 失败 |",
        "| --- | ---: | ---: | ---: |",
    ])
    lines.extend(f"| {'标称' if profile == 'nominal' else '应力'} | {item['trials']} | {item['successes']} | {item['failures']} |" for profile, item in by_profile.items())
    lines.extend([
        "",
        "同一 `scene seed` 下的成败转变：",
        "",
        "| 转变 | 配对数 |",
        "| --- | ---: |",
    ])
    lines.extend(f"| {name} | {value} |" for name, value in validation["paired_outcome_transitions"].items())
    lines.extend([
        "",
        "## 失败阶段",
        "",
        "| 阶段 | 失败数 |",
        "| --- | ---: |",
    ])
    lines.extend(f"| {name} | {value} |" for name, value in sorted(failure_stages.items()))
    lines.extend([
        "",
        "## 下一步",
        "",
        "使用按 scene seed 划分的训练/验证集训练接触失败监测器；只有在独立的新 seed 上满足预注册的配对评测条件时，才把该候选写入最终方案。",
    ])
    args.markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"json: {args.json_output}")
    print(f"markdown: {args.markdown_output}")


if __name__ == "__main__":
    main()
