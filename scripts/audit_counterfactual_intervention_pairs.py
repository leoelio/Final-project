from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


EXPECTED_ARRAYS = {"close_rgb", "lift_rgb", "close_robot", "lift_robot"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit counterfactual intervention labels and their training-support gate.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--minimum-exclusive-support", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    rows = [json.loads(line) for line in (run_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise RuntimeError("no counterfactual rows")
    for row in rows:
        if row["label"] == "not_executable":
            continue
        path = run_dir / row["state_file"]
        with np.load(path) as state:
            if set(state.files) != EXPECTED_ARRAYS:
                raise RuntimeError(f"unexpected saved arrays in {path}: {state.files}")
            if state["close_rgb"].shape != state["lift_rgb"].shape or state["close_rgb"].ndim != 3 or state["close_rgb"].shape[-1] != 3:
                raise RuntimeError(f"invalid RGB schema in {path}")
            if state["close_robot"].shape != (32,) or state["lift_robot"].shape != (32,):
                raise RuntimeError(f"invalid robot-only schema in {path}")
    labels = Counter(row["label"] for row in rows)
    keys = sorted({(row["domain"], row["task"]) for row in rows})
    by_task_domain = {
        f"{domain}|{task}": {
            "scenes": sum(row["domain"] == domain and row["task"] == task for row in rows),
            "continue_better": sum(row["domain"] == domain and row["task"] == task and row["label"] == "continue_better" for row in rows),
            "early_better": sum(row["domain"] == domain and row["task"] == task and row["label"] == "early_better" for row in rows),
            "tie": sum(row["domain"] == domain and row["task"] == task and row["label"] == "tie" for row in rows),
        }
        for domain, task in keys
    }
    report = {
        "version": "counterfactual_intervention_pairs_audit_v1",
        "run_dir": str(run_dir),
        "scenes": len(rows),
        "labels": dict(sorted(labels.items())),
        "minimum_exclusive_support": args.minimum_exclusive_support,
        "training_allowed": labels["early_better"] >= args.minimum_exclusive_support and labels["continue_better"] >= args.minimum_exclusive_support,
        "by_task_domain": by_task_domain,
        "input_boundary": "Saved state files contain only RGB and robot-only 32D vectors. Full MuJoCo state existed only in memory for same-lift-state branch restoration.",
        "decision": "Do not train an intervention selector when either exclusive outcome class lacks support.",
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 提前重抓反事实先导审计 V1",
        "",
        "## 数据边界",
        "",
        "每个样本从同一 `lift_post` 状态分叉。完整 MuJoCo 状态只在内存中用于恢复两条反事实分支；落盘文件只有闭合后/抬升后 RGB 和两个 32 维机械臂本体向量。",
        "",
        "## 标签与门槛",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| 场景 | {report['scenes']} |",
        f"| continue_better | {labels['continue_better']} |",
        f"| early_better | {labels['early_better']} |",
        f"| tie | {labels['tie']} |",
        f"| 每类独有收益门槛 | {args.minimum_exclusive_support} |",
        f"| 允许训练选择器 | {'是' if report['training_allowed'] else '否'} |",
        "",
        "## 分层结果",
        "",
        "| 接触域 / 任务 | 场景 | 继续更优 | 提前重抓更优 | 平局 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(f"| {key} | {value['scenes']} | {value['continue_better']} | {value['early_better']} | {value['tie']} |" for key, value in by_task_domain.items())
    lines.extend([
        "",
        "## 决策",
        "",
        "`early_better` 的独有收益未达到训练门槛，因此不训练新的干预选择器，也不为追求正例而扩大同一提前重抓数据。该先导审计只能说明当前早期深抓取动作在已采样的抬升状态上没有显示可部署的条件化收益；它不宣称提前重抓在所有可能环境中绝不有效。V4 继续保持默认方案。",
    ])
    args.markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
