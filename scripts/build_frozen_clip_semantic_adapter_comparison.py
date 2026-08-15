from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VERSION = "frozen_clip_semantic_adapter_same_protocol_comparison_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare local linear and Kaggle bottleneck frozen-CLIP semantic adapters.")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "frozen_clip_semantic_adapter_same_protocol_comparison.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "frozen_clip_semantic_adapter_same_protocol_comparison.csv")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / f"{VERSION}.json")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ood_totals(payload: dict, condition: str) -> dict[str, int]:
    rows = [row for row in payload["rows"] if row["condition"] == condition]
    return {
        "episodes": len(rows),
        "task_successes": sum(int(row["task_success"]) for row in rows),
        "semantic_correct": sum(int(row["semantic_correct"]) for row in rows),
        "strict_grasp_successes": sum(int(row["strict_grasp_success"]) for row in rows),
    }


def md_row(values: list[object]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"


def main() -> None:
    args = parse_args()
    local_model = ROOT / "outputs" / "clip_semantic_waypoint" / "clip_semantic_waypoint_core_v2_v1_20260721_110325.npz"
    local_core = ROOT / "outputs" / "evaluations" / "core_v2_clip_semantic_waypoint_v1.json"
    local_ood = ROOT / "outputs" / "evaluations" / "core_v2_clip_semantic_ood_generalization_v1.json"
    remote_model = ROOT / "outputs" / "clip_semantic_waypoint" / "kaggle_clip_semantic_adapter_core_v2_v1_kernel_v3.npz"
    remote_core = ROOT / "outputs" / "evaluations" / "kaggle_clip_semantic_adapter_core_v2_v1.json"
    remote_ood = ROOT / "outputs" / "evaluations" / "kaggle_clip_semantic_adapter_core_v2_ood_v1.json"
    for path in (local_model, local_core, local_ood, remote_model, remote_core, remote_ood):
        if not path.exists():
            raise FileNotFoundError(path)

    with np.load(local_model) as data:
        local_metadata = json.loads(data["metadata"].item())
        local_params = int(data["weights"].size + data["bias"].size)
    with np.load(remote_model) as data:
        remote_metadata = json.loads(data["metadata"].item())
        remote_params = int(data["down_weight"].size + data["down_bias"].size + data["up_weight"].size + data["up_bias"].size)

    local_core_data = read_json(local_core)
    remote_core_data = read_json(remote_core)
    local_ood_data = read_json(local_ood)
    remote_ood_data = read_json(remote_ood)
    if len(local_ood_data.get("rows", [])) != 80 or len(remote_ood_data.get("rows", [])) != 80:
        raise RuntimeError("Both OOD evaluations must contain the same 80-episode protocol")

    methods = [
        {
            "version": "clip_semantic_waypoint_core_v2_v1",
            "label": "本地线性 CLIP 意图头",
            "head": "1024->4 linear",
            "execution": "local",
            "trainable_params": local_params,
            "train_time_seconds": float(local_metadata["train_time_seconds"]),
            "canonical_success": str(local_core_data["summary"]["success"]),
            "canonical_strict": str(local_core_data["summary"]["strict_grasp_success"]),
            "ood": {condition: ood_totals(local_ood_data, condition) for condition in ("paraphrase", "hard_distractors")},
        },
        {
            "version": "kaggle_clip_semantic_adapter_core_v2_v1",
            "label": "Kaggle 瓶颈 CLIP 适配器",
            "head": "1024->16->4 ReLU",
            "execution": "Kaggle CPU fallback",
            "trainable_params": remote_params,
            "train_time_seconds": float(remote_metadata["train_time_seconds"]),
            "canonical_success": f"{remote_core_data['local_closed_loop']['successes']}/{remote_core_data['local_closed_loop']['episodes']}",
            "canonical_strict": f"{remote_core_data['local_closed_loop']['strict_grasp_successes']}/{remote_core_data['local_closed_loop']['episodes']}",
            "ood": {condition: ood_totals(remote_ood_data, condition) for condition in ("paraphrase", "hard_distractors")},
        },
    ]

    rows = []
    for method in methods:
        for condition in ("paraphrase", "hard_distractors"):
            total = method["ood"][condition]
            rows.append(
                {
                    "version": method["version"],
                    "method": method["label"],
                    "head": method["head"],
                    "execution": method["execution"],
                    "trainable_params": method["trainable_params"],
                    "train_time_seconds": f"{method['train_time_seconds']:.4f}",
                    "canonical_success": method["canonical_success"],
                    "canonical_strict_grasp": method["canonical_strict"],
                    "condition": condition,
                    "episodes": total["episodes"],
                    "task_successes": total["task_successes"],
                    "semantic_correct": total["semantic_correct"],
                    "strict_grasp_successes": total["strict_grasp_successes"],
                }
            )

    deltas = {
        condition: methods[0]["ood"][condition]["task_successes"] - methods[1]["ood"][condition]["task_successes"]
        for condition in ("paraphrase", "hard_distractors")
    }
    result = {
        "version": VERSION,
        "protocol": "Same frozen CLIP encoder, four intents, structured waypoint executor, and 80-episode OOD protocol. Only the intent head and training runtime differ.",
        "methods": methods,
        "rows": rows,
        "local_minus_kaggle_task_success_deltas": deltas,
        "conclusion": "The larger Kaggle bottleneck adapter matches canonical closed-loop success but does not improve this OOD protocol over the local linear head.",
        "boundary": "This compares high-level semantic intent heads plus the same structured executor. It is not an end-to-end VLA or OpenVLA LoRA comparison.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = list(rows[0])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# 冻结 CLIP 语义适配器同协议对照",
        "",
        f"版本：`{VERSION}`",
        "",
        "## 实验设计",
        "",
        "- 两种方法冻结相同的 `openai/clip-vit-base-patch32`，使用四类任务意图和相同的结构化 waypoint executor。",
        "- 两种方法在 Core V2 四任务规范闭环均为 `20/20` 严格抓放成功；因此只比较未见语言改写和 hard 多物体干扰的语义选择。",
        "- OOD 协议固定为 60 条未见英文改写和 20 条 hard 干扰场景。端到端成功要求语义正确、严格抓取和目标放置同时成立。",
        "",
        "## 结果",
        "",
        md_row(["方法", "头部", "可训练参数", "训练时间 (s)", "规范闭环", "条件", "任务成功", "语义正确", "严格抓取"]),
        md_row(["---"] * 9),
    ]
    for row in rows:
        lines.append(
            md_row(
                [
                    row["method"],
                    row["head"],
                    f"{int(row['trainable_params']):,}",
                    row["train_time_seconds"],
                    row["canonical_success"],
                    "未见改写" if row["condition"] == "paraphrase" else "hard 多物体干扰",
                    f"{row['task_successes']}/{row['episodes']}",
                    f"{row['semantic_correct']}/{row['episodes']}",
                    f"{row['strict_grasp_successes']}/{row['episodes']}",
                ]
            )
        )
    lines.extend(
        [
            "",
            "## 可写结论",
            "",
            f"- 在本协议中，Kaggle 1024->16->4 适配器的未见改写任务成功比本地线性头低 `{deltas['paraphrase']}` 条（48/60 对 51/60）；hard 干扰低 `{deltas['hard_distractors']}` 条（19/20 对 20/20）。",
            "- 参数更多、训练更久的瓶颈头没有改善当前四意图任务的 OOD 泛化；规范闭环同为 20/20，不能只报告规范任务成功来声称适配器更强。",
            "- 这一结论支持保留线性头作为轻量对照，并将 Kaggle 瓶颈头作为远程训练可复现的负向消融。",
            "",
            "## 论文边界",
            "",
            "- 本对照只比较冻结 CLIP 的高层语义意图头；接触、抓取和放置均由同一个结构化 executor 完成。",
            "- 不能写成端到端 VLA、OpenVLA LoRA、GPU 加速训练或真实 WidowX 验证结果。",
            "",
            "## 复现",
            "",
            "```powershell",
            f'& "{ROOT / ".venv" / "Scripts" / "python.exe"}" "{ROOT / "scripts" / "build_frozen_clip_semantic_adapter_comparison.py"}"',
            "```",
            "",
        ]
    )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"comparison_md: {args.output_md}", flush=True)
    print(f"comparison_csv: {args.output_csv}", flush=True)
    print(f"comparison_json: {args.output_json}", flush=True)
    print(f"paraphrase_delta_local_minus_kaggle: {deltas['paraphrase']}", flush=True)
    print(f"hard_distractor_delta_local_minus_kaggle: {deltas['hard_distractors']}", flush=True)


if __name__ == "__main__":
    main()
