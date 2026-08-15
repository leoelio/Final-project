from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from torch_runtime import ensure_torch_path  # noqa: E402

ensure_torch_path()

from run_clip_action_head import load_clip  # noqa: E402
from run_clip_semantic_waypoint import configure_env, load_policy, rollout_episode  # noqa: E402
from widowx_env import TASKS  # noqa: E402


VERSION = "core_v2_clip_semantic_ood_generalization_v1"
SPECS = (
    {
        "key": "blue_to_blue",
        "task": "place_blue_cube_blue_pad",
        "train_complexity": "medium",
        "paraphrases": (
            "put the azure block on the blue disk",
            "move the cobalt cube onto the blue circle",
            "set the blue block on its matching pad",
        ),
    },
    {
        "key": "blue_to_red",
        "task": "place_blue_cube_red_pad",
        "train_complexity": "medium",
        "paraphrases": (
            "put the azure block on the red disk",
            "move the blue block onto the crimson circle",
            "set the cobalt cube on the red pad",
        ),
    },
    {
        "key": "red_to_red",
        "task": "place_red_cube_red_pad",
        "train_complexity": "medium",
        "paraphrases": (
            "put the crimson block on the red disk",
            "move the scarlet cube onto the red circle",
            "set the red block on its matching pad",
        ),
    },
    {
        "key": "leftmost_cube",
        "task": "move_leftmost_cube_to_bowl",
        "train_complexity": "language",
        "paraphrases": (
            "put the cube furthest to the left in the bowl",
            "move the left-most block into the bowl",
            "place the westernmost cube inside the bowl",
        ),
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate paraphrase and distractor generalization for the frozen CLIP semantic-waypoint policy.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--version", default=VERSION, help="Result version written to the report and JSON payload.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--workspace-profile", choices=("legacy", "core_v2"), default="core_v2")
    parser.add_argument("--instruction-normalization", choices=("none", "desktop_alias_v1"), default="none")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--camera", default="top_rgb")
    parser.add_argument("--arm-kp", type=float, default=150.0)
    parser.add_argument("--arm-force", type=float, default=100.0)
    parser.add_argument("--gripper-kp", type=float, default=1200.0)
    parser.add_argument("--gripper-force", type=float, default=200.0)
    parser.add_argument("--friction", type=float, default=5.0)
    parser.add_argument("--place-tcp-z", type=float, default=0.041)
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "core_v2_clip_semantic_ood_generalization.csv")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / f"{VERSION}.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "core_v2_clip_semantic_ood_generalization.md")
    parser.add_argument("--report-only", action="store_true", help="Rebuild the Markdown report from an existing output JSON.")
    return parser.parse_args()


def md_row(values: list[object]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for condition in ("paraphrase", "hard_distractors"):
        for task_key in (spec["key"] for spec in SPECS):
            selected = [row for row in rows if row["condition"] == condition and row["task_key"] == task_key]
            if not selected:
                continue
            result.append(
                {
                    "condition": condition,
                    "task_key": task_key,
                    "episodes": len(selected),
                    "task_success": f"{sum(int(row['task_success']) for row in selected)}/{len(selected)}",
                    "semantic_correct": f"{sum(int(row['semantic_correct']) for row in selected)}/{len(selected)}",
                    "strict_grasp_success": f"{sum(int(row['strict_grasp_success']) for row in selected)}/{len(selected)}",
                    "mean_target_distance": float(np.mean([float(row["target_distance"]) for row in selected])),
                }
            )
    return result


def write_report(
    path: Path,
    rows: list[dict[str, object]],
    summary: list[dict[str, object]],
    args: argparse.Namespace,
    model_metadata: dict[str, object] | None = None,
) -> None:
    totals = {}
    for condition in ("paraphrase", "hard_distractors"):
        selected = [row for row in rows if row["condition"] == condition]
        totals[condition] = {
            "episodes": len(selected),
            "task_success": f"{sum(int(row['task_success']) for row in selected)}/{len(selected)}",
            "semantic_correct": f"{sum(int(row['semantic_correct']) for row in selected)}/{len(selected)}",
            "strict_grasp_success": f"{sum(int(row['strict_grasp_success']) for row in selected)}/{len(selected)}",
        }
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    lines = [
        "# Core V2 CLIP 语义-结构化执行 OOD 泛化报告",
        "",
        f"版本：`{args.version}`",
        "",
        "## 协议",
        "",
        "- 使用已冻结的 `openai/clip-vit-base-patch32` 四类意图 adapter；不重新训练、不加入新示范。",
        f"- 指令规范化：`{args.instruction_normalization}`。`desktop_alias_v1` 只把桌面任务的颜色、形状、目标区域和空间同义词映射到训练词表，不读取任务标签。",
        "- `paraphrase`：每项任务 3 条训练中未出现的英文改写，每条在 5 个新 seed 上评测，共 60 个 episode。",
        "- `hard_distractors`：保留原始指令，但把场景复杂度改为 `hard`，使红/蓝/绿/黄方块、绿球、红圆柱等物体同时出现；每项 5 个新 seed，共 20 个 episode。",
        "- 任务成功要求语义意图正确、严格抓取成功且最终放入目标。严格抓取为物体相对初始高度上升至少 `0.06 m`，并在 TCP `0.06 m` 内累计至少 50 个仿真步。",
        "",
        "## 条件汇总",
        "",
        md_row(["条件", "episode", "端到端任务成功", "语义正确", "严格抓取"]),
        md_row(["---", "---:", "---:", "---:", "---:"]),
    ]
    if model_metadata and str(model_metadata.get("language_augmentation", "none")) != "none":
        lines.insert(11, f"- 训练时使用 `{model_metadata['language_augmentation']}` 语义改写增强；{model_metadata.get('augmentation_protocol', '')}")
    labels = {"paraphrase": "未见指令改写", "hard_distractors": "全物体干扰"}
    lines.extend(md_row([labels[key], item["episodes"], item["task_success"], item["semantic_correct"], item["strict_grasp_success"]]) for key, item in totals.items())
    lines.extend([
        "",
        "## 分任务结果",
        "",
        md_row(["条件", "任务", "episode", "端到端任务成功", "语义正确", "严格抓取", "平均目标距离 (m)"]),
        md_row(["---", "---", "---:", "---:", "---:", "---:", "---:"]),
    ])
    lines.extend(md_row([
        labels[str(item["condition"])],
        item["task_key"],
        item["episodes"],
        item["task_success"],
        item["semantic_correct"],
        item["strict_grasp_success"],
        f'{float(item["mean_target_distance"]):.4f}',
    ]) for item in summary)
    lines.extend([
        "",
        "## 解释与边界",
        "",
        "- 本实验检验的是冻结 CLIP 的图文意图分类是否能承受同义词和更多场景干扰物；接触、抓取和放置仍由 scripted waypoint expert 执行。",
        "- 因此它不能证明端到端 VLA 的语言泛化或控制泛化；连续 CLIP action-head 的基线仍应引用 `docs/core_v2_pretrained_vlm_action_head_report.md` 中的 `0/20` 结果。",
        "- 完整逐 episode 指令、预测意图、对象选择、严格抓取和落点数据保存在 CSV/JSON 中。视频只保留一个 7 物体成功例和一个真实语言误判例，避免用重复视频代替统计结果。",
        "- `outputs/videos/clip_semantic_ood_hard_leftmost_cube_seed1300.mp4`：7 个物体同时出现时，正确选择最左方块并放入碗。",
        "- `outputs/videos/clip_semantic_ood_paraphrase_blue_to_red_seed700.mp4`：`put the azure block on the red disk` 的真实语义误判例。",
        "",
        "## 复现命令",
        "",
        "```powershell",
        "$env:VLA_TORCH_PACKAGE_DIR=\"D:\\vla_torch_cuda_pkgs\"",
        f'& "{python}" "{ROOT / "scripts" / "evaluate_clip_semantic_ood_generalization.py"}" --model "{args.model}" --episodes {args.episodes} --workspace-profile {args.workspace_profile} --image-size {args.image_size} --camera {args.camera} --arm-kp {args.arm_kp:g} --arm-force {args.arm_force:g} --gripper-kp {args.gripper_kp:g} --gripper-force {args.gripper_force:g} --friction {args.friction:g} --place-tcp-z {args.place_tcp_z:g}',
        "```",
        "",
        "## 交互式 viewer 命令",
        "",
        "```powershell",
        "$env:VLA_TORCH_PACKAGE_DIR=\"D:\\vla_torch_cuda_pkgs\"",
        f'& "{python}" "{ROOT / "scripts" / "run_clip_semantic_waypoint.py"}" --model "{args.model}" --task move_leftmost_cube_to_bowl --complexity hard --workspace-profile {args.workspace_profile} --seed 1300 --episodes 1 --viewer --duration 45 --speed 0.25 --image-size {args.image_size} --camera {args.camera} --arm-kp {args.arm_kp:g} --arm-force {args.arm_force:g} --gripper-kp {args.gripper_kp:g} --gripper-force {args.gripper_force:g} --friction {args.friction:g} --place-tcp-z {args.place_tcp_z:g}',
        "```",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.report_only:
        payload = json.loads(args.output_json.read_text(encoding="utf-8"))
        write_report(args.output_md, payload["rows"], payload["summary"], args, payload.get("model_metadata"))
        print(f"report_md: {args.output_md}", flush=True)
        return
    policy = load_policy(args.model)
    clip_model, processor = load_clip(str(policy["metadata"]["clip_model"]))
    rows: list[dict[str, object]] = []
    for task_index, spec in enumerate(SPECS):
        for phrase_index, instruction in enumerate(spec["paraphrases"]):
            for offset in range(args.episodes):
                seed = 600 + task_index * 100 + phrase_index * args.episodes + offset
                args.task = str(spec["task"])
                args.complexity = str(spec["train_complexity"])
                args.instruction = instruction
                result = rollout_episode(args, policy, clip_model, processor, seed)
                result.update({"condition": "paraphrase", "task_key": spec["key"], "phrase_index": phrase_index + 1})
                rows.append(result)
                print(f"condition=paraphrase task={spec['key']} phrase={phrase_index + 1} seed={seed} semantic={result['semantic_correct']} task_success={result['task_success']}", flush=True)
        for offset in range(args.episodes):
            seed = 1000 + task_index * 100 + offset
            args.task = str(spec["task"])
            args.complexity = "hard"
            args.instruction = TASKS[args.task].instruction
            result = rollout_episode(args, policy, clip_model, processor, seed)
            result.update({"condition": "hard_distractors", "task_key": spec["key"], "phrase_index": 0})
            rows.append(result)
            print(f"condition=hard_distractors task={spec['key']} seed={seed} semantic={result['semantic_correct']} task_success={result['task_success']}", flush=True)

    summary = aggregate(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "condition", "task_key", "phrase_index", "seed", "task", "complexity", "instruction", "normalized_instruction", "predicted_intent",
        "semantic_correct", "success", "task_success", "placed", "strict_grasp_success", "max_object_z",
        "lifted_steps_near_tcp", "selected_object", "target_geom", "target_distance", "out_of_table",
    ]
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    payload = {
        "version": str(args.version),
        "model": str(args.model),
        "model_metadata": policy["metadata"],
        "episodes_per_condition": args.episodes,
        "rows": rows,
        "summary": summary,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.output_md, rows, summary, args, policy["metadata"])
    print(f"rows: {len(rows)}", flush=True)
    print(f"report_md: {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
