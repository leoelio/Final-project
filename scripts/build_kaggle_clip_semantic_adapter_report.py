from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VERSION = "kaggle_clip_semantic_adapter_core_v2_v1"
MODEL = ROOT / "outputs" / "clip_semantic_waypoint" / f"{VERSION}_kernel_v3.npz"
REMOTE_DIR = ROOT / "outputs" / "kaggle_remote" / f"{VERSION}_kernel_v3"
LOCAL_BASELINE_MODEL = ROOT / "outputs" / "clip_semantic_waypoint" / "clip_semantic_waypoint_core_v2_v1_20260721_110325.npz"
LOCAL_BASELINE_EVAL = ROOT / "outputs" / "evaluations" / "core_v2_clip_semantic_waypoint_v1.json"
VIDEO = ROOT / "outputs" / "videos" / "kaggle_clip_semantic_adapter_core_v2_v1_hard_leftmost_seed1900.mp4"
VIDEO_METADATA = VIDEO.with_suffix(".json")
OOD_VERSION = "kaggle_clip_semantic_adapter_core_v2_ood_v1"
OOD_JSON = ROOT / "outputs" / "evaluations" / f"{OOD_VERSION}.json"
OOD_CSV = ROOT / "docs" / f"{OOD_VERSION}.csv"
OOD_REPORT = ROOT / "docs" / f"{OOD_VERSION}_report.md"
OOD_FAILURE_VIDEO = ROOT / "outputs" / "videos" / "kaggle_clip_semantic_adapter_core_v2_ood_paraphrase_blue_to_red_seed700.mp4"
OOD_FAILURE_VIDEO_METADATA = OOD_FAILURE_VIDEO.with_suffix(".json")
TASKS = (
    ("place_blue_cube_blue_pad", "medium", "blue_blue"),
    ("place_blue_cube_red_pad", "medium", "blue_red"),
    ("place_red_cube_red_pad", "medium", "red_red"),
    ("move_leftmost_cube_to_bowl", "hard", "leftmost"),
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    metrics = read_json(REMOTE_DIR / f"{VERSION}_metrics.json")
    remote = metrics["metadata"]
    if remote["version"] != VERSION or remote["episode_samples"] != 79:
        raise RuntimeError("unexpected Kaggle result contract")
    if remote["validation_samples"] != 16:
        raise RuntimeError("unexpected Kaggle validation split")
    if not VIDEO.exists() or not VIDEO_METADATA.exists():
        raise FileNotFoundError("Kaggle remote video evidence is missing")
    required_ood = (OOD_JSON, OOD_CSV, OOD_REPORT, OOD_FAILURE_VIDEO, OOD_FAILURE_VIDEO_METADATA)
    missing_ood = [path for path in required_ood if not path.exists()]
    if missing_ood:
        raise FileNotFoundError(f"Kaggle OOD evidence is missing: {missing_ood}")
    video_metadata = read_json(VIDEO_METADATA)
    video_summary = video_metadata["summary"]
    if not video_summary["success"] or not video_summary["strict_grasp_success"]:
        raise RuntimeError("Kaggle remote video does not show strict success")
    ood_payload = read_json(OOD_JSON)
    if ood_payload["version"] != OOD_VERSION or len(ood_payload["rows"]) != 80:
        raise RuntimeError("unexpected Kaggle OOD evaluation contract")
    ood_totals = {}
    for condition in ("paraphrase", "hard_distractors"):
        selected = [row for row in ood_payload["rows"] if row["condition"] == condition]
        ood_totals[condition] = {
            "episodes": len(selected),
            "task_successes": sum(int(row["task_success"]) for row in selected),
            "semantic_correct": sum(int(row["semantic_correct"]) for row in selected),
            "strict_grasp_successes": sum(int(row["strict_grasp_success"]) for row in selected),
        }
    if ood_totals["paraphrase"]["episodes"] != 60 or ood_totals["hard_distractors"]["episodes"] != 20:
        raise RuntimeError("unexpected Kaggle OOD condition counts")
    ood_failure_metadata = read_json(OOD_FAILURE_VIDEO_METADATA)
    ood_failure_summary = ood_failure_metadata["summary"]
    if ood_failure_summary["semantic_correct"] or ood_failure_summary["task_success"]:
        raise RuntimeError("Kaggle OOD failure video is not a semantic failure")
    with np.load(LOCAL_BASELINE_MODEL) as baseline_model:
        baseline_meta = json.loads(baseline_model["metadata"].item())
        baseline_params = int(baseline_model["weights"].size + baseline_model["bias"].size)
    baseline_eval = read_json(LOCAL_BASELINE_EVAL)
    baseline_summary = baseline_eval["summary"]

    rows: list[dict] = []
    summaries: list[dict] = []
    for task, complexity, suffix in TASKS:
        summary = read_json(ROOT / "outputs" / "evaluations" / f"kaggle_clip_semantic_adapter_core_v2_{suffix}.json")
        if summary["version"] != VERSION or summary["task"] != task or summary["complexity"] != complexity:
            raise RuntimeError(f"unexpected local evaluation contract: {suffix}")
        summaries.append(summary)
        rows.extend(summary["rows"])
    if len(rows) != 20 or not all(row["success"] and row["strict_grasp_success"] for row in rows):
        raise RuntimeError("local closed-loop evaluation is incomplete")

    local = {
        "episodes": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "strict_grasp_successes": sum(int(row["strict_grasp_success"]) for row in rows),
        "semantic_correct": sum(int(row["semantic_correct"]) for row in rows),
        "mean_target_distance": sum(float(row["target_distance"]) for row in rows) / len(rows),
        "task_summaries": [
            {
                "task": summary["task"],
                "complexity": summary["complexity"],
                "success": summary["success"],
                "strict_grasp_success": summary["strict_grasp_success"],
                "semantic_correct": summary["semantic_correct"],
                "mean_target_distance": summary["mean_target_distance"],
            }
            for summary in summaries
        ],
    }
    result = {
        "version": VERSION,
        "stage": "remote_frozen_vlm_semantic_adapter_plus_local_mujoco_executor",
        "method_boundary": "Kaggle trains only a frozen CLIP 1024-to-16-to-4 task-intent adapter. MuJoCo action execution uses the structured waypoint expert. This is not an end-to-end VLA policy and not OpenVLA LoRA.",
        "remote_training": remote,
        "local_closed_loop": local,
        "local_ood_generalization": {
            "version": OOD_VERSION,
            "protocol": "60 unseen instruction paraphrases plus 20 hard distractor episodes; frozen model, no additional training",
            "totals": ood_totals,
            "failure_video": str(OOD_FAILURE_VIDEO.relative_to(ROOT)),
            "failure_video_summary": {
                "instruction": ood_failure_summary["instruction"],
                "predicted_intent": ood_failure_summary["predicted_intent"],
                "semantic_correct": bool(ood_failure_summary["semantic_correct"]),
                "task_success": bool(ood_failure_summary["task_success"]),
            },
        },
        "same_protocol_baseline": {
            "version": "clip_semantic_waypoint_core_v2_v1",
            "method": "frozen_clip_linear_intent_head_plus_structured_waypoint_executor",
            "trainable_params": baseline_params,
            "episode_samples": baseline_meta["samples"],
            "train_samples": baseline_meta["train_samples"],
            "validation_samples": baseline_meta["val_samples"],
            "train_accuracy": baseline_meta["train_accuracy"],
            "validation_accuracy": baseline_meta["val_accuracy"],
            "train_time_seconds": baseline_meta["train_time_seconds"],
            "local_closed_loop": baseline_summary,
        },
        "artifacts": {
            "model": str(MODEL.relative_to(ROOT)),
            "model_sha256": sha256(MODEL),
            "remote_outputs": str(REMOTE_DIR.relative_to(ROOT)),
            "video": str(VIDEO.relative_to(ROOT)),
            "video_metadata": str(VIDEO_METADATA.relative_to(ROOT)),
            "ood_json": str(OOD_JSON.relative_to(ROOT)),
            "ood_csv": str(OOD_CSV.relative_to(ROOT)),
            "ood_report": str(OOD_REPORT.relative_to(ROOT)),
            "ood_failure_video": str(OOD_FAILURE_VIDEO.relative_to(ROOT)),
            "ood_failure_video_metadata": str(OOD_FAILURE_VIDEO_METADATA.relative_to(ROOT)),
            "video_frames": int(video_metadata["frames"]),
            "video_fps": int(video_metadata["fps"]),
            "kaggle_kernel": "https://www.kaggle.com/code/luxunyu/widowx-mujoco-clip-semantic-adapter-v1",
            "kaggle_dataset": "https://www.kaggle.com/datasets/luxunyu/widowx-mujoco-core-v2-rlds-source-v1",
        },
        "runtime_incidents": [
            "Kernel v1: Kaggle automatically extracted the input archive; the script only searched for the archive.",
            "Kernel v2: Kaggle Transformers returned BaseModelOutputWithPooling; pooler_output compatibility was added.",
            "Kernel v3: completed. Kaggle allocated a P100, but its compute capability was incompatible with the preinstalled PyTorch, so the script recorded CPU fallback rather than claiming GPU acceleration.",
        ],
    }
    output_json = ROOT / "outputs" / "evaluations" / f"{VERSION}.json"
    output_csv = ROOT / "docs" / f"{VERSION}.csv"
    output_md = ROOT / "docs" / f"{VERSION}_report.md"
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        fields = [
            "seed", "task", "complexity", "instruction", "predicted_intent", "semantic_correct",
            "selected_object", "target_geom", "success", "strict_grasp_success", "target_distance",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    output_md.write_text(
        f"# Kaggle 冻结 CLIP 语义适配器实验记录\n\n"
        f"版本：`{VERSION}`\n\n"
        "## 方法边界\n\n"
        "冻结 `openai/clip-vit-base-patch32`，只训练 1024->16->4 的 ReLU 瓶颈任务意图适配器；MuJoCo 中由已验证的结构化 waypoint expert 执行抓取和放置。"
        "它不是端到端 VLA，也不是 OpenVLA LoRA。\n\n"
        "## Kaggle 训练\n\n"
        f"- 私有数据集：79 条成功 episode、2528 个 image/state/action step。\n"
        f"- 实际训练样本：63 条；分层验证：16 条。\n"
        f"- 可训练参数：{remote['trainable_adapter_params']:,}；冻结编码器参数：{remote['frozen_encoder_params']:,}。\n"
        f"- 训练/验证意图准确率：{remote['train_accuracy']:.1%} / {remote['validation_accuracy']:.1%}。\n"
        f"- 训练时长：{remote['train_time_seconds']:.2f} 秒。\n"
        f"- 实际设备：`{remote['device']}`。Kaggle 分配 P100，但与预装 PyTorch 不兼容，脚本自动 CPU fallback；不得表述为 GPU 加速训练。\n\n"
        "## 同协议对照\n\n"
        "两种方法均冻结相同 CLIP 编码器，使用相同 79 条示范和 63/16 分层划分，并由相同的结构化 waypoint executor 进行 20 条 Core V2 闭环测试。\n\n"
        "| 方法 | 训练头 | 可训练参数 | 训练时间 (s) | 训练/验证意图准确率 | 严格闭环成功 |\n"
        "| --- | --- | ---: | ---: | --- | --- |\n"
        f"| 本地线性 CLIP 头 | 1024->4 | {baseline_params:,} | {baseline_meta['train_time_seconds']:.2f} | {baseline_meta['train_accuracy']:.1%} / {baseline_meta['val_accuracy']:.1%} | {baseline_summary['success']} |\n"
        f"| Kaggle 瓶颈适配器 | 1024->16->4 | {remote['trainable_adapter_params']:,} | {remote['train_time_seconds']:.2f} | {remote['train_accuracy']:.1%} / {remote['validation_accuracy']:.1%} | {local['successes']}/{local['episodes']} |\n\n"
        "## 本机 MuJoCo 严格闭环\n\n"
        f"- 总成功率：{local['successes']}/{local['episodes']}。\n"
        f"- 严格抓取成功率：{local['strict_grasp_successes']}/{local['episodes']}。\n"
        f"- 语义意图正确率：{local['semantic_correct']}/{local['episodes']}。\n"
        f"- 平均最终目标距离：{local['mean_target_distance']:.4f} m。\n\n"
        "## 仿真视频证据\n\n"
        f"- 视频：`{VIDEO.relative_to(ROOT)}`。\n"
        f"- 元数据：`{VIDEO_METADATA.relative_to(ROOT)}`，{video_metadata['frames']} 帧、{video_metadata['fps']} fps。\n"
        f"- 同一 hard 场景 seed=1900：`success={video_summary['success']}`、`strict_grasp_success={video_summary['strict_grasp_success']}`、最终目标距离 {video_summary['target_distance']:.4f} m。\n\n"
        "| 任务 | 复杂度 | 成功 | 严格抓取 | 语义正确 | 平均目标距离 (m) |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        + "".join(
            f"| `{item['task']}` | `{item['complexity']}` | {item['success']} | {item['strict_grasp_success']} | {item['semantic_correct']} | {item['mean_target_distance']:.4f} |\n"
            for item in local['task_summaries']
        )
        + "\n## 已修复运行记录\n\n"
        "仅保留两个不重复的环境兼容问题：v1 的自动解压输入路径、v2 的 Transformers 输出包装。v3 是唯一计入正式结果的运行版本。诊断日志保存在 `outputs/kaggle_remote/`。\n\n"
        "## 复现命令\n\n"
        "Kaggle 重新提交（令牌只放在当前终端环境变量中，不写入文件）：\n\n"
        "```powershell\n"
        "$env:KAGGLE_API_TOKEN=\"<your Kaggle API token>\"\n"
        f"& \"{python}\" -m kaggle kernels push -p \"{ROOT / 'kaggle' / 'kernels' / 'widowx_mujoco_clip_semantic_adapter_v1'}\" --accelerator gpu --timeout 3600\n"
        "```\n\n"
        "本机无窗口批量评测：\n\n"
        "```powershell\n"
        "$env:VLA_TORCH_PACKAGE_DIR=\"D:\\vla_torch_cuda_pkgs\"\n"
        f"& \"{python}\" \"{ROOT / 'scripts' / 'evaluate_clip_semantic_waypoint.py'}\" --model \"{MODEL}\" --version {VERSION} --task move_leftmost_cube_to_bowl --complexity hard --seed 1900 --episodes 5 --output-csv \"{ROOT / 'docs' / 'kaggle_clip_semantic_adapter_core_v2_leftmost.csv'}\" --output-json \"{ROOT / 'outputs' / 'evaluations' / 'kaggle_clip_semantic_adapter_core_v2_leftmost.json'}\"\n"
        "```\n\n"
        "本机交互式 MuJoCo viewer：\n\n"
        "```powershell\n"
        "$env:VLA_TORCH_PACKAGE_DIR=\"D:\\vla_torch_cuda_pkgs\"\n"
        f"& \"{python}\" \"{ROOT / 'scripts' / 'run_clip_semantic_waypoint.py'}\" --model \"{MODEL}\" --task move_leftmost_cube_to_bowl --complexity hard --seed 1900 --episodes 1 --viewer --duration 60 --speed 0.05\n"
        "```\n",
        encoding="utf-8",
    )
    with output_md.open("a", encoding="utf-8") as file:
        file.write(
            "\n## Kaggle 适配器的语言与干扰泛化\n\n"
            "该评测冻结 Kaggle 训练所得模型，不增加演示、不再训练。`paraphrase` 使用 12 条训练中未出现的英文改写，"
            "`hard_distractors` 在多物体干扰场景保留原始指令。端到端成功必须同时满足语义正确、严格抓取和目标放置。\n\n"
            "| 条件 | episode | 端到端任务成功 | 语义正确 | 严格抓取 |\n"
            "| --- | ---: | ---: | ---: | ---: |\n"
            f"| 未见指令改写 | {ood_totals['paraphrase']['episodes']} | {ood_totals['paraphrase']['task_successes']}/{ood_totals['paraphrase']['episodes']} | {ood_totals['paraphrase']['semantic_correct']}/{ood_totals['paraphrase']['episodes']} | {ood_totals['paraphrase']['strict_grasp_successes']}/{ood_totals['paraphrase']['episodes']} |\n"
            f"| hard 多物体干扰 | {ood_totals['hard_distractors']['episodes']} | {ood_totals['hard_distractors']['task_successes']}/{ood_totals['hard_distractors']['episodes']} | {ood_totals['hard_distractors']['semantic_correct']}/{ood_totals['hard_distractors']['episodes']} | {ood_totals['hard_distractors']['strict_grasp_successes']}/{ood_totals['hard_distractors']['episodes']} |\n\n"
            "关键负例：对于 `put the azure block on the red disk`，模型将蓝色物体误判为红色物体任务。"
            "该片段中抓取动作本身可能发生，但 `semantic_correct=False` 且 `task_success=False`，不能计作目标任务成功。\n\n"
            f"- OOD JSON：`{OOD_JSON.relative_to(ROOT)}`\n"
            f"- OOD CSV：`{OOD_CSV.relative_to(ROOT)}`\n"
            f"- 代表性语义误判视频：`{OOD_FAILURE_VIDEO.relative_to(ROOT)}`\n"
            f"- 视频结果：`predicted_intent={ood_failure_summary['predicted_intent']}`，`semantic_correct={ood_failure_summary['semantic_correct']}`，`task_success={ood_failure_summary['task_success']}`。\n\n"
            "论文边界：该结果只说明冻结 CLIP 小适配器在本四意图任务的语义分类泛化，不说明端到端 VLA 控制、OpenVLA LoRA 或真实 WidowX 的泛化。\n"
        )
    print(f"report: {output_md}")
    print(f"csv: {output_csv}")
    print(f"json: {output_json}")
    print(f"success: {local['successes']}/{local['episodes']}")


if __name__ == "__main__":
    main()
