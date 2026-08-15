from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VERSION = "robot_vla_action_head_handoff_v1"
DEFAULT_BRIDGE_JSON = ROOT / "outputs" / "evaluations" / "openvla_dataset_bridge_v1.json"
DEFAULT_FEASIBILITY_JSON = ROOT / "outputs" / "evaluations" / "openvla_feasibility_check_v1.json"
DEFAULT_OUTPUT_JSON = ROOT / "outputs" / "evaluations" / f"{VERSION}.json"
DEFAULT_OUTPUT_MD = ROOT / "docs" / "robot_vla_action_head_handoff.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the handoff contract for real robot VLA action-head training.")
    parser.add_argument("--bridge-json", type=Path, default=DEFAULT_BRIDGE_JSON)
    parser.add_argument("--feasibility-json", type=Path, default=DEFAULT_FEASIBILITY_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def ps_command(script: str) -> str:
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    return f'& "{python}" "{ROOT / script}"'


def build_handoff(bridge: dict[str, Any], feasibility: dict[str, Any]) -> dict[str, Any]:
    checks = feasibility.get("feasibility", {}).get("checks", {})
    gpu_memory_gb = float(checks.get("gpu_memory_gb", 0.0))
    bridge_samples = int(bridge.get("samples_exported", 0))
    can_run_real_vla_locally = bool(gpu_memory_gb >= 27.0 and checks.get("openvla_cache_exists"))

    return {
        "version": VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed_prerequisite",
        "purpose": "Define the gate and handoff package for running real robot VLA frozen-backbone action-head training outside this 6GB local machine.",
        "local_verdict": {
            "can_run_real_vla_locally": can_run_real_vla_locally,
            "gpu_memory_gb": gpu_memory_gb,
            "bridge_samples": bridge_samples,
            "openvla_cache_exists": bool(checks.get("openvla_cache_exists")),
            "reason": "本机 6GB GPU 不满足真实 OpenVLA/OFT action-head 或 LoRA 训练显存要求；该版本只能作为远端运行交接门禁。",
        },
        "input_contract": {
            "bridge_samples_jsonl": bridge.get("jsonl_path", "data/vla_bridge/openvla_dataset_bridge_v1/samples.jsonl"),
            "bridge_manifest": "data/vla_bridge/openvla_dataset_bridge_v1/manifest.json",
            "preview_grid": bridge.get("preview_path", "data/vla_bridge/openvla_dataset_bridge_v1/preview_grid.png"),
            "required_fields": [
                "image",
                "instruction",
                "state.qpos",
                "state.qvel",
                "state.ctrl",
                "state.tcp",
                "state.objects",
                "action",
                "episode_index",
                "source_step",
            ],
            "minimum_samples": 60,
        },
        "remote_runtime_gate": {
            "required_backbone": "真实机器人预训练 VLA/VLM 表征，例如 openvla/openvla-7b 或 OpenVLA-OFT checkpoint",
            "minimum_gpu_memory_gb": 27,
            "recommended_gpu_memory_gb": 48,
            "required_packages": ["torch", "transformers", "accelerate", "peft"],
            "recommended_packages": ["flash_attn", "bitsandbytes", "torchvision"],
            "must_record": ["GPU 型号", "GPU 显存", "torch 版本", "CUDA 版本", "模型 checkpoint", "是否冻结 backbone", "可训练参数量"],
        },
        "output_contract": {
            "model_artifact": "outputs/robot_vla_action_head/robot_vla_action_head_lite_v1.*",
            "feature_cache": "outputs/robot_vla_action_head/openvla_feature_cache_v1.*",
            "evaluation_json": "outputs/evaluations/robot_vla_action_head_lite_v1.json",
            "train_range_video": "outputs/videos/robot_vla_action_head_lite_v1_seed0.mp4",
            "language_video": "outputs/videos/robot_vla_action_head_lite_v1_language_seed200.mp4",
            "report": "docs/robot_vla_action_head_lite_report.md",
            "required_metrics": [
                "train_range_success",
                "heldout_success",
                "language_success",
                "target_distance",
                "grasp_success",
                "object_z",
                "trainable_params",
                "train_time_seconds",
                "peak_gpu_memory_mb",
            ],
        },
        "acceptance_gate": {
            "may_register_as_completed": [
                "确认使用真实机器人预训练 VLA/VLM 表征，而不是本地 hand-crafted object/language/CLIP proxy。",
                "输出 train-range、held-out、language/spatial 三类评测。",
                "保存至少两个 viewer 或离屏视频：主任务 seed0 和语言/空间任务 seed200。",
                "将 success、grasp_success 和 object_z 同时写入评测 JSON/CSV。",
                "补齐 model_resource_summary.csv、video_evidence_index.csv、failure_mode_taxonomy.csv 和 experiment_versions.json。",
            ],
            "must_remain_planned_if": [
                "只使用当前 object_language_action_head_lite_v1、vision_language_action_head_lite_v1 或 clip_action_head_lite_v1 的本地 proxy 特征。",
                "没有真实 VLA checkpoint、feature cache 或远端运行日志。",
                "没有可复现评测和视频证据。",
            ],
        },
        "reference_sources": [
            {
                "name": "OpenVLA README",
                "url": "https://github.com/openvla/openvla/blob/main/README.md",
                "use": "LoRA fine-tuning reference, 27GB+ lower-bound note, 72GB batch-size note, troubleshooting guidance.",
            },
            {
                "name": "OpenVLA-OFT GitHub",
                "url": "https://github.com/moojink/openvla-oft",
                "use": "OFT inference and training system requirements.",
            },
            {
                "name": "OpenVLA-OFT website FAQ",
                "url": "https://openvla-oft.github.io/",
                "use": "OFT minimum/recommended memory examples for training and inference.",
            },
        ],
        "paper_boundary": "只能写成真实 robot VLA action-head 的运行交接门禁，不能写成 robot_vla_action_head_lite_v1 已完成，更不能写成 OpenVLA LoRA、OFT、Isaac 或真实 WidowX 验证完成。",
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_md(path: Path, data: dict[str, Any]) -> None:
    verdict = data["local_verdict"]
    input_contract = data["input_contract"]
    runtime = data["remote_runtime_gate"]
    output_contract = data["output_contract"]
    acceptance = data["acceptance_gate"]
    lines = [
        "# Robot VLA Action-Head 运行交接门禁",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：为后续 `robot_vla_action_head_lite_v1` 定义真实机器人 VLA 表征接入的输入、运行环境、输出和入包门禁。该文件不是策略模型，不参与成功率比较，也不能替代真实 OpenVLA/OFT 训练。",
        "",
        "## 1. 本机结论",
        "",
        f"- 本机是否可直接运行真实 robot VLA action-head：{'是' if verdict['can_run_real_vla_locally'] else '否'}",
        f"- 检测到 GPU 显存：`{verdict['gpu_memory_gb']}GB`",
        f"- OpenVLA bridge 样本数：`{verdict['bridge_samples']}`",
        f"- OpenVLA 本地缓存：{'是' if verdict['openvla_cache_exists'] else '否'}",
        f"- 判断：{verdict['reason']}",
        "",
        "## 2. 输入契约",
        "",
        "远端或大显存机器运行真实 VLA action-head 时，必须使用当前 bridge 数据：",
        "",
        "```text",
        f"samples_jsonl: {input_contract['bridge_samples_jsonl']}",
        f"manifest: {input_contract['bridge_manifest']}",
        f"preview_grid: {input_contract['preview_grid']}",
        f"minimum_samples: {input_contract['minimum_samples']}",
        "```",
        "",
        "每条样本必须包含字段：",
        "",
        "```text",
        *input_contract["required_fields"],
        "```",
        "",
        "## 3. 远端运行门槛",
        "",
        f"- Backbone：{runtime['required_backbone']}",
        f"- 最低显存：`{runtime['minimum_gpu_memory_gb']}GB`",
        f"- 建议显存：`{runtime['recommended_gpu_memory_gb']}GB+`",
        f"- 必需依赖：{', '.join(runtime['required_packages'])}",
        f"- 建议依赖：{', '.join(runtime['recommended_packages'])}",
        "",
        "必须记录：",
        "",
        "```text",
        *runtime["must_record"],
        "```",
        "",
        "## 4. 输出契约",
        "",
        "| 输出 | 路径 |",
        "| --- | --- |",
    ]
    for key, value in output_contract.items():
        if isinstance(value, list):
            continue
        lines.append(f"| `{key}` | `{value}` |")

    lines.extend(
        [
            "",
            "必须写入的指标：",
            "",
            "```text",
            *output_contract["required_metrics"],
            "```",
            "",
            "## 5. 入包门禁",
            "",
            "只有同时满足以下条件，`robot_vla_action_head_lite_v1` 才能从 planned 变成正式方法版本：",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in acceptance["may_register_as_completed"])
    lines.extend(
        [
            "",
            "如果出现以下任一情况，必须保持 planned：",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in acceptance["must_remain_planned_if"])

    lines.extend(
        [
            "",
            "## 6. 本地复现/重建命令",
            "",
            "```powershell",
            ps_command("scripts/export_openvla_dataset_bridge.py") + " --episodes 6 --steps-per-episode 12 --image-size 128",
            '$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"',
            ps_command("scripts/check_openvla_feasibility.py"),
            ps_command("scripts/build_robot_vla_action_head_handoff.py"),
            "```",
            "",
            "## 7. 资料依据",
            "",
        ]
    )
    for item in data["reference_sources"]:
        lines.append(f"- {item['name']}：{item['url']}。用途：{item['use']}")

    lines.extend(
        [
            "",
            "## 8. 论文表述边界",
            "",
            f"- {data['paper_boundary']}",
            "",
            "可以写：当前已经完成真实 robot VLA action-head 的数据输入和远端运行门禁，明确本机不能直接完成真实 OpenVLA/OFT 后训练。",
            "",
            "不能写：`robot_vla_action_head_lite_v1`、OpenVLA LoRA、OpenVLA-OFT、Isaac 或真实 WidowX 验证已经完成。",
            "",
            f"生成时间：{data['generated_at']}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    bridge = read_json(args.bridge_json)
    feasibility = read_json(args.feasibility_json)
    data = build_handoff(bridge, feasibility)
    write_json(args.output_json, data)
    write_md(args.output_md, data)
    print(f"version: {VERSION}", flush=True)
    print(f"output_json: {args.output_json}", flush=True)
    print(f"report: {args.output_md}", flush=True)
    print(f"can_run_real_vla_locally: {data['local_verdict']['can_run_real_vla_locally']}", flush=True)


if __name__ == "__main__":
    main()
