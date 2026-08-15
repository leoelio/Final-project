from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from torch_runtime import ensure_torch_path  # noqa: E402
from vlm_runtime import ensure_vlm_path  # noqa: E402


VERSION = "openvla_feasibility_check_v1"
DEFAULT_OUTPUT_JSON = ROOT / "outputs" / "evaluations" / f"{VERSION}.json"
DEFAULT_REPORT = ROOT / "docs" / "openvla_feasibility_report.md"
DEFAULT_BRIDGE_JSON = ROOT / "outputs" / "evaluations" / "openvla_dataset_bridge_v1.json"

SOURCE_NOTES = [
    {
        "name": "OpenVLA README",
        "url": "https://github.com/openvla/openvla/blob/main/README.md",
        "note": "OpenVLA LoRA fine-tuning example uses one A100 80GB; README notes smaller GPU is possible if it has at least about 27GB memory by changing batch size.",
    },
    {
        "name": "OpenVLA finetune.py",
        "url": "https://github.com/openvla/openvla/blob/main/vla-scripts/finetune.py",
        "note": "OpenVLA LoRA script notes PEFT is required and reports 48GB GPU for batch size 12, 80GB GPU for batch size 24.",
    },
    {
        "name": "OpenVLA-OFT system requirements",
        "url": "https://github.com/moojink/openvla-oft",
        "note": "OpenVLA-OFT lists roughly 16-18GB for inference and 27-80GB for training depending on setup.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether this local machine is suitable for real OpenVLA/robot VLA fine-tuning.")
    parser.add_argument("--bridge-json", type=Path, default=DEFAULT_BRIDGE_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def import_status(module_name: str) -> dict[str, Any]:
    try:
        module = __import__(module_name)
    except Exception as exc:
        return {"available": False, "error": repr(exc)}
    return {"available": True, "version": str(getattr(module, "__version__", "unknown"))}


def torch_status() -> dict[str, Any]:
    ensure_torch_path()
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": repr(exc)}

    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_gb": round(float(props.total_memory) / (1024**3), 3),
            }
        )
    return {
        "available": True,
        "version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": str(getattr(torch.version, "cuda", None)),
        "device_count": int(torch.cuda.device_count()),
        "devices": devices,
    }


def find_hf_model_cache(model_id: str) -> list[str]:
    model_path_name = "models--" + model_id.replace("/", "--")
    candidates = []
    env_home = os.environ.get("HF_HOME")
    if env_home:
        candidates.append(Path(env_home) / "hub" / model_path_name)
    candidates.extend(
        [
            Path.home() / ".cache" / "huggingface" / "hub" / model_path_name,
            ROOT / ".cache" / "huggingface" / "hub" / model_path_name,
        ]
    )
    return [str(path) for path in candidates if path.exists()]


def read_bridge(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "error": f"missing {path}"}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return {
        "available": True,
        "version": data.get("version", ""),
        "samples_exported": int(data.get("samples_exported", 0)),
        "jsonl_path": data.get("jsonl_path", ""),
        "preview_path": data.get("preview_path", ""),
    }


def feasibility(torch: dict[str, Any], packages: dict[str, dict[str, Any]], bridge: dict[str, Any]) -> dict[str, Any]:
    gpu_memories = [float(item["total_memory_gb"]) for item in torch.get("devices", [])]
    best_gpu_gb = max(gpu_memories) if gpu_memories else 0.0
    missing_required = [name for name in ("peft", "accelerate") if not packages[name].get("available")]
    missing_optional = [name for name in ("bitsandbytes", "flash_attn") if not packages[name].get("available")]

    checks = {
        "torch_cuda_available": bool(torch.get("cuda_available", False)),
        "gpu_memory_gb": best_gpu_gb,
        "bridge_samples": int(bridge.get("samples_exported", 0)) if bridge.get("available") else 0,
        "peft_available": bool(packages["peft"].get("available")),
        "accelerate_available": bool(packages["accelerate"].get("available")),
        "bitsandbytes_available": bool(packages["bitsandbytes"].get("available")),
        "openvla_cache_exists": bool(find_hf_model_cache("openvla/openvla-7b")),
    }

    if best_gpu_gb >= 48 and not missing_required and bridge.get("available"):
        status = "可以尝试 OpenVLA LoRA，但仍建议先小 batch smoke test"
        recommended_next = "在隔离环境安装 OpenVLA 依赖，先用 1-2 条 bridge 样本做 LoRA smoke test。"
    elif best_gpu_gb >= 27 and not missing_required and bridge.get("available"):
        status = "可能可以尝试极小 batch OpenVLA LoRA，但风险较高"
        recommended_next = "优先用云端 48GB+ GPU；若本机尝试，必须使用低 batch、短序列和严格显存监控。"
    else:
        status = "本机不适合直接训练真实 OpenVLA/机器人 VLA LoRA"
        recommended_next = "保留 bridge 数据入口，在本机继续做小型视觉/ACT/action-head 实验；真实 OpenVLA 训练迁移到 48GB+ GPU 或云端。"

    reasons = []
    if best_gpu_gb < 27:
        reasons.append(f"最大 GPU 显存约 {best_gpu_gb:.1f}GB，低于 OpenVLA README/OFT 文档中的 27GB 级训练下限。")
    if missing_required:
        reasons.append("缺少关键训练依赖：" + "、".join(missing_required))
    if missing_optional:
        reasons.append("缺少量化/高效注意力常用依赖：" + "、".join(missing_optional))
    if not bridge.get("available"):
        reasons.append("OpenVLA bridge 样本还不可用。")
    if not checks["openvla_cache_exists"]:
        reasons.append("本机未发现 openvla/openvla-7b 的 Hugging Face 缓存。")

    return {
        "status": status,
        "recommended_next": recommended_next,
        "checks": checks,
        "blocking_reasons": reasons,
    }


def collect(args: argparse.Namespace) -> dict[str, Any]:
    ensure_torch_path()
    ensure_vlm_path()
    torch = torch_status()
    packages = {
        "transformers": import_status("transformers"),
        "huggingface_hub": import_status("huggingface_hub"),
        "peft": import_status("peft"),
        "accelerate": import_status("accelerate"),
        "bitsandbytes": import_status("bitsandbytes"),
        "flash_attn": import_status("flash_attn"),
        "torchvision": import_status("torchvision"),
    }
    bridge = read_bridge(args.bridge_json)
    result = {
        "version": VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.executable,
        "torch": torch,
        "packages": packages,
        "bridge": bridge,
        "model_cache": {
            "openvla/openvla-7b": find_hf_model_cache("openvla/openvla-7b"),
        },
        "source_notes": SOURCE_NOTES,
    }
    result["feasibility"] = feasibility(torch, packages, bridge)
    return result


def yes_no(value: bool) -> str:
    return "是" if value else "否"


def write_report(data: dict[str, Any], path: Path) -> None:
    torch = data["torch"]
    feasibility_data = data["feasibility"]
    packages = data["packages"]
    bridge = data["bridge"]
    devices = torch.get("devices", [])
    gpu_text = "；".join(f"{item['name']} / {item['total_memory_gb']}GB" for item in devices) or "未检测到 CUDA GPU"

    lines = [
        "# OpenVLA 本地可行性检查",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：检查当前本机是否适合直接进行真实 OpenVLA/机器人 VLA action-head、Adapter 或 LoRA 后训练。该报告不新增策略方法，不参与成功率比较；它用于决定下一阶段是在本机继续做小型 proxy，还是迁移到更大显存 GPU。",
        "",
        "## 当前结论",
        "",
        f"- 判定：{feasibility_data['status']}",
        f"- 建议：{feasibility_data['recommended_next']}",
        f"- GPU：`{gpu_text}`",
        f"- Torch CUDA 可用：{yes_no(bool(torch.get('cuda_available', False)))}",
        f"- OpenVLA bridge 样本：{bridge.get('samples_exported', 0) if bridge.get('available') else '不可用'}",
        f"- openvla/openvla-7b 本地缓存：{yes_no(bool(data['model_cache']['openvla/openvla-7b']))}",
        "",
        "## 关键依赖",
        "",
        "| 包 | 可用 | 版本/错误 |",
        "| --- | --- | --- |",
    ]
    for name, status in packages.items():
        detail = status.get("version") if status.get("available") else status.get("error")
        lines.append(f"| `{name}` | {yes_no(bool(status.get('available')))} | `{detail}` |")

    lines.extend(
        [
            "",
            "## 阻塞原因",
            "",
        ]
    )
    reasons = feasibility_data.get("blocking_reasons", [])
    if reasons:
        lines.extend(f"- {reason}" for reason in reasons)
    else:
        lines.append("- 未检测到硬性阻塞；仍建议从 smoke test 开始。")

    lines.extend(
        [
            "",
            "## 公开资料依据",
            "",
        ]
    )
    for item in data["source_notes"]:
        lines.append(f"- {item['name']}：{item['url']}。{item['note']}")

    lines.extend(
        [
            "",
            "## 本项目下一步",
            "",
            "- 本机继续维护 `openvla_dataset_bridge_v1`、CLIP/action-head proxy、小型视觉 ACT 和 MuJoCo viewer 评测。",
            "- 若要推进 `robot_vla_action_head_lite_v1`，建议换到 48GB+ GPU 或云端环境，先只做冻结表征 + action head smoke test。",
            "- 若只能使用本机 6GB GPU，不应把失败归因于 OpenVLA 方法无效；应写成环境显存约束。",
            "",
            "## 复现命令",
            "",
            "```powershell",
            f'$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"',
            f'& "{ROOT / ".venv" / "Scripts" / "python.exe"}" "{ROOT / "scripts" / "check_openvla_feasibility.py"}"',
            "```",
            "",
            "## 论文表述边界",
            "",
            "- 可以写：已完成本机 OpenVLA/机器人 VLA 后训练可行性审计，当前 6GB GPU 不满足真实 OpenVLA LoRA 训练的显存条件。",
            "- 不能写：OpenVLA LoRA、真实机器人 VLA action head、Isaac 或真实 WidowX 验证已经完成。",
            "",
            f"生成时间：{data['generated_at']}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data = collect(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(data, args.output_md)
    print(f"version: {VERSION}", flush=True)
    print(f"status: {data['feasibility']['status']}", flush=True)
    print(f"output_json: {args.output_json}", flush=True)
    print(f"report: {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
