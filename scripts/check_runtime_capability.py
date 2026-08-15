from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from torch_runtime import DEFAULT_TORCH_PACKAGE_DIR, TORCH_PACKAGE_DIR, ensure_torch_path  # noqa: E402
from vlm_runtime import VLM_PACKAGE_DIR, ensure_vlm_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local runtime capability for ACT/VLM/VLA experiments.")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / "runtime_capability_v1.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "runtime_capability_report.md")
    return parser.parse_args()


def run_command(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        return {"ok": False, "error": repr(exc), "stdout": "", "stderr": ""}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def import_status(module_name: str) -> dict[str, Any]:
    try:
        module = __import__(module_name)
    except Exception as exc:
        return {"available": False, "error": repr(exc)}
    return {"available": True, "version": str(getattr(module, "__version__", "unknown"))}


def torch_status() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": repr(exc)}

    cuda_available = bool(torch.cuda.is_available())
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": int(props.total_memory),
            }
        )
    cuda_smoke = {"ok": False, "skipped": True}
    if cuda_available:
        try:
            tensor = torch.ones((16, 16), device="cuda")
            result = (tensor @ tensor).sum()
            torch.cuda.synchronize()
            cuda_smoke = {"ok": True, "skipped": False, "value": float(result.detach().cpu().item())}
        except Exception as exc:
            cuda_smoke = {"ok": False, "skipped": False, "error": repr(exc)}

    return {
        "available": True,
        "version": str(torch.__version__),
        "cuda_available": cuda_available,
        "cuda_version": str(getattr(torch.version, "cuda", None)),
        "device_count": int(torch.cuda.device_count()),
        "devices": devices,
        "cuda_smoke": cuda_smoke,
    }


def collect() -> dict[str, Any]:
    ensure_torch_path()
    ensure_vlm_path()
    return {
        "version": "runtime_capability_v1",
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "paths": {
            "vla_torch_package_dir_env": str(os.environ.get("VLA_TORCH_PACKAGE_DIR", "")),
            "default_torch_package_dir": str(DEFAULT_TORCH_PACKAGE_DIR),
            "torch_package_dir": str(TORCH_PACKAGE_DIR),
            "torch_package_dir_exists": TORCH_PACKAGE_DIR.exists(),
            "vlm_package_dir": str(VLM_PACKAGE_DIR),
            "vlm_package_dir_exists": VLM_PACKAGE_DIR.exists(),
        },
        "nvidia_smi": run_command(["nvidia-smi"]),
        "nvidia_query": run_command(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"]
        ),
        "packages": {
            "mujoco": import_status("mujoco"),
            "numpy": import_status("numpy"),
            "torch": torch_status(),
            "transformers": import_status("transformers"),
            "torchvision": import_status("torchvision"),
            "PIL": import_status("PIL"),
            "psutil": import_status("psutil"),
        },
    }


def yes_no(value: bool) -> str:
    return "是" if value else "否"


def write_markdown(data: dict[str, Any], path: Path) -> None:
    packages = data["packages"]
    torch = packages["torch"]
    nvidia_query = data["nvidia_query"]
    gpu_line = nvidia_query["stdout"].splitlines()[0] if nvidia_query.get("stdout") else "未检测到 nvidia-smi GPU 输出"
    torch_cuda = bool(torch.get("cuda_available", False))
    cuda_smoke = torch.get("cuda_smoke", {})

    if torch_cuda:
        next_phase_lines = [
            "- 当前可通过 `VLA_TORCH_PACKAGE_DIR` 加载 CUDA Torch；小型 PyTorch ACT、视觉 encoder 和缓存表征实验可以开始尝试 GPU 路线。",
            "- 本机 GPU 显存约 6GB，仍不适合直接全量微调大型 VLA 或 OpenVLA；真实 VLA 路线应优先冻结表征 + action head，LoRA/Adapter 只能从低分辨率、小 batch、短 horizon 开始。",
            "- 当前 CPU Torch 路径仍保留；不设置 `VLA_TORCH_PACKAGE_DIR` 时可以回退到原有轻量 baseline 运行方式。",
            "- 论文中仍应把目前的 CLIP、RGB pooled features、Adapter/LoRA-style 结果写成本地代理对照，不写成真实机器人 VLA 后训练结果。",
        ]
    else:
        next_phase_lines = [
            "- 当前机器有 NVIDIA GPU，但当前 Python 运行时加载到的是 CPU 版 Torch；直接做大 VLA 或 OpenVLA LoRA 不现实。",
            "- MuJoCo、CPU Torch 和 Transformers 路径已经能支撑小型 ACT、CLIP/action-head 代理和离线表征缓存实验。",
            "- 下一步若接入真实机器人 VLA，应先安装或配置 CUDA 版 Torch，并优先选择冻结表征 + action head；LoRA/Adapter 只能从低分辨率、小 batch、短 horizon 开始。",
            "- 论文中应把目前的 CLIP、RGB pooled features、Adapter/LoRA-style 结果写成本地代理对照，不写成真实机器人 VLA 后训练结果。",
        ]

    lines = [
        "# 运行环境能力检查",
        "",
        "版本：`runtime_capability_v1`",
        "",
        "用途：记录当前机器是否适合继续做 ACT、视觉模型、VLM/VLA 表征和 LoRA/Adapter 后训练实验。",
        "",
        "## 当前结论",
        "",
        f"- GPU：`{gpu_line}`",
        f"- MuJoCo 可用：{yes_no(packages['mujoco']['available'])}",
        f"- Torch 可用：{yes_no(torch.get('available', False))}",
        f"- Torch CUDA 可用：{yes_no(torch.get('cuda_available', False))}",
        f"- CUDA smoke test：{yes_no(bool(cuda_smoke.get('ok', False)))}",
        f"- Transformers 可用：{yes_no(packages['transformers']['available'])}",
        f"- TorchVision 可用：{yes_no(packages['torchvision']['available'])}",
        "",
        "## 对下一阶段的影响",
        "",
        *next_phase_lines,
        "",
        "## Python 与路径",
        "",
        f"- Python：`{data['python']['version'].splitlines()[0]}`",
        f"- 可执行文件：`{data['python']['executable']}`",
        f"- 系统：`{data['python']['platform']}`",
        f"- `VLA_TORCH_PACKAGE_DIR`：`{data['paths']['vla_torch_package_dir_env'] or '未设置'}`",
        f"- 默认 Torch 额外包目录：`{data['paths']['default_torch_package_dir']}`",
        f"- 当前 Torch 额外包目录：`{data['paths']['torch_package_dir']}`，存在：{yes_no(data['paths']['torch_package_dir_exists'])}",
        f"- VLM 额外包目录：`{data['paths']['vlm_package_dir']}`，存在：{yes_no(data['paths']['vlm_package_dir_exists'])}",
        "",
        "## 关键包",
        "",
        "| 包 | 可用 | 版本/错误 |",
        "| --- | --- | --- |",
    ]
    for name in ("mujoco", "numpy", "torch", "transformers", "torchvision", "PIL", "psutil"):
        status = packages[name]
        detail = status.get("version") if status.get("available") else status.get("error")
        if name == "torch" and status.get("available"):
            detail = f"{status.get('version')}，cuda_available={status.get('cuda_available')}，cuda_version={status.get('cuda_version')}，cuda_smoke={status.get('cuda_smoke', {}).get('ok')}"
        lines.append(f"| `{name}` | {yes_no(status.get('available', False))} | `{detail}` |")

    lines.extend(
        [
            "",
            "## 复现命令",
            "",
            "```powershell",
            f'& "{ROOT / ".venv" / "Scripts" / "python.exe"}" "{ROOT / "scripts" / "check_runtime_capability.py"}"',
            "",
            f'& "{ROOT / "scripts" / "check_cuda_torch_runtime.ps1"}"',
            "```",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data = collect()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(data, args.output_md)
    print(f"runtime_json: {args.output_json}", flush=True)
    print(f"runtime_report: {args.output_md}", flush=True)
    torch = data["packages"]["torch"]
    print(f"torch_available: {torch.get('available')}", flush=True)
    print(f"torch_cuda_available: {torch.get('cuda_available', False)}", flush=True)


if __name__ == "__main__":
    main()
