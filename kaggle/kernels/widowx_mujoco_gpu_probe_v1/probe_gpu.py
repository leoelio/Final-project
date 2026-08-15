from __future__ import annotations

import json
from pathlib import Path
import subprocess


def nvidia_smi() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def main() -> None:
    import torch

    cuda_available = torch.cuda.is_available()
    total_memory_mb = 0
    torch_name = None
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        total_memory_mb = properties.total_memory // (1024 * 1024)
        torch_name = properties.name

    payload = {
        "version": "widowx_mujoco_gpu_probe_v1",
        "cuda_available": cuda_available,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_device_name": torch_name,
        "total_memory_mb": total_memory_mb,
        "nvidia_smi": nvidia_smi() if cuda_available else None,
        "openvla_lora_minimum_27gb": total_memory_mb >= 27 * 1024,
        "openvla_recommended_48gb": total_memory_mb >= 48 * 1024,
        "smolvla_smoke_test_candidate": cuda_available and total_memory_mb >= 12 * 1024,
    }
    output = Path("/kaggle/working/gpu_probe_v1.json")
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
