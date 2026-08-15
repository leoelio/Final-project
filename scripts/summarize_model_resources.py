from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


TRAINABLE_KEYS = ("weights",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize model size, trainable parameters, and data scale.")
    parser.add_argument("--versions", type=Path, default=ROOT / "docs" / "experiment_versions.json")
    parser.add_argument("--summary", type=Path, default=ROOT / "docs" / "evaluation_summary.csv")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "model_resource_summary.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "model_resource_summary.md")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_summary(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as file:
        return {row["version"]: row for row in csv.DictReader(file)}


def as_json_metadata(value: np.ndarray) -> dict[str, Any]:
    raw = value.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not raw:
        return {}
    return json.loads(str(raw))


def trainable_array_name(name: str) -> bool:
    if name in TRAINABLE_KEYS:
        return True
    if name.startswith("adapter_") or name.startswith("lora_"):
        return True
    if name.startswith("phase") and ("_w" in name or "_b" in name):
        return True
    if name.startswith("base_"):
        return False
    return (name.startswith("w") or name.startswith("b")) and name[1:].isdigit()


def format_number(value: Any) -> str:
    if value in ("", None):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def format_mse(value: Any) -> str:
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def infer_model_note(version: str, artifact: Path, metadata: dict[str, Any], trainable_params: int) -> str:
    method = str(metadata.get("method", ""))
    if version.startswith("expert"):
        return "规则 expert，无学习参数，用作 oracle 和数据生成器。"
    if version.startswith("replay"):
        return "示范轨迹回放，不是策略模型，用于验证数据可复现。"
    if version.startswith("knn"):
        return "非参数检索 baseline；可训练参数为 0，但需要存储示范样本。"
    if "trajectory_knn" in method or "trajectory_knn" in version:
        return "历史观测检索动作块 baseline；可训练参数为 0，但需要存储轨迹窗口和动作块。"
    if "torch_state_transformer_act_cvae" in method or "torch_act_cvae" in version:
        latent_dim = metadata.get("latent_dim", "")
        suffix = f"；latent_dim={latent_dim}" if latent_dim != "" else ""
        return f"PyTorch state-only ACT-CVAE-lite baseline，训练 posterior latent、推理使用 zero latent，不含视觉 encoder{suffix}。"
    if "visual_feature_transformer_act" in method or "visual_feature_act" in version:
        return "MuJoCo 离线重渲染 RGB pooled features + 语言/本体状态的 Transformer ACT-lite；不是完整 CNN 视觉 ACT。"
    if "visual_act_cnn_cvae" in method or "visual_act_cnn_cvae" in version:
        return "小型 CNN 编码 MuJoCo RGB，拼接语言 token 和本体状态，再用 Transformer ACT-CVAE 输出动作块；这是本地视觉 ACT-CVAE-lite baseline，不是官方完整 ACT。"
    if "torch_state_diffusion_policy" in method or "torch_diffusion_policy" in version:
        steps = metadata.get("diffusion_steps", "")
        device = metadata.get("device", "")
        peak = metadata.get("peak_vram_mb", "")
        return f"PyTorch state-only Diffusion Policy 动作块 baseline，使用历史状态条件和 DDPM 噪声预测；不是视觉 Diffusion Policy。diffusion_steps={steps}，device={device}，peak_vram_mb={format_mse(peak)}。"
    if "torch_act_state_chunk_cuda" in version:
        device = metadata.get("device", "")
        peak = metadata.get("peak_vram_mb", "")
        suffix = f"；device={device}，peak_vram_mb={format_mse(peak)}" if peak != "" else ""
        return f"与 state-only PyTorch ACT 同结构的 CUDA 资源对照版本，不是策略结构改进{suffix}。"
    if "phase_conditioned_torch_state_transformer_act" in method or "phase_conditioned_torch_act" in version:
        return "state-only PyTorch ACT 追加离散阶段 one-hot 的动作块 baseline；用于检验阶段条件是否能改善接触/抬升，不是完整视觉 ACT。"
    if "torch_state_transformer_act" in method or "torch_act" in version:
        return "PyTorch Transformer encoder/decoder 动作块 baseline，更接近标准 ACT，但仍只使用状态特征。"
    if "structured_waypoint_policy" in method or "structured_waypoint_policy" in version:
        return "结构化单次 waypoint 控制 baseline；使用目标物和目标区域状态，不是 learned VLA。"
    if "diffusion" in method or "diffusion" in version:
        return "NumPy DDPM 风格动作块 baseline；不是官方完整 PyTorch Diffusion Policy。"
    if "reward_weighted" in method or "reward_weighted" in version:
        return "基于 attempt 偏好和 dense shaping 权重训练的 reward-weighted BC/action-head 后训练代理；不是 RL。"
    if "phase_conditioned" in method or "phase_conditioned" in version:
        return "显式拆分 approach、grasp、lift、transfer、place/release 五个阶段动作头；属于阶段条件 action-head 代理，不是 pretrained VLA。"
    if "vision_language_action_head" in method or "vision_language_action_head" in version:
        return "冻结 MuJoCo RGB 视觉代理特征 + 语言 token + 轻量 action head；不是 pretrained VLM/VLA。"
    if "clip_action_head" in method or "clip_action_head" in version:
        frozen = metadata.get("frozen_encoder_params", "")
        suffix = f"；冻结 CLIP encoder 参数约 {format_number(frozen)}" if frozen else ""
        return f"冻结 pretrained CLIP 图像/文本编码器，只训练轻量 MLP action head；这是 VLM 表征代理，不是机器人 VLA{suffix}。"
    if "adapter_object_language_action_head" in method or version.startswith("adapter_action_head"):
        return "冻结 object-language action-head 主干，仅训练小型 Adapter 残差模块；本地 PEFT 代理实验，不是 pretrained VLA Adapter。"
    if "lora_object_language_action_head" in method or version.startswith("lora_action_head"):
        return "冻结 object-language action-head 主干，仅训练 LoRA-style 低秩输出残差；本地 PEFT 代理实验，不是 pretrained VLA LoRA。"
    if "trajectory_conditioned" in method or "trajectory_conditioned" in version:
        return "带历史观测的动作块 MLP，作为 ACT-lite/trajectory-conditioned baseline。"
    if "chunk" in method or "act_lite" in version:
        return "短动作块 MLP，作为轻量 ACT-style baseline。"
    if "action_head" in method or "action_head" in version:
        return "轻量 action-head 代理基线；不是 pretrained VLM/VLA。"
    if trainable_params > 0:
        return "普通监督模仿学习 baseline。"
    if artifact.suffix == ".npz":
        return "NPZ artifact。"
    return "非模型脚本 artifact。"


def summarize_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        metadata = as_json_metadata(data["metadata"]) if "metadata" in data.files else {}
        trainable_params = sum(int(data[name].size) for name in data.files if trainable_array_name(name))
        if metadata.get("trainable_params") is not None:
            trainable_params = int(metadata["trainable_params"])
        trainable_arrays = ",".join(name for name in data.files if trainable_array_name(name))

        stored_samples = (
            metadata.get("samples")
            or metadata.get("source_samples")
            or metadata.get("train_samples")
            or (int(len(data["actions"])) if "actions" in data.files else "")
        )
        feature_dim = metadata.get("feature_dim") or metadata.get("observation_dim") or ""
        action_dim = metadata.get("action_dim") or ""
        horizon = metadata.get("horizon") or ""
        history = metadata.get("history") or ""
        diffusion_steps = metadata.get("diffusion_steps") or ""
        train_time_seconds = metadata.get("train_time_seconds") or metadata.get("elapsed_seconds") or ""
        vram_mb = metadata.get("peak_vram_mb") or ""

    return {
        "metadata": metadata,
        "trainable_params": trainable_params,
        "trainable_arrays": trainable_arrays,
        "stored_samples": stored_samples,
        "feature_dim": feature_dim,
        "action_dim": action_dim,
        "horizon": horizon,
        "history": history,
        "diffusion_steps": diffusion_steps,
        "train_mse": metadata.get("train_mse", ""),
        "val_mse": metadata.get("val_mse", ""),
        "train_time_seconds": train_time_seconds,
        "peak_vram_mb": vram_mb,
    }


def summarize_pt(path: Path) -> dict[str, Any]:
    from torch_runtime import ensure_torch_path

    ensure_torch_path()
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    trainable_params = int(metadata.get("trainable_params", 0))
    if not trainable_params and "model_state" in checkpoint:
        trainable_params = sum(int(value.numel()) for value in checkpoint["model_state"].values())

    return {
        "metadata": metadata,
        "trainable_params": trainable_params,
        "trainable_arrays": "model_state",
        "stored_samples": metadata.get("source_samples") or metadata.get("train_chunks") or "",
        "feature_dim": metadata.get("observation_dim", ""),
        "action_dim": metadata.get("action_dim", ""),
        "horizon": metadata.get("horizon", ""),
        "history": metadata.get("history", ""),
        "diffusion_steps": metadata.get("diffusion_steps", ""),
        "train_mse": metadata.get("train_mse_norm", metadata.get("train_noise_mse_norm", "")),
        "val_mse": metadata.get("val_mse_norm", metadata.get("val_noise_mse_norm", "")),
        "train_time_seconds": metadata.get("train_time_seconds", ""),
        "peak_vram_mb": metadata.get("peak_vram_mb", ""),
    }


def summarize_method(method: dict[str, Any], summary_row: dict[str, str]) -> dict[str, Any]:
    artifact = ROOT / method["artifact"]
    artifact_size_mb = artifact.stat().st_size / (1024 * 1024) if artifact.exists() else 0.0
    npz_summary: dict[str, Any] = {
        "metadata": {},
        "trainable_params": 0,
        "trainable_arrays": "",
        "stored_samples": "",
        "feature_dim": "",
        "action_dim": "",
        "horizon": "",
        "history": "",
        "diffusion_steps": "",
        "train_mse": "",
        "val_mse": "",
        "train_time_seconds": "",
        "peak_vram_mb": "",
    }
    if artifact.suffix == ".npz" and artifact.exists():
        npz_summary = summarize_npz(artifact)
    elif artifact.suffix == ".pt" and artifact.exists():
        npz_summary = summarize_pt(artifact)

    note = infer_model_note(
        method["version"],
        artifact,
        npz_summary["metadata"],
        int(npz_summary["trainable_params"]),
    )
    return {
        "version": method["version"],
        "method": method["method"],
        "stage": method["stage"],
        "artifact": method["artifact"],
        "artifact_size_mb": f"{artifact_size_mb:.3f}",
        "trainable_params": str(npz_summary["trainable_params"]),
        "trainable_arrays": npz_summary["trainable_arrays"],
        "stored_samples": str(npz_summary["stored_samples"]),
        "feature_dim": str(npz_summary["feature_dim"]),
        "action_dim": str(npz_summary["action_dim"]),
        "history": str(npz_summary["history"]),
        "horizon": str(npz_summary["horizon"]),
        "diffusion_steps": str(npz_summary["diffusion_steps"]),
        "train_mse": format_mse(npz_summary["train_mse"]),
        "val_mse": format_mse(npz_summary["val_mse"]),
        "train_time_seconds": str(npz_summary["train_time_seconds"]),
        "peak_vram_mb": str(npz_summary["peak_vram_mb"]),
        "train_range_success": summary_row.get("train_range_success", method.get("train_range_success", "")),
        "heldout_success": summary_row.get("heldout_success", method.get("heldout_success", "")),
        "note": note,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "version",
        "method",
        "stage",
        "artifact",
        "artifact_size_mb",
        "trainable_params",
        "trainable_arrays",
        "stored_samples",
        "feature_dim",
        "action_dim",
        "history",
        "horizon",
        "diffusion_steps",
        "train_mse",
        "val_mse",
        "train_time_seconds",
        "peak_vram_mb",
        "train_range_success",
        "heldout_success",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(values) + " |"


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    learned = [row for row in rows if int(row["trainable_params"]) > 0]
    largest = max(learned, key=lambda row: int(row["trainable_params"])) if learned else None

    lines = [
        "# 模型资源与规模汇总",
        "",
        "用途：记录每个已登记方法的 artifact 大小、可训练参数量、训练样本量和闭环评测结果，用于回答“轻量化后训练是否省算力/省参数/省数据”的论文问题。",
        "",
        "说明：",
        "",
        "- `可训练参数` 只统计 `.npz` 中 `weights`、`w*`、`b*` 等可学习数组；expert、replay 和 kNN 的可训练参数记为 0。",
        "- `kNN BC` 是非参数检索方法，虽然可训练参数为 0，但需要保存示范观测和动作样本，因此模型文件明显更大。",
        "- `训练耗时` 和 `峰值显存` 当前没有在旧训练脚本中记录，后续接入 PyTorch/LoRA/Adapter 时应加入计时和显存日志。",
        "- `Diffusion Policy-lite`、`ACT-lite` 都是本项目 NumPy 轻量 baseline，不能写成官方完整实现。",
        "",
        "## 资源对比表",
        "",
        md_row(
            [
                "版本",
                "方法",
                "可训练参数",
                "存储样本",
                "特征/观测维度",
                "动作维度",
                "历史/动作块",
                "模型大小 MB",
                "训练/留出成功率",
            ]
        ),
        md_row(["---", "---", "---:", "---:", "---:", "---:", "---", "---:", "---"]),
    ]

    for row in rows:
        history_horizon = "/".join(part for part in [row["history"], row["horizon"]] if part)
        lines.append(
            md_row(
                [
                    f"`{row['version']}`",
                    row["method"],
                    format_number(row["trainable_params"]),
                    format_number(row["stored_samples"]),
                    format_number(row["feature_dim"]),
                    format_number(row["action_dim"]),
                    history_horizon,
                    row["artifact_size_mb"],
                    f"{row['train_range_success']} / {row['heldout_success']}",
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 离线误差与资源备注",
            "",
            md_row(["版本", "train MSE", "val MSE", "训练耗时 s", "峰值显存 MB", "备注"]),
            md_row(["---", "---:", "---:", "---:", "---:", "---"]),
        ]
    )
    for row in rows:
        lines.append(
            md_row(
                [
                    f"`{row['version']}`",
                    row["train_mse"],
                    row["val_mse"],
                    row["train_time_seconds"],
                    row["peak_vram_mb"],
                    row["note"],
                ]
            )
        )

    lines.extend(["", "## 当前可写入论文的阶段性结论", ""])
    if largest:
        lines.append(
            f"1. 当前最重的可训练模型是 `{largest['version']}`，约 {format_number(largest['trainable_params'])} 个可训练参数，仍属于轻量级 MLP/action-chunk 范围。"
        )
    lines.extend(
        [
            "2. `Linear BC` 的参数量很小且离线 MSE 很低，但闭环成功率为 0，说明离线回归误差不能直接代表机械臂闭环操作能力。",
            "3. `kNN BC` 在训练范围能成功，但需要存储大量示范样本，留出范围下降明显，更像轨迹记忆而不是泛化策略。",
            "4. `Object-Language Action Head-lite` 与 `Multi-task Object-Language Action Head-lite` 参数量保持轻量，但当前符号特征和 MLP action head 仍不足以稳定解决语言/空间泛化。",
            "5. `Reward-Weighted Action Head-lite` 已补充轻量后训练代理，但当前 reward weighting 没有带来闭环成功率提升。",
            "6. `Frozen CLIP Action Head-lite` 已接入 pretrained VLM 表征代理：CLIP encoder 冻结不计入可训练参数，action head 训练耗时和闭环成功率单独记录。",
            "7. 下一阶段如果接入真正机器人 VLA、LoRA 或 Adapter，应在同一张表中继续补充 trainable parameters、训练耗时、显存和不同数据规模下的成功率。",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    versions = read_json(args.versions)
    summary_rows = read_summary(args.summary)
    rows = [
        summarize_method(method, summary_rows.get(method["version"], {}))
        for method in versions["methods"]
    ]

    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows)

    print(f"resource_csv: {args.output_csv}", flush=True)
    print(f"resource_markdown: {args.output_md}", flush=True)
    print(f"methods: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
