from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import zipfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "robot_vla_remote_run_pack_v1"
TARGET_VERSION = "robot_vla_action_head_lite_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a remote run pack for real Robot VLA action-head training.")
    parser.add_argument("--bridge-json", type=Path, default=ROOT / "outputs" / "evaluations" / "openvla_dataset_bridge_v1.json")
    parser.add_argument("--rlds-source-json", type=Path, default=ROOT / "outputs" / "evaluations" / "widowx_mujoco_rlds_source_v1.json")
    parser.add_argument("--rlds-source-validation-json", type=Path, default=ROOT / "outputs" / "evaluations" / "widowx_mujoco_rlds_source_validation_v1.json")
    parser.add_argument("--handoff-json", type=Path, default=ROOT / "outputs" / "evaluations" / "robot_vla_action_head_handoff_v1.json")
    parser.add_argument("--feasibility-json", type=Path, default=ROOT / "outputs" / "evaluations" / "openvla_feasibility_check_v1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "robot_vla_remote_run_pack" / VERSION)
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / f"{VERSION}.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "robot_vla_remote_run_pack.md")
    parser.add_argument("--preview-samples", type=int, default=12)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def ps_command(script: str) -> str:
    return f'& "{PYTHON}" "{ROOT / script}"'


def copy_bridge_tree(bridge_output_dir: Path, pack_dir: Path) -> list[str]:
    target = pack_dir / bridge_output_dir.relative_to(ROOT)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(bridge_output_dir, target)
    return sorted(rel(path) for path in target.rglob("*") if path.is_file())


def copy_rlds_builder(pack_dir: Path) -> list[str]:
    source = ROOT / "scripts" / "remote_openvla" / "widowx_mujoco_pick_place_dataset_builder.py"
    if not source.exists():
        raise FileNotFoundError(source)
    target_dir = pack_dir / "rlds_builder"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    shutil.copy2(source, target)
    return [rel(target)]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(sample, ensure_ascii=False, separators=(",", ":")) for sample in samples]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_remote_result_schema() -> dict[str, Any]:
    return {
        "version": TARGET_VERSION,
        "status": "remote_result_required",
        "required_top_level_fields": [
            "version",
            "status",
            "backbone_checkpoint",
            "uses_real_robot_vla_features",
            "hardware",
            "training",
            "evaluation",
            "artifacts",
            "paper_boundary",
        ],
        "hardware": {
            "required": ["gpu_name", "gpu_memory_gb", "torch_version", "cuda_version"],
            "minimum_gpu_memory_gb": 27,
            "recommended_gpu_memory_gb": 48,
        },
        "training": {
            "required": [
                "trainable_params",
                "train_time_seconds",
                "peak_gpu_memory_mb",
                "backbone_frozen",
                "feature_cache_path",
            ]
        },
        "data_contract": {
            "required": [
                "rlds_dataset_name",
                "rlds_dataset_statistics",
                "action_representation",
                "dataset_adapter_commit",
            ]
        },
        "evaluation": {
            "required": [
                "train_range_success",
                "heldout_success",
                "language_success",
                "target_distance",
                "grasp_success",
                "object_z",
            ]
        },
        "artifacts": {
            "required": [
                "model_artifact",
                "evaluation_json",
                "train_range_video",
                "language_video",
                "report",
            ]
        },
    }


def build_remote_result_template() -> dict[str, Any]:
    return {
        "version": TARGET_VERSION,
        "status": "fill_after_remote_training",
        "backbone_checkpoint": "",
        "uses_real_robot_vla_features": False,
        "hardware": {
            "gpu_name": "",
            "gpu_memory_gb": 0,
            "torch_version": "",
            "cuda_version": "",
        },
        "training": {
            "trainable_params": 0,
            "train_time_seconds": 0,
            "peak_gpu_memory_mb": 0,
            "backbone_frozen": True,
            "feature_cache_path": "outputs/robot_vla_action_head/openvla_feature_cache_v1.*",
        },
        "data_contract": {
            "rlds_dataset_name": "",
            "rlds_dataset_statistics": "",
            "action_representation": "",
            "dataset_adapter_commit": "",
        },
        "evaluation": {
            "train_range_success": "0/5",
            "heldout_success": "0/5",
            "language_success": "0/5",
            "target_distance": None,
            "grasp_success": "0/5",
            "object_z": None,
        },
        "artifacts": {
            "model_artifact": "outputs/robot_vla_action_head/robot_vla_action_head_lite_v1.*",
            "evaluation_json": "outputs/evaluations/robot_vla_action_head_lite_v1.json",
            "train_range_video": "outputs/videos/robot_vla_action_head_lite_v1_seed0.mp4",
            "language_video": "outputs/videos/robot_vla_action_head_lite_v1_language_seed200.mp4",
            "report": "docs/robot_vla_action_head_lite_report.md",
        },
        "paper_boundary": "只有回填真实机器人预训练 VLA/VLM 表征、远端训练日志、评测 JSON 和视频证据后，才能把 robot_vla_action_head_lite_v1 登记为正式方法。",
    }


def write_remote_commands(path: Path, pack: dict[str, Any]) -> None:
    lines = [
        "# Robot VLA 远端运行命令模板",
        "",
        "这些命令是远端 48GB+ GPU/云端机器上的执行模板，不在当前 6GB 本机直接运行。",
        "",
        "## 1. 解压运行包",
        "",
        "```powershell",
        "Expand-Archive .\\robot_vla_remote_run_pack_v1.zip -DestinationPath .\\robot_vla_remote_run_pack_v1 -Force",
        "Set-Location .\\robot_vla_remote_run_pack_v1",
        "```",
        "",
        "## 2. 安装依赖",
        "",
        "```powershell",
        "python -m pip install torch transformers accelerate peft torchvision",
        "python -m pip install bitsandbytes flash-attn",
        "python -m pip install tensorflow tensorflow-datasets",
        "```",
        "",
        "## 3. 先完成 RLDS 接入",
        "",
        "当前 `data/vla_bridge/widowx_mujoco_rlds_source_v1/` 已包含经过验证的 episode source，但仍不是已注册的 TFDS/RLDS 数据集。先按 `RLDS_INTEGRATION.md` 构建、注册并完成 smoke test。",
        "",
        "```powershell",
        "$env:WIDOWX_MUJOCO_RLDS_SOURCE_DIR = (Resolve-Path .\\data\\vla_bridge\\widowx_mujoco_rlds_source_v1)",
        "tfds build .\\rlds_builder --data_dir <RLDS_DATA_ROOT>",
        "Get-Content .\\RLDS_INTEGRATION.md",
        "```",
        "",
        "## 4. 完成注册后的 LoRA 命令模板",
        "",
        "只在 RLDS builder、OpenVLA dataset config 和 transform 已真实提交并记录 commit 后使用。`<...>` 是远端机器的实际路径，不能原样执行。",
        "",
        "```powershell",
        "Set-Location <OPENVLA_REPOSITORY>",
        "torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py `",
        "  --vla_path openvla/openvla-7b `",
        "  --data_root_dir <RLDS_DATA_ROOT> `",
        "  --dataset_name widowx_mujoco_pick_place `",
        "  --run_root_dir <RUN_ROOT> `",
        "  --adapter_tmp_dir <ADAPTER_TMP_ROOT> `",
        "  --batch_size 1 `",
        "  --grad_accumulation_steps 12 `",
        "  --max_steps 5000 `",
        "  --save_steps 500 `",
        "  --learning_rate 5e-4 `",
        "  --shuffle_buffer_size 1000",
        "```",
        "",
        "## 5. 回填结果",
        "",
        "远端完成后必须把以下文件带回本仓库，再按正式入包 gate 更新评测表、资源表、视频证据和失败模式分类。",
        "",
        "```text",
        *pack["required_remote_return_files"],
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_rlds_integration(path: Path, pack: dict[str, Any]) -> None:
    lines = [
        "# OpenVLA RLDS 接入检查单",
        "",
        "状态：`pre_rlds_source_ready`",
        "",
        "当前 `widowx_mujoco_rlds_source_v1` 包含 79 个成功 episode 的 image/state/action source。它不是 TensorFlow Datasets 生成的 RLDS 数据集，因此不能把 episode NPZ 直接传给 `vla-scripts/finetune.py`。动作已从原始 MuJoCo absolute actuator control target 转为 8D joint delta + gripper command。",
        "",
        "## 必须完成的四项工作",
        "",
        "1. 用 TFDS/RLDS builder 按 episode 导出 `steps`：每步至少包含 `observation.image`、`observation.state`、`action`、`discount`、`reward`、`is_first`、`is_last`、`is_terminal` 和 `language_instruction`。",
        "2. 保持 source v1 的 8D `JOINT_POS` 动作：6 个 WidowX joint delta、1 个零填充 joint delta、1 个二值夹爪开合命令。不要改回绝对 control target；控制频率和夹爪编码必须写入数据集文档。",
        "3. 在 OpenVLA 源码中为 `widowx_mujoco_pick_place` 注册 `StateEncoding.JOINT` 和 `ActionEncoding.JOINT_POS` 的 dataset config 及 standardization transform；记录这个改动所在 commit。LoRA `finetune.py` 依赖这一注册，而不是直接读取 NPZ。",
        "4. 先执行 10-step smoke test，确认 run directory 中保存 dataset statistics；再启动完整 LoRA。动作归一化统计必须与部署时的反归一化键一致。",
        "",
        "## 回填前最低验证",
        "",
        "- `tfds build` 或等价构建命令成功，且能读取 train split。",
        "- 随机抽取至少一个 episode，检查图像、instruction、state 和 action 长度对齐。",
        "- 记录 `rlds_dataset_name`、dataset statistics 路径、动作表示和 OpenVLA adapter commit。",
        "- 远端评测必须在当前 MuJoCo Core V2 的主任务、留出任务和语言/空间任务上分别运行，并保存视频。",
        "",
        "## OpenVLA 注册骨架",
        "",
        "将以下 config 加到 OpenVLA 的 `OXE_DATASET_CONFIGS`，并将 identity transform 登记到 `OXE_STANDARDIZATION_TRANSFORMS`。builder 已输出这些字段名，不能替换为旧 bridge 的 JSONL 字段。",
        "",
        "```python",
        '"widowx_mujoco_pick_place": {',
        '    "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},',
        '    "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},',
        '    "state_obs_keys": ["joint_state", "gripper_state"],',
        "    \"state_encoding\": StateEncoding.JOINT,",
        "    \"action_encoding\": ActionEncoding.JOINT_POS,",
        "},",
        "",
        "def widowx_mujoco_pick_place_transform(traj):",
        "    return traj",
        "```",
        "",
        "## 官方参考",
        "",
        "- OpenVLA README: https://github.com/openvla/openvla#fine-tuning-openvla-via-lora",
        "- OpenVLA finetune.py: https://github.com/openvla/openvla/blob/main/vla-scripts/finetune.py",
        "- RLDS builder example: https://github.com/kpertsch/rlds_dataset_builder",
        "",
        "## 论文边界",
        "",
        "完成本检查单只是获得真实训练的前提；只有远端训练、闭环评测、视频和结果回填全部完成，才能把它称为真实 OpenVLA LoRA 结果。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_pack_readme(path: Path, pack: dict[str, Any]) -> None:
    lines = [
        "# Robot VLA 远端运行包",
        "",
        f"版本：`{VERSION}`",
        "",
        f"目标计划版本：`{TARGET_VERSION}`",
        "",
        "用途：把当前 MuJoCo bridge 数据、真实 robot VLA action-head handoff 契约、远端运行命令模板和结果回填 schema 打包，供 48GB+ GPU 或云端机器继续执行真实 VLA/VLM 表征 action-head 后训练。",
        "",
        "重要边界：本包不是策略模型，不包含真实 OpenVLA/OFT 训练结果，不参与当前 25 个 MuJoCo 方法的成功率比较。",
        "",
        "## 1. 包内容摘要",
        "",
        "- `REMOTE_RUN_COMMANDS.md`：远端依赖、RLDS 前置工作、LoRA 命令模板与回填要求。",
        "- `RLDS_INTEGRATION.md`：episode source 与真正 OpenVLA 输入之间的缺口及验收清单。",
        "- `rlds_builder/widowx_mujoco_pick_place_dataset_builder.py`：远端 `tfds build` 可直接使用的 RLDS builder。",
        f"- `{pack['rlds_source_dir']}/`：{pack['rlds_source_episodes']} 条成功 episode、{pack['rlds_source_steps']} 个 image/state/action source step，尚未注册为 RLDS。",
        "- `remote_result_schema.json` / `remote_result_template.json`：回填字段与文件契约。",
        "- `run_config.json`：本机可行性、bridge 与 handoff 元数据。",
        "",
        "## 2. 输入数据",
        "",
        f"- RLDS source episode：`{pack['rlds_source_episodes']}` 条",
        f"- RLDS source step：`{pack['rlds_source_steps']}` 条",
        f"- image size：`{pack['rlds_source_image_size']}`",
        f"- source：`{pack['rlds_source_dir']}`",
        f"- manifest：`{pack['rlds_source_manifest']}`",
        "- 当前状态：`pre_rlds_source_ready`，必须先完成包内 `RLDS_INTEGRATION.md`。",
        "- 动作是经过验证的 8D JOINT_POS source，不是原始 MuJoCo absolute control target。",
        "",
        "## 3. 远端门槛",
        "",
        f"- 最低显存：`{pack['minimum_gpu_memory_gb']}GB`",
        f"- 建议显存：`{pack['recommended_gpu_memory_gb']}GB+`",
        "- 必须使用真实机器人预训练 VLA/VLM 表征，不能使用当前本地 hand-crafted proxy 冒充。",
        "",
        "## 4. 回填文件",
        "",
        "```text",
        *pack["required_remote_return_files"],
        "```",
        "",
        "## 5. 本地重建命令",
        "",
        "```powershell",
        ps_command("scripts/build_robot_vla_remote_run_pack.py"),
        ps_command("scripts/build_next_experiment_registry.py"),
        ps_command("scripts/build_final_artifact_manifest.py"),
        '$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"',
        ps_command("scripts/verify_experiment_artifacts.py"),
        "```",
        "",
        "## 6. 论文边界",
        "",
        "- 可以写：真实 robot VLA action-head 的远端运行包已经完成，数据、命令模板、结果 schema 和回填 gate 已固定。",
        "- 不能写：`robot_vla_action_head_lite_v1`、OpenVLA LoRA、OpenVLA-OFT、Isaac 或真实 WidowX 验证已经完成。",
        "",
        f"生成时间：{pack['generated_at']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def make_archive(pack_dir: Path) -> Path:
    archive = pack_dir.with_suffix(".zip")
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(pack_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(pack_dir).as_posix())
    return archive


def build_pack(args: argparse.Namespace) -> dict[str, Any]:
    bridge = read_json(args.bridge_json)
    rlds_source = read_json(args.rlds_source_json)
    rlds_source_validation = read_json(args.rlds_source_validation_json)
    handoff = read_json(args.handoff_json)
    feasibility = read_json(args.feasibility_json)
    if rlds_source.get("version") != "widowx_mujoco_rlds_source_v1":
        raise RuntimeError("unexpected RLDS source version")
    if rlds_source.get("status") != "rlds_source_ready_not_registered":
        raise RuntimeError("RLDS source must remain ready but unregistered")
    if rlds_source_validation.get("source_version") != rlds_source["version"]:
        raise RuntimeError("RLDS source validation is linked to the wrong source version")
    if int(rlds_source_validation.get("episodes_validated", 0)) != int(rlds_source.get("episodes_exported", 0)):
        raise RuntimeError("RLDS source validation episode count does not match source manifest")
    if int(rlds_source_validation.get("steps_validated", 0)) != int(rlds_source.get("steps_exported", 0)):
        raise RuntimeError("RLDS source validation step count does not match source manifest")
    bridge_output_dir = ROOT / bridge["output_dir"]
    rlds_source_dir = ROOT / rlds_source["output_dir"]
    samples_path = ROOT / bridge["jsonl_path"]
    samples = read_jsonl(samples_path)
    if len(samples) < int(handoff["input_contract"]["minimum_samples"]):
        raise RuntimeError(f"bridge samples are below handoff minimum: {len(samples)}")

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    copied_bridge_files = copy_bridge_tree(bridge_output_dir, args.output_dir)
    copied_rlds_source_files = copy_bridge_tree(rlds_source_dir, args.output_dir)
    copied_rlds_builder_files = copy_rlds_builder(args.output_dir)

    docs_dir = args.output_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for doc in (
        ROOT / "docs" / "openvla_dataset_bridge_report.md",
        ROOT / "docs" / "widowx_mujoco_rlds_source_v1_report.md",
        ROOT / "docs" / "widowx_mujoco_rlds_source_validation_v1.md",
        ROOT / "docs" / "openvla_feasibility_report.md",
        ROOT / "docs" / "robot_vla_action_head_handoff.md",
    ):
        shutil.copy2(doc, docs_dir / doc.name)

    schema = build_remote_result_schema()
    template = build_remote_result_template()
    write_json(args.output_dir / "run_config.json", {
        "version": VERSION,
        "status": "completed_prerequisite",
        "target_planned_version": TARGET_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "depends_on": [
            "openvla_dataset_bridge_v1",
            "widowx_mujoco_rlds_source_v1",
            "widowx_mujoco_rlds_source_validation_v1",
            "openvla_feasibility_check_v1",
            "robot_vla_action_head_handoff_v1",
        ],
        "bridge": bridge,
        "rlds_source": rlds_source,
        "rlds_source_validation": rlds_source_validation,
        "local_feasibility": feasibility.get("feasibility", {}),
        "handoff": handoff,
        "paper_boundary": "这是远端运行包，不是 robot_vla_action_head_lite_v1 的完成结果。",
    })
    write_json(args.output_dir / "remote_result_schema.json", schema)
    write_json(args.output_dir / "remote_result_template.json", template)
    write_jsonl(args.output_dir / "samples_preview.jsonl", samples[: args.preview_samples])

    pack: dict[str, Any] = {
        "version": VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed_prerequisite",
        "target_planned_version": TARGET_VERSION,
        "pack_dir": rel(args.output_dir),
        "archive_path": "",
        "bridge_samples": len(samples),
        "preview_samples": min(args.preview_samples, len(samples)),
        "image_size": int(bridge.get("image_size", 0)),
        "bridge_samples_jsonl": bridge["jsonl_path"],
        "bridge_manifest": bridge["manifest_path"],
        "bridge_preview": bridge["preview_path"],
        "rlds_source_version": rlds_source["version"],
        "rlds_source_validation_version": rlds_source_validation["version"],
        "rlds_source_dir": rlds_source["output_dir"],
        "rlds_source_manifest": rlds_source["manifest_path"],
        "rlds_source_episodes": int(rlds_source["episodes_exported"]),
        "rlds_source_steps": int(rlds_source["steps_exported"]),
        "rlds_source_image_size": int(rlds_source["image_size"]),
        "minimum_gpu_memory_gb": int(handoff["remote_runtime_gate"]["minimum_gpu_memory_gb"]),
        "recommended_gpu_memory_gb": int(handoff["remote_runtime_gate"]["recommended_gpu_memory_gb"]),
        "required_remote_return_files": [
            "outputs/robot_vla_action_head/robot_vla_action_head_lite_v1.*",
            "outputs/robot_vla_action_head/openvla_feature_cache_v1.*",
            "outputs/evaluations/robot_vla_action_head_lite_v1.json",
            "outputs/videos/robot_vla_action_head_lite_v1_seed0.mp4",
            "outputs/videos/robot_vla_action_head_lite_v1_language_seed200.mp4",
            "docs/robot_vla_action_head_lite_report.md",
        ],
        "packaged_files": [],
        "paper_boundary": "只能写成真实 robot VLA action-head 的远端运行包，不能写成真实 VLA 后训练结果。",
    }
    write_remote_commands(args.output_dir / "REMOTE_RUN_COMMANDS.md", pack)
    write_rlds_integration(args.output_dir / "RLDS_INTEGRATION.md", pack)
    write_pack_readme(args.output_dir / "README_REMOTE_RUN.md", pack)

    packaged_files = sorted(rel(path) for path in args.output_dir.rglob("*") if path.is_file())
    pack["packaged_files"] = packaged_files
    write_pack_readme(args.output_dir / "README_REMOTE_RUN.md", pack)
    archive = make_archive(args.output_dir)
    pack["archive_path"] = rel(archive)
    pack["copied_bridge_files"] = copied_bridge_files
    pack["copied_rlds_source_files"] = copied_rlds_source_files
    pack["copied_rlds_builder_files"] = copied_rlds_builder_files
    pack["packaged_file_count"] = len(packaged_files)

    write_json(args.output_json, pack)
    write_pack_readme(args.output_md, pack)
    return pack


def main() -> None:
    args = parse_args()
    pack = build_pack(args)
    print(f"robot_vla_remote_run_pack_md: {args.output_md}", flush=True)
    print(f"robot_vla_remote_run_pack_json: {args.output_json}", flush=True)
    print(f"robot_vla_remote_run_pack_archive: {pack['archive_path']}", flush=True)
    print(f"rlds_source_steps: {pack['rlds_source_steps']}", flush=True)
    print(f"packaged_files: {pack['packaged_file_count']}", flush=True)


if __name__ == "__main__":
    main()
