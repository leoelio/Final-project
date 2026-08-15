from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_clip_action_head import reset_to_saved_state  # noqa: E402
from train_clip_semantic_waypoint import LANGUAGE_AUGMENTATIONS, TASK_LABELS  # noqa: E402
from train_vision_language_action_head import attempt_start_index, selected_attempts  # noqa: E402
from widowx_env import WidowXTabletopEnv  # noqa: E402
from widowx_env.demo_dataset import read_metadata  # noqa: E402
from widowx_env.vision_grounding import load_calibration  # noqa: E402


RUN_DIRS = (
    ROOT / "data" / "demos" / "core_v2_place_blue_cube_blue_pad_medium_train20_v1",
    ROOT / "data" / "demos" / "kaggle_scale_place_blue_cube_blue_pad_medium_80_v1",
    ROOT / "data" / "demos" / "core_v2_place_blue_cube_red_pad_medium_train20_v1",
    ROOT / "data" / "demos" / "kaggle_scale_place_blue_cube_red_pad_medium_80_v1",
    ROOT / "data" / "demos" / "core_v2_place_red_cube_red_pad_medium_train20_v1",
    ROOT / "data" / "demos" / "kaggle_scale_place_red_cube_red_pad_medium_80_v1",
    ROOT / "data" / "demos" / "core_v2_move_leftmost_cube_to_bowl_language_train20_v1",
    ROOT / "data" / "demos" / "kaggle_scale_move_leftmost_cube_to_bowl_language_80_v1",
    ROOT / "data" / "demos" / "kaggle_scale_move_leftmost_cube_to_bowl_language_80_v1_sampling_failure",
    ROOT / "data" / "demos" / "kaggle_scale_move_leftmost_cube_to_bowl_language_80_v1_resume2",
)
RESERVED = {
    "place_blue_cube_blue_pad": set(range(20, 25)),
    "place_blue_cube_red_pad": set(range(120, 125)),
    "place_red_cube_red_pad": set(range(220, 225)),
    "move_leftmost_cube_to_bowl": set(range(420, 425)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export seed-disjoint MuJoCo initial RGB, language, and waypoint labels as a Kaggle patch-pointer training pack.")
    parser.add_argument("--audit-json", type=Path, default=ROOT / "outputs" / "evaluations" / "kaggle_spatial_data_collection_v1.json")
    parser.add_argument("--run-dirs", type=Path, nargs="+", default=RUN_DIRS)
    parser.add_argument("--calibration", type=Path, default=ROOT / "outputs" / "rgb_grounding" / "top_rgb_core_v2_calibration_v1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "kaggle_patch_pointer_v1")
    parser.add_argument("--archive", type=Path, default=ROOT / "outputs" / "kaggle_patch_pointer_v1.zip")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / "kaggle_patch_pointer_pack_v1.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "kaggle_patch_pointer_pack_v1.md")
    parser.add_argument("--pack-version", default="kaggle_patch_pointer_v1")
    parser.add_argument("--model-version", default="clip_patch_pointer_kaggle_v1")
    return parser.parse_args()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def world_to_pixel(matrix: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return np.linalg.solve(matrix[:, :2], np.asarray(xy, dtype=np.float32) - matrix[:, 2]).astype(np.float32)


def restore_initial(env: WidowXTabletopEnv, run_dir: Path, metadata: dict) -> None:
    with np.load(run_dir / str(metadata["trajectory_file"])) as data:
        attempts = selected_attempts(data, metadata, include_failures=False)
        start = attempt_start_index(data, int(attempts[0]))
        reset_to_saved_state(env, data["attempt_start_qpos"][start], data["attempt_start_qvel"][start], data["attempt_start_ctrl"][start])


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n", encoding="utf-8")


def make_notebook(path: Path, model_version: str) -> None:
    code = [
        "# Frozen CLIP Patch-Pointer Training on Kaggle\n",
        "from pathlib import Path\n",
        "DATASET = Path('/kaggle/input/widowx-mujoco-patch-pointer-v1')\n",
        "WORK = Path('/kaggle/working')\n",
        "!pip install -q transformers==4.57.0\n",
        "!cp {DATASET}/scripts/kaggle_train_patch_pointer.py {WORK}/kaggle_train_patch_pointer.py\n",
        f"!python {{WORK}}/kaggle_train_patch_pointer.py --dataset-root {{DATASET}} --output {{WORK}}/{model_version}.pt --metrics {{WORK}}/{model_version}.json --epochs 300 --device cuda\n",
        f"print('Download {model_version}.pt and {model_version}.json, then run the local MuJoCo evaluator documented in KAGGLE_UPLOAD.md.')\n",
    ]
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# MuJoCo Frozen CLIP Patch-Pointer Training\n", "Enable Internet so Kaggle can download the CLIP checkpoint, attach the exported dataset, then run the next cell.\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.x"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


def make_archive(source: Path, archive: Path) -> None:
    if archive.exists():
        raise FileExistsError(f"archive already exists: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(source).as_posix())


def main() -> None:
    args = parse_args()
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    if not audit.get("kaggle_export_ready"):
        raise RuntimeError("data audit does not permit a Kaggle export")
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    calibration = load_calibration(args.calibration)
    if calibration.image_size != 224 or calibration.camera != "top_rgb" or calibration.workspace_profile != "core_v2":
        raise ValueError("Kaggle export requires the Core V2 224px top_rgb calibration")
    args.output_dir.mkdir(parents=True)
    images_dir = args.output_dir / "images"
    images_dir.mkdir()
    env = WidowXTabletopEnv(seed=0, image_size=(224, 224), camera="top_rgb", workspace_profile="core_v2")
    renderer = mujoco.Renderer(env.model, height=224, width=224)
    rows: list[dict] = []
    source_counts: dict[str, int] = {}
    seen: set[tuple[str, int]] = set()
    try:
        for run_dir in args.run_dirs:
            count = 0
            for metadata in sorted(read_metadata(run_dir), key=lambda row: int(row["episode_index"])):
                if not metadata.get("success"):
                    continue
                task = str(metadata["task"])
                seed = int(metadata["seed"])
                if seed in RESERVED[task]:
                    raise RuntimeError(f"reserved holdout seed found during export: {task} seed {seed}")
                if (task, seed) in seen:
                    raise RuntimeError(f"duplicate successful source seed during export: {task} seed {seed}")
                seen.add((task, seed))
                env.reset(task=task, complexity=str(metadata["complexity"]), seed=seed)
                restore_initial(env, run_dir, metadata)
                renderer.update_scene(env.data, camera="top_rgb")
                image_name = f"{task}_seed_{seed}.png"
                image_path = images_dir / image_name
                from PIL import Image

                Image.fromarray(renderer.render().astype(np.uint8), mode="RGB").save(image_path)
                source_xy = np.asarray(metadata["initial_objects"][str(metadata["target_object"])][:2], dtype=np.float32)
                source_uv = np.clip(world_to_pixel(calibration.matrix, source_xy), 0.0, float(calibration.image_size - 1))
                split = "validation" if seed % 10 == 9 else "train"
                rows.append(
                    {
                        "id": f"{task}_seed_{seed}",
                        "image": f"images/{image_name}",
                        "task": task,
                        "task_label": TASK_LABELS.index(task),
                        "seed": seed,
                        "split": split,
                        "instruction_variants": [str(metadata["instruction"]), *LANGUAGE_AUGMENTATIONS[task]],
                        "source_xy_m": source_xy.round(6).tolist(),
                        "source_pixel_uv": source_uv.round(4).tolist(),
                        "source_pixel_uv_normalized": (source_uv / float(calibration.image_size - 1)).astype(float).round(7).tolist(),
                        "source_run_dir": relative(run_dir),
                        "source_trajectory": str(metadata["trajectory_file"]),
                    }
                )
                count += 1
            source_counts[relative(run_dir)] = count
    finally:
        renderer.close()
    rows.sort(key=lambda row: (row["task"], row["seed"]))
    samples_path = args.output_dir / "samples.jsonl"
    write_jsonl(samples_path, rows)
    content_hash = hashlib.sha256()
    for path in [samples_path, *sorted(images_dir.glob("*.png"))]:
        content_hash.update(path.name.encode("utf-8"))
        content_hash.update(path.read_bytes())
    per_split = {split: sum(row["split"] == split for row in rows) for split in ("train", "validation")}
    manifest = {
        "version": args.pack_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method_boundary": "This pack trains a frozen CLIP spatial patch-pointer head, not OpenVLA or a full VLA policy. MuJoCo object truth supplies offline labels only; runtime uses RGB, instruction, fixed calibration, and model predictions.",
        "task_labels": list(TASK_LABELS),
        "workspace_profile": "core_v2",
        "camera": "top_rgb",
        "image_size": 224,
        "calibration": calibration.to_dict(),
        "samples": len(rows),
        "per_split": per_split,
        "per_task": {task: sum(row["task"] == task for row in rows) for task in TASK_LABELS},
        "source_counts": source_counts,
        "reserved_holdout_seeds": {task: sorted(seeds) for task, seeds in RESERVED.items()},
        "dataset_content_sha256": content_hash.hexdigest(),
        "sample_schema": {"image": "relative PNG", "instruction_variants": "canonical plus three task-preserving variants", "source_pixel_uv_normalized": "supervision label", "split": "seed-deterministic train/validation"},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    scripts_dir = args.output_dir / "scripts"
    scripts_dir.mkdir()
    shutil.copy2(ROOT / "scripts" / "kaggle_train_patch_pointer.py", scripts_dir / "kaggle_train_patch_pointer.py")
    (args.output_dir / "requirements.txt").write_text("torch\ntransformers==4.57.0\npillow\nnumpy\n", encoding="utf-8")
    make_notebook(args.output_dir / "kaggle_train_patch_pointer.ipynb", args.model_version)
    upload = f"""# Kaggle 上传与训练

1. 上传本目录或同名 ZIP 为 Kaggle Dataset，建议名称 `widowx-mujoco-patch-pointer-v1`。
2. 创建 Kaggle Notebook，开启 GPU 和 Internet，并附加这个 Dataset。
3. 上传或打开 `kaggle_train_patch_pointer.ipynb`，确认输入目录为 `/kaggle/input/widowx-mujoco-patch-pointer-v1`，运行训练。
4. 下载 `{args.model_version}.pt` 与 `{args.model_version}.json` 到本机后，使用固定 MuJoCo 留出 `20-24/120-124/220-224/420-424` 运行 `scripts/evaluate_clip_patch_pointer.py`。

该训练包只对应冻结 CLIP patch-token 空间指针头。它不是 OpenVLA LoRA 运行包，不能以其训练结果宣称完成 OpenVLA、VLA foundation-model 或真实机械臂实验。
"""
    (args.output_dir / "KAGGLE_UPLOAD.md").write_text(upload, encoding="utf-8")
    make_archive(args.output_dir, args.archive)
    result = {
        "version": f"{args.pack_version}_pack",
        "dataset_dir": relative(args.output_dir),
        "archive": relative(args.archive),
        "samples": len(rows),
        "per_split": per_split,
        "per_task": manifest["per_task"],
        "dataset_content_sha256": manifest["dataset_content_sha256"],
        "audit": relative(args.audit_json),
        "kaggle_notebook": relative(args.output_dir / "kaggle_train_patch_pointer.ipynb"),
        "training_script": relative(scripts_dir / "kaggle_train_patch_pointer.py"),
        "decision": "Ready for Kaggle frozen-CLIP patch-pointer long training; OpenVLA LoRA remains a separate unmet remote path.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = f"""# Kaggle 冻结 CLIP Patch 指针训练包

版本：`{args.pack_version}_pack`

- 样本：`{result['samples']}` 条初始 RGB scene；训练/验证为 `{per_split['train']}/{per_split['validation']}` scene。
- 任务：`{manifest['per_task']}`。
- 内容 hash：`{result['dataset_content_sha256']}`。
- 数据审计：`{result['audit']}`，成功 seed 无重复且不包含固定 MuJoCo 留出。
- Kaggle notebook：`{result['kaggle_notebook']}`。
- ZIP：`{result['archive']}`。

训练输出必须回填本机，以固定 MuJoCo 20 episode 留出做闭环评测；Kaggle 内部验证指标不能替代该闭环证据。

方法边界：该包训练冻结 CLIP 的空间 patch-pointer 头，不是 OpenVLA、OpenVLA LoRA 或端到端 VLA 成果。
"""
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
