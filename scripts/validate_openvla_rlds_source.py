from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VERSION = "widowx_mujoco_rlds_source_validation_v1"
SOURCE_VERSION = "widowx_mujoco_rlds_source_v1"
DEFAULT_SOURCE_DIR = ROOT / "data" / "vla_bridge" / SOURCE_VERSION
DEFAULT_JSON = ROOT / "outputs" / "evaluations" / f"{VERSION}.json"
DEFAULT_REPORT = ROOT / "docs" / f"{VERSION}.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the versioned episode source before building the remote RLDS dataset.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_episode(source_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / row["episode_path"]
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        required = {
            "image_paths",
            "state",
            "action",
            "discount",
            "reward",
            "is_first",
            "is_last",
            "is_terminal",
            "language_instruction",
            "task",
            "source_steps",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise KeyError(f"{path.name} missing fields: {missing}")
        count = len(data["image_paths"])
        if count < 2 or data["state"].shape != (count, 8) or data["action"].shape != (count, 8):
            raise ValueError(f"{path.name} has invalid state/action shape")
        if not np.isfinite(data["state"]).all() or not np.isfinite(data["action"]).all():
            raise ValueError(f"{path.name} contains non-finite values")
        if not bool(data["is_first"][0]) or int(np.sum(data["is_first"])) != 1:
            raise ValueError(f"{path.name} has invalid is_first flags")
        for key in ("is_last", "is_terminal"):
            if not bool(data[key][-1]) or int(np.sum(data[key])) != 1:
                raise ValueError(f"{path.name} has invalid {key} flags")
        if not np.allclose(data["action"][:, 6], 0.0):
            raise ValueError(f"{path.name} has non-zero padded joint action")
        gripper_values = set(np.unique(data["action"][:, 7]).astype(float).tolist())
        if not gripper_values.issubset({0.0, 1.0}):
            raise ValueError(f"{path.name} gripper action is not binary: {sorted(gripper_values)}")
        image_paths = [str(item) for item in data["image_paths"]]
        missing_images = [image for image in image_paths if not (source_dir / image).exists()]
        if missing_images:
            raise FileNotFoundError(f"{path.name} missing image: {missing_images[0]}")
        return {
            "episode_path": row["episode_path"],
            "task": str(data["task"].item()),
            "steps": count,
            "image_count": len(image_paths),
            "action_abs_max": float(np.abs(data["action"][:, :6]).max()),
        }


def write_report(summary: dict[str, Any], path: Path) -> None:
    task_rows = "\n".join(f"| `{task}` | {count} |" for task, count in summary["task_episode_counts"].items())
    lines = [
        "# WidowX MuJoCo RLDS 源数据验证",
        "",
        f"版本：`{VERSION}`",
        "",
        f"- 源版本：`{SOURCE_VERSION}`",
        f"- episode：`{summary['episodes_validated']}`",
        f"- 样本：`{summary['steps_validated']}`",
        f"- state/action shape：`{summary['state_shape']}` / `{summary['action_shape']}`",
        f"- 图像文件：`{summary['images_validated']}`",
        f"- 关节增量绝对值最大值：`{summary['joint_delta_abs_max']:.6f}`",
        "",
        "## 任务分布",
        "",
        "| 任务 | 成功 episode |",
        "| --- | ---: |",
        task_rows,
        "",
        "## 已验证约束",
        "",
        "- 所有 episode 都有完整图像、state、action、语言和 RLDS 终止字段。",
        "- state/action 都是 `(steps, 8)`，第七关节为零填充。",
        "- 夹爪 action 只包含 0 或 1，且每条轨迹只有一个 first/last/terminal。",
        "- 所有 image path 均可读取。",
        "",
        "## 论文边界",
        "",
        "- 可以写：待注册 RLDS source 的数据结构和动作语义已通过本地完整性验证。",
        "- 不能写：TFDS/RLDS 已在远端构建、OpenVLA LoRA 或真实机器人验证已经完成。",
        "",
        f"生成时间：{summary['generated_at']}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest_path = args.source_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    if manifest.get("version") != SOURCE_VERSION:
        raise RuntimeError("source manifest has an unexpected version")
    rows = [validate_episode(args.source_dir, row) for row in manifest.get("episode_manifest", [])]
    if not rows:
        raise ValueError("source manifest has no episodes")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["task"]] = counts.get(row["task"], 0) + 1
    summary = {
        "version": VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_version": SOURCE_VERSION,
        "source_dir": relative(args.source_dir),
        "episodes_validated": len(rows),
        "steps_validated": sum(int(row["steps"]) for row in rows),
        "images_validated": sum(int(row["image_count"]) for row in rows),
        "state_shape": [8],
        "action_shape": [8],
        "task_episode_counts": dict(sorted(counts.items())),
        "joint_delta_abs_max": max(float(row["action_abs_max"]) for row in rows),
        "paper_boundary": "本验证只覆盖待注册 RLDS source 的结构完整性，不代表远端 TFDS 构建或 OpenVLA 训练已经完成。",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(summary, args.report)
    print(f"version: {VERSION}", flush=True)
    print(f"episodes_validated: {summary['episodes_validated']}", flush=True)
    print(f"steps_validated: {summary['steps_validated']}", flush=True)
    print(f"images_validated: {summary['images_validated']}", flush=True)


if __name__ == "__main__":
    main()
