from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from vlm_runtime import ensure_vlm_path  # noqa: E402
from widowx_env import WidowXTabletopEnv  # noqa: E402
from widowx_env.demo_dataset import read_metadata  # noqa: E402

ensure_vlm_path()

from PIL import Image, ImageDraw  # noqa: E402


VERSION = "openvla_dataset_bridge_v1"
DEFAULT_RUN_DIR = ROOT / "data" / "demos" / "place_blue_cube_blue_pad_medium_20260702_051752"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "vla_bridge" / VERSION
DEFAULT_REPORT = ROOT / "docs" / "openvla_dataset_bridge_report.md"
DEFAULT_JSON = ROOT / "outputs" / "evaluations" / f"{VERSION}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a small MuJoCo demonstration preview in image/instruction/state/action format for future VLA adaptation.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--steps-per-episode", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--camera", default="top_rgb")
    parser.add_argument("--include-failures", action="store_true")
    return parser.parse_args()


def pre_step_array(series: np.ndarray, start_value: np.ndarray, indices: np.ndarray) -> np.ndarray:
    values = series[indices]
    if len(indices) == 1:
        return start_value[None, ...].astype(np.float32)
    return np.concatenate([start_value[None, ...], values[:-1]], axis=0).astype(np.float32)


def attempt_start_index(data: np.lib.npyio.NpzFile, attempt_id: int) -> int:
    matches = np.flatnonzero(data["attempt_start_ids"] == attempt_id)
    if len(matches) == 0:
        raise ValueError(f"attempt {attempt_id} has no saved start state")
    return int(matches[0])


def selected_attempt(metadata: dict[str, Any], data: np.lib.npyio.NpzFile) -> int:
    if metadata.get("success"):
        return int(metadata["attempts"])
    attempts = sorted(int(item) for item in np.unique(data["attempt_ids"]))
    if not attempts:
        raise ValueError("trajectory contains no attempts")
    return attempts[-1]


def sample_indices(indices: np.ndarray, count: int) -> np.ndarray:
    if len(indices) <= count:
        return indices.astype(np.int32)
    positions = np.linspace(0, len(indices) - 1, count, dtype=np.int32)
    return indices[positions].astype(np.int32)


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def restore_state(env: WidowXTabletopEnv, qpos: np.ndarray, qvel: np.ndarray, ctrl: np.ndarray) -> None:
    env.data.qpos[:] = qpos
    env.data.qvel[:] = qvel
    env.data.ctrl[:] = ctrl
    mujoco.mj_forward(env.model, env.data)


def build_sample(
    metadata: dict[str, Any],
    trajectory_path: Path,
    env: WidowXTabletopEnv,
    renderer: mujoco.Renderer,
    image_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with np.load(trajectory_path) as data:
        attempt_id = selected_attempt(metadata, data)
        indices = np.flatnonzero(data["attempt_ids"] == attempt_id)
        if len(indices) == 0:
            return rows
        start = attempt_start_index(data, attempt_id)
        sampled = sample_indices(indices, max(1, args.steps_per_episode))
        local_lookup = {int(source_index): local_index for local_index, source_index in enumerate(indices)}
        qpos = pre_step_array(data["qpos"], data["attempt_start_qpos"][start], indices)
        qvel = pre_step_array(data["qvel"], data["attempt_start_qvel"][start], indices)
        ctrl = pre_step_array(data["ctrl"], data["attempt_start_ctrl"][start], indices)
        tcp = pre_step_array(data["tcp"], data["attempt_start_tcp"][start], indices)
        objects = pre_step_array(data["object_positions"], data["attempt_start_object_positions"][start], indices)
        object_names = [str(item) for item in data["object_names"].tolist()]

        env.reset(task=str(metadata["task"]), complexity=str(metadata["complexity"]), seed=int(metadata["seed"]))
        for source_step in sampled:
            local = local_lookup[int(source_step)]
            restore_state(env, qpos[local], qvel[local], ctrl[local])
            renderer.update_scene(env.data, camera=env.camera)
            rgb = renderer.render().astype(np.uint8)
            image_name = f"episode_{int(metadata['episode_index']):06d}_step_{int(source_step):06d}.png"
            image_path = image_dir / image_name
            Image.fromarray(rgb, mode="RGB").save(image_path)

            rows.append(
                {
                    "version": VERSION,
                    "source_run_dir": rel(args.run_dir),
                    "episode_index": int(metadata["episode_index"]),
                    "seed": int(metadata["seed"]),
                    "attempt_id": int(attempt_id),
                    "source_step": int(source_step),
                    "task": str(metadata["task"]),
                    "complexity": str(metadata["complexity"]),
                    "instruction": str(metadata["instruction"]),
                    "target_object": metadata.get("target_object"),
                    "target_geom": metadata.get("target_geom"),
                    "active_objects": list(metadata.get("active_objects", [])),
                    "image": rel(image_path),
                    "state": {
                        "qpos": qpos[local].astype(float).round(6).tolist(),
                        "qvel": qvel[local].astype(float).round(6).tolist(),
                        "ctrl": ctrl[local].astype(float).round(6).tolist(),
                        "tcp": tcp[local].astype(float).round(6).tolist(),
                        "objects": {
                            name: objects[local, index].astype(float).round(6).tolist()
                            for index, name in enumerate(object_names)
                        },
                    },
                    "action": data["actions"][source_step].astype(float).round(6).tolist(),
                    "action_space": "MuJoCo WidowX actuator control target, 7 dimensions",
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_preview(image_paths: list[Path], output: Path, columns: int = 4, tile: int = 160) -> None:
    if not image_paths:
        return
    rows = int(np.ceil(len(image_paths) / columns))
    canvas = Image.new("RGB", (columns * tile, rows * tile), "white")
    draw = ImageDraw.Draw(canvas)
    for index, path in enumerate(image_paths):
        image = Image.open(path).convert("RGB").resize((tile, tile - 18))
        x = (index % columns) * tile
        y = (index // columns) * tile
        canvas.paste(image, (x, y))
        draw.text((x + 4, y + tile - 16), path.stem[-18:], fill=(0, 0, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# OpenVLA 数据桥接报告",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：把当前 MuJoCo WidowX demonstration 导出为 `image + instruction + state + action` 的小规模预览样本，作为后续真实 VLA/OpenVLA action-head、Adapter 或 LoRA 接入前的数据桥接检查。这个版本不是策略模型，不参与成功率比较。",
        "",
        "## 输出文件",
        "",
        "```text",
        summary["jsonl_path"],
        summary["manifest_path"],
        summary["preview_path"],
        summary["summary_json_path"],
        "```",
        "",
        "## 导出统计",
        "",
        f"- 源示范目录：`{summary['run_dir']}`",
        f"- episode 数：`{summary['episodes_exported']}`",
        f"- 样本数：`{summary['samples_exported']}`",
        f"- 图像尺寸：`{summary['image_size']}x{summary['image_size']}`",
        f"- 相机：`{summary['camera']}`",
        f"- 每个样本字段：`image`、`instruction`、`state.qpos/qvel/ctrl/tcp/objects`、`action`、`episode_index`、`source_step`",
        "",
        "## 当前 VLA 接入判断",
        "",
        "- OpenVLA 官方模型是 7B 级 VLA，适合用作真实机器人 VLA 后训练目标，但本机 6GB GPU 不适合直接加载并训练完整模型。",
        "- 官方 OpenVLA 仓库推荐在算力不足时使用 LoRA；其 fine-tuning 脚本说明 LoRA 仍以 48GB/80GB 级 GPU 作为参考配置。",
        "- OpenVLA-OFT 文档给出的推理和训练显存也明显高于当前 6GB GPU，因此本机下一步更适合先做数据格式、特征缓存和 action-head 接口，而不是直接训练 OpenVLA。",
        "- 本桥接版本的意义是把当前 MuJoCo demonstration 明确整理成 VLA 训练需要的基本字段；真正写作 `robot_vla_action_head_lite_v1` 前，仍必须实际接入机器人预训练 VLA/VLM 表征。",
        "",
        "参考入口：",
        "",
        "- OpenVLA GitHub：https://github.com/openvla/openvla",
        "- OpenVLA 7B Hugging Face：https://huggingface.co/openvla/openvla-7b",
        "- OpenVLA-OFT：https://openvla-oft.github.io/",
        "",
        "## 复现命令",
        "",
        "```powershell",
        f'& "{ROOT / ".venv" / "Scripts" / "python.exe"}" "{ROOT / "scripts" / "export_openvla_dataset_bridge.py"}" --episodes {summary["episodes_requested"]} --steps-per-episode {summary["steps_per_episode"]} --image-size {summary["image_size"]}',
        "```",
        "",
        "## 论文表述边界",
        "",
        "- 可以写：已经建立 MuJoCo demonstration 到 VLA 样本结构的桥接导出，后续可复用该字段接入 OpenVLA/机器人 VLA 表征。",
        "- 不能写：OpenVLA LoRA、真实机器人 VLA action head、Isaac 或真实 WidowX 验证已经完成。",
        "",
        f"生成时间：{summary['generated_at']}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    metadata_rows = [item for item in read_metadata(args.run_dir) if args.include_failures or item.get("success")]
    metadata_rows = sorted(metadata_rows, key=lambda item: int(item["episode_index"]))[: max(1, args.episodes)]

    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    env = WidowXTabletopEnv(seed=0, image_size=(args.image_size, args.image_size), camera=args.camera)
    renderer = mujoco.Renderer(env.model, height=args.image_size, width=args.image_size)
    samples: list[dict[str, Any]] = []
    try:
        for metadata in metadata_rows:
            samples.extend(build_sample(metadata, args.run_dir / metadata["trajectory_file"], env, renderer, image_dir, args))
    finally:
        renderer.close()

    jsonl_path = args.output_dir / "samples.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    preview_path = args.output_dir / "preview_grid.png"
    write_jsonl(jsonl_path, samples)
    image_paths = [ROOT / sample["image"] for sample in samples[:24]]
    make_preview(image_paths, preview_path)

    generated_at = datetime.now().isoformat(timespec="seconds")
    summary = {
        "version": VERSION,
        "generated_at": generated_at,
        "run_dir": rel(args.run_dir),
        "output_dir": rel(args.output_dir),
        "jsonl_path": rel(jsonl_path),
        "manifest_path": rel(manifest_path),
        "preview_path": rel(preview_path),
        "report_path": rel(args.report),
        "summary_json_path": rel(args.summary_json),
        "episodes_requested": int(args.episodes),
        "steps_per_episode": int(args.steps_per_episode),
        "episodes_exported": len({sample["episode_index"] for sample in samples}),
        "samples_exported": len(samples),
        "image_size": int(args.image_size),
        "camera": args.camera,
        "include_failures": bool(args.include_failures),
        "schema": {
            "image": "relative PNG path",
            "instruction": "task language string",
            "state": "MuJoCo qpos/qvel/ctrl/tcp/object positions before action",
            "action": "7D MuJoCo actuator control target",
        },
    }
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, args.report)
    print(f"version: {VERSION}", flush=True)
    print(f"samples_exported: {len(samples)}", flush=True)
    print(f"jsonl_path: {jsonl_path}", flush=True)
    print(f"preview_path: {preview_path}", flush=True)
    print(f"report_path: {args.report}", flush=True)


if __name__ == "__main__":
    main()
