from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from vlm_runtime import ensure_vlm_path
from widowx_env import WidowXTabletopEnv
from widowx_env.demo_dataset import read_metadata


ensure_vlm_path()

from PIL import Image, ImageDraw


VERSION = "widowx_mujoco_rlds_source_v1"
DEFAULT_RUN_DIRS = (
    ROOT / "data" / "demos" / "core_v2_place_blue_cube_blue_pad_medium_train20_v1",
    ROOT / "data" / "demos" / "core_v2_place_blue_cube_red_pad_medium_train20_v1",
    ROOT / "data" / "demos" / "core_v2_place_red_cube_red_pad_medium_train20_v1",
    ROOT / "data" / "demos" / "core_v2_move_leftmost_cube_to_bowl_language_train20_v1",
)
DEFAULT_OUTPUT_DIR = ROOT / "data" / "vla_bridge" / VERSION
DEFAULT_REPORT = ROOT / "docs" / f"{VERSION}_report.md"
DEFAULT_JSON = ROOT / "outputs" / "evaluations" / f"{VERSION}.json"
OPEN_GRIPPER = 0.037
CLOSED_GRIPPER = 0.015


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export successful Core V2 demos as episode NPZ source files for a future RLDS builder."
    )
    parser.add_argument("--run-dirs", type=Path, nargs="+", default=DEFAULT_RUN_DIRS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--steps-per-episode", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--camera", default="top_rgb")
    return parser.parse_args()


def relative(path: Path, parent: Path = ROOT) -> str:
    try:
        return path.relative_to(parent).as_posix()
    except ValueError:
        return path.as_posix()


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


def successful_attempt(metadata: dict[str, Any]) -> int:
    if not metadata.get("success"):
        raise ValueError("RLDS source exporter only accepts successful demonstrations")
    return int(metadata["attempts"])


def sample_local_indices(length: int, count: int) -> np.ndarray:
    if count < 2:
        raise ValueError("steps-per-episode must be at least 2")
    if length < count:
        raise ValueError(f"trajectory has only {length} steps, fewer than requested {count}")
    return np.linspace(0, length - 1, count, dtype=np.int32)


def gripper_fraction(qpos: np.ndarray) -> float:
    return float(np.clip((float(qpos[6]) - CLOSED_GRIPPER) / (OPEN_GRIPPER - CLOSED_GRIPPER), 0.0, 1.0))


def gripper_open_command(ctrl: np.ndarray) -> float:
    threshold = (OPEN_GRIPPER + CLOSED_GRIPPER) / 2.0
    return float(float(ctrl[6]) >= threshold)


def joint_state(qpos: np.ndarray) -> np.ndarray:
    state = np.zeros(8, dtype=np.float32)
    state[:6] = qpos[:6]
    state[7] = gripper_fraction(qpos)
    return state


def joint_delta_action(current_qpos: np.ndarray, next_qpos: np.ndarray, next_ctrl: np.ndarray) -> np.ndarray:
    action = np.zeros(8, dtype=np.float32)
    action[:6] = next_qpos[:6] - current_qpos[:6]
    action[7] = gripper_open_command(next_ctrl)
    return action


def restore_state(env: WidowXTabletopEnv, qpos: np.ndarray, qvel: np.ndarray, ctrl: np.ndarray) -> None:
    env.data.qpos[:] = qpos
    env.data.qvel[:] = qvel
    env.data.ctrl[:] = ctrl
    mujoco.mj_forward(env.model, env.data)


def export_episode(
    metadata: dict[str, Any],
    run_dir: Path,
    output_dir: Path,
    episode_id: int,
    env: WidowXTabletopEnv,
    renderer: mujoco.Renderer,
    args: argparse.Namespace,
) -> dict[str, Any]:
    trajectory_path = run_dir / str(metadata["trajectory_file"])
    image_dir = output_dir / "images"
    episode_dir = output_dir / "episodes"

    with np.load(trajectory_path) as data:
        attempt_id = successful_attempt(metadata)
        source_indices = np.flatnonzero(data["attempt_ids"] == attempt_id)
        if len(source_indices) == 0:
            raise ValueError(f"attempt {attempt_id} has no trajectory rows")
        start = attempt_start_index(data, attempt_id)
        qpos = pre_step_array(data["qpos"], data["attempt_start_qpos"][start], source_indices)
        qvel = pre_step_array(data["qvel"], data["attempt_start_qvel"][start], source_indices)
        ctrl = pre_step_array(data["ctrl"], data["attempt_start_ctrl"][start], source_indices)
        local_indices = sample_local_indices(len(source_indices), args.steps_per_episode)

        env.reset(
            task=str(metadata["task"]),
            complexity=str(metadata["complexity"]),
            seed=int(metadata["seed"]),
        )
        states: list[np.ndarray] = []
        images: list[str] = []
        source_steps: list[int] = []
        for local_index in local_indices:
            restore_state(env, qpos[local_index], qvel[local_index], ctrl[local_index])
            renderer.update_scene(env.data, camera=args.camera)
            image_name = f"episode_{episode_id:04d}_step_{int(source_indices[local_index]):04d}.png"
            image_path = image_dir / image_name
            Image.fromarray(renderer.render().astype(np.uint8), mode="RGB").save(image_path)
            images.append(relative(image_path, output_dir))
            states.append(joint_state(qpos[local_index]))
            source_steps.append(int(source_indices[local_index]))

        state_array = np.asarray(states, dtype=np.float32)
        action_array = np.zeros_like(state_array)
        for index in range(len(local_indices) - 1):
            current = int(local_indices[index])
            following = int(local_indices[index + 1])
            action_array[index] = joint_delta_action(qpos[current], qpos[following], ctrl[following])
        action_array[-1, 7] = gripper_open_command(ctrl[int(local_indices[-1])])

    episode_path = episode_dir / f"episode_{episode_id:04d}.npz"
    np.savez_compressed(
        episode_path,
        image_paths=np.asarray(images),
        state=state_array,
        action=action_array,
        discount=np.ones(len(state_array), dtype=np.float32),
        reward=np.concatenate([np.zeros(len(state_array) - 1, dtype=np.float32), np.ones(1, dtype=np.float32)]),
        is_first=np.arange(len(state_array)) == 0,
        is_last=np.arange(len(state_array)) == len(state_array) - 1,
        is_terminal=np.arange(len(state_array)) == len(state_array) - 1,
        language_instruction=np.asarray(str(metadata["instruction"])),
        task=np.asarray(str(metadata["task"])),
        source_run_dir=np.asarray(relative(run_dir)),
        source_episode_index=np.asarray(int(metadata["episode_index"]), dtype=np.int32),
        source_steps=np.asarray(source_steps, dtype=np.int32),
    )
    return {
        "episode_id": episode_id,
        "episode_path": relative(episode_path),
        "source_run_dir": relative(run_dir),
        "source_episode_index": int(metadata["episode_index"]),
        "task": str(metadata["task"]),
        "instruction": str(metadata["instruction"]),
        "steps": len(state_array),
        "image_paths": images,
        "action_min": action_array.min(axis=0).round(6).tolist(),
        "action_max": action_array.max(axis=0).round(6).tolist(),
    }


def write_preview(output_dir: Path, image_paths: list[str]) -> None:
    selected = image_paths[:24]
    if not selected:
        return
    tile = 160
    columns = 4
    rows = int(np.ceil(len(selected) / columns))
    preview = Image.new("RGB", (columns * tile, rows * tile), "white")
    draw = ImageDraw.Draw(preview)
    for index, image_path in enumerate(selected):
        image = Image.open(output_dir / image_path).convert("RGB").resize((tile, tile - 18))
        x = (index % columns) * tile
        y = (index // columns) * tile
        preview.paste(image, (x, y))
        draw.text((x + 4, y + tile - 16), Path(image_path).stem[-17:], fill=(0, 0, 0))
    preview.save(output_dir / "preview_grid.png")


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# WidowX MuJoCo RLDS 源数据报告",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：将 Core V2 的成功示范整理成逐 episode 的 NPZ 源文件，供远端 TFDS/RLDS builder 构建并注册到 OpenVLA。它不是已经构建完成的 RLDS 数据集，也不是任何策略训练结果。",
        "",
        "## 数据规模",
        "",
        f"- 成功 episode：`{summary['episodes_exported']}`",
        f"- 总步骤：`{summary['steps_exported']}`",
        f"- 每个 episode：`{summary['steps_per_episode']}` 个低频动作片段",
        f"- 图像：`{summary['image_size']}x{summary['image_size']}`，相机 `{summary['camera']}`",
        f"- 任务：`{'`、`'.join(summary['tasks'])}`",
        "",
        "## 动作与状态表示",
        "",
        "- state：8D，`6 个 WidowX 关节位置 + 1 个零填充关节 + 1 个连续夹爪开合状态`，对应 OpenVLA `StateEncoding.JOINT`。",
        "- action：8D，`6 个相邻低频状态的关节增量 + 1 个零填充关节增量 + 1 个二值夹爪开合命令`，对应 OpenVLA `ActionEncoding.JOINT_POS`。",
        "- 原始 MuJoCo absolute actuator control target 未直接作为动作标签使用；每个非终点 action 从当前和下一采样时刻的关节状态计算得到。",
        "",
        "## 后续远端步骤",
        "",
        "1. 用 TFDS/RLDS builder 读取 `episodes/episode_*.npz` 并生成 `steps` 数据集。",
        "2. 在 OpenVLA 注册 `widowx_mujoco_pick_place` 的 dataset config 和 standardization transform。",
        "3. 先完成 10-step smoke test，并检查 OpenVLA run directory 的 dataset statistics。",
        "4. 再运行 LoRA，并通过 `robot_vla_remote_result_intake_v1` 回填评测、视频和资源数据。",
        "",
        "## 文件",
        "",
        "```text",
        summary["output_dir"],
        summary["manifest_path"],
        summary["preview_path"],
        summary["summary_json_path"],
        "```",
        "",
        "## 复现命令",
        "",
        "```powershell",
        f'& "{ROOT / ".venv" / "Scripts" / "python.exe"}" "{ROOT / "scripts" / "export_openvla_rlds_source.py"}"',
        "```",
        "",
        "## 论文边界",
        "",
        "- 可以写：已完成 Core V2 成功示范到 RLDS source episode 的动作语义转换与版本化保存。",
        "- 不能写：RLDS 已注册、OpenVLA LoRA、Isaac 或真实 WidowX 验证已经完成。",
        "",
        f"生成时间：{summary['generated_at']}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output already exists: {args.output_dir}")
    if args.steps_per_episode < 2:
        raise ValueError("steps-per-episode must be at least 2")

    args.output_dir.mkdir(parents=True)
    (args.output_dir / "images").mkdir()
    (args.output_dir / "episodes").mkdir()
    env = WidowXTabletopEnv(seed=0, image_size=(args.image_size, args.image_size), camera=args.camera, workspace_profile="core_v2")
    renderer = mujoco.Renderer(env.model, height=args.image_size, width=args.image_size)
    episode_rows: list[dict[str, Any]] = []
    try:
        for run_dir in args.run_dirs:
            for metadata in read_metadata(run_dir):
                if metadata.get("success"):
                    episode_rows.append(export_episode(metadata, run_dir, args.output_dir, len(episode_rows), env, renderer, args))
    finally:
        renderer.close()

    if not episode_rows:
        shutil.rmtree(args.output_dir)
        raise ValueError("no successful episodes were exported")

    image_paths = [path for row in episode_rows for path in row["image_paths"]]
    all_actions = []
    for row in episode_rows:
        with np.load(ROOT / row["episode_path"]) as data:
            all_actions.append(data["action"])
    action_values = np.concatenate(all_actions, axis=0)
    write_preview(args.output_dir, image_paths)

    summary = {
        "version": VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "rlds_source_ready_not_registered",
        "output_dir": relative(args.output_dir),
        "manifest_path": relative(args.output_dir / "manifest.json"),
        "preview_path": relative(args.output_dir / "preview_grid.png"),
        "summary_json_path": relative(args.summary_json),
        "source_run_dirs": [relative(path) for path in args.run_dirs],
        "episodes_exported": len(episode_rows),
        "steps_exported": len(image_paths),
        "steps_per_episode": int(args.steps_per_episode),
        "image_size": int(args.image_size),
        "camera": args.camera,
        "tasks": sorted({row["task"] for row in episode_rows}),
        "episode_manifest": episode_rows,
        "state_representation": "JOINT: 6 WidowX arm qpos + zero pad + continuous gripper open fraction",
        "action_representation": "JOINT_POS: 6 sampled qpos deltas + zero pad + binary gripper open command",
        "action_shape": [8],
        "action_min": action_values.min(axis=0).round(6).tolist(),
        "action_max": action_values.max(axis=0).round(6).tolist(),
        "paper_boundary": "这是待远端 TFDS/RLDS builder 使用的 source episode，不是已注册 RLDS 数据集或真实 OpenVLA 训练结果。",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(summary, args.report)
    print(f"version: {VERSION}", flush=True)
    print(f"episodes_exported: {summary['episodes_exported']}", flush=True)
    print(f"steps_exported: {summary['steps_exported']}", flush=True)
    print(f"output_dir: {args.output_dir}", flush=True)
    print(f"report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
