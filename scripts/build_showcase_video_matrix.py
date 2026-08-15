from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from widowx_env import WidowXTabletopEnv


ASSET_DIR = ROOT / "showcase_assets" / "method_task_layout_matrix_v1"
MANIFEST_PATH = ASSET_DIR / "manifest.json"
JS_PATH = ASSET_DIR / "video_matrix.js"
CLIP_MODEL = ROOT / "runtime_assets" / "clip_semantic_waypoint_core_v2_v1_20260721_110325.npz"
CALIBRATION = ROOT / "runtime_assets" / "top_rgb_core_v2_calibration_v1.json"
STRUCTURED_MODEL = ROOT / "outputs" / "structured_waypoint_policy" / "structured_waypoint_policy_20260720_065456.npz"

METHODS = (
    {
        "id": "structured",
        "name": "Structured Waypoint Reference",
        "short_name": "Structured waypoint",
        "family": "state-conditioned control reference",
        "description": "Uses MuJoCo state for object and target positions; it is a control reference, not a language policy.",
    },
    {
        "id": "clip_open_loop",
        "name": "Frozen CLIP Semantic Waypoint",
        "short_name": "Frozen CLIP open loop",
        "family": "frozen vision-language intent + RGB geometry",
        "description": "Frozen CLIP selects one of four closed task intents, then a structured executor performs one attempt without RGB retry.",
    },
    {
        "id": "v4_rgb_feedback",
        "name": "Final RGB Feedback V4",
        "short_name": "Final V4",
        "family": "frozen vision-language intent + RGB feedback",
        "description": "The final MuJoCo system: frozen CLIP intent, top-view RGB grounding, structured execution, and at most one bounded RGB retry.",
    },
)

TASKS = (
    {
        "id": "blue_to_blue",
        "task": "place_blue_cube_blue_pad",
        "name": "Blue cube to blue pad",
        "name_zh": "蓝色立方体到蓝色放置盘",
        "focus": "colour and object identity",
    },
    {
        "id": "red_to_red",
        "task": "place_red_cube_red_pad",
        "name": "Red cube to red pad",
        "name_zh": "红色立方体到红色放置盘",
        "focus": "same-colour object and target",
    },
    {
        "id": "leftmost_to_bowl",
        "task": "move_leftmost_cube_to_bowl",
        "name": "Leftmost cube to white bowl",
        "name_zh": "最左方块到白碗",
        "focus": "spatial reference and language",
    },
)

LAYOUTS = (
    {
        "id": "standard",
        "name": "Standard layout",
        "name_zh": "标准三物体布局",
        "seed": 3300,
        "complexity": "medium",
        "purpose": "reference tabletop arrangement",
    },
    {
        "id": "repositioned",
        "name": "Repositioned layout",
        "name_zh": "同任务重排布局",
        "seed": 3301,
        "complexity": "medium",
        "purpose": "same task with the target object moved",
    },
    {
        "id": "hard_distractors",
        "name": "Distractor-rich layout",
        "name_zh": "七物体干扰布局",
        "seed": 3302,
        "complexity": "hard",
        "purpose": "same task with all distractors present",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the evidence-traceable 3 x 3 x 3 MuJoCo showcase video matrix.")
    parser.add_argument("--render", action="store_true", help="Render missing clips. Without this flag only the layout preflight and manifest are written.")
    parser.add_argument("--force", action="store_true", help="Re-render clips that already have both video and metadata files.")
    parser.add_argument("--method", choices=[item["id"] for item in METHODS], default=None, help="Render one method only.")
    parser.add_argument("--task", choices=[item["id"] for item in TASKS], default=None, help="Render one task only.")
    parser.add_argument("--layout", choices=[item["id"] for item in LAYOUTS], default=None, help="Render one layout only.")
    return parser.parse_args()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def select(items: tuple[dict, ...], item_id: str | None) -> tuple[dict, ...]:
    return tuple(item for item in items if item_id is None or item["id"] == item_id)


def layout_snapshot(task: dict, layout: dict) -> dict:
    env = WidowXTabletopEnv(seed=layout["seed"], workspace_profile="core_v2")
    obs = env.reset(task=task["task"], complexity=layout["complexity"], seed=layout["seed"])
    target_object = str(obs["target_object"])
    target_position = env.object_position(target_object)
    object_positions = {
        name: [round(float(value), 5) for value in env.object_position(name)[:2]]
        for name in obs["active_objects"]
    }
    return {
        "seed": layout["seed"],
        "complexity": layout["complexity"],
        "active_objects": list(obs["active_objects"]),
        "selected_target_object": target_object,
        "selected_target_xy_m": [round(float(value), 5) for value in target_position[:2]],
        "object_positions_xy_m": object_positions,
    }


def verify_layouts(tasks: tuple[dict, ...], layouts: tuple[dict, ...]) -> dict[str, dict[str, dict]]:
    snapshots: dict[str, dict[str, dict]] = {}
    for task in tasks:
        task_snapshots = {layout["id"]: layout_snapshot(task, layout) for layout in layouts}
        coordinates = [tuple(item["selected_target_xy_m"]) for item in task_snapshots.values()]
        if len(set(coordinates)) != len(coordinates):
            raise RuntimeError(f"layout seeds do not move the target object for {task['id']}: {coordinates}")
        snapshots[task["id"]] = task_snapshots
    return snapshots


def base_physics_args(layout: dict) -> list[str]:
    return [
        "--task", layout["task"],
        "--complexity", layout["complexity"],
        "--workspace-profile", "core_v2",
        "--seed", str(layout["seed"]),
        "--arm-kp", "150",
        "--arm-force", "100",
        "--gripper-kp", "1200",
        "--gripper-force", "200",
        "--friction", "5",
        "--place-tcp-z", "0.041",
    ]


def render_command(method: dict, task: dict, layout: dict, video_path: Path) -> list[str]:
    run = {**layout, "task": task["task"]}
    common = base_physics_args(run)
    if method["id"] == "structured":
        return [
            sys.executable,
            str(ROOT / "scripts" / "export_video.py"),
            "--method", "structured_waypoint_policy",
            "--version", f"matrix_v1_{method['id']}__{task['id']}__{layout['id']}",
            "--model", str(STRUCTURED_MODEL),
            "--output", str(video_path),
            "--width", "640", "--height", "480", "--fps", "24", "--frame-stride", "12",
            *common,
        ]
    if method["id"] == "clip_open_loop":
        return [
            sys.executable,
            str(ROOT / "scripts" / "export_video.py"),
            "--method", "clip_semantic_waypoint",
            "--version", f"matrix_v1_{method['id']}__{task['id']}__{layout['id']}",
            "--model", str(CLIP_MODEL),
            "--output", str(video_path),
            "--width", "640", "--height", "480", "--fps", "24", "--frame-stride", "12",
            *common,
        ]
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_clip_semantic_rgb_feedback.py"),
        "--model", str(CLIP_MODEL),
        "--calibration", str(CALIBRATION),
        "--video-path", str(video_path),
        "--width", "640", "--height", "480", "--fps", "24", "--frame-stride", "12",
        "--feedback-attempts", "1", "--recovery-search", "table", "--no-viewer",
        *common,
    ]


def viewer_command(method: dict, task: dict, layout: dict) -> list[str]:
    run = {**layout, "task": task["task"]}
    common = base_physics_args(run)
    if method["id"] == "structured":
        return [sys.executable, str(ROOT / "scripts" / "run_structured_waypoint_policy.py"), "--model", str(STRUCTURED_MODEL), "--viewer", "--duration", "90", "--speed", "0.04", *common]
    if method["id"] == "clip_open_loop":
        return [sys.executable, str(ROOT / "scripts" / "run_clip_semantic_waypoint.py"), "--model", str(CLIP_MODEL), "--viewer", "--duration", "90", "--speed", "0.04", *common]
    return [sys.executable, str(ROOT / "scripts" / "run_clip_semantic_rgb_feedback.py"), "--model", str(CLIP_MODEL), "--calibration", str(CALIBRATION), "--viewer", "--duration", "90", "--speed", "0.04", "--feedback-attempts", "1", "--recovery-search", "table", *common]


def quote_command(command: list[str]) -> str:
    return " ".join(f'"{item}"' if " " in item else item for item in command)


def load_result(metadata_path: Path) -> tuple[bool | None, dict]:
    if not metadata_path.exists():
        return None, {}
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    summary = data[0] if isinstance(data, list) else data.get("summary", {})
    success = summary.get("task_success", summary.get("success"))
    return bool(success) if success is not None else None, summary


def write_manifest(entries: list[dict], layout_snapshots: dict[str, dict[str, dict]]) -> None:
    script_path = ROOT / "scripts" / "build_showcase_video_matrix.py"
    manifest = {
        "version": "method_task_layout_matrix_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "MuJoCo-only WidowX tabletop rollouts. Every video is generated from the method, task, layout and seed recorded in this file.",
        "matrix_shape": {"methods": len(METHODS), "tasks": len(TASKS), "layouts": len(LAYOUTS), "expected_videos": len(METHODS) * len(TASKS) * len(LAYOUTS)},
        "methods": METHODS,
        "tasks": TASKS,
        "layouts": LAYOUTS,
        "layout_snapshots": layout_snapshots,
        "entries": entries,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    JS_PATH.write_text(f"window.SHOWCASE_VIDEO_MATRIX = {json.dumps(manifest, ensure_ascii=False)};\n", encoding="utf-8")
    lines = [
        "# 方法-任务-布局视频矩阵 V1",
        "",
        "本目录提供展示页使用的 3 方法 × 3 任务 × 3 布局 = 27 条 MuJoCo 录像。",
        "每个 MP4 都由对应策略脚本实际执行并由 MuJoCo 离屏渲染得到，不能跨单元替换或复用。",
        "",
        "## 方法",
        "",
        "- Structured Waypoint Reference：使用 MuJoCo 物体与目标状态的结构化控制参考，不是语言策略。",
        "- Frozen CLIP Semantic Waypoint：冻结 CLIP 闭集意图 + 顶视 RGB 定位 + 无重试的结构化执行。",
        "- Final RGB Feedback V4：冻结 CLIP 闭集意图 + 顶视 RGB 定位 + 至多一次有界 RGB 重试。",
        "",
        "## 任务",
        "",
        "- 蓝色立方体到蓝色放置盘（场景中是蓝盘，不是蓝碗）。",
        "- 红色立方体到红色放置盘。",
        "- 最左方块到白碗。",
        "",
        "## 布局",
        "",
        "- 标准三物体布局：seed 3300，medium。",
        "- 同任务重排布局：seed 3301，medium；目标物位置发生变化。",
        "- 七物体干扰布局：seed 3302，hard；目标物位置和干扰物同时变化。",
        "",
        "## 追溯规则",
        "",
        "- `manifest.json` 的每一条记录保存 method_id、task_id、layout_id、seed、complexity、MP4 路径、运行 JSON、渲染命令、viewer 命令和 task_success。",
        "- 同一 task + layout 在三种方法下使用同一个 seed 与 complexity，保证视频切换是同一初始桌面的对照。",
        "- `video_matrix.js` 是展示页可直接加载的同一份清单；不要手工修改其视频路径。",
        "",
        "## 重建",
        "",
        "```powershell",
        f'& "{sys.executable}" "{script_path}" --render',
        "```",
        "",
        "如需单独查看某格的交互仿真，复制 `manifest.json` 对应条目的 `viewer_command`，它会打开 MuJoCo viewer。",
        "",
    ]
    (ASSET_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    methods = select(METHODS, args.method)
    tasks = select(TASKS, args.task)
    layouts = select(LAYOUTS, args.layout)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    layout_snapshots = verify_layouts(tasks, layouts)
    entries: list[dict] = []

    for method in methods:
        for task in tasks:
            for layout in layouts:
                stem = f"{method['id']}__{task['id']}__{layout['id']}"
                video_path = ASSET_DIR / f"{stem}.mp4"
                metadata_path = video_path.with_suffix(".json")
                command = render_command(method, task, layout, video_path)
                if args.render and (args.force or not (video_path.exists() and metadata_path.exists())):
                    print(f"rendering: {stem}", flush=True)
                    subprocess.run(command, cwd=ROOT, check=True)
                success, summary = load_result(metadata_path)
                entries.append({
                    "id": stem,
                    "method_id": method["id"],
                    "task_id": task["id"],
                    "layout_id": layout["id"],
                    "video": rel(video_path),
                    "metadata": rel(metadata_path),
                    "render_command": quote_command(command),
                    "viewer_command": quote_command(viewer_command(method, task, layout)),
                    "task_success": success,
                    "summary": summary,
                })

    write_manifest(entries, layout_snapshots)
    rendered = sum(item["task_success"] is not None for item in entries)
    print(f"matrix_manifest: {MANIFEST_PATH}", flush=True)
    print(f"video_entries_with_metadata: {rendered}/{len(entries)}", flush=True)


if __name__ == "__main__":
    main()
