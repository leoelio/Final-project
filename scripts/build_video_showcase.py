from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHOWCASE_DIR = ROOT / "outputs" / "showcase"
MANIFEST_PATH = SHOWCASE_DIR / "video_showcase_manifest.json"
DOC_PATH = ROOT / "docs" / "video_showcase.md"


LANGUAGE_CLIPS = [
    {
        "version": "expert_scripted_language_v1",
        "method": "Expert language oracle",
        "stage": "language_oracle",
        "train_range_success": "not_applicable",
        "heldout_success": "4/5",
        "clip": "outputs/videos/expert_scripted_language_v1_seed200.mp4",
    },
    {
        "version": "structured_waypoint_policy_v1_language_eval",
        "method": "Structured waypoint language eval",
        "stage": "structured_control_baseline",
        "train_range_success": "not_applicable",
        "heldout_success": "4/5",
        "clip": "outputs/videos/structured_waypoint_policy_v1_language_seed200.mp4",
    },
    {
        "version": "object_language_action_head_lite_v1_language_eval",
        "method": "Single-task action head language eval",
        "stage": "vla_action_head_proxy",
        "train_range_success": "not_applicable",
        "heldout_success": "0/5",
        "clip": "outputs/videos/object_language_action_head_lite_v1_language_seed200.mp4",
    },
    {
        "version": "reward_weighted_action_head_lite_v1_language_eval",
        "method": "Reward-weighted action head language eval",
        "stage": "reward_weighted_bc_post_training",
        "train_range_success": "not_applicable",
        "heldout_success": "0/5",
        "clip": "outputs/videos/reward_weighted_action_head_lite_v1_language_seed200.mp4",
    },
    {
        "version": "phase_conditioned_action_head_lite_v1_language_eval",
        "method": "Phase-conditioned action head language eval",
        "stage": "phase_conditioned_action_head_proxy",
        "train_range_success": "not_applicable",
        "heldout_success": "0/5",
        "clip": "outputs/videos/phase_conditioned_action_head_lite_v1_language_seed200.mp4",
    },
    {
        "version": "torch_act_cvae_state_chunk_v1_language_eval",
        "method": "ACT-CVAE-lite language eval",
        "stage": "torch_act_cvae_baseline",
        "train_range_success": "not_applicable",
        "heldout_success": "0/5",
        "clip": "outputs/videos/torch_act_cvae_state_chunk_v1_language_seed200.mp4",
    },
    {
        "version": "torch_act_state_chunk_cuda_v1_language_eval",
        "method": "CUDA Torch ACT language eval",
        "stage": "torch_act_baseline",
        "train_range_success": "not_applicable",
        "heldout_success": "0/5",
        "clip": "outputs/videos/torch_act_state_chunk_cuda_v1_language_seed200.mp4",
    },
    {
        "version": "torch_diffusion_policy_state_chunk_v1_language_eval",
        "method": "PyTorch state diffusion policy language eval",
        "stage": "torch_diffusion_policy_baseline",
        "train_range_success": "not_applicable",
        "heldout_success": "0/5",
        "clip": "outputs/videos/torch_diffusion_policy_state_chunk_v1_language_seed200.mp4",
    },
    {
        "version": "visual_feature_act_lite_v1_language_eval",
        "method": "Visual-feature ACT-lite language eval",
        "stage": "visual_feature_act_baseline",
        "train_range_success": "not_applicable",
        "heldout_success": "0/5",
        "clip": "outputs/videos/visual_feature_act_lite_v1_language_seed200.mp4",
    },
    {
        "version": "vision_language_action_head_lite_v1_language_eval",
        "method": "Vision-language action head language eval",
        "stage": "vla_action_head_proxy",
        "train_range_success": "not_applicable",
        "heldout_success": "0/5",
        "clip": "outputs/videos/vision_language_action_head_lite_v1_language_seed200.mp4",
    },
    {
        "version": "clip_action_head_lite_v1_language_eval",
        "method": "Frozen CLIP action head language eval",
        "stage": "pretrained_vlm_action_head_proxy",
        "train_range_success": "not_applicable",
        "heldout_success": "0/5",
        "clip": "outputs/videos/clip_action_head_lite_v1_language_seed200.mp4",
    },
    {
        "version": "adapter_action_head_lite_v1_language_eval",
        "method": "Adapter action head language eval",
        "stage": "peft_action_head_proxy",
        "train_range_success": "not_applicable",
        "heldout_success": "0/5",
        "clip": "outputs/videos/adapter_action_head_lite_v1_language_seed200.mp4",
    },
    {
        "version": "lora_action_head_lite_v1_language_eval",
        "method": "LoRA-style action head language eval",
        "stage": "peft_action_head_proxy",
        "train_range_success": "not_applicable",
        "heldout_success": "0/5",
        "clip": "outputs/videos/lora_action_head_lite_v1_language_seed200.mp4",
    },
    {
        "version": "multi_task_object_action_head_lite_v1_language_eval",
        "method": "Multi-task action head language eval",
        "stage": "multi_task_action_head_proxy",
        "train_range_success": "not_applicable",
        "heldout_success": "0/5",
        "clip": "outputs/videos/multi_task_object_action_head_lite_v1_language_seed400.mp4",
    },
]


PRESET_OUTPUTS = {
    "core": "core_methods_grid.mp4",
    "registered": "all_registered_methods_grid.mp4",
    "language": "language_generalization_grid.mp4",
}


CORE_VERSIONS = [
    "expert_scripted_v1",
    "structured_waypoint_policy_v1",
    "replay_demo_v1",
    "linear_bc_v1",
    "knn_bc_v1",
    "mlp_bc_v1",
    "torch_act_state_chunk_v1",
    "torch_act_state_chunk_cuda_v1",
    "torch_act_cvae_state_chunk_v1",
    "torch_diffusion_policy_state_chunk_v1",
    "visual_feature_act_lite_v1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build side-by-side rollout videos for defense/demo comparison.")
    parser.add_argument("--versions", type=Path, default=ROOT / "docs" / "experiment_versions.json")
    parser.add_argument("--preset", choices=("all", "core", "registered", "language"), default="all")
    parser.add_argument("--output-dir", type=Path, default=SHOWCASE_DIR)
    parser.add_argument("--doc", type=Path, default=DOC_PATH)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--tile-height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=24)
    return parser.parse_args()


def read_versions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["methods"]


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def ffmpeg_path() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg was not found on PATH")
    return exe


def ffprobe_video(path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,duration,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"no video stream found: {path}")
    return streams[0]


def metadata_for(clip_path: Path) -> dict:
    metadata_path = clip_path.with_suffix(".json")
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def clip_result(clip_path: Path) -> str:
    summary = metadata_for(clip_path).get("summary", {})
    if "success" in summary:
        return f"success {summary['success']}"
    if "steps_replayed" in summary:
        return f"replay {summary['steps_replayed']}"
    return "result unknown"


def escape_drawtext(value: str) -> str:
    return value.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def font_file() -> Path | None:
    for candidate in (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/segoeui.ttf")):
        if candidate.exists():
            return candidate
    return None


def drawtext_filter(text: str, y: int, font: Path | None) -> str:
    options = []
    if font:
        options.append(f"fontfile='{escape_drawtext(font.as_posix())}'")
    options.extend(
        [
            f"text='{escape_drawtext(text)}'",
            "x=8",
            f"y={y}",
            "fontcolor=white",
            "fontsize=14",
            "box=1",
            "boxcolor=black@0.55",
            "boxborderw=4",
        ]
    )
    return "drawtext=" + ":".join(options)


def short_text(value: str, limit: int = 42) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def build_filter(clips: list[dict], tile_width: int, tile_height: int) -> str:
    if len(clips) <= 5:
        columns = len(clips)
    elif len(clips) <= 6:
        columns = 3
    elif len(clips) <= 10:
        columns = 4
    elif len(clips) <= 15:
        columns = 5
    else:
        columns = min(4, max(1, math.ceil(math.sqrt(len(clips)))))
    font = font_file()
    filters = []
    for index, clip in enumerate(clips):
        clip_path = resolve(clip["clip"])
        line1 = short_text(clip["version"], 40)
        line2 = short_text(
            f"{clip['method']} | train {clip['train_range_success']} held {clip['heldout_success']} | {clip_result(clip_path)}",
            58,
        )
        filters.append(
            f"[{index}:v]"
            f"scale={tile_width}:{tile_height}:force_original_aspect_ratio=decrease,"
            f"pad={tile_width}:{tile_height}:(ow-iw)/2:(oh-ih)/2:black,"
            "setsar=1,"
            f"{drawtext_filter(line1, 8, font)},"
            f"{drawtext_filter(line2, 32, font)}"
            f"[v{index}]"
        )
    layout = "|".join(f"{(index % columns) * tile_width}_{(index // columns) * tile_height}" for index in range(len(clips)))
    inputs = "".join(f"[v{index}]" for index in range(len(clips)))
    return ";".join(filters) + f";{inputs}xstack=inputs={len(clips)}:layout={layout}:fill=black[out]"


def build_showcase(name: str, clips: list[dict], args: argparse.Namespace) -> dict:
    output = args.output_dir / PRESET_OUTPUTS[name]
    output.parent.mkdir(parents=True, exist_ok=True)

    for clip in clips:
        clip_path = resolve(clip["clip"])
        if not clip_path.exists():
            raise FileNotFoundError(clip_path)
        ffprobe_video(clip_path)

    command = [ffmpeg_path(), "-y", "-loglevel", "error"]
    for clip in clips:
        command.extend(["-stream_loop", "-1", "-t", f"{args.duration:.3f}", "-i", str(resolve(clip["clip"]))])
    command.extend(
        [
            "-filter_complex",
            build_filter(clips, args.tile_width, args.tile_height),
            "-map",
            "[out]",
            "-an",
            "-r",
            str(args.fps),
            "-t",
            f"{args.duration:.3f}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
    )
    subprocess.run(command, check=True)
    stream = ffprobe_video(output)
    return {
        "preset": name,
        "output": str(output.relative_to(ROOT)),
        "clips": [clip["version"] for clip in clips],
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "duration": float(stream.get("duration", 0.0)),
    }


def select_presets(args: argparse.Namespace, methods: list[dict]) -> dict[str, list[dict]]:
    by_version = {method["version"]: method for method in methods}
    presets = {
        "core": [by_version[version] for version in CORE_VERSIONS],
        "registered": methods,
        "language": LANGUAGE_CLIPS,
    }
    if args.preset == "all":
        return presets
    return {args.preset: presets[args.preset]}


def powershell_command() -> str:
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    script = ROOT / "scripts" / "build_video_showcase.py"
    return f'& "{python}" "{script}" --preset all --duration 12 --tile-width 320 --tile-height 240 --fps 24'


def write_doc(path: Path, manifest: dict) -> None:
    rows = []
    for item in manifest["showcases"]:
        rows.append(
            "| "
            f"`{item['preset']}` | "
            f"`{item['output']}` | "
            f"{item['width']}x{item['height']} | "
            f"{item['duration']:.1f}s | "
            f"{len(item['clips'])} |"
        )

    lines = [
        "# 视频展示素材索引",
        "",
        "版本：`video_showcase_v1`",
        "",
        "用途：把已经生成的 MuJoCo rollout 片段整理成答辩和阶段汇报可直接播放的对比视频。该脚本不重新运行仿真，只读取 `outputs/videos` 中已有的 mp4 和 JSON 元数据。",
        "",
        "## 已生成的总览视频",
        "",
        "| 预设 | 输出文件 | 分辨率 | 时长 | 包含片段数 |",
        "| --- | --- | ---: | ---: | ---: |",
        *rows,
        "",
        "## 推荐使用方式",
        "",
        "1. `core_methods_grid.mp4`：用于快速解释任务、示范 replay、弱 BC、记忆型 kNN、MLP 和 PyTorch ACT 的差异。",
        "2. `all_registered_methods_grid.mp4`：用于总览当前登记的正式方法版本。",
        "3. `language_generalization_grid.mp4`：用于展示语言/空间泛化任务中 expert 与 action-head 代理方法的差距。",
        "4. `outputs/presentation_clips/00_defense_video_reel.mp4`：用于答辩时按阶段快速播放完整实验链路；详细索引见 `docs/presentation_video_pack.md`。",
        "",
        "## 重新生成命令",
        "",
        "```powershell",
        powershell_command(),
        "```",
        "",
        "## 片段来源",
        "",
        "正式单任务片段来自 `docs/experiment_versions.json` 的 `clip` 字段；语言任务额外片段来自 `outputs/videos/*language_seed*.mp4`。",
        "",
        "## 注意",
        "",
        "这些总览视频只用于展示和讲解，不替代 `docs/evaluation_summary.csv`、`docs/language_generalization_summary.csv` 和 `docs/data_efficiency_summary.csv` 中的量化评测结果。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    methods = read_versions(args.versions)
    presets = select_presets(args, methods)
    showcases = [build_showcase(name, clips, args) for name, clips in presets.items()]
    manifest = {
        "version": "video_showcase_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "showcases": showcases,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_doc(args.doc, manifest)
    print(f"manifest_path: {MANIFEST_PATH}", flush=True)
    print(f"doc_path: {args.doc}", flush=True)
    for item in showcases:
        print(f"showcase: {item['preset']} -> {item['output']}", flush=True)


if __name__ == "__main__":
    main()
