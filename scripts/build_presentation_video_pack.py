from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "presentation_clips"
DOC_PATH = ROOT / "docs" / "presentation_video_pack.md"
MANIFEST_PATH = OUTPUT_DIR / "presentation_video_pack_manifest.json"
CANDIDATE_DIAGNOSTIC_CSV = ROOT / "docs" / "candidate_diagnostic_video_index.csv"


LANGUAGE_CLIPS = {
    "expert_scripted_language_v1": {
        "method": "Expert language oracle",
        "stage": "language_oracle",
        "train_range_success": "n/a",
        "heldout_success": "4/5",
        "clip": "outputs/videos/expert_scripted_language_v1_seed200.mp4",
    },
    "structured_waypoint_policy_v1_language_eval": {
        "method": "Structured waypoint language",
        "stage": "structured_control_baseline",
        "train_range_success": "n/a",
        "heldout_success": "4/5",
        "clip": "outputs/videos/structured_waypoint_policy_v1_language_seed200.mp4",
    },
    "object_language_action_head_lite_v1_language_eval": {
        "method": "Object-language head language",
        "stage": "vla_action_head_proxy",
        "train_range_success": "n/a",
        "heldout_success": "0/5",
        "clip": "outputs/videos/object_language_action_head_lite_v1_language_seed200.mp4",
    },
    "clip_action_head_lite_v1_language_eval": {
        "method": "Frozen CLIP head language",
        "stage": "pretrained_vlm_action_head_proxy",
        "train_range_success": "n/a",
        "heldout_success": "0/5",
        "clip": "outputs/videos/clip_action_head_lite_v1_language_seed200.mp4",
    },
    "vision_language_action_head_lite_v1_language_eval": {
        "method": "Vision-language head language",
        "stage": "vla_action_head_proxy",
        "train_range_success": "n/a",
        "heldout_success": "0/5",
        "clip": "outputs/videos/vision_language_action_head_lite_v1_language_seed200.mp4",
    },
    "multi_task_object_action_head_lite_v1_language_eval": {
        "method": "Multi-task head language",
        "stage": "multi_task_action_head_proxy",
        "train_range_success": "n/a",
        "heldout_success": "0/5",
        "clip": "outputs/videos/multi_task_object_action_head_lite_v1_language_seed400.mp4",
    },
}


DOMAIN_RANDOMIZATION_CLIPS = {
    "domain_randomization_structured_low_friction_v1": {
        "method": "Structured waypoint low-friction",
        "stage": "domain_randomization_proxy",
        "train_range_success": "2/2",
        "heldout_success": "MuJoCo proxy",
        "clip": "outputs/videos/domain_randomization_structured_low_friction_seed0.mp4",
    },
    "domain_randomization_trajectory_knn_low_friction_v1": {
        "method": "Trajectory-kNN low-friction",
        "stage": "domain_randomization_proxy",
        "train_range_success": "1/2",
        "heldout_success": "MuJoCo proxy",
        "clip": "outputs/videos/domain_randomization_trajectory_knn_low_friction_seed0.mp4",
    },
    "domain_randomization_visual_act_cnn_cvae_low_friction_v1": {
        "method": "Visual ACT-CNN-CVAE low-friction",
        "stage": "domain_randomization_proxy",
        "train_range_success": "0/2",
        "heldout_success": "MuJoCo proxy",
        "clip": "outputs/videos/domain_randomization_visual_act_cnn_cvae_low_friction_seed0.mp4",
    },
}


PACKS = [
    {
        "key": "01_task_data_oracle",
        "title": "阶段 1：任务、示范与结构化上界",
        "purpose": "证明 MuJoCo WidowX 桌面任务、示范 replay 和结构化阶段控制链路可用。",
        "versions": ["expert_scripted_v1", "replay_demo_v1", "structured_waypoint_policy_v1"],
    },
    {
        "key": "02_basic_bc_baselines",
        "title": "阶段 2：普通 BC baseline",
        "purpose": "展示 Linear BC、kNN BC、MLP BC 在训练范围和泛化上的差异。",
        "versions": ["linear_bc_v1", "knn_bc_v1", "mlp_bc_v1"],
    },
    {
        "key": "03_trajectory_act_diffusion",
        "title": "阶段 3：Trajectory / ACT / Diffusion",
        "purpose": "展示动作块、历史轨迹、ACT-style、CVAE 和 diffusion action-chunk baseline。",
        "versions": [
            "trajectory_conditioned_chunk_bc_v2",
            "trajectory_knn_chunk_bc_v1",
            "torch_act_state_chunk_cuda_v1",
            "torch_act_cvae_state_chunk_v1",
            "torch_diffusion_policy_state_chunk_v1",
            "visual_feature_act_lite_v1",
        ],
    },
    {
        "key": "04_action_head_peft_proxy",
        "title": "阶段 4：VLA action-head / PEFT proxy",
        "purpose": "展示 object/vision/language action-head、阶段条件、reward-weighted、Adapter/LoRA-style proxy。",
        "versions": [
            "object_language_action_head_lite_v1",
            "phase_conditioned_action_head_lite_v1",
            "reward_weighted_action_head_lite_v1",
            "adapter_action_head_lite_v1",
            "lora_action_head_lite_v1",
            "clip_action_head_lite_v1",
            "vision_language_action_head_lite_v1",
            "multi_task_object_action_head_lite_v1",
        ],
    },
    {
        "key": "05_language_generalization",
        "title": "阶段 5：语言 / 空间泛化",
        "purpose": "对比 language oracle、结构化策略和 learned/action-head proxy 在 leftmost-to-bowl 任务上的差距。",
        "versions": [
            "expert_scripted_language_v1",
            "structured_waypoint_policy_v1_language_eval",
            "object_language_action_head_lite_v1_language_eval",
            "clip_action_head_lite_v1_language_eval",
            "vision_language_action_head_lite_v1_language_eval",
            "multi_task_object_action_head_lite_v1_language_eval",
        ],
    },
    {
        "key": "06_domain_randomization_proxy",
        "title": "阶段 6：MuJoCo Domain Randomization 代理",
        "purpose": "展示低摩擦、弱夹爪扰动下结构化强对照、trajectory-kNN 和 Visual ACT-CNN-CVAE-lite 的鲁棒性差异。",
        "versions": [
            "domain_randomization_structured_low_friction_v1",
            "domain_randomization_trajectory_knn_low_friction_v1",
            "domain_randomization_visual_act_cnn_cvae_low_friction_v1",
        ],
    },
    {
        "key": "07_candidate_diagnostics",
        "title": "阶段 7：候选诊断与失败模式",
        "purpose": "集中展示 trajectory/ACT/contact-stage/preference 后训练候选的成功指标、抓取失败和论文边界。",
        "candidate_source": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build presentation-ready staged MuJoCo video clips.")
    parser.add_argument("--versions", type=Path, default=ROOT / "docs" / "experiment_versions.json")
    parser.add_argument("--candidate-diagnostic-csv", type=Path, default=CANDIDATE_DIAGNOSTIC_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--doc", type=Path, default=DOC_PATH)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--no-master", action="store_true", help="Only build per-stage videos, not the concatenated reel.")
    return parser.parse_args()


def read_methods(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return {method["version"]: method for method in data["methods"]}


def read_candidate_methods(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    candidates: dict[str, dict] = {}
    for row in rows:
        version = row["版本"]
        candidates[version] = {
            "version": version,
            "method": row["方法定位"],
            "stage": "candidate_method_diagnosis",
            "train_range_success": "candidate",
            "heldout_success": row["结果"],
            "clip": row["视频文件"],
        }
    return candidates


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
        "stream=codec_name,width,height,duration,nb_frames",
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
    metadata = clip_path.with_suffix(".json")
    if not metadata.exists():
        return {}
    return json.loads(metadata.read_text(encoding="utf-8-sig"))


def clip_result(clip_path: Path, method: dict) -> str:
    summary = metadata_for(clip_path).get("summary", {})
    if "success" in summary:
        return f"result success={summary['success']}"
    if "steps_replayed" in summary:
        return f"replay steps={summary['steps_replayed']}"
    train = method.get("train_range_success", "n/a")
    held = method.get("heldout_success", "n/a")
    return f"train {train} held {held}"


def font_file() -> Path | None:
    for candidate in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ):
        if candidate.exists():
            return candidate
    return None


def escape_drawtext(value: str) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("%", "\\%")
        .replace("\n", " ")
    )


def drawtext(text: str, x: int, y: int, size: int, font: Path | None, box: bool = True) -> str:
    options = []
    if font:
        options.append(f"fontfile='{escape_drawtext(font.as_posix())}'")
    options.extend(
        [
            f"text='{escape_drawtext(text)}'",
            f"x={x}",
            f"y={y}",
            "fontcolor=white",
            f"fontsize={size}",
        ]
    )
    if box:
        options.extend(["box=1", "boxcolor=black@0.62", "boxborderw=6"])
    return "drawtext=" + ":".join(options)


def short(value: str, limit: int) -> str:
    value = str(value)
    return value if len(value) <= limit else value[: limit - 3] + "..."


def layout_for(count: int, width: int, height: int) -> tuple[int, int, int, int]:
    if count <= 4:
        columns = 2
    elif count <= 6:
        columns = 3
    else:
        columns = 4
    rows = math.ceil(count / columns)
    return columns, rows, width // columns, height // rows


def build_stage_filter(pack: dict, clips: list[dict], width: int, height: int) -> str:
    columns, _rows, tile_w, tile_h = layout_for(len(clips), width, height)
    font = font_file()
    filters = []
    for index, clip in enumerate(clips):
        method = clip["method"]
        clip_path = resolve(method["clip"])
        line1 = short(method["version"], 42)
        line2 = short(
            f"{method['method']} | train {method.get('train_range_success', 'n/a')} held {method.get('heldout_success', 'n/a')}",
            62,
        )
        line3 = short(clip_result(clip_path, method), 48)
        filters.append(
            f"[{index}:v]"
            f"scale={tile_w}:{tile_h}:force_original_aspect_ratio=decrease,"
            f"pad={tile_w}:{tile_h}:(ow-iw)/2:(oh-ih)/2:black,"
            "setsar=1,"
            f"{drawtext(line1, 8, 54, 16, font)},"
            f"{drawtext(line2, 8, 82, 14, font)},"
            f"{drawtext(line3, 8, 108, 14, font)}"
            f"[v{index}]"
        )
    inputs = "".join(f"[v{index}]" for index in range(len(clips)))
    layout = "|".join(f"{(index % columns) * tile_w}_{(index // columns) * tile_h}" for index in range(len(clips)))
    title = drawtext(pack["title"], 18, 14, 28, font)
    purpose = drawtext(short(pack["purpose"], 70), 18, height - 42, 18, font)
    return (
        ";".join(filters)
        + f";{inputs}xstack=inputs={len(clips)}:layout={layout}:fill=black,"
        + f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
        + f"{title},{purpose}[out]"
    )


def clip_for(version: str, methods: dict[str, dict], candidates: dict[str, dict]) -> dict:
    if version in methods:
        return {**methods[version]}
    if version in candidates:
        return {**candidates[version]}
    if version in LANGUAGE_CLIPS:
        item = LANGUAGE_CLIPS[version]
        return {"version": version, **item}
    if version in DOMAIN_RANDOMIZATION_CLIPS:
        item = DOMAIN_RANDOMIZATION_CLIPS[version]
        return {"version": version, **item}
    raise KeyError(version)


def versions_for_pack(pack: dict, candidates: dict[str, dict]) -> list[str]:
    if pack.get("candidate_source"):
        return list(candidates)
    return list(pack["versions"])


def build_stage_video(pack: dict, methods: dict[str, dict], candidates: dict[str, dict], args: argparse.Namespace) -> dict:
    output = args.output_dir / f"{pack['key']}.mp4"
    versions = versions_for_pack(pack, candidates)
    clips = [{"version": version, "method": clip_for(version, methods, candidates)} for version in versions]
    for clip in clips:
        clip_path = resolve(clip["method"]["clip"])
        if not clip_path.exists():
            raise FileNotFoundError(clip_path)
        ffprobe_video(clip_path)

    command = [ffmpeg_path(), "-y", "-loglevel", "error"]
    for clip in clips:
        command.extend(["-stream_loop", "-1", "-t", f"{args.duration:.3f}", "-i", str(resolve(clip["method"]["clip"]))])
    command.extend(
        [
            "-filter_complex",
            build_stage_filter(pack, clips, args.width, args.height),
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
        "key": pack["key"],
        "title": pack["title"],
        "purpose": pack["purpose"],
        "output": str(output.relative_to(ROOT)),
        "clips": [clip["version"] for clip in clips],
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "duration": float(stream.get("duration", 0.0)),
    }


def build_master(stage_items: list[dict], args: argparse.Namespace) -> dict:
    output = args.output_dir / "00_defense_video_reel.mp4"
    command = [ffmpeg_path(), "-y", "-loglevel", "error"]
    for item in stage_items:
        command.extend(["-i", str(resolve(item["output"]))])
    concat_inputs = "".join(f"[{index}:v]" for index in range(len(stage_items)))
    command.extend(
        [
            "-filter_complex",
            f"{concat_inputs}concat=n={len(stage_items)}:v=1:a=0[out]",
            "-map",
            "[out]",
            "-an",
            "-r",
            str(args.fps),
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
        "key": "00_defense_video_reel",
        "title": "答辩阶段总览视频",
        "output": str(output.relative_to(ROOT)),
        "clips": [item["key"] for item in stage_items],
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "duration": float(stream.get("duration", 0.0)),
    }


def powershell_command(args: argparse.Namespace) -> str:
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    script = ROOT / "scripts" / "build_presentation_video_pack.py"
    return (
        f'& "{python}" "{script}" --duration {args.duration:g} '
        f"--width {args.width} --height {args.height} --fps {args.fps}"
    )


def write_doc(path: Path, manifest: dict, args: argparse.Namespace) -> None:
    master_duration = (manifest.get("master") or {}).get("duration", args.duration * len(manifest["stages"]))
    lines = [
        "# 答辩视频片段包",
        "",
        "版本：`presentation_video_pack_v1`",
        "",
        "用途：把已有 MuJoCo rollout 视频按毕业设计研究阶段剪成可直接播放的短片段。该脚本只读取 `outputs/videos` 中的固定视频，不重新运行 MuJoCo。",
        "",
        "## 输出文件",
        "",
        "| 类型 | 输出文件 | 分辨率 | 时长 | 内容 |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    master = manifest.get("master")
    if master:
        lines.append(
            f"| 总览 reel | `{master['output']}` | {master['width']}x{master['height']} | {master['duration']:.1f}s | 串联全部阶段短片 |"
        )
    for item in manifest["stages"]:
        lines.append(
            f"| 阶段短片 | `{item['output']}` | {item['width']}x{item['height']} | {item['duration']:.1f}s | {item['title']} |"
        )

    lines.extend(
        [
            "",
            "## 阶段说明",
            "",
        ]
    )
    for item in manifest["stages"]:
        lines.extend(
            [
                f"### {item['title']}",
                "",
                item["purpose"],
                "",
                "包含版本：",
                "",
            ]
        )
        for version in item["clips"]:
            lines.append(f"- `{version}`")
        if item["key"] == "07_candidate_diagnostics":
            lines.extend(
                [
                    "",
                    "候选来源：`docs/candidate_diagnostic_video_index.csv`，对应版本 `candidate_diagnostic_video_index_v1`。",
                ]
            )
        lines.append("")

    lines.extend(
        [
            "## 推荐播放顺序",
            "",
            f"1. 先播放 `00_defense_video_reel.mp4`，用 {master_duration:.0f} 秒左右快速说明完整实验链路。",
            f"2. 论文答辩中如需展开某一阶段，再单独播放对应 `01` 到 `{len(manifest['stages']):02d}` 的阶段短片。",
            "3. 若需要更完整的全量方法并排展示，使用 `outputs/showcase/all_registered_methods_grid.mp4`；若需要单个方法细看，回到 `outputs/videos/*.mp4`。",
            "",
            "## 讲稿与时间线",
            "",
            "```text",
            "docs\\video_presentation_storyboard.md",
            "docs\\video_presentation_storyboard.html",
            "```",
            "",
            f"说明：`video_presentation_storyboard_v1` 给出 {master_duration:.0f} 秒总览 reel 的逐段时间线、每个阶段短片的讲稿提示、量化证据引用和论文红线。",
            "",
            "## 重新生成命令",
            "",
            "```powershell",
            powershell_command(args),
            f'& "{ROOT / ".venv" / "Scripts" / "python.exe"}" "{ROOT / "scripts" / "build_video_presentation_storyboard.py"}"',
            "```",
            "",
            "## 注意",
            "",
            "这些短片只作为可视化证据和讲解材料，量化结论仍以 `docs/evaluation_summary.csv`、`docs/language_generalization_summary.csv`、`docs/domain_randomization_summary.csv`、`docs/model_resource_summary.csv` 和 `docs/method_stage_audit.md` 为准。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = read_methods(args.versions)
    candidates = read_candidate_methods(args.candidate_diagnostic_csv)
    stages = [build_stage_video(pack, methods, candidates, args) for pack in PACKS]
    master = None if args.no_master else build_master(stages, args)
    manifest = {
        "version": "presentation_video_pack_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds_per_stage": args.duration,
        "stages": stages,
        "master": master,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_doc(args.doc, manifest, args)
    print(f"presentation_manifest: {MANIFEST_PATH}", flush=True)
    print(f"presentation_doc: {args.doc}", flush=True)
    if master:
        print(f"master_reel: {master['output']}", flush=True)
    for item in stages:
        print(f"stage_clip: {item['key']} -> {item['output']}", flush=True)


if __name__ == "__main__":
    main()
