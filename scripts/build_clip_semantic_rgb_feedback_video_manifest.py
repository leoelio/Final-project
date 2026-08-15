from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "outputs" / "evaluations" / "clip_semantic_rgb_feedback_closed_loop_v1.json"
VIDEO_DIR = ROOT / "presentation_videos" / "rgb_feedback_loop_v1"
OUTPUT_JSON = VIDEO_DIR / "manifest.json"
OUTPUT_MD = ROOT / "docs" / "clip_semantic_rgb_feedback_video_evidence.md"


CASES = (
    {
        "file": "01_nominal_open_loop_spatial_failure_seed722.mp4",
        "mode": "rgb_open_loop",
        "domain": "nominal",
        "task": "move_leftmost_cube_to_bowl",
        "seed": 722,
        "purpose": "单次 RGB 定位在空间任务中的失败对照。",
    },
    {
        "file": "02_nominal_visual_retry_recovery_seed722.mp4",
        "mode": "rgb_visual_retry",
        "domain": "nominal",
        "task": "move_leftmost_cube_to_bowl",
        "seed": 722,
        "purpose": "首次未到目标后，视觉重定位一次并恢复成功。",
    },
    {
        "file": "03_mild_contact_unrecovered_seed722.mp4",
        "mode": "rgb_visual_retry",
        "domain": "mild_contact_shift",
        "task": "move_leftmost_cube_to_bowl",
        "seed": 722,
        "purpose": "物体不再位于可安全重定位的源工作区，保留为不可恢复边界。",
    },
    {
        "file": "04_low_contact_retry_recovery_seed724.mp4",
        "mode": "rgb_visual_retry",
        "domain": "low_contact_shift",
        "task": "move_leftmost_cube_to_bowl",
        "seed": 724,
        "purpose": "低接触条件下首次抓取失败后，通过 RGB 重定位恢复成功。",
    },
)


def main() -> None:
    evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    rows = {
        (item["mode"], item["domain"], item["task"], int(item["seed"])): item
        for item in evaluation["rows"]
    }
    entries = []
    for case in CASES:
        path = VIDEO_DIR / case["file"]
        metadata_path = path.with_suffix(".json")
        key = (case["mode"], case["domain"], case["task"], case["seed"])
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)
        if key not in rows:
            raise KeyError(f"evaluation row not found: {key}")
        row = rows[key]
        entries.append(
            {
                **case,
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "metadata_path": str(metadata_path.relative_to(ROOT)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "success": bool(row["success"]),
                "attempt_count": int(row["attempt_count"]),
                "recovery_triggered": bool(row["recovery_triggered"]),
                "recovery_reason": row["recovery_reason"],
                "target_distance_m": float(row["target_distance_m"]),
            }
        )
    OUTPUT_JSON.write_text(json.dumps({"version": "rgb_feedback_video_manifest_v1", "entries": entries}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# RGB 视觉反馈闭环视频证据",
        "",
        "版本：`rgb_feedback_video_manifest_v1`",
        "",
        "这四条视频按失败对照、名义域恢复、扰动边界和低接触恢复排列。它们均对应 `clip_semantic_rgb_feedback_closed_loop_v1` 的同 seed 量化记录，不再另存重复的常规成功片段。",
        "",
        "| 编号 | 作用 | 域 | 策略 | seed | 成功 | 尝试次数 | 视频 |",
        "| --- | --- | --- | --- | ---: | --- | ---: | --- |",
    ]
    for index, entry in enumerate(entries, start=1):
        lines.append(
            f"| {index} | {entry['purpose']} | `{entry['domain']}` | `{entry['mode']}` | {entry['seed']} | "
            f"{entry['success']} | {entry['attempt_count']} | `{entry['path']}` |"
        )
    lines.extend(
        [
            "",
            "每个 MP4 同目录的 JSON 保存该次执行的完整 attempt log；成功率与平均距离必须引用主评测报告，而不是由视频推断。",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"manifest_path: {OUTPUT_JSON}")
    print(f"evidence_path: {OUTPUT_MD}")
    print(f"videos: {len(entries)}")


if __name__ == "__main__":
    main()
