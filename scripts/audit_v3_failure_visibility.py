from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_rgb_frontend_end_to_end import TASKS  # noqa: E402
from evaluate_rgb_recovery_profiles import DOMAINS, rollout_args  # noqa: E402
from run_clip_action_head import load_clip  # noqa: E402
from run_clip_semantic_rgb_feedback import configure_env, render_top_rgb, rollout  # noqa: E402
from run_clip_semantic_waypoint import load_policy  # noqa: E402
from widowx_env.vision_grounding import (  # noqa: E402
    CUBE_MAX_AREA,
    CUBE_MIN_AREA,
    RECOVERY_CUBE_MIN_FILL_RATIO,
    color_mask,
    connected_components,
    load_calibration,
    source_workspace_mask,
    static_target_exclusion_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline visibility audit for final V3 failures after their first attempt.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--input-json",
        type=Path,
        default=ROOT / "outputs" / "evaluations" / "rgb_occlusion_recovery_v3_extended_v1.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "outputs" / "evaluations" / "rgb_occlusion_recovery_v3_failure_visibility_audit_v1.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=ROOT / "docs" / "rgb_occlusion_recovery_v3_failure_visibility_audit_v1.md",
    )
    parser.add_argument("--audit-version", default="rgb_occlusion_recovery_v3_failure_visibility_audit_v1")
    parser.add_argument("--stage-label", default="V3")
    return parser.parse_args()


def in_table(world_xy: np.ndarray) -> bool:
    return bool(0.18 <= world_xy[0] <= 0.62 and -0.28 <= world_xy[1] <= 0.28)


def main() -> None:
    args = parse_args()
    evaluation = json.loads(args.input_json.read_text(encoding="utf-8"))
    failures = [row for row in evaluation["rows"] if not row["task_success"]]
    policy = load_policy(args.model)
    calibration = load_calibration(ROOT / "outputs" / "rgb_grounding" / "top_rgb_core_v2_calibration_v1.json")
    clip_model, processor = load_clip(str(policy["metadata"]["clip_model"]))
    rows = []
    for index, original in enumerate(failures, start=1):
        config = rollout_args(original["task"], TASKS[original["task"]], "standard", DOMAINS[original["domain"]])
        config.feedback_attempts = 0
        env, obs = configure_env(config, int(original["seed"]))
        result = rollout(config, policy, clip_model, processor, calibration, int(original["seed"]), env=env, obs=obs)
        image = render_top_rgb(env, config.image_size, config.camera)
        color = result["selected_object"].split("_", 1)[0]
        mask = color_mask(image, color) & ~static_target_exclusion_mask(image.shape[:2], calibration, color)
        source_mask = source_workspace_mask(image.shape[:2], calibration)
        truth_xy = env.object_position(obs["target_object"])[:2]
        candidates = []
        for region in connected_components(mask):
            world_xy = calibration.pixel_to_world(region.center_uv)[:2]
            error = float(np.linalg.norm(world_xy - truth_xy))
            valid_shape = CUBE_MIN_AREA <= region.area <= CUBE_MAX_AREA and region.fill_ratio >= RECOVERY_CUBE_MIN_FILL_RATIO
            candidates.append(
                {
                    "area_px": int(region.area),
                    "fill_ratio": round(float(region.fill_ratio), 4),
                    "world_xy": world_xy.round(5).tolist(),
                    "truth_error_m": error,
                    "valid_recovery_shape": valid_shape,
                    "in_source_workspace": bool(source_mask[int(round(region.center_uv[1])), int(round(region.center_uv[0]))]),
                    "in_table": in_table(world_xy),
                }
            )
        correct = [item for item in candidates if item["valid_recovery_shape"] and item["truth_error_m"] <= 0.04]
        source_correct = [item for item in correct if item["in_source_workspace"]]
        table_correct = [item for item in correct if item["in_table"]]
        rows.append(
            {
                "domain": original["domain"],
                "task": original["task"],
                "seed": original["seed"],
                "original_recovery_reason": original["recovery_reason"],
                "first_replay_target_distance_m": result["target_distance"],
                "original_final_target_distance_m": original["target_distance_m"],
                "correct_candidate_in_source_workspace": bool(source_correct),
                "correct_candidate_in_broader_table": bool(table_correct),
                "correct_candidate_outside_source_workspace": bool(table_correct and not source_correct),
                "truth_xy": truth_xy.round(5).tolist(),
                "candidates": candidates,
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    summary = {
        "version": args.audit_version,
        "input_version": evaluation["version"],
        "episodes": len(rows),
        "correct_candidate_in_source_workspace": sum(row["correct_candidate_in_source_workspace"] for row in rows),
        "correct_candidate_in_broader_table": sum(row["correct_candidate_in_broader_table"] for row in rows),
        "correct_candidate_outside_source_workspace": sum(row["correct_candidate_outside_source_workspace"] for row in rows),
        "runtime_boundary": "This is an offline diagnostic. MuJoCo truth only labels whether an RGB component matches the intended cube; it is not used by the deployed policy.",
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# {args.stage_label} 最终失败后的 RGB 可见性审计",
        "",
        f"该脚本只复现 {args.stage_label} 最终失败场景的首轮动作，并在动作结束后离线检查 RGB 连通域。MuJoCo 真值只用于标签核对，不能作为运行时恢复输入。",
        "",
        "| 指标 | 结果 |",
        "| --- | ---: |",
        f"| {args.stage_label} 最终失败 | {summary['episodes']} |",
        f"| 合格方块仍在原源工作区 | {summary['correct_candidate_in_source_workspace']}/{summary['episodes']} |",
        f"| 合格方块仍在较大桌面范围 | {summary['correct_candidate_in_broader_table']}/{summary['episodes']} |",
        f"| 合格方块只在原源区外 | {summary['correct_candidate_outside_source_workspace']}/{summary['episodes']} |",
        "",
        "| 接触域 | 任务 | seed | 原终止原因 | 源区可恢复 | 桌面可见 | 仅源区外 |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['domain']}` | `{row['task']}` | {row['seed']} | `{row['original_recovery_reason']}` | "
            f"{int(row['correct_candidate_in_source_workspace'])} | {int(row['correct_candidate_in_broader_table'])} | "
            f"{int(row['correct_candidate_outside_source_workspace'])} |"
        )
    lines.extend(
        [
            "",
            "只有“合格方块在较大桌面范围可见且不在原源工作区”的比例足够高时，才值得把部署策略的恢复范围从源工作区扩展到桌面范围；否则应优先采集接触阶段示范，而非扩大视觉搜索。",
        ]
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"output_json: {args.output_json}")


if __name__ == "__main__":
    main()
