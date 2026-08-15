from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from widowx_env import TASKS, WidowXTabletopEnv  # noqa: E402
from widowx_env.vision_grounding import (  # noqa: E402
    draw_detection_overlay,
    load_calibration,
    locate_leftmost_cube,
    locate_object,
    write_ppm,
)


TASKS_TO_EVALUATE = (
    "place_blue_cube_blue_pad",
    "place_blue_cube_red_pad",
    "place_red_cube_red_pad",
    "move_leftmost_cube_to_bowl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit RGB-only object localization against MuJoCo state used only as evaluation truth.")
    parser.add_argument("--calibration", type=Path, default=ROOT / "outputs" / "rgb_grounding" / "top_rgb_core_v2_calibration_v1.json")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / "rgb_grounding_core_v2_v1.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "rgb_grounding_core_v2_v1.csv")
    parser.add_argument("--overlay-dir", type=Path, default=ROOT / "outputs" / "rgb_grounding" / "overlays")
    parser.add_argument("--tasks", nargs="+", choices=TASKS_TO_EVALUATE, default=TASKS_TO_EVALUATE)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--camera", default="top_rgb")
    return parser.parse_args()


def resolve_visual_source(image: np.ndarray, calibration, task: str) -> tuple[str, np.ndarray, object]:
    if task == "move_leftmost_cube_to_bowl":
        return locate_leftmost_cube(image, calibration)
    target_object = TASKS[task].target_object
    assert target_object is not None
    position, detection = locate_object(image, calibration, target_object)
    return target_object, position, detection


def main() -> None:
    args = parse_args()
    calibration = load_calibration(args.calibration)
    rows: list[dict] = []
    renderer = None
    for task_index, task in enumerate(args.tasks):
        complexity = "language" if task == "move_leftmost_cube_to_bowl" else "medium"
        for offset in range(args.episodes):
            seed = args.seed + task_index * 100 + offset
            env = WidowXTabletopEnv(
                seed=seed,
                image_size=(args.image_size, args.image_size),
                camera=args.camera,
                workspace_profile="core_v2",
            )
            obs = env.reset(task=task, complexity=complexity, seed=seed)
            renderer = mujoco.Renderer(env.model, height=args.image_size, width=args.image_size)
            try:
                renderer.update_scene(env.data, camera=args.camera)
                image = renderer.render().copy()
                selected_name, visual_position, detection = resolve_visual_source(image, calibration, task)
                oracle_name = str(obs["target_object"])
                oracle_position = env.object_position(oracle_name).copy()
                error_m = float(np.linalg.norm(visual_position[:2] - oracle_position[:2]))
                selected_correct = selected_name == oracle_name
                write_ppm(args.overlay_dir / f"{task}_seed{seed}.ppm", draw_detection_overlay(image, [detection]))
                row = {
                    "task": task,
                    "seed": seed,
                    "selected_name": selected_name,
                    "oracle_name": oracle_name,
                    "selected_correct": selected_correct,
                    "detected": True,
                    "position_error_m": error_m,
                    "area_px": int(detection.area),
                    "fill_ratio": float(detection.fill_ratio),
                }
            except LookupError as error:
                row = {
                    "task": task,
                    "seed": seed,
                    "selected_name": "not_detected",
                    "oracle_name": str(obs["target_object"]),
                    "selected_correct": False,
                    "detected": False,
                    "position_error_m": None,
                    "area_px": 0,
                    "fill_ratio": 0.0,
                    "error": str(error),
                }
            finally:
                renderer.close()
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    detected_rows = [row for row in rows if row["detected"]]
    summary = {
        "version": "rgb_grounding_core_v2_v1",
        "method": "top_rgb_color_shape_localization",
        "calibration": str(args.calibration),
        "runtime_state_use": "none for source localization; MuJoCo state is queried only after inference for evaluation.",
        "episodes": len(rows),
        "detected": sum(int(row["detected"]) for row in rows),
        "selection_correct": sum(int(row["selected_correct"]) for row in rows),
        "mean_position_error_m": float(np.mean([row["position_error_m"] for row in detected_rows])) if detected_rows else None,
        "max_position_error_m": float(np.max([row["position_error_m"] for row in detected_rows])) if detected_rows else None,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary_path: {args.output_json}")
    print(f"detection_rate: {summary['detected']}/{summary['episodes']}")
    print(f"selection_rate: {summary['selection_correct']}/{summary['episodes']}")
    print(f"mean_position_error_m: {summary['mean_position_error_m']}")


if __name__ == "__main__":
    main()
