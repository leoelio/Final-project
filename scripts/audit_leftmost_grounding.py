from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from widowx_env import WidowXTabletopEnv  # noqa: E402
from widowx_env.vision_grounding import load_calibration, locate_leftmost_cube  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline audit for visually separable leftmost-cube task generation.")
    parser.add_argument("--calibration", type=Path, default=ROOT / "outputs" / "rgb_grounding" / "top_rgb_core_v2_calibration_v1.json")
    parser.add_argument("--seed", type=int, default=3200)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / "leftmost_grounding_audit_v1.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "leftmost_grounding_audit_v1.md")
    return parser.parse_args()


def render_rgb(env: WidowXTabletopEnv) -> np.ndarray:
    renderer = mujoco.Renderer(env.model, height=224, width=224)
    try:
        renderer.update_scene(env.data, camera="top_rgb")
        return renderer.render().copy()
    finally:
        renderer.close()


def main() -> None:
    args = parse_args()
    calibration = load_calibration(args.calibration)
    records: list[dict] = []
    for offset in range(args.episodes):
        seed = args.seed + offset
        env = WidowXTabletopEnv(seed=seed, image_size=(224, 224), camera="top_rgb", workspace_profile="core_v2")
        obs = env.reset(task="move_leftmost_cube_to_bowl", complexity="language", seed=seed)
        cube_positions = sorted(
            ((name, env.object_position(name)[:2].copy()) for name in obs["active_objects"] if name.endswith("_cube")),
            key=lambda item: float(item[1][0]),
        )
        image = render_rgb(env)
        selected, estimate, detection = locate_leftmost_cube(image, calibration)
        truth_name = str(obs["target_object"])
        truth = env.object_position(truth_name)[:2]
        records.append(
            {
                "seed": seed,
                "truth_name": truth_name,
                "selected_name": selected,
                "selection_correct": selected == truth_name,
                "minimum_cube_x_gap_m": float(cube_positions[1][1][0] - cube_positions[0][1][0]),
                "position_error_m": float(np.linalg.norm(estimate[:2] - truth)),
                "detection_area_px": int(detection.area),
                "offline_cube_x": {name: float(position[0]) for name, position in cube_positions},
            }
        )
    summary = {
        "version": "leftmost_grounding_audit_v1",
        "seed_range": f"{args.seed}-{args.seed + args.episodes - 1}",
        "episodes": len(records),
        "selection_correct": sum(record["selection_correct"] for record in records),
        "minimum_observed_x_gap_m": float(min(record["minimum_cube_x_gap_m"] for record in records)),
        "mean_position_error_m": float(np.mean([record["position_error_m"] for record in records])),
        "max_position_error_m": float(max(record["position_error_m"] for record in records)),
        "records": records,
        "runtime_boundary": "The task policy uses RGB, calibration, and the leftmost-cube rule only. Object identities and x gaps are offline audit labels only.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 最左方块视觉选择审计",
        "",
        f"任务 `move_leftmost_cube_to_bowl`，seed `{summary['seed_range']}`。环境生成保证前两名方块的真实 x 间隔至少为 0.03 m；真值只用于离线核验。",
        "",
        "| 指标 | 结果 |",
        "| --- | ---: |",
        f"| 视觉选择正确 | {summary['selection_correct']}/{summary['episodes']} |",
        f"| 最小真实 x 间隔 | {summary['minimum_observed_x_gap_m']:.4f} m |",
        f"| 平均定位误差 | {summary['mean_position_error_m']:.4f} m |",
        f"| 最大定位误差 | {summary['max_position_error_m']:.4f} m |",
    ]
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
