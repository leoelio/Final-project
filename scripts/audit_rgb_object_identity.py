from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from widowx_env import WidowXTabletopEnv  # noqa: E402
from widowx_env.vision_grounding import (  # noqa: E402
    color_mask,
    connected_components,
    load_calibration,
    source_workspace_mask,
    static_target_exclusion_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline RGB identity audit for same-color cube/cylinder distractors.")
    parser.add_argument("--task", default="place_blue_cube_red_pad")
    parser.add_argument("--complexity", default="medium")
    parser.add_argument("--source-object", default="blue_cube")
    parser.add_argument("--same-color-objects", default=None, help="Comma-separated truth labels used only by the offline audit.")
    parser.add_argument("--seed", type=int, default=3400)
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--calibration", type=Path, default=ROOT / "outputs" / "rgb_grounding" / "top_rgb_core_v2_calibration_v1.json")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / "rgb_object_identity_audit_v1.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "rgb_object_identity_audit_v1.md")
    return parser.parse_args()


def render_top_rgb(env: WidowXTabletopEnv) -> np.ndarray:
    renderer = mujoco.Renderer(env.model, height=224, width=224)
    try:
        renderer.update_scene(env.data, camera="top_rgb")
        return renderer.render().copy()
    finally:
        renderer.close()


def select(candidates: list[dict], rule: str) -> dict | None:
    if rule == "current":
        eligible = [row for row in candidates if row["fill_ratio"] >= 0.75]
        return max(eligible, key=lambda row: 3.0 * row["fill_ratio"] + 0.001 * row["area"], default=None)
    if rule == "largest_area":
        eligible = [row for row in candidates if row["fill_ratio"] >= 0.70]
        return max(eligible, key=lambda row: row["area"], default=None)
    raise KeyError(rule)


def feature_range(rows: list[dict], field: str) -> list[float] | None:
    if not rows:
        return None
    return [round(float(min(row[field] for row in rows)), 3), round(float(max(row[field] for row in rows)), 3)]


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    color = args.source_object.split("_", 1)[0]
    calibration = load_calibration(args.calibration)
    component_rows: list[dict] = []
    scene_rows: list[dict] = []
    same_color_objects = (
        [name.strip() for name in args.same_color_objects.split(",") if name.strip()]
        if args.same_color_objects
        else [f"{color}_cube", f"{color}_cylinder"]
    )
    if args.source_object not in same_color_objects:
        raise ValueError("source object must be included in --same-color-objects")
    for offset in range(args.episodes):
        seed = args.seed + offset
        env = WidowXTabletopEnv(seed=seed, image_size=(224, 224), camera="top_rgb", workspace_profile="core_v2")
        env.reset(task=args.task, complexity=args.complexity, seed=seed)
        image = render_top_rgb(env)
        mask = color_mask(image, color)
        mask &= source_workspace_mask(image.shape[:2], calibration)
        mask &= ~static_target_exclusion_mask(image.shape[:2], calibration, color)
        source_xy = env.object_position(args.source_object)[:2].copy()
        candidates: list[dict] = []
        for region in connected_components(mask):
            if not 20 <= region.area <= 650:
                continue
            position = calibration.pixel_to_world(region.center_uv)[:2]
            distances = {name: float(np.linalg.norm(position - env.object_position(name)[:2])) for name in same_color_objects}
            nearest = min(distances, key=distances.get)
            row = {
                "seed": seed,
                "area": int(region.area),
                "fill_ratio": float(region.fill_ratio),
                "bbox_width": int(region.bbox[2] - region.bbox[0] + 1),
                "bbox_height": int(region.bbox[3] - region.bbox[1] + 1),
                "aspect_ratio": float((region.bbox[2] - region.bbox[0] + 1) / (region.bbox[3] - region.bbox[1] + 1)),
                "position_xy": position.round(5).tolist(),
                "offline_nearest_object": nearest,
                "offline_source_error_m": float(np.linalg.norm(position - source_xy)),
            }
            candidates.append(row)
            component_rows.append(row)
        selected = {rule: select(candidates, rule) for rule in ("current", "largest_area")}
        scene_rows.append(
            {
                "seed": seed,
                "source_truth_xy": source_xy.round(5).tolist(),
                "candidate_count": len(candidates),
                "rules": {
                    rule: None
                    if row is None
                    else {
                        "offline_nearest_object": row["offline_nearest_object"],
                        "offline_source_error_m": row["offline_source_error_m"],
                        "correct_within_4cm": bool(row["offline_source_error_m"] <= 0.04),
                    }
                    for rule, row in selected.items()
                },
            }
        )
    rule_summary = {}
    for rule in ("current", "largest_area"):
        selected_rows = [row["rules"][rule] for row in scene_rows if row["rules"][rule] is not None]
        rule_summary[rule] = {
            "detected": len(selected_rows),
            "correct_within_4cm": sum(bool(row["correct_within_4cm"]) for row in selected_rows),
            "mean_source_error_m": float(np.mean([row["offline_source_error_m"] for row in selected_rows])) if selected_rows else None,
        }
    grouped = {name: [row for row in component_rows if row["offline_nearest_object"] == name] for name in same_color_objects}
    payload = {
        "version": "rgb_object_identity_audit_v1",
        "task": args.task,
        "source_object": args.source_object,
        "seed_range": f"{args.seed}-{args.seed + args.episodes - 1}",
        "episodes": args.episodes,
        "rule_summary": rule_summary,
        "component_feature_ranges": {
            name: {field: feature_range(rows, field) for field in ("area", "fill_ratio", "aspect_ratio")}
            for name, rows in grouped.items()
        },
        "component_label_counts": dict(Counter(row["offline_nearest_object"] for row in component_rows)),
        "scene_rows": scene_rows,
        "runtime_boundary": "MuJoCo object positions are used only to label RGB components and score rules offline. The runtime detector uses RGB, calibration, static target layout, and deterministic candidate rules only.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# RGB 同色异形对象身份审计",
        "",
        f"任务 `{args.task}`，目标对象 `{args.source_object}`，seed `{payload['seed_range']}`。MuJoCo 真值只用于离线标注和评分。",
        "",
        "| 规则 | 检出 | 源方块 4 cm 内正确 | 平均源点误差 |",
        "| --- | ---: | ---: | ---: |",
        *[
            f"| `{rule}` | {item['detected']}/{args.episodes} | {item['correct_within_4cm']}/{args.episodes} | {item['mean_source_error_m']:.4f} m |"
            for rule, item in rule_summary.items()
        ],
        "",
        "| 离线真值标签 | 连通域数 | 面积范围 px | 填充率范围 | 长宽比范围 |",
        "| --- | ---: | --- | --- | --- |",
        *[
            f"| `{name}` | {payload['component_label_counts'].get(name, 0)} | {payload['component_feature_ranges'][name]['area']} | "
            f"{payload['component_feature_ranges'][name]['fill_ratio']} | {payload['component_feature_ranges'][name]['aspect_ratio']} |"
            for name in same_color_objects
        ],
        "",
        "该审计用于选定下一版运行时规则，不应被表述为运行时使用对象真值。",
    ]
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key not in {"scene_rows"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
