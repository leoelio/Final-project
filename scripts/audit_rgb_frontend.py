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
from widowx_env.vision_grounding import (  # noqa: E402
    connected_components,
    draw_detection_overlay,
    color_mask,
    load_calibration,
    locate_object,
    write_ppm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit RGB source grounding before any control rollout.")
    parser.add_argument("--calibration", type=Path, default=ROOT / "outputs" / "rgb_grounding" / "top_rgb_core_v2_calibration_v1.json")
    parser.add_argument("--seed", type=int, default=2600)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / "rgb_frontend_audit_v1.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "rgb_frontend_audit_v1.md")
    parser.add_argument("--image-dir", type=Path, default=ROOT / "outputs" / "rgb_frontend_audit" / "v1")
    return parser.parse_args()


def render_rgb(env: WidowXTabletopEnv, segmentation: bool = False) -> np.ndarray:
    renderer = mujoco.Renderer(env.model, height=224, width=224)
    try:
        if segmentation:
            renderer.enable_segmentation_rendering()
        renderer.update_scene(env.data, camera="top_rgb")
        return renderer.render().copy()
    finally:
        renderer.close()


def legacy_detection(image: np.ndarray, calibration) -> tuple[np.ndarray | None, int | None]:
    """Reproduce the pre-workspace-clip rule for an offline before/after audit."""
    candidates: list[tuple[float, np.ndarray, int]] = []
    for region in connected_components(color_mask(image, "blue")):
        world = calibration.pixel_to_world(region.center_uv)
        in_workspace = 0.21 <= world[0] <= 0.47 and -0.13 <= world[1] <= 0.13
        if in_workspace and region.fill_ratio >= 0.80 and 20 <= region.area <= 650:
            candidates.append((3.0 * region.fill_ratio - 0.001 * region.area, world, region.area))
    if not candidates:
        return None, None
    _, position, area = max(candidates, key=lambda item: item[0])
    return position, area


def segmentation_bbox(env: WidowXTabletopEnv) -> tuple[int, list[int] | None]:
    segmentation = render_rgb(env, segmentation=True)
    geom_id = env.model.geom("blue_cube_geom").id
    mask = segmentation[..., 0] == geom_id
    v, u = np.nonzero(mask)
    if not len(u):
        return 0, None
    return int(mask.sum()), [int(u.min()), int(v.min()), int(u.max()), int(v.max())]


def main() -> None:
    args = parse_args()
    calibration = load_calibration(args.calibration)
    records: list[dict] = []
    args.image_dir.mkdir(parents=True, exist_ok=True)
    for offset in range(args.episodes):
        seed = args.seed + offset
        env = WidowXTabletopEnv(seed=seed, image_size=(224, 224), camera="top_rgb", workspace_profile="core_v2")
        env.reset(task="place_blue_cube_red_pad", complexity="medium", seed=seed)
        image = render_rgb(env)
        truth = env.object_position("blue_cube")[:2]
        legacy_position, legacy_area = legacy_detection(image, calibration)
        try:
            revised_position, revised_detection = locate_object(image, calibration, "blue_cube")
            revised_error = float(np.linalg.norm(revised_position[:2] - truth))
            revised_area = int(revised_detection.area)
            revised_overlay = draw_detection_overlay(image, [revised_detection])
        except LookupError:
            revised_position = None
            revised_error = None
            revised_area = None
            revised_overlay = image
        visible_pixels, visible_bbox = segmentation_bbox(env)
        legacy_error = None if legacy_position is None else float(np.linalg.norm(legacy_position[:2] - truth))
        record = {
            "seed": seed,
            "legacy_detected": legacy_position is not None,
            "legacy_area_px": legacy_area,
            "legacy_error_m": legacy_error,
            "revised_detected": revised_position is not None,
            "revised_area_px": revised_area,
            "revised_error_m": revised_error,
            "offline_truth_xy": truth.round(5).tolist(),
            "offline_visible_blue_cube_pixels": visible_pixels,
            "offline_visible_blue_cube_bbox": visible_bbox,
        }
        if not record["legacy_detected"] or (legacy_error is not None and legacy_error > 0.04):
            stem = f"blue_cube_seed{seed}"
            write_ppm(args.image_dir / f"{stem}_top.ppm", image)
            write_ppm(args.image_dir / f"{stem}_revised_overlay.ppm", revised_overlay)
            record["saved_top_image"] = str(args.image_dir / f"{stem}_top.ppm")
            record["saved_overlay_image"] = str(args.image_dir / f"{stem}_revised_overlay.ppm")
        records.append(record)
    summary = {
        "version": "rgb_frontend_audit_v1",
        "seed_range": f"{args.seed}-{args.seed + args.episodes - 1}",
        "task": "place_blue_cube_red_pad",
        "episodes": len(records),
        "legacy_detected": sum(record["legacy_detected"] for record in records),
        "revised_detected": sum(record["revised_detected"] for record in records),
        "legacy_correct_within_4cm": sum(record["legacy_error_m"] is not None and record["legacy_error_m"] <= 0.04 for record in records),
        "revised_correct_within_4cm": sum(record["revised_error_m"] is not None and record["revised_error_m"] <= 0.04 for record in records),
        "records": records,
        "runtime_boundary": "The deployed detector uses RGB, calibration, the static source workspace, and color rules only. MuJoCo segmentation and object position are offline audit labels only.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# RGB 初始定位审计",
        "",
        f"蓝方块到红盘，seed `{summary['seed_range']}`。运行时只用 RGB 与标定；物体位置和分割只用于离线审计。",
        "",
        "| 规则 | 检出 | 位置误差不超过 4 cm |",
        "| --- | ---: | ---: |",
        f"| 修改前连通域规则 | {summary['legacy_detected']}/{summary['episodes']} | {summary['legacy_correct_within_4cm']}/{summary['episodes']} |",
        f"| 工作区裁剪后的规则 | {summary['revised_detected']}/{summary['episodes']} | {summary['revised_correct_within_4cm']}/{summary['episodes']} |",
        "",
        "修改前的失败图像及修订后叠加图仅用于诊断，保存在输出图像目录。",
    ]
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
