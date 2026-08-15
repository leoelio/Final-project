from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from widowx_env import WidowXTabletopEnv  # noqa: E402
from widowx_env.vision_grounding import (  # noqa: E402
    detect_colored_regions,
    fit_plane_calibration,
    save_calibration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate the fixed top RGB camera to the Core V2 tabletop plane.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "rgb_grounding" / "top_rgb_core_v2_calibration_v1.json")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--camera", default="top_rgb")
    parser.add_argument("--workspace-profile", choices=("legacy", "core_v2"), default="core_v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = WidowXTabletopEnv(
        seed=0,
        image_size=(args.image_size, args.image_size),
        camera=args.camera,
        workspace_profile=args.workspace_profile,
    )
    env.reset(task="place_blue_cube_blue_pad", complexity="easy", seed=0)
    anchors_xy = np.array(
        [
            [0.24, -0.09],
            [0.34, -0.09],
            [0.44, -0.09],
            [0.24, 0.00],
            [0.34, 0.00],
            [0.44, 0.00],
            [0.24, 0.09],
            [0.34, 0.09],
            [0.44, 0.09],
        ],
        dtype=float,
    )
    pixels: list[np.ndarray] = []
    records: list[dict] = []
    renderer = mujoco.Renderer(env.model, height=args.image_size, width=args.image_size)
    try:
        for anchor in anchors_xy:
            env._set_free_object_pose("yellow_cube", np.array([anchor[0], anchor[1], 0.026], dtype=float))
            mujoco.mj_forward(env.model, env.data)
            renderer.update_scene(env.data, camera=args.camera)
            regions = detect_colored_regions(renderer.render(), "yellow")
            if not regions:
                raise RuntimeError(f"yellow cube was not detected at {anchor.tolist()}")
            region = max(regions, key=lambda item: item.area)
            pixels.append(region.center_uv)
            records.append({"world_xy": anchor.tolist(), "pixel_uv": region.center_uv.tolist(), "area": region.area})
    finally:
        renderer.close()
    calibration = fit_plane_calibration(
        np.asarray(pixels, dtype=float),
        anchors_xy,
        image_size=args.image_size,
        camera=args.camera,
        workspace_profile=args.workspace_profile,
    )
    save_calibration(args.output, calibration, records)
    print(f"calibration_path: {args.output}")
    print(f"anchors: {len(records)}")
    print(f"rms_error_m: {calibration.rms_error_m:.6f}")


if __name__ == "__main__":
    main()
