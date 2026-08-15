from __future__ import annotations

"""Check that a clean source checkout contains the files needed for the core demo."""

from pathlib import Path
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from widowx_env import WidowXTabletopEnv  # noqa: E402


REQUIRED_FILES = (
    "assets/mujoco/tabletop_wx250s_scene.xml",
    "external/wx250s_assets/wx250s.xml",
    "external/wx250s_assets/LICENSE",
    "runtime_assets/clip_semantic_waypoint_core_v2_v1_20260721_110325.npz",
    "runtime_assets/top_rgb_core_v2_calibration_v1.json",
    "runtime_assets/final_closure_audit_v1.json",
    "scripts/run_clip_semantic_rgb_feedback.py",
)


def main() -> None:
    missing = [relative for relative in REQUIRED_FILES if not (ROOT / relative).is_file()]
    if missing:
        raise SystemExit("Repository bundle is incomplete:\n- " + "\n- ".join(missing))

    with np.load(ROOT / "runtime_assets" / "clip_semantic_waypoint_core_v2_v1_20260721_110325.npz") as model:
        if not model.files:
            raise SystemExit("Runtime policy artifact is empty.")

    env = WidowXTabletopEnv(seed=0, image_size=(96, 96))
    observation = env.reset(task="place_blue_cube_blue_pad", complexity="medium", seed=0)
    image = env.render_rgb()
    checks = {
        "mujoco_version": mujoco.__version__,
        "action_size": env.action_size,
        "instruction": observation["instruction"],
        "rgb_shape": tuple(image.shape),
        "rgb_nonblank": bool(image.max() > image.min()),
    }
    if not checks["rgb_nonblank"]:
        raise SystemExit("MuJoCo RGB renderer returned a blank image.")
    print("Repository bundle verification passed.")
    for key, value in checks.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
