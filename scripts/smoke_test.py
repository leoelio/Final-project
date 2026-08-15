from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from widowx_env import TASKS, WidowXTabletopEnv


def main() -> None:
    env = WidowXTabletopEnv(seed=7, image_size=(96, 128))
    obs = env.reset(task="place_blue_cube_red_pad", complexity="medium")
    for _ in range(20):
        _, metrics = env.step(env.home_ctrl)
    image = env.render_rgb()

    active_positions = np.array([obs["objects"][name][:2] for name in obs["active_objects"]])
    if len(active_positions) > 1:
        distances = [
            np.linalg.norm(active_positions[i] - active_positions[j])
            for i in range(len(active_positions))
            for j in range(i + 1, len(active_positions))
        ]
        min_distance = float(min(distances))
    else:
        min_distance = float("inf")

    checks = {
        "tasks_defined": len(TASKS) >= 5,
        "action_size_is_wx250s_actuators": env.action_size == 7,
        "active_objects_nonoverlap": min_distance > 0.08,
        "render_nonblank": bool(image.max() > image.min()),
        "finite_qpos": bool(np.isfinite(env.data.qpos).all()),
        "finite_qvel": bool(np.isfinite(env.data.qvel).all()),
        "metrics_have_success": "success" in metrics,
    }
    for name, ok in checks.items():
        print(f"{name}: {ok}")
    print("instruction:", obs["instruction"])
    print("active_objects:", ", ".join(obs["active_objects"]))
    print("target_object:", obs["target_object"])
    print("min_active_object_distance:", round(min_distance, 4))
    print("render_shape:", image.shape)
    print("metrics:", metrics)

    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
