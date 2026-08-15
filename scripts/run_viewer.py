from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
import sys

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from widowx_env import TASKS, WidowXTabletopEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open the WidowX tabletop MuJoCo viewer.")
    parser.add_argument("--task", choices=sorted(TASKS), default="place_blue_cube_red_pad")
    parser.add_argument("--complexity", choices=("easy", "medium", "hard", "language"), default="medium")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to keep the viewer open. 0 means until closed.")
    parser.add_argument("--motion", choices=("idle", "sweep"), default="idle")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = WidowXTabletopEnv(seed=args.seed)
    obs = env.reset(task=args.task, complexity=args.complexity, seed=args.seed)
    print(f"instruction: {obs['instruction']}", flush=True)
    print(f"active_objects: {', '.join(obs['active_objects'])}", flush=True)
    print("viewer: close the MuJoCo window to exit", flush=True)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        start = time.time()
        while viewer.is_running():
            if args.duration and time.time() - start > args.duration:
                break
            if args.motion == "sweep":
                t = time.time() - start
                action = env.home_ctrl.copy()
                action[0] += 0.35 * np.sin(t)
                action[1] += 0.15 * np.sin(0.7 * t)
                env.step(action)
            else:
                env.step(env.home_ctrl)
            viewer.sync()
            time.sleep(0.01)
    print("final_metrics:", env.metrics(), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
