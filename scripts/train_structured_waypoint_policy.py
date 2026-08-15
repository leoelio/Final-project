from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save a structured single-attempt waypoint policy artifact.")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data" / "demos" / "place_blue_cube_blue_pad_medium_20260702_051752")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "structured_waypoint_policy")
    parser.add_argument("--model-prefix", default="structured_waypoint_policy")
    parser.add_argument("--approach-z", type=float, default=0.12)
    parser.add_argument("--grasp-z", type=float, default=0.008)
    parser.add_argument("--lift-z", type=float, default=0.18)
    parser.add_argument("--transfer-z", type=float, default=0.18)
    parser.add_argument("--place-tcp-z", type=float, default=0.055)
    parser.add_argument("--retreat-z", type=float, default=0.16)
    parser.add_argument("--open-gripper", type=float, default=0.037)
    parser.add_argument("--close-gripper", type=float, default=0.015)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / f"{args.model_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    params = np.asarray(
        [
            args.approach_z,
            args.grasp_z,
            args.lift_z,
            args.transfer_z,
            args.place_tcp_z,
            args.retreat_z,
            args.open_gripper,
            args.close_gripper,
        ],
        dtype=np.float32,
    )
    metadata = {
        "method": "structured_waypoint_policy",
        "run_dir": str(args.run_dir),
        "samples": 0,
        "feature_dim": "object_target_state",
        "action_dim": 7,
        "train_time_seconds": time.time() - started,
        "peak_vram_mb": 0.0,
        "approach_z": float(args.approach_z),
        "grasp_z": float(args.grasp_z),
        "lift_z": float(args.lift_z),
        "transfer_z": float(args.transfer_z),
        "place_tcp_z": float(args.place_tcp_z),
        "retreat_z": float(args.retreat_z),
        "open_gripper": float(args.open_gripper),
        "close_gripper": float(args.close_gripper),
        "note": "Single-attempt object/target-conditioned waypoint policy; structured control baseline, not learned VLA.",
    }
    np.savez_compressed(model_path, params=params, metadata=json.dumps(metadata, ensure_ascii=False))
    print(f"model_path: {model_path}", flush=True)
    print(f"params: {params.tolist()}", flush=True)
    print(f"train_time_seconds: {metadata['train_time_seconds']:.6f}", flush=True)


if __name__ == "__main__":
    main()
