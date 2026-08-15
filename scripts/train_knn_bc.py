from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from widowx_env.demo_dataset import latest_run_dir, load_demo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a NumPy kNN behavior cloning baseline.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "knn_bc")
    parser.add_argument("--stride", type=int, default=4, help="Keep every Nth sample to make closed-loop kNN fast enough.")
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--all-attempts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or latest_run_dir()
    dataset = load_demo_dataset(
        run_dir,
        successful_only=not args.include_failures,
        successful_attempt_only=not args.all_attempts,
    )

    stride = max(1, int(args.stride))
    keep = np.arange(0, len(dataset.actions), stride)
    observations = dataset.observations[keep].astype(np.float32)
    actions = dataset.actions[keep].astype(np.float32)

    x_mean = observations.mean(axis=0)
    x_std = observations.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    observations_norm = ((observations - x_mean) / x_std).astype(np.float32)

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / f"knn_bc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    meta = {
        "run_dir": str(run_dir),
        "source_samples": int(len(dataset.actions)),
        "samples": int(len(actions)),
        "stride": stride,
        "observation_dim": int(observations.shape[1]),
        "action_dim": int(actions.shape[1]),
        "successful_only": not args.include_failures,
        "successful_attempt_only": not args.all_attempts,
    }
    np.savez_compressed(
        model_path,
        observations_norm=observations_norm,
        actions=actions,
        phases=observations[:, -3].astype(np.float32),
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        action_min=actions.min(axis=0).astype(np.float32),
        action_max=actions.max(axis=0).astype(np.float32),
        metadata=json.dumps(meta, ensure_ascii=False),
    )

    print(f"run_dir: {run_dir}", flush=True)
    print(f"model_path: {model_path}", flush=True)
    print(f"source_samples: {meta['source_samples']}", flush=True)
    print(f"samples: {meta['samples']}", flush=True)
    print(f"stride: {stride}", flush=True)
    print(f"observation_dim: {meta['observation_dim']}", flush=True)
    print(f"action_dim: {meta['action_dim']}", flush=True)


if __name__ == "__main__":
    main()
