from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from widowx_env.demo_dataset import latest_run_dir, load_demo_dataset, read_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect saved demonstration samples.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--all-attempts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or latest_run_dir()
    metadata = read_metadata(run_dir)
    dataset = load_demo_dataset(
        run_dir,
        successful_only=not args.include_failures,
        successful_attempt_only=not args.all_attempts,
    )

    print(f"run_dir: {run_dir}", flush=True)
    print(f"episodes_in_metadata: {len(metadata)}", flush=True)
    print(f"segments_loaded: {len(dataset.segments)}", flush=True)
    print(f"samples: {len(dataset.actions)}", flush=True)
    print(f"observation_shape: {dataset.observations.shape}", flush=True)
    print(f"action_shape: {dataset.actions.shape}", flush=True)
    print(f"episode_indices: {sorted(np.unique(dataset.episode_indices).astype(int).tolist())}", flush=True)
    print(f"attempt_ids: {sorted(np.unique(dataset.attempt_ids).astype(int).tolist())}", flush=True)
    print(f"action_mean: {np.round(dataset.actions.mean(axis=0), 4).tolist()}", flush=True)
    print(f"action_std: {np.round(dataset.actions.std(axis=0), 4).tolist()}", flush=True)


if __name__ == "__main__":
    main()
