from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_chunk_bc import augment_relative_features, observation_layout  # noqa: E402
from widowx_env.demo_dataset import latest_run_dir, load_demo_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a trajectory-conditioned kNN action-chunk BC baseline.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "trajectory_knn_bc")
    parser.add_argument("--model-prefix", default="trajectory_knn_chunk_bc")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--sample-stride", type=int, default=16)
    parser.add_argument("--augment-relative", action="store_true")
    parser.add_argument("--no-augment-relative", dest="augment_relative", action="store_false")
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--all-attempts", action="store_true")
    parser.set_defaults(augment_relative=False)
    return parser.parse_args()


def build_samples(
    raw_observations: np.ndarray,
    model_observations: np.ndarray,
    actions: np.ndarray,
    segments: list[dict],
    horizon: int,
    history: int,
    sample_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    chunks: list[np.ndarray] = []
    phases: list[float] = []
    offset = 0
    for segment in segments:
        length = int(segment["steps"])
        segment_slice = slice(offset, offset + length)
        offset += length
        if length < horizon + history - 1:
            continue

        raw_segment = raw_observations[segment_slice]
        model_segment = model_observations[segment_slice]
        action_segment = actions[segment_slice]
        for start in range(history - 1, length - horizon + 1, sample_stride):
            xs.append(model_segment[start - history + 1: start + 1].reshape(-1))
            chunks.append(action_segment[start: start + horizon])
            phases.append(float(raw_segment[start, -3]))

    if not xs:
        raise ValueError("no trajectory kNN samples were built; reduce --history, --horizon, or --sample-stride")

    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(chunks, dtype=np.float32),
        np.asarray(phases, dtype=np.float32),
    )


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or latest_run_dir()
    dataset = load_demo_dataset(
        run_dir,
        successful_only=not args.include_failures,
        successful_attempt_only=not args.all_attempts,
    )
    raw_observations = dataset.observations.astype(np.float32)
    model_observations = raw_observations
    layout = observation_layout()
    if args.augment_relative:
        model_observations = augment_relative_features(model_observations, layout)

    horizon = max(2, int(args.horizon))
    history = max(1, int(args.history))
    sample_stride = max(1, int(args.sample_stride))
    x, action_chunks, phases = build_samples(
        raw_observations,
        model_observations,
        dataset.actions.astype(np.float32),
        dataset.segments,
        horizon,
        history,
        sample_stride,
    )

    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    x_norm = ((x - x_mean) / x_std).astype(np.float32)

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / f"{args.model_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    meta = {
        "method": "trajectory_conditioned_knn_action_chunk_bc",
        "run_dir": str(run_dir),
        "source_samples": int(len(dataset.actions)),
        "samples": int(len(x_norm)),
        "raw_observation_dim": int(raw_observations.shape[1]),
        "single_observation_dim": int(model_observations.shape[1]),
        "observation_dim": int(x_norm.shape[1]),
        "action_dim": int(dataset.actions.shape[1]),
        "horizon": horizon,
        "history": history,
        "sample_stride": sample_stride,
        "augment_relative": bool(args.augment_relative),
        "layout": layout,
        "successful_only": not args.include_failures,
        "successful_attempt_only": not args.all_attempts,
    }
    np.savez_compressed(
        model_path,
        observations_norm=x_norm,
        action_chunks=action_chunks,
        phases=phases,
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        action_min=dataset.actions.min(axis=0).astype(np.float32),
        action_max=dataset.actions.max(axis=0).astype(np.float32),
        metadata=json.dumps(meta, ensure_ascii=False),
    )

    print(f"run_dir: {run_dir}", flush=True)
    print(f"model_path: {model_path}", flush=True)
    print(f"source_samples: {meta['source_samples']}", flush=True)
    print(f"samples: {meta['samples']}", flush=True)
    print(f"single_observation_dim: {meta['single_observation_dim']}", flush=True)
    print(f"observation_dim: {meta['observation_dim']}", flush=True)
    print(f"action_dim: {meta['action_dim']}", flush=True)
    print(f"horizon: {horizon}", flush=True)
    print(f"history: {history}", flush=True)
    print(f"sample_stride: {sample_stride}", flush=True)


if __name__ == "__main__":
    main()
