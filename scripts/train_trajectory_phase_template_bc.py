from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from widowx_env.demo_dataset import phase_features, read_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a phase-binned trajectory template BC baseline.")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data" / "demos" / "place_blue_cube_blue_pad_medium_20260702_051752")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "trajectory_phase_template_bc")
    parser.add_argument("--model-prefix", default="trajectory_phase_template_bc")
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--min-bin-samples", type=int, default=64)
    parser.add_argument("--feature-mode", choices=("planned", "state"), default="planned")
    parser.add_argument("--include-failures", action="store_true")
    return parser.parse_args()


def pre_step_array(series: np.ndarray, start_value: np.ndarray, indices: np.ndarray) -> np.ndarray:
    values = series[indices]
    if len(indices) == 1:
        return start_value[None, ...].astype(np.float32)
    return np.concatenate([start_value[None, ...], values[:-1]], axis=0).astype(np.float32)


def attempt_start_index(data: np.lib.npyio.NpzFile, attempt_id: int) -> int:
    matches = np.flatnonzero(data["attempt_start_ids"] == attempt_id)
    if len(matches) == 0:
        raise ValueError(f"attempt {attempt_id} has no start state")
    return int(matches[0])


def episode_features(
    data: np.lib.npyio.NpzFile,
    metadata: dict,
    attempt_id: int,
    sample_stride: int,
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.flatnonzero(data["attempt_ids"] == attempt_id)
    if len(indices) == 0:
        raise ValueError(f"attempt {attempt_id} has no saved steps")
    indices = indices[:: max(1, int(sample_stride))]

    object_names = [str(item) for item in data["object_names"].tolist()]
    target_object = str(metadata["target_object"])
    if target_object not in object_names:
        raise ValueError(f"target object {target_object!r} is not in object_names")
    object_index = object_names.index(target_object)

    start_index = attempt_start_index(data, attempt_id)
    qpos = pre_step_array(data["qpos"], data["attempt_start_qpos"][start_index], indices)
    ctrl = pre_step_array(data["ctrl"], data["attempt_start_ctrl"][start_index], indices)
    tcp = pre_step_array(data["tcp"], data["attempt_start_tcp"][start_index], indices)
    objects = pre_step_array(data["object_positions"], data["attempt_start_object_positions"][start_index], indices)
    object_pos = objects[:, object_index, :]
    target = np.asarray(metadata["target_position"] or [0.0, 0.0, 0.0], dtype=np.float32)
    target_pos = np.repeat(target[None, :], len(indices), axis=0)
    initial_object = np.asarray(metadata["initial_objects"][target_object], dtype=np.float32)
    initial_object_pos = np.repeat(initial_object[None, :], len(indices), axis=0)
    local_phase = np.linspace(0.0, 1.0, len(np.flatnonzero(data["attempt_ids"] == attempt_id)), dtype=np.float32)[:: max(1, int(sample_stride))]

    if feature_mode == "planned":
        features = np.concatenate(
            [
                initial_object_pos,
                target_pos,
                initial_object_pos - target_pos,
                phase_features(local_phase),
            ],
            axis=1,
        ).astype(np.float32)
    else:
        features = np.concatenate(
            [
                qpos[:, :6],
                ctrl,
                tcp,
                object_pos,
                target_pos,
                object_pos - tcp,
                object_pos - target_pos,
                tcp - target_pos,
                initial_object_pos,
                initial_object_pos - target_pos,
                phase_features(local_phase),
            ],
            axis=1,
        ).astype(np.float32)
    actions = data["actions"][indices].astype(np.float32)
    return features, actions, local_phase.astype(np.float32)


def build_arrays(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    features: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    phases: list[np.ndarray] = []
    used_rows: list[dict] = []
    for metadata in read_metadata(args.run_dir):
        if not args.include_failures and not bool(metadata["success"]):
            continue
        attempt_id = int(metadata["attempts"]) if bool(metadata["success"]) else 1
        with np.load(args.run_dir / metadata["trajectory_file"]) as data:
            x, y, phase = episode_features(data, metadata, attempt_id, args.sample_stride, args.feature_mode)
        features.append(x)
        actions.append(y)
        phases.append(phase)
        used_rows.append(metadata)
    if not features:
        raise ValueError(f"no training samples loaded from {args.run_dir}")
    return (
        np.concatenate(features, axis=0).astype(np.float32),
        np.concatenate(actions, axis=0).astype(np.float32),
        np.concatenate(phases, axis=0).astype(np.float32),
        used_rows,
    )


def fit_bin_weights(x_norm: np.ndarray, y: np.ndarray, phases: np.ndarray, bins: int, ridge: float, min_samples: int) -> tuple[np.ndarray, np.ndarray]:
    action_dim = y.shape[1]
    weights = np.zeros((bins, x_norm.shape[1] + 1, action_dim), dtype=np.float32)
    counts = np.zeros(bins, dtype=np.int32)
    bin_ids = np.minimum(bins - 1, np.floor(phases * bins).astype(np.int32))
    for bin_id in range(bins):
        radius = 0
        mask = bin_ids == bin_id
        while int(mask.sum()) < min_samples and radius < bins:
            radius += 1
            mask = np.abs(bin_ids - bin_id) <= radius
        x_bin = x_norm[mask]
        y_bin = y[mask]
        counts[bin_id] = int(len(x_bin))
        x_aug = np.concatenate([x_bin, np.ones((len(x_bin), 1), dtype=np.float32)], axis=1)
        reg = np.eye(x_aug.shape[1], dtype=np.float32) * float(ridge)
        reg[-1, -1] = 0.0
        lhs = x_aug.T @ x_aug + reg
        rhs = x_aug.T @ y_bin
        weights[bin_id] = np.linalg.solve(lhs, rhs).astype(np.float32)
    return weights, counts


def main() -> None:
    args = parse_args()
    bins = max(8, int(args.bins))
    x, y, phases, rows = build_arrays(args)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    x_norm = ((x - x_mean) / x_std).astype(np.float32)
    weights, counts = fit_bin_weights(x_norm, y, phases, bins, float(args.ridge), int(args.min_bin_samples))

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / f"{args.model_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    metadata = {
        "method": "trajectory_phase_template_bc",
        "run_dir": str(args.run_dir),
        "source_episodes": int(len(rows)),
        "samples": int(len(x)),
        "feature_dim": int(x.shape[1]),
        "action_dim": int(y.shape[1]),
        "bins": bins,
        "sample_stride": int(args.sample_stride),
        "ridge": float(args.ridge),
        "min_bin_samples": int(args.min_bin_samples),
        "feature_mode": str(args.feature_mode),
        "include_failures": bool(args.include_failures),
        "note": "Phase-binned ridge action template baseline for trajectory-conditioned BC; not full ACT.",
    }
    np.savez_compressed(
        model_path,
        weights=weights,
        bin_counts=counts,
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        action_min=y.min(axis=0).astype(np.float32),
        action_max=y.max(axis=0).astype(np.float32),
        metadata=json.dumps(metadata, ensure_ascii=False),
    )
    print(f"run_dir: {args.run_dir}", flush=True)
    print(f"model_path: {model_path}", flush=True)
    print(f"source_episodes: {metadata['source_episodes']}", flush=True)
    print(f"samples: {metadata['samples']}", flush=True)
    print(f"feature_dim: {metadata['feature_dim']}", flush=True)
    print(f"action_dim: {metadata['action_dim']}", flush=True)
    print(f"bins: {bins}", flush=True)
    print(f"min_bin_count: {int(counts.min())}", flush=True)
    print(f"max_bin_count: {int(counts.max())}", flush=True)


if __name__ == "__main__":
    main()
