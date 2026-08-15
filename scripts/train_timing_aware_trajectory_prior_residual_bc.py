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

from timing_aware_trajectory_prior_residual_common import (  # noqa: E402
    TIMING_STAGES,
    VERSION,
    build_plan,
    build_segments,
    make_config,
    prior_action_for_step,
    residual_feature,
    segment_for_step,
    total_steps,
)
from train_trajectory_prior_residual_bc import fit_ridge, successful_attempt_id  # noqa: E402
from widowx_env import WidowXTabletopEnv  # noqa: E402
from widowx_env.demo_dataset import read_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train timing-aware trajectory-prior residual BC.")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data" / "demos" / "contact_stage_demo_place_blue_cube_blue_pad_medium_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "timing_aware_trajectory_prior_residual_bc")
    parser.add_argument("--model-prefix", default=VERSION)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--residual-clip-quantile", type=float, default=0.995)
    parser.add_argument("--approach-z", type=float, default=0.12)
    parser.add_argument("--grasp-z", type=float, default=0.008)
    parser.add_argument("--lift-z", type=float, default=0.18)
    parser.add_argument("--include-failures", action="store_true")
    return parser.parse_args()


def configure_env(metadata: dict) -> tuple[WidowXTabletopEnv, dict]:
    seed = int(metadata["seed"])
    env = WidowXTabletopEnv(seed=seed)
    env.set_arm_actuator_strength(kp=150.0, force_limit=100.0)
    env.set_gripper_actuator_strength(kp=900.0, force_limit=180.0)
    env.set_grasp_contact_friction(sliding=3.0)
    obs = env.reset(task=str(metadata["task"]), complexity=str(metadata["complexity"]), seed=seed)
    return env, obs


def episode_arrays(metadata: dict, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    env, obs = configure_env(metadata)
    target_geom = str(metadata.get("target_geom") or env.task.target_geom)
    object_name = str(metadata.get("target_object") or obs["target_object"])
    config = make_config(args.approach_z, args.grasp_z, args.lift_z)
    plan = build_plan(env, object_name, target_geom, config)
    segments = build_segments(env, plan, config)
    prior_total = total_steps(segments)

    path = args.run_dir / metadata["trajectory_file"]
    with np.load(path) as data:
        attempt_id = successful_attempt_id(metadata)
        all_indices = np.flatnonzero(data["attempt_ids"] == attempt_id)
        if len(all_indices) == 0:
            raise ValueError(f"attempt {attempt_id} has no saved steps in {path}")
        selected_positions = np.arange(len(all_indices), dtype=np.int32)[:: max(1, int(args.sample_stride))]
        selected_indices = all_indices[selected_positions]
        actions = data["actions"][selected_indices].astype(np.float32)

    target_position = np.asarray(metadata["target_position"] or plan["target_position"], dtype=np.float32)
    initial_object = np.asarray(metadata["initial_objects"][object_name], dtype=np.float32)
    demo_denom = max(1, len(all_indices) - 1)
    prior_denom = max(1, prior_total - 1)

    features = []
    priors = []
    for local_step in selected_positions:
        global_phase = float(local_step) / demo_denom
        prior_step = min(int(round(global_phase * prior_denom)), prior_total - 1)
        segment = segment_for_step(segments, prior_step)
        priors.append(prior_action_for_step(segments, prior_step))
        features.append(residual_feature(initial_object, target_position, global_phase, segment, prior_step))

    return np.stack(features).astype(np.float32), actions, np.stack(priors).astype(np.float32)


def build_training_arrays(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    features: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    priors: list[np.ndarray] = []
    used_rows: list[dict] = []
    for metadata in read_metadata(args.run_dir):
        if not args.include_failures and not bool(metadata["success"]):
            continue
        x, y, base = episode_arrays(metadata, args)
        features.append(x)
        actions.append(y)
        priors.append(base)
        used_rows.append(metadata)
    if not features:
        raise ValueError(f"no usable demonstrations found under {args.run_dir}")
    return (
        np.concatenate(features, axis=0).astype(np.float32),
        np.concatenate(actions, axis=0).astype(np.float32),
        np.concatenate(priors, axis=0).astype(np.float32),
        used_rows,
    )


def main() -> None:
    args = parse_args()
    x, actions, priors, rows = build_training_arrays(args)
    residuals = (actions - priors).astype(np.float32)

    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    x_norm = ((x - x_mean) / x_std).astype(np.float32)
    weights = fit_ridge(x_norm, residuals, float(args.ridge))

    q = float(np.clip(args.residual_clip_quantile, 0.5, 1.0))
    residual_abs = np.quantile(np.abs(residuals), q, axis=0).astype(np.float32)
    residual_abs[residual_abs < 1e-5] = 1e-5

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / f"{args.model_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    metadata = {
        "version": VERSION,
        "method": "timing_aware_trajectory_prior_residual_bc",
        "run_dir": str(args.run_dir),
        "source_episodes": int(len(rows)),
        "samples": int(len(x)),
        "feature_dim": int(x.shape[1]),
        "action_dim": int(actions.shape[1]),
        "sample_stride": int(args.sample_stride),
        "ridge": float(args.ridge),
        "residual_clip_quantile": q,
        "approach_z": float(args.approach_z),
        "grasp_z": float(args.grasp_z),
        "lift_z": float(args.lift_z),
        "include_failures": bool(args.include_failures),
        "timing_stages": [stage.stage for stage in TIMING_STAGES],
        "alignment": "demo_global_phase_to_timing_prior_step",
        "note": "中文说明：时序感知 trajectory-prior residual BC 候选；把强闭合、闭合保持和抬升保持写入分阶段轨迹先验，不是完整官方 ACT，也不是 VLA 后训练。",
    }
    np.savez_compressed(
        model_path,
        weights=weights,
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        residual_min=(-residual_abs).astype(np.float32),
        residual_max=residual_abs.astype(np.float32),
        action_min=actions.min(axis=0).astype(np.float32),
        action_max=actions.max(axis=0).astype(np.float32),
        metadata=json.dumps(metadata, ensure_ascii=False),
    )
    print(f"run_dir: {args.run_dir}", flush=True)
    print(f"model_path: {model_path}", flush=True)
    print(f"source_episodes: {metadata['source_episodes']}", flush=True)
    print(f"samples: {metadata['samples']}", flush=True)
    print(f"feature_dim: {metadata['feature_dim']}", flush=True)
    print(f"timing_stages: {', '.join(metadata['timing_stages'])}", flush=True)
    print(f"mean_abs_residual: {float(np.mean(np.abs(residuals))):.8f}", flush=True)
    print(f"max_abs_residual: {float(np.max(np.abs(residuals))):.8f}", flush=True)


if __name__ == "__main__":
    main()
