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
from train_trajectory_knn_bc import build_samples  # noqa: E402
from widowx_env.demo_dataset import latest_run_dir, load_demo_dataset, read_metadata  # noqa: E402


VERSION = "preference_trajectory_post_training_v1_candidate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a preference-weighted trajectory-kNN post-training candidate.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "preference_post_training")
    parser.add_argument("--model-prefix", default="preference_trajectory_post_training")
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--sample-stride", type=int, default=16)
    parser.add_argument("--augment-relative", action="store_true")
    parser.add_argument("--no-augment-relative", dest="augment_relative", action="store_false")
    parser.add_argument("--distance-temperature", type=float, default=0.08)
    parser.add_argument("--preference-mode", choices=("distance", "episode_rank", "tcp_lift_rank"), default="distance")
    parser.add_argument("--rank-decay", type=float, default=0.45)
    parser.add_argument("--success-multiplier", type=float, default=2.0)
    parser.add_argument("--placed-multiplier", type=float, default=1.4)
    parser.add_argument("--tcp-lift-multiplier", type=float, default=2.5)
    parser.add_argument("--failed-attempt-multiplier", type=float, default=0.45)
    parser.add_argument("--failed-episode-multiplier", type=float, default=0.25)
    parser.add_argument("--out-of-table-multiplier", type=float, default=0.2)
    parser.add_argument("--min-preference", type=float, default=0.05)
    parser.add_argument("--lift-threshold", type=float, default=0.085)
    parser.add_argument("--tcp-lift-threshold", type=float, default=0.12)
    parser.set_defaults(augment_relative=False)
    return parser.parse_args()


def object_names(data: np.lib.npyio.NpzFile) -> list[str]:
    names = []
    for item in data["object_names"]:
        if isinstance(item, bytes):
            names.append(item.decode("utf-8"))
        else:
            names.append(str(item))
    return names


def out_of_table(pos: np.ndarray) -> bool:
    return bool(pos[0] < -0.25 or pos[0] > 0.85 or abs(pos[1]) > 0.38)


def attempt_preferences(args: argparse.Namespace, run_dir: Path) -> tuple[dict[tuple[int, int], dict[str, float | bool]], dict[str, int | float]]:
    preferences: dict[tuple[int, int], dict[str, float | bool]] = {}
    counts = {
        "episodes": 0,
        "successful_episodes": 0,
        "attempts": 0,
        "preferred_attempts": 0,
        "placed_attempts": 0,
        "tcp_lift_attempts": 0,
        "failed_attempts": 0,
        "out_of_table_attempts": 0,
    }
    distances: list[float] = []
    weights: list[float] = []

    for metadata in read_metadata(run_dir):
        counts["episodes"] += 1
        episode_success = bool(metadata["success"])
        counts["successful_episodes"] += int(episode_success)
        target = np.asarray(metadata["target_position"] or [0.0, 0.0, 0.0], dtype=np.float32)
        target_object = str(metadata["target_object"])
        final_attempt = int(metadata["attempts"])

        with np.load(run_dir / metadata["trajectory_file"]) as data:
            names = object_names(data)
            target_index = names.index(target_object)
            attempt_infos = []
            for attempt_id in sorted(int(item) for item in np.unique(data["attempt_ids"])):
                indices = np.flatnonzero(data["attempt_ids"] == attempt_id)
                if len(indices) == 0:
                    continue
                attempt_object_positions = data["object_positions"][indices, target_index].astype(np.float32)
                attempt_tcp = data["tcp"][indices].astype(np.float32)
                final_pos = attempt_object_positions[-1]
                target_distance = float(np.linalg.norm(final_pos[:2] - target[:2]))
                table_violation = out_of_table(final_pos)
                placed = bool(np.isfinite(target_distance) and target_distance < 0.065 and final_pos[2] < 0.08)
                tcp_object_distances = np.linalg.norm(attempt_tcp - attempt_object_positions, axis=1)
                max_object_z = float(np.max(attempt_object_positions[:, 2]))
                lifted = attempt_object_positions[:, 2] >= float(args.lift_threshold)
                if np.any(lifted):
                    min_tcp_object_distance_while_lifted = float(np.min(tcp_object_distances[lifted]))
                else:
                    min_tcp_object_distance_while_lifted = float("inf")
                tcp_lift_success = bool(
                    max_object_z >= float(args.lift_threshold)
                    and min_tcp_object_distance_while_lifted < float(args.tcp_lift_threshold)
                )
                if args.preference_mode == "tcp_lift_rank":
                    preferred = bool(episode_success and attempt_id == final_attempt and placed and tcp_lift_success)
                else:
                    preferred = bool(episode_success and attempt_id == final_attempt and placed)
                attempt_infos.append(
                    {
                        "attempt_id": attempt_id,
                        "target_distance": target_distance,
                        "object_z": float(final_pos[2]),
                        "max_object_z": max_object_z,
                        "min_tcp_object_distance": float(np.min(tcp_object_distances)),
                        "min_tcp_object_distance_while_lifted": min_tcp_object_distance_while_lifted,
                        "tcp_lift_success": tcp_lift_success,
                        "placed": placed,
                        "out_of_table": table_violation,
                        "preferred": preferred,
                    }
                )

            ranked_attempts = sorted(
                attempt_infos,
                key=lambda item: (
                    not bool(item["preferred"]),
                    not bool(item["tcp_lift_success"]) if args.preference_mode == "tcp_lift_rank" else False,
                    bool(item["out_of_table"]),
                    float(item["target_distance"]),
                    -float(item["max_object_z"]) if args.preference_mode == "tcp_lift_rank" else 0.0,
                    float(item["min_tcp_object_distance"]),
                ),
            )
            rank_by_attempt = {int(item["attempt_id"]): rank for rank, item in enumerate(ranked_attempts)}

            for info in attempt_infos:
                attempt_id = int(info["attempt_id"])
                target_distance = float(info["target_distance"])
                table_violation = bool(info["out_of_table"])
                preferred = bool(info["preferred"])
                weight = float(args.min_preference) + float(np.exp(-target_distance / max(args.distance_temperature, 1e-6)))
                if args.preference_mode == "episode_rank":
                    rank = int(rank_by_attempt[attempt_id])
                    weight = float(args.min_preference) + float(np.exp(-rank / max(args.rank_decay, 1e-6)))
                elif args.preference_mode == "tcp_lift_rank":
                    rank = int(rank_by_attempt[attempt_id])
                    weight = float(args.min_preference) + float(np.exp(-rank / max(args.rank_decay, 1e-6)))
                    if bool(info["placed"]):
                        weight *= float(args.placed_multiplier)
                    if bool(info["tcp_lift_success"]):
                        weight *= float(args.tcp_lift_multiplier)
                if preferred:
                    weight *= float(args.success_multiplier)
                else:
                    weight *= float(args.failed_attempt_multiplier)
                if not episode_success:
                    weight *= float(args.failed_episode_multiplier)
                if table_violation:
                    weight *= float(args.out_of_table_multiplier)
                weight = max(float(args.min_preference), float(weight))

                counts["attempts"] += 1
                counts["preferred_attempts"] += int(preferred)
                counts["placed_attempts"] += int(bool(info["placed"]))
                counts["tcp_lift_attempts"] += int(bool(info["tcp_lift_success"]))
                counts["failed_attempts"] += int(not preferred)
                counts["out_of_table_attempts"] += int(table_violation)
                distances.append(target_distance)
                weights.append(weight)
                preferences[(int(metadata["episode_index"]), int(attempt_id))] = {
                    "preferred": preferred,
                    "episode_success": episode_success,
                    "target_distance": target_distance,
                    "object_z": float(info["object_z"]),
                    "max_object_z": float(info["max_object_z"]),
                    "min_tcp_object_distance": float(info["min_tcp_object_distance"]),
                    "min_tcp_object_distance_while_lifted": float(info["min_tcp_object_distance_while_lifted"]),
                    "tcp_lift_success": bool(info["tcp_lift_success"]),
                    "placed": bool(info["placed"]),
                    "out_of_table": table_violation,
                    "rank": int(rank_by_attempt[attempt_id]),
                    "weight": weight,
                }

    counts["mean_attempt_target_distance"] = float(np.mean(distances)) if distances else 0.0
    counts["mean_preference_weight"] = float(np.mean(weights)) if weights else 0.0
    return preferences, counts


def sample_segment_ids(segments: list[dict], horizon: int, history: int, sample_stride: int) -> np.ndarray:
    ids: list[int] = []
    for segment_id, segment in enumerate(segments):
        length = int(segment["steps"])
        if length < horizon + history - 1:
            continue
        for _ in range(history - 1, length - horizon + 1, sample_stride):
            ids.append(segment_id)
    if not ids:
        raise ValueError("no segment ids were built; reduce --history, --horizon, or --sample-stride")
    return np.asarray(ids, dtype=np.int32)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or latest_run_dir()
    dataset = load_demo_dataset(run_dir, successful_only=False, successful_attempt_only=False)
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
    segment_ids = sample_segment_ids(dataset.segments, horizon, history, sample_stride)
    preference_by_attempt, preference_summary = attempt_preferences(args, run_dir)
    segment_preferences = []
    for segment in dataset.segments:
        key = (int(segment["episode_index"]), int(segment["attempt_id"]))
        segment_preferences.append(float(preference_by_attempt[key]["weight"]))
    sample_preferences = np.asarray([segment_preferences[int(segment_id)] for segment_id in segment_ids], dtype=np.float32)

    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    x_norm = ((x - x_mean) / x_std).astype(np.float32)

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / f"{args.model_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    meta = {
        "version": str(args.version),
        "method": "preference_weighted_trajectory_knn_post_training_candidate",
        "run_dir": str(run_dir),
        "source_samples": int(len(dataset.actions)),
        "samples": int(len(x_norm)),
        "segments": int(len(dataset.segments)),
        "raw_observation_dim": int(raw_observations.shape[1]),
        "single_observation_dim": int(model_observations.shape[1]),
        "observation_dim": int(x_norm.shape[1]),
        "action_dim": int(dataset.actions.shape[1]),
        "horizon": horizon,
        "history": history,
        "sample_stride": sample_stride,
        "augment_relative": bool(args.augment_relative),
        "layout": layout,
        "preference_source": "scripted demo episode/attempt outcome proxy",
        "preference_strategy": (
            "episode-level ranked attempt preference with success/failure/out-of-table multipliers"
            if args.preference_mode == "episode_rank"
            else "episode-level ranked attempt preference with placed/tcp-lift/out-of-table multipliers"
            if args.preference_mode == "tcp_lift_rank"
            else "trajectory-level target-distance reward with success/failure/out-of-table multipliers"
        ),
        "preference_mode": str(args.preference_mode),
        "distance_temperature": float(args.distance_temperature),
        "rank_decay": float(args.rank_decay),
        "success_multiplier": float(args.success_multiplier),
        "placed_multiplier": float(args.placed_multiplier),
        "tcp_lift_multiplier": float(args.tcp_lift_multiplier),
        "failed_attempt_multiplier": float(args.failed_attempt_multiplier),
        "failed_episode_multiplier": float(args.failed_episode_multiplier),
        "out_of_table_multiplier": float(args.out_of_table_multiplier),
        "min_preference": float(args.min_preference),
        "lift_threshold": float(args.lift_threshold),
        "tcp_lift_threshold": float(args.tcp_lift_threshold),
        "preference_summary": preference_summary,
        "successful_only": False,
        "successful_attempt_only": False,
    }
    np.savez_compressed(
        model_path,
        observations_norm=x_norm,
        action_chunks=action_chunks,
        phases=phases,
        sample_preferences=sample_preferences,
        segment_ids=segment_ids,
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
    print(f"segments: {meta['segments']}", flush=True)
    print(f"preference_summary: {preference_summary}", flush=True)
    print(f"sample_preference_min: {float(sample_preferences.min()):.4f}", flush=True)
    print(f"sample_preference_mean: {float(sample_preferences.mean()):.4f}", flush=True)
    print(f"sample_preference_max: {float(sample_preferences.max()):.4f}", flush=True)


if __name__ == "__main__":
    main()
