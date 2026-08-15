from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .tabletop_env import OBJECTS, PROJECT_ROOT, WidowXTabletopEnv


@dataclass(frozen=True)
class DemoDataset:
    observations: np.ndarray
    actions: np.ndarray
    episode_indices: np.ndarray
    attempt_ids: np.ndarray
    source_steps: np.ndarray
    segments: list[dict[str, Any]]


def latest_run_dir(demos_root: Path | None = None) -> Path:
    root = demos_root or PROJECT_ROOT / "data" / "demos"
    candidates = [path for path in root.glob("*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no demo run folders found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_metadata(run_dir: Path) -> list[dict[str, Any]]:
    metadata_path = run_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata file not found: {metadata_path}")
    return [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def phase_features(phase: float | np.ndarray) -> np.ndarray:
    phase_array = np.asarray(phase, dtype=np.float32)
    return np.stack(
        [
            phase_array,
            np.sin(np.pi * phase_array),
            np.cos(np.pi * phase_array),
        ],
        axis=-1,
    ).astype(np.float32)


def observation_from_env(env: WidowXTabletopEnv, target_position: np.ndarray, phase: float = 0.0) -> np.ndarray:
    object_positions = np.stack([env.object_position(name) for name in OBJECTS])
    return observation_vector(
        env.data.qpos,
        env.data.qvel,
        env.data.ctrl,
        env.tcp_position(),
        object_positions,
        target_position,
        phase_features(phase),
    )


def observation_vector(
    qpos: np.ndarray,
    qvel: np.ndarray,
    ctrl: np.ndarray,
    tcp: np.ndarray,
    object_positions: np.ndarray,
    target_position: np.ndarray,
    phase: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(qpos, dtype=np.float32).reshape(-1),
            np.asarray(qvel, dtype=np.float32).reshape(-1),
            np.asarray(ctrl, dtype=np.float32).reshape(-1),
            np.asarray(tcp, dtype=np.float32).reshape(-1),
            np.asarray(object_positions, dtype=np.float32).reshape(-1),
            np.asarray(target_position, dtype=np.float32).reshape(-1),
            np.asarray(phase, dtype=np.float32).reshape(-1),
        ]
    ).astype(np.float32)


def load_demo_dataset(
    run_dir: Path,
    successful_only: bool = True,
    successful_attempt_only: bool = True,
) -> DemoDataset:
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episode_indices: list[np.ndarray] = []
    attempt_ids: list[np.ndarray] = []
    source_steps: list[np.ndarray] = []
    segments: list[dict[str, Any]] = []

    for metadata in read_metadata(run_dir):
        if successful_only and not metadata["success"]:
            continue

        path = run_dir / metadata["trajectory_file"]
        with np.load(path) as data:
            attempts = _attempts_to_use(data, metadata, successful_attempt_only)
            for attempt_id in attempts:
                indices = np.flatnonzero(data["attempt_ids"] == attempt_id)
                if len(indices) == 0:
                    continue

                x = _build_observations(data, indices, metadata, attempt_id)
                y = data["actions"][indices].astype(np.float32)

                observations.append(x)
                actions.append(y)
                episode_indices.append(np.full(len(indices), int(metadata["episode_index"]), dtype=np.int32))
                attempt_ids.append(np.full(len(indices), attempt_id, dtype=np.int16))
                source_steps.append(indices.astype(np.int32))
                segments.append(
                    {
                        "episode_index": int(metadata["episode_index"]),
                        "seed": int(metadata["seed"]),
                        "attempt_id": int(attempt_id),
                        "steps": int(len(indices)),
                        "success": bool(metadata["success"]),
                    }
                )

    if not observations:
        raise ValueError(f"no samples loaded from {run_dir}")

    return DemoDataset(
        observations=np.concatenate(observations, axis=0),
        actions=np.concatenate(actions, axis=0),
        episode_indices=np.concatenate(episode_indices, axis=0),
        attempt_ids=np.concatenate(attempt_ids, axis=0),
        source_steps=np.concatenate(source_steps, axis=0),
        segments=segments,
    )


def _attempts_to_use(data: np.lib.npyio.NpzFile, metadata: dict[str, Any], successful_attempt_only: bool) -> list[int]:
    if successful_attempt_only and metadata["success"]:
        return [int(metadata["attempts"])]
    return sorted(int(item) for item in np.unique(data["attempt_ids"]))


def _build_observations(
    data: np.lib.npyio.NpzFile,
    indices: np.ndarray,
    metadata: dict[str, Any],
    attempt_id: int,
) -> np.ndarray:
    start_index = _attempt_start_index(data, attempt_id)
    target = np.asarray(metadata["target_position"] or [0.0, 0.0, 0.0], dtype=np.float32)

    qpos = _pre_step_array(data["qpos"], data["attempt_start_qpos"][start_index], indices)
    qvel = _pre_step_array(data["qvel"], data["attempt_start_qvel"][start_index], indices)
    ctrl = _pre_step_array(data["ctrl"], data["attempt_start_ctrl"][start_index], indices)
    tcp = _pre_step_array(data["tcp"], data["attempt_start_tcp"][start_index], indices)
    objects = _pre_step_array(
        data["object_positions"],
        data["attempt_start_object_positions"][start_index],
        indices,
    )

    target_features = np.repeat(target[None, :], len(indices), axis=0)
    local_phase = np.linspace(0.0, 1.0, len(indices), dtype=np.float32)
    phase = phase_features(local_phase)
    return np.concatenate(
        [
            qpos,
            qvel,
            ctrl,
            tcp,
            objects.reshape(len(indices), -1),
            target_features,
            phase,
        ],
        axis=1,
    ).astype(np.float32)


def _attempt_start_index(data: np.lib.npyio.NpzFile, attempt_id: int) -> int:
    required = (
        "attempt_start_ids",
        "attempt_start_qpos",
        "attempt_start_qvel",
        "attempt_start_ctrl",
        "attempt_start_tcp",
        "attempt_start_object_positions",
    )
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"demo file is missing fields required for BC loading: {missing}")

    matches = np.flatnonzero(data["attempt_start_ids"] == attempt_id)
    if len(matches) == 0:
        raise ValueError(f"attempt {attempt_id} has no saved start state")
    return int(matches[0])


def _pre_step_array(series: np.ndarray, start_value: np.ndarray, indices: np.ndarray) -> np.ndarray:
    values = series[indices]
    if len(indices) == 1:
        return start_value[None, ...].astype(np.float32)
    return np.concatenate([start_value[None, ...], values[:-1]], axis=0).astype(np.float32)
