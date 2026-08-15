from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from widowx_env import WidowXTabletopEnv
from widowx_env.demo_dataset import phase_features
from widowx_env.scripted_expert import PickPlaceConfig, PickPlaceExpert, interpolate_action


VERSION = "trajectory_prior_residual_bc_v1_candidate"
STAGES = (
    ("approach", "approach", "approach_steps"),
    ("descend", "grasp_open", "descend_steps"),
    ("close", "grasp_closed", "close_steps"),
    ("lift", "lift_closed", "lift_steps"),
    ("transfer", "transfer_closed", "transfer_steps"),
    ("place_descend", "place_closed", "place_descend_steps"),
    ("release", "place_open", "open_steps"),
    ("retreat", "retreat_open", "retreat_steps"),
    ("hold", "retreat_open", "hold_steps"),
)


@dataclass(frozen=True)
class StageSegment:
    stage: str
    stage_id: int
    start_step: int
    end_step: int
    start_action: np.ndarray
    target_action: np.ndarray

    @property
    def steps(self) -> int:
        return max(1, self.end_step - self.start_step)


def make_config(
    approach_z: float = 0.12,
    grasp_z: float = 0.008,
    lift_z: float = 0.18,
    place_tcp_z: float = 0.055,
) -> PickPlaceConfig:
    return PickPlaceConfig(
        approach_z_offset=float(approach_z),
        grasp_z_offset=float(grasp_z),
        lift_z_offset=float(lift_z),
        place_tcp_z=float(place_tcp_z),
    )


def build_plan(env: WidowXTabletopEnv, object_name: str, target_geom: str, config: PickPlaceConfig) -> dict:
    expert = PickPlaceExpert(env, config)
    return expert.plan(object_name, target_geom)


def build_segments(env: WidowXTabletopEnv, plan: dict, config: PickPlaceConfig) -> list[StageSegment]:
    segments: list[StageSegment] = []
    start_action = env.data.ctrl.copy().astype(np.float32)
    cursor = 0
    for stage_id, (stage, action_key, steps_attr) in enumerate(STAGES):
        steps = int(getattr(config, steps_attr))
        target_action = plan["actions"][action_key].astype(np.float32)
        segments.append(
            StageSegment(
                stage=stage,
                stage_id=stage_id,
                start_step=cursor,
                end_step=cursor + steps,
                start_action=start_action.copy(),
                target_action=target_action.copy(),
            )
        )
        cursor += steps
        start_action = target_action.copy()
    return segments


def total_steps(segments: list[StageSegment]) -> int:
    return int(segments[-1].end_step)


def segment_for_step(segments: list[StageSegment], step: int) -> StageSegment:
    clipped = min(max(0, int(step)), total_steps(segments) - 1)
    for segment in segments:
        if segment.start_step <= clipped < segment.end_step:
            return segment
    return segments[-1]


def prior_action_for_step(segments: list[StageSegment], step: int) -> np.ndarray:
    segment = segment_for_step(segments, step)
    local_step = int(step) - int(segment.start_step)
    return interpolate_action(segment.start_action, segment.target_action, local_step, segment.steps).astype(np.float32)


def stage_phase(segment: StageSegment, step: int) -> float:
    return float(np.clip((int(step) - segment.start_step) / max(1, segment.steps - 1), 0.0, 1.0))


def one_hot(index: int, size: int) -> np.ndarray:
    values = np.zeros(int(size), dtype=np.float32)
    values[int(index)] = 1.0
    return values


def residual_feature(
    initial_object: np.ndarray,
    target_position: np.ndarray,
    global_phase: float,
    segment: StageSegment,
    step: int,
) -> np.ndarray:
    initial = np.asarray(initial_object, dtype=np.float32)
    target = np.asarray(target_position, dtype=np.float32)
    return np.concatenate(
        [
            initial,
            target,
            initial - target,
            phase_features(float(global_phase)),
            phase_features(stage_phase(segment, step)),
            one_hot(segment.stage_id, len(STAGES)),
        ]
    ).astype(np.float32)


def load_residual_model(path: Path) -> dict:
    with np.load(path) as data:
        return {
            "weights": data["weights"].astype(np.float32),
            "x_mean": data["x_mean"].astype(np.float32),
            "x_std": data["x_std"].astype(np.float32),
            "residual_min": data["residual_min"].astype(np.float32),
            "residual_max": data["residual_max"].astype(np.float32),
            "action_min": data["action_min"].astype(np.float32),
            "action_max": data["action_max"].astype(np.float32),
            "metadata": json.loads(data["metadata"].item()),
        }


def predict_residual(model: dict, feature: np.ndarray) -> np.ndarray:
    x = ((feature - model["x_mean"]) / model["x_std"]).astype(np.float32)
    x_aug = np.concatenate([x, np.ones(1, dtype=np.float32)])
    residual = x_aug @ model["weights"]
    return np.clip(residual, model["residual_min"], model["residual_max"]).astype(np.float32)
