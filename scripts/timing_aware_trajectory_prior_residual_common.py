from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trajectory_prior_residual_common import (
    StageSegment,
    build_plan,
    load_residual_model,
    make_config,
    one_hot,
    predict_residual,
    prior_action_for_step,
    segment_for_step,
    stage_phase,
    total_steps,
)
from widowx_env import WidowXTabletopEnv
from widowx_env.demo_dataset import phase_features
from widowx_env.scripted_expert import PickPlaceConfig


VERSION = "timing_aware_trajectory_prior_residual_bc_v1_candidate"


@dataclass(frozen=True)
class TimingAwareStage:
    stage: str
    action_key: str
    steps: int | str
    gripper: float | None = None


TIMING_STAGES = (
    TimingAwareStage("approach", "approach", "approach_steps"),
    TimingAwareStage("descend", "grasp_open", "descend_steps", 0.037),
    TimingAwareStage("close", "grasp_closed", 420, 0.004),
    TimingAwareStage("close_hold", "grasp_closed", 280, 0.004),
    TimingAwareStage("pre_lift", "lift_closed", "lift_steps", 0.004),
    TimingAwareStage("lift", "lift_closed", 220, 0.004),
    TimingAwareStage("transfer", "transfer_closed", "transfer_steps", 0.004),
    TimingAwareStage("place_descend", "place_closed", "place_descend_steps", 0.004),
    TimingAwareStage("release", "place_open", "open_steps"),
    TimingAwareStage("retreat", "retreat_open", "retreat_steps"),
    TimingAwareStage("hold", "retreat_open", "hold_steps"),
)


def with_gripper(action: np.ndarray, value: float | None) -> np.ndarray:
    output = action.astype(np.float32).copy()
    if value is not None:
        output[6] = float(value)
    return output


def stage_steps(config: PickPlaceConfig, spec: TimingAwareStage) -> int:
    if isinstance(spec.steps, str):
        return int(getattr(config, spec.steps))
    return int(spec.steps)


def build_segments(env: WidowXTabletopEnv, plan: dict, config: PickPlaceConfig) -> list[StageSegment]:
    segments: list[StageSegment] = []
    start_action = env.data.ctrl.copy().astype(np.float32)
    cursor = 0
    for stage_id, spec in enumerate(TIMING_STAGES):
        steps = stage_steps(config, spec)
        target_action = with_gripper(plan["actions"][spec.action_key], spec.gripper)
        segments.append(
            StageSegment(
                stage=spec.stage,
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
            one_hot(segment.stage_id, len(TIMING_STAGES)),
        ]
    ).astype(np.float32)
