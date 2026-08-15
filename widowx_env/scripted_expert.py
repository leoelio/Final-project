from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Any, Callable

import mujoco
import numpy as np

from .ik_controller import DampedLeastSquaresIK
from .tabletop_env import WidowXTabletopEnv


StepRecorder = Callable[[np.ndarray, WidowXTabletopEnv], None]


def new_motion_trace(env: WidowXTabletopEnv) -> dict[str, float | int | bool]:
    object_z = float(env.object_position(env.episode_target_object)[2])
    return {
        "initial_object_z": object_z,
        "max_object_z": object_z,
        "lifted_steps_near_tcp": 0,
        "strict_grasp_success": False,
    }


def update_motion_trace(env: WidowXTabletopEnv, trace: dict[str, float | int | bool]) -> None:
    object_position = env.object_position(env.episode_target_object)
    tcp_object_distance = float(np.linalg.norm(env.tcp_position() - object_position))
    trace["max_object_z"] = max(float(trace["max_object_z"]), float(object_position[2]))

    lifted = float(object_position[2]) - float(trace["initial_object_z"]) >= 0.06
    if lifted and tcp_object_distance < 0.06:
        trace["lifted_steps_near_tcp"] = int(trace["lifted_steps_near_tcp"]) + 1
    trace["strict_grasp_success"] = int(trace["lifted_steps_near_tcp"]) >= 50


def capture_state(env: WidowXTabletopEnv) -> dict[str, np.ndarray | float]:
    return {
        "qpos": env.data.qpos.copy(),
        "qvel": env.data.qvel.copy(),
        "ctrl": env.data.ctrl.copy(),
        "time": float(env.data.time),
    }


def restore_state(env: WidowXTabletopEnv, state: dict[str, np.ndarray | float]) -> None:
    env.data.qpos[:] = state["qpos"]
    env.data.qvel[:] = state["qvel"]
    env.data.ctrl[:] = state["ctrl"]
    env.data.time = float(state["time"])
    mujoco.mj_forward(env.model, env.data)


@dataclass(frozen=True)
class PickConfig:
    approach_z_offset: float = 0.12
    grasp_z_offset: float = 0.008
    lift_z_offset: float = 0.18
    ik_tolerance: float = 0.012
    open_gripper: float = 0.037
    close_gripper: float = 0.015
    approach_steps: int = 260
    descend_steps: int = 220
    close_steps: int = 260
    lift_steps: int = 420
    hold_steps: int = 160


@dataclass(frozen=True)
class PickPlaceConfig(PickConfig):
    transfer_z_offset: float = 0.18
    place_tcp_z: float = 0.055
    retreat_z_offset: float = 0.16
    transfer_steps: int = 700
    place_descend_steps: int = 320
    open_steps: int = 220
    retreat_steps: int = 280


def make_arm_action(env: WidowXTabletopEnv, arm_qpos: np.ndarray, gripper: float) -> np.ndarray:
    action = env.home_ctrl.copy()
    action[:6] = arm_qpos[:6]
    action[6] = gripper
    return action


def interpolate_action(start: np.ndarray, end: np.ndarray, step: int, steps: int) -> np.ndarray:
    blend = min(1.0, (step + 1) / max(1, steps))
    return (1.0 - blend) * start + blend * end


class PickOnlyExpert:
    def __init__(self, env: WidowXTabletopEnv, config: PickConfig | None = None) -> None:
        self.env = env
        self.config = config or PickConfig()

    def plan(self, object_name: str) -> dict[str, Any]:
        return self.plan_from_position(self.env.object_position(object_name).copy())

    def plan_from_position(self, object_position: np.ndarray) -> dict[str, Any]:
        """Plan from an externally supplied object position, such as an RGB localization result."""
        cfg = self.config
        ik = DampedLeastSquaresIK(self.env, tolerance=cfg.ik_tolerance)
        state = capture_state(self.env)
        object_position = np.asarray(object_position, dtype=float).copy()

        try:
            approach = ik.solve(object_position + np.array([0.0, 0.0, cfg.approach_z_offset], dtype=float))
            grasp = ik.solve(object_position + np.array([0.0, 0.0, cfg.grasp_z_offset], dtype=float))
            lift = ik.solve(object_position + np.array([0.0, 0.0, cfg.lift_z_offset], dtype=float))

            return {
                "object_position": object_position,
                "approach": approach,
                "grasp": grasp,
                "lift": lift,
                "actions": {
                    "approach": make_arm_action(self.env, approach.qpos, cfg.open_gripper),
                    "grasp_open": make_arm_action(self.env, grasp.qpos, cfg.open_gripper),
                    "grasp_closed": make_arm_action(self.env, grasp.qpos, cfg.close_gripper),
                    "lift_closed": make_arm_action(self.env, lift.qpos, cfg.close_gripper),
                },
            }
        finally:
            restore_state(self.env, state)

    def execute(
        self,
        plan: dict[str, Any],
        viewer=None,
        record_step: StepRecorder | None = None,
        speed: float = 1.0,
    ) -> dict[str, float | bool]:
        cfg = self.config
        actions = plan["actions"]

        motion_trace = new_motion_trace(self.env)
        self._track(actions["approach"], cfg.approach_steps, viewer, record_step, speed, motion_trace)
        self._track(actions["grasp_open"], cfg.descend_steps, viewer, record_step, speed, motion_trace)
        self._track(actions["grasp_closed"], cfg.close_steps, viewer, record_step, speed, motion_trace)
        self._track(actions["lift_closed"], cfg.lift_steps, viewer, record_step, speed, motion_trace)
        self._track(actions["lift_closed"], cfg.hold_steps, viewer, record_step, speed, motion_trace)

        object_name = self.env.episode_target_object
        object_position = self.env.object_position(object_name)
        tcp_position = self.env.tcp_position()
        initial_object_position = plan["object_position"]
        lifted = bool(object_position[2] > 0.08)
        lift_delta = float(object_position[2] - initial_object_position[2])
        tcp_object_distance = float(np.linalg.norm(tcp_position - object_position))
        return {
            "success": lifted and bool(motion_trace["strict_grasp_success"]),
            "object_z": float(object_position[2]),
            "lift_delta": lift_delta,
            "tcp_object_distance": tcp_object_distance,
            "max_object_z": float(motion_trace["max_object_z"]),
            "lifted_steps_near_tcp": int(motion_trace["lifted_steps_near_tcp"]),
            "strict_grasp_success": bool(motion_trace["strict_grasp_success"]),
            "contact_count": float(self.env.data.ncon),
            "out_of_table": bool(object_position[0] < -0.25 or object_position[0] > 0.85 or abs(object_position[1]) > 0.38),
        }

    def _track(
        self,
        target_action: np.ndarray,
        steps: int,
        viewer=None,
        record_step: StepRecorder | None = None,
        speed: float = 1.0,
        motion_trace: dict[str, float | int | bool] | None = None,
    ) -> None:
        start_action = self.env.data.ctrl.copy()
        dt = float(self.env.model.opt.timestep)
        for step in range(steps):
            action = interpolate_action(start_action, target_action, step, steps)
            self.env.step(action)
            if motion_trace is not None:
                update_motion_trace(self.env, motion_trace)
            if record_step is not None:
                record_step(action, self.env)
            if viewer is not None:
                viewer.sync()
                if speed > 0:
                    time.sleep(dt / speed)


class PickPlaceExpert(PickOnlyExpert):
    def __init__(self, env: WidowXTabletopEnv, config: PickPlaceConfig | None = None) -> None:
        super().__init__(env, config or PickPlaceConfig())
        self.config: PickPlaceConfig

    def plan(self, object_name: str, target_geom: str) -> dict[str, Any]:
        return self.plan_from_positions(
            self.env.object_position(object_name).copy(),
            self.env.target_position(target_geom).copy(),
            target_geom=target_geom,
        )

    def plan_from_positions(
        self,
        object_position: np.ndarray,
        target_position: np.ndarray,
        target_geom: str = "vision_target",
    ) -> dict[str, Any]:
        """Plan entirely from supplied object/target positions; MuJoCo state is only used for IK."""
        cfg = self.config
        state = capture_state(self.env)
        plan = super().plan_from_position(object_position)
        ik = DampedLeastSquaresIK(self.env, tolerance=cfg.ik_tolerance)
        target_position = np.asarray(target_position, dtype=float).copy()

        try:
            self.env.data.qpos[:6] = plan["lift"].qpos
            mujoco.mj_forward(self.env.model, self.env.data)

            transfer_target = target_position + np.array([0.0, 0.0, cfg.transfer_z_offset], dtype=float)
            place_target = np.array([target_position[0], target_position[1], cfg.place_tcp_z], dtype=float)
            retreat_target = target_position + np.array([0.0, 0.0, cfg.retreat_z_offset], dtype=float)

            transfer = ik.solve(transfer_target)
            place = ik.solve(place_target)
            retreat = ik.solve(retreat_target)

            plan["target_geom"] = target_geom
            plan["target_position"] = target_position
            plan["transfer"] = transfer
            plan["place"] = place
            plan["retreat"] = retreat
            plan["actions"].update(
                {
                    "transfer_closed": make_arm_action(self.env, transfer.qpos, cfg.close_gripper),
                    "place_closed": make_arm_action(self.env, place.qpos, cfg.close_gripper),
                    "place_open": make_arm_action(self.env, place.qpos, cfg.open_gripper),
                    "retreat_open": make_arm_action(self.env, retreat.qpos, cfg.open_gripper),
                }
            )
            return plan
        finally:
            restore_state(self.env, state)

    def execute(
        self,
        plan: dict[str, Any],
        viewer=None,
        record_step: StepRecorder | None = None,
        speed: float = 1.0,
    ) -> dict[str, float | bool]:
        cfg = self.config
        actions = plan["actions"]

        motion_trace = new_motion_trace(self.env)
        self._track(actions["approach"], cfg.approach_steps, viewer, record_step, speed, motion_trace)
        self._track(actions["grasp_open"], cfg.descend_steps, viewer, record_step, speed, motion_trace)
        self._track(actions["grasp_closed"], cfg.close_steps, viewer, record_step, speed, motion_trace)
        self._track(actions["lift_closed"], cfg.lift_steps, viewer, record_step, speed, motion_trace)
        lift_tcp_object_distance = float(np.linalg.norm(self.env.tcp_position() - self.env.object_position(self.env.episode_target_object)))
        self._track(actions["transfer_closed"], cfg.transfer_steps, viewer, record_step, speed, motion_trace)
        transfer_tcp_object_distance = float(np.linalg.norm(self.env.tcp_position() - self.env.object_position(self.env.episode_target_object)))
        self._track(actions["place_closed"], cfg.place_descend_steps, viewer, record_step, speed, motion_trace)
        place_tcp_object_distance = float(np.linalg.norm(self.env.tcp_position() - self.env.object_position(self.env.episode_target_object)))
        self._track(actions["place_open"], cfg.open_steps, viewer, record_step, speed, motion_trace)
        self._track(actions["retreat_open"], cfg.retreat_steps, viewer, record_step, speed, motion_trace)
        self._track(actions["retreat_open"], cfg.hold_steps, viewer, record_step, speed, motion_trace)

        object_name = self.env.episode_target_object
        object_position = self.env.object_position(object_name)
        target_position = plan["target_position"]
        target_distance = float(np.linalg.norm(object_position[:2] - target_position[:2]))
        placed = bool(target_distance < 0.065 and object_position[2] < 0.08)
        return {
            "success": placed and bool(motion_trace["strict_grasp_success"]),
            "placed": placed,
            "object_z": float(object_position[2]),
            "target_distance": target_distance,
            "tcp_object_distance": float(np.linalg.norm(self.env.tcp_position() - object_position)),
            "max_object_z": float(motion_trace["max_object_z"]),
            "lifted_steps_near_tcp": int(motion_trace["lifted_steps_near_tcp"]),
            "strict_grasp_success": bool(motion_trace["strict_grasp_success"]),
            "held_after_lift": bool(lift_tcp_object_distance < 0.06),
            "held_after_transfer": bool(transfer_tcp_object_distance < 0.06),
            "held_before_release": bool(place_tcp_object_distance < 0.06),
            "lift_tcp_object_distance": lift_tcp_object_distance,
            "transfer_tcp_object_distance": transfer_tcp_object_distance,
            "place_tcp_object_distance": place_tcp_object_distance,
            "contact_count": float(self.env.data.ncon),
            "out_of_table": bool(object_position[0] < -0.25 or object_position[0] > 0.85 or abs(object_position[1]) > 0.38),
        }


@dataclass(frozen=True)
class ContactFusionConfig(PickPlaceConfig):
    """One MuJoCo state-feedback retry for a detected transport-time object drop."""

    close_gripper: float = 0.007
    close_steps: int = 420
    max_regrasp_attempts: int = 1
    hold_distance: float = 0.085
    recovery_close_gripper: float = 0.007
    recovery_close_steps: int = 420


class ContactFusionPickPlaceExpert(PickPlaceExpert):
    """Structured execution with a single contact/proximity-confirmed regrasp attempt."""

    def __init__(self, env: WidowXTabletopEnv, config: ContactFusionConfig | None = None) -> None:
        super().__init__(env, config or ContactFusionConfig())
        self.config: ContactFusionConfig

    def execute(
        self,
        plan: dict[str, Any],
        viewer=None,
        record_step: StepRecorder | None = None,
        speed: float = 1.0,
    ) -> dict[str, float | int | bool]:
        object_name = self.env.episode_target_object
        target_geom = str(plan["target_geom"])
        attempts: list[dict[str, float | int | bool]] = []
        active_plan = plan
        active_config: PickPlaceConfig = self.config

        for attempt_index in range(self.config.max_regrasp_attempts + 1):
            outcome = self._execute_attempt(active_plan, active_config, viewer, record_step, speed)
            attempts.append(outcome)
            if bool(outcome["placed"]):
                return self._summary(active_plan, attempts)
            if attempt_index >= self.config.max_regrasp_attempts:
                break

            # The recovery plan is generated from the observed post-drop state; no simulator reset is used.
            active_config = replace(
                self.config,
                close_gripper=self.config.recovery_close_gripper,
                close_steps=self.config.recovery_close_steps,
            )
            recovery_expert = PickPlaceExpert(self.env, active_config)
            active_plan = recovery_expert.plan(object_name, target_geom)

        return self._summary(active_plan, attempts)

    def _execute_attempt(
        self,
        plan: dict[str, Any],
        config: PickPlaceConfig,
        viewer,
        record_step: StepRecorder | None,
        speed: float,
    ) -> dict[str, float | int | bool]:
        actions = plan["actions"]
        trace = new_motion_trace(self.env)
        self._track(actions["approach"], config.approach_steps, viewer, record_step, speed, trace)
        self._track(actions["grasp_open"], config.descend_steps, viewer, record_step, speed, trace)
        self._track(actions["grasp_closed"], config.close_steps, viewer, record_step, speed, trace)
        self._track(actions["lift_closed"], config.lift_steps, viewer, record_step, speed, trace)
        grasp_confirmed = bool(trace["strict_grasp_success"])
        transport_held = grasp_confirmed and self._track_while_held(
            actions["transfer_closed"], config.transfer_steps, viewer, record_step, speed, trace
        )
        place_held = transport_held and self._track_while_held(
            actions["place_closed"], config.place_descend_steps, viewer, record_step, speed, trace
        )
        if place_held:
            self._track(actions["place_open"], config.open_steps, viewer, record_step, speed, trace)
            self._track(actions["retreat_open"], config.retreat_steps, viewer, record_step, speed, trace)
            self._track(actions["retreat_open"], config.hold_steps, viewer, record_step, speed, trace)

        object_position = self.env.object_position(self.env.episode_target_object)
        target_distance = float(np.linalg.norm(object_position[:2] - plan["target_position"][:2]))
        placed = bool(place_held and target_distance < 0.065 and object_position[2] < 0.08)
        return {
            "placed": placed,
            "grasp_confirmed": grasp_confirmed,
            "transport_held": transport_held,
            "failure_stage": "complete" if placed else ("grasp" if not grasp_confirmed else "transport" if not transport_held else "placement"),
            "max_object_z": float(trace["max_object_z"]),
            "lifted_steps_near_tcp": int(trace["lifted_steps_near_tcp"]),
            "target_distance": target_distance,
        }

    def _track_while_held(
        self,
        target_action: np.ndarray,
        steps: int,
        viewer,
        record_step: StepRecorder | None,
        speed: float,
        trace: dict[str, float | int | bool],
    ) -> bool:
        start_action = self.env.data.ctrl.copy()
        dt = float(self.env.model.opt.timestep)
        for step in range(steps):
            action = interpolate_action(start_action, target_action, step, steps)
            self.env.step(action)
            update_motion_trace(self.env, trace)
            if record_step is not None:
                record_step(action, self.env)
            if step >= 30 and not self._object_is_held():
                return False
            if viewer is not None:
                viewer.sync()
                if speed > 0:
                    time.sleep(dt / speed)
        return True

    def _object_is_held(self) -> bool:
        object_position = self.env.object_position(self.env.episode_target_object)
        return bool(np.linalg.norm(self.env.tcp_position() - object_position) < self.config.hold_distance)

    def _summary(self, plan: dict[str, Any], attempts: list[dict[str, float | int | bool]]) -> dict[str, float | int | bool]:
        object_position = self.env.object_position(self.env.episode_target_object)
        target_distance = float(np.linalg.norm(object_position[:2] - plan["target_position"][:2]))
        placed = bool(target_distance < 0.065 and object_position[2] < 0.08)
        return {
            "success": placed and any(bool(item["grasp_confirmed"]) for item in attempts),
            "placed": placed,
            "object_z": float(object_position[2]),
            "target_distance": target_distance,
            "tcp_object_distance": float(np.linalg.norm(self.env.tcp_position() - object_position)),
            "max_object_z": max(float(item["max_object_z"]) for item in attempts),
            "lifted_steps_near_tcp": sum(int(item["lifted_steps_near_tcp"]) for item in attempts),
            "strict_grasp_success": any(bool(item["grasp_confirmed"]) for item in attempts),
            "contact_regrasp_attempts": max(0, len(attempts) - 1),
            "transport_hold_confirmed": bool(attempts[-1]["transport_held"]),
            "contact_recovery_reason": str(attempts[0]["failure_stage"]),
            "contact_count": float(self.env.data.ncon),
            "out_of_table": bool(object_position[0] < -0.25 or object_position[0] > 0.85 or abs(object_position[1]) > 0.38),
        }
