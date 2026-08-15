from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import sys
import time

import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from widowx_env import TASKS, WidowXTabletopEnv  # noqa: E402
from widowx_env.scripted_expert import PickPlaceConfig, PickPlaceExpert, interpolate_action  # noqa: E402


VERSION = "contact_stage_subpolicy_v1_candidate"


@dataclass
class StageTrace:
    steps_taken: int = 0
    max_object_z: float = 0.0
    max_contact_count: float = 0.0
    min_tcp_object_distance: float | None = None
    min_tcp_object_distance_while_lifted: float | None = None
    first_lift_step: int | None = None
    first_tcp_lift_step: int | None = None
    ever_grasp_success: bool = False
    ever_tcp_lift_success: bool = False
    stage_steps: dict[str, int] = field(default_factory=dict)
    stage_min_tcp_distance: dict[str, float] = field(default_factory=dict)
    stage_max_object_z: dict[str, float] = field(default_factory=dict)
    stage_max_contact_count: dict[str, float] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a contact-traced staged scripted subpolicy.")
    parser.add_argument("--task", choices=sorted(TASKS), default="place_blue_cube_blue_pad")
    parser.add_argument("--complexity", choices=("easy", "medium", "hard", "language"), default="medium")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-viewer", dest="viewer", action="store_false")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--speed", type=float, default=0.05)
    parser.add_argument("--arm-kp", type=float, default=150.0)
    parser.add_argument("--arm-force", type=float, default=100.0)
    parser.add_argument("--gripper-kp", type=float, default=900.0)
    parser.add_argument("--gripper-force", type=float, default=180.0)
    parser.add_argument("--friction", type=float, default=3.0)
    parser.add_argument("--approach-z", type=float, default=0.12)
    parser.add_argument("--grasp-z", type=float, default=0.008)
    parser.add_argument("--lift-z", type=float, default=0.18)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--lift-threshold", type=float, default=0.085)
    parser.add_argument("--tcp-lift-threshold", type=float, default=0.12)
    parser.add_argument("--log-every", type=int, default=500)
    parser.set_defaults(viewer=True)
    return parser.parse_args()


def configure_env(args: argparse.Namespace, seed: int) -> tuple[WidowXTabletopEnv, dict]:
    env = WidowXTabletopEnv(seed=seed)
    env.set_arm_actuator_strength(kp=float(args.arm_kp), force_limit=float(args.arm_force))
    env.set_gripper_actuator_strength(kp=float(args.gripper_kp), force_limit=float(args.gripper_force))
    env.set_grasp_contact_friction(sliding=float(args.friction))
    obs = env.reset(task=str(args.task), complexity=str(args.complexity), seed=seed)
    return env, obs


def make_expert(args: argparse.Namespace, env: WidowXTabletopEnv) -> PickPlaceExpert:
    config = PickPlaceConfig(
        approach_z_offset=float(args.approach_z),
        grasp_z_offset=float(args.grasp_z),
        lift_z_offset=float(args.lift_z),
    )
    return PickPlaceExpert(env, config)


def update_trace(args: argparse.Namespace, trace: StageTrace, env: WidowXTabletopEnv, stage: str) -> None:
    metrics = env.metrics()
    object_position = env.object_position(env.episode_target_object)
    tcp_object_distance = float(np.linalg.norm(env.tcp_position() - object_position))
    object_z = float(metrics["object_z"])
    contact_count = float(metrics["contact_count"])

    trace.steps_taken += 1
    trace.max_object_z = max(trace.max_object_z, object_z)
    trace.max_contact_count = max(trace.max_contact_count, contact_count)
    trace.min_tcp_object_distance = tcp_object_distance if trace.min_tcp_object_distance is None else min(trace.min_tcp_object_distance, tcp_object_distance)
    trace.stage_steps[stage] = int(trace.stage_steps.get(stage, 0)) + 1
    trace.stage_min_tcp_distance[stage] = min(float(trace.stage_min_tcp_distance.get(stage, float("inf"))), tcp_object_distance)
    trace.stage_max_object_z[stage] = max(float(trace.stage_max_object_z.get(stage, -float("inf"))), object_z)
    trace.stage_max_contact_count[stage] = max(float(trace.stage_max_contact_count.get(stage, 0.0)), contact_count)

    lifted = object_z >= float(args.lift_threshold)
    trace.ever_grasp_success = trace.ever_grasp_success or bool(metrics["grasp_success"])
    if lifted:
        if trace.first_lift_step is None:
            trace.first_lift_step = trace.steps_taken
        trace.min_tcp_object_distance_while_lifted = (
            tcp_object_distance
            if trace.min_tcp_object_distance_while_lifted is None
            else min(trace.min_tcp_object_distance_while_lifted, tcp_object_distance)
        )
        if tcp_object_distance < float(args.tcp_lift_threshold):
            trace.ever_tcp_lift_success = True
            if trace.first_tcp_lift_step is None:
                trace.first_tcp_lift_step = trace.steps_taken


def track_stage(
    args: argparse.Namespace,
    env: WidowXTabletopEnv,
    target_action: np.ndarray,
    steps: int,
    stage: str,
    trace: StageTrace,
    viewer=None,
) -> None:
    start_action = env.data.ctrl.copy()
    dt = float(env.model.opt.timestep)
    for index in range(int(steps)):
        action = interpolate_action(start_action, target_action, index, int(steps))
        env.step(action)
        update_trace(args, trace, env, stage)
        if args.log_every > 0 and trace.steps_taken % int(args.log_every) == 0:
            metrics = env.metrics()
            print(
                f"step={trace.steps_taken} stage={stage} success={metrics['success']} "
                f"target_distance={float(metrics['target_distance']):.4f} object_z={float(metrics['object_z']):.4f} "
                f"contact_count={float(metrics['contact_count']):.0f}",
                flush=True,
            )
        if viewer is not None:
            viewer.sync()
            if float(args.speed) > 0:
                time.sleep(dt / float(args.speed))


def execute_attempt(args: argparse.Namespace, env: WidowXTabletopEnv, expert: PickPlaceExpert, object_name: str, trace: StageTrace, viewer=None) -> dict:
    if not env.task.target_geom:
        raise ValueError(f"task {env.task.name!r} does not define a place target")
    plan = expert.plan(object_name, env.task.target_geom)
    actions = plan["actions"]
    cfg = expert.config

    pre_lift_steps = [
        ("approach", actions["approach"], cfg.approach_steps),
        ("descend", actions["grasp_open"], cfg.descend_steps),
        ("close", actions["grasp_closed"], cfg.close_steps),
        ("lift", actions["lift_closed"], cfg.lift_steps),
    ]
    for stage, action, steps in pre_lift_steps:
        track_stage(args, env, action, steps, stage, trace, viewer)

    metrics = env.metrics()
    object_position = env.object_position(env.episode_target_object)
    tcp_object_distance = float(np.linalg.norm(env.tcp_position() - object_position))
    lifted_for_transfer = bool(float(metrics["object_z"]) >= float(args.lift_threshold) and tcp_object_distance < float(args.tcp_lift_threshold))
    if not lifted_for_transfer:
        return {"continued_to_place": False, "success": False, "lifted_for_transfer": False}

    place_steps = [
        ("transfer", actions["transfer_closed"], cfg.transfer_steps),
        ("place_descend", actions["place_closed"], cfg.place_descend_steps),
        ("release", actions["place_open"], cfg.open_steps),
        ("retreat", actions["retreat_open"], cfg.retreat_steps),
        ("hold", actions["retreat_open"], cfg.hold_steps),
    ]
    for stage, action, steps in place_steps:
        track_stage(args, env, action, steps, stage, trace, viewer)

    return {"continued_to_place": True, "success": bool(env.metrics()["success"]), "lifted_for_transfer": True}


def summarize_trace(args: argparse.Namespace, env: WidowXTabletopEnv, obs: dict, trace: StageTrace, attempts: int, attempt_summary: dict) -> dict:
    metrics = env.metrics()
    return {
        "version": VERSION,
        "seed": int(obs.get("seed", args.seed)),
        "task": str(args.task),
        "complexity": str(args.complexity),
        "instruction": str(obs["instruction"]),
        "active_objects": list(obs["active_objects"]),
        "target_object": str(obs["target_object"]),
        "success": bool(metrics["success"]),
        "target_distance": float(metrics["target_distance"]),
        "object_z": float(metrics["object_z"]),
        "max_object_z": float(trace.max_object_z),
        "height_threshold_hit": bool(trace.max_object_z >= float(args.lift_threshold)),
        "ever_grasp_success": bool(trace.ever_grasp_success),
        "tcp_grasp_lift_success": bool(trace.ever_tcp_lift_success and trace.max_object_z >= float(args.lift_threshold)),
        "strict_grasp_lift_success": bool(trace.ever_grasp_success and trace.max_object_z >= float(args.lift_threshold)),
        "grasp_success": bool(metrics["grasp_success"]),
        "out_of_table": bool(metrics["out_of_table"]),
        "contact_count": float(metrics["contact_count"]),
        "max_contact_count": float(trace.max_contact_count),
        "min_tcp_object_distance": trace.min_tcp_object_distance,
        "min_tcp_object_distance_while_lifted": trace.min_tcp_object_distance_while_lifted,
        "attempts": int(attempts),
        "continued_to_place": bool(attempt_summary.get("continued_to_place", False)),
        "steps_taken": int(trace.steps_taken),
        "first_lift_step": trace.first_lift_step,
        "first_tcp_lift_step": trace.first_tcp_lift_step,
        "stage_steps": {key: int(value) for key, value in trace.stage_steps.items()},
        "stage_min_tcp_distance": {key: float(value) for key, value in trace.stage_min_tcp_distance.items()},
        "stage_max_object_z": {key: float(value) for key, value in trace.stage_max_object_z.items()},
        "stage_max_contact_count": {key: float(value) for key, value in trace.stage_max_contact_count.items()},
    }


def rollout_with_env(args: argparse.Namespace, env: WidowXTabletopEnv, obs: dict, viewer=None) -> dict:
    if env.task.kind != "place":
        raise ValueError("contact-stage subpolicy currently targets place tasks")
    expert = make_expert(args, env)
    object_name = str(obs["target_object"])
    trace = StageTrace()
    attempt_summary: dict = {"success": False}
    attempts = 0
    for attempt in range(1, max(0, int(args.retries)) + 2):
        attempts = attempt
        attempt_summary = execute_attempt(args, env, expert, object_name, trace, viewer)
        if bool(env.metrics()["success"]):
            break
    return summarize_trace(args, env, obs, trace, attempts, attempt_summary)


def run_episode(args: argparse.Namespace, seed: int, viewer=None) -> dict:
    env, obs = configure_env(args, seed)
    obs = dict(obs)
    obs["seed"] = seed
    summary = rollout_with_env(args, env, obs, viewer)
    print("episode_summary:", summary, flush=True)
    return summary


def main() -> None:
    args = parse_args()
    summaries = []
    for episode in range(int(args.episodes)):
        seed = int(args.seed) + episode
        if args.viewer and episode == 0:
            env, obs = configure_env(args, seed)
            obs = dict(obs)
            obs["seed"] = seed
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                summary = rollout_with_env(args, env, obs, viewer)
                start = time.time()
                while viewer.is_running():
                    viewer.sync()
                    if args.duration and time.time() - start > float(args.duration):
                        break
                    time.sleep(0.01)
        else:
            summary = run_episode(args, seed, viewer=None)
        summaries.append(summary)
    successes = sum(1 for item in summaries if item["success"])
    print(f"success_rate: {successes}/{len(summaries)} = {successes / max(1, len(summaries)):.3f}", flush=True)


if __name__ == "__main__":
    main()
