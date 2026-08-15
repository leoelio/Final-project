from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_bc_policy import postprocess_action, print_step, unsafe_reason  # noqa: E402
from run_contact_stage_subpolicy import StageTrace, summarize_trace, update_trace  # noqa: E402
from trajectory_prior_residual_common import (  # noqa: E402
    VERSION,
    build_plan,
    build_segments,
    load_residual_model,
    make_config,
    predict_residual,
    prior_action_for_step,
    residual_feature,
    segment_for_step,
    total_steps,
)
from widowx_env import TASKS, WidowXTabletopEnv  # noqa: E402
from widowx_env.demo_dataset import read_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trajectory-prior residual BC in MuJoCo.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--task", choices=sorted(TASKS), default=None)
    parser.add_argument("--complexity", choices=("easy", "medium", "hard", "language"), default=None)
    parser.add_argument("--workspace-profile", choices=("legacy", "core_v2"), default="legacy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2840)
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
    parser.add_argument("--place-tcp-z", type=float, default=0.055)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--clip-actions", action="store_true")
    parser.add_argument("--no-clip-actions", dest="clip_actions", action="store_false")
    parser.add_argument("--action-alpha", type=float, default=1.0)
    parser.add_argument("--max-arm-delta", type=float, default=0.02)
    parser.add_argument("--max-gripper-delta", type=float, default=0.0008)
    parser.add_argument("--require-lift-before-transfer", action="store_true")
    parser.add_argument("--no-require-lift-before-transfer", dest="require_lift_before_transfer", action="store_false")
    parser.add_argument("--lift-threshold", type=float, default=0.085)
    parser.add_argument("--tcp-lift-threshold", type=float, default=0.12)
    parser.add_argument("--stop-on-unsafe", action="store_true")
    parser.add_argument("--no-stop-on-unsafe", dest="stop_on_unsafe", action="store_false")
    parser.add_argument("--log-every", type=int, default=500)
    parser.set_defaults(viewer=True)
    parser.set_defaults(clip_actions=True)
    parser.set_defaults(stop_on_unsafe=True)
    parser.set_defaults(require_lift_before_transfer=True)
    return parser.parse_args()


def latest_model() -> Path:
    candidates = sorted(
        (ROOT / "outputs" / "trajectory_prior_residual_bc").glob("trajectory_prior_residual_bc_v1_candidate_*.npz"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError("no trajectory-prior residual BC model found")
    return candidates[-1]


def infer_task_defaults(model: dict) -> tuple[str, str]:
    first = read_metadata(Path(model["metadata"]["run_dir"]))[0]
    return str(first["task"]), str(first["complexity"])


def configure_env(args: argparse.Namespace, seed: int, task: str, complexity: str) -> tuple[WidowXTabletopEnv, dict]:
    env = WidowXTabletopEnv(seed=seed, workspace_profile=args.workspace_profile)
    env.set_arm_actuator_strength(kp=float(args.arm_kp), force_limit=float(args.arm_force))
    env.set_gripper_actuator_strength(kp=float(args.gripper_kp), force_limit=float(args.gripper_force))
    env.set_grasp_contact_friction(sliding=float(args.friction))
    obs = env.reset(task=task, complexity=complexity, seed=seed)
    return env, obs


def lifted_for_transfer(args: argparse.Namespace, env: WidowXTabletopEnv) -> bool:
    metrics = env.metrics()
    object_position = env.object_position(env.episode_target_object)
    tcp_object_distance = float(np.linalg.norm(env.tcp_position() - object_position))
    return bool(float(metrics["object_z"]) >= float(args.lift_threshold) and tcp_object_distance < float(args.tcp_lift_threshold))


def rollout_with_env(
    args: argparse.Namespace,
    model: dict,
    env: WidowXTabletopEnv,
    obs: dict,
    seed: int,
    task: str,
    complexity: str,
    viewer=None,
) -> dict:
    if env.task.kind != "place" or not env.task.target_geom:
        raise ValueError("trajectory-prior residual policy currently requires a place task")

    config = make_config(args.approach_z, args.grasp_z, args.lift_z, args.place_tcp_z)
    plan = build_plan(env, str(obs["target_object"]), env.task.target_geom, config)
    segments = build_segments(env, plan, config)
    prior_steps = total_steps(segments)
    steps = min(int(args.steps), prior_steps)
    target_position = env.target_position(env.task.target_geom).astype(np.float32)
    initial_object = env.object_position(env.episode_target_object).astype(np.float32)
    ctrl_min = env.model.actuator_ctrlrange[:, 0]
    ctrl_max = env.model.actuator_ctrlrange[:, 1]
    dt = float(env.model.opt.timestep)
    previous_action = env.data.ctrl.copy()
    trace = StageTrace()
    action_norms: list[float] = []
    residual_norms: list[float] = []
    stop_reason = None
    continued_to_place = False

    for step in range(steps):
        segment = segment_for_step(segments, step)
        phase = step / max(1, prior_steps - 1)
        base_action = prior_action_for_step(segments, step)
        feature = residual_feature(initial_object, target_position, phase, segment, step)
        residual = predict_residual(model, feature) * float(args.residual_scale)
        raw_action = (base_action + residual).astype(np.float32)
        action = postprocess_action(args, model, raw_action, previous_action, ctrl_min, ctrl_max)
        delta = action - previous_action
        env.step(action)
        previous_action = action
        action_norms.append(float(np.linalg.norm(action)))
        residual_norms.append(float(np.linalg.norm(residual)))
        update_trace(args, trace, env, segment.stage)
        metrics = env.metrics()
        if segment.stage in {"transfer", "place_descend", "release", "retreat", "hold"}:
            continued_to_place = True
        if args.log_every > 0 and (step == 0 or (step + 1) % int(args.log_every) == 0):
            print_step(step + 1, metrics, action, delta)
            print(f"stage={segment.stage} residual_norm={float(np.linalg.norm(residual)):.6f}", flush=True)
        if (
            bool(args.require_lift_before_transfer)
            and segment.stage == "lift"
            and step + 1 == segment.end_step
            and not lifted_for_transfer(args, env)
        ):
            stop_reason = "lift gate failed before transfer"
            print(f"stopped_early: step={step + 1}, reason={stop_reason}", flush=True)
            break
        stop_reason = unsafe_reason(metrics) if args.stop_on_unsafe else None
        if stop_reason:
            print(f"stopped_early: step={step + 1}, reason={stop_reason}", flush=True)
            break
        if viewer is not None:
            viewer.sync()
            if float(args.speed) > 0:
                time.sleep(dt / float(args.speed))

    summary = summarize_trace(
        args,
        env,
        {**dict(obs), "seed": seed},
        trace,
        attempts=1,
        attempt_summary={
            "continued_to_place": continued_to_place,
            "success": bool(env.metrics()["success"]),
            "lifted_for_transfer": lifted_for_transfer(args, env),
        },
    )
    summary.update(
        {
            "version": str(model["metadata"].get("version", VERSION)),
            "steps_taken": int(trace.steps_taken),
            "stop_reason": stop_reason,
            "mean_action_norm": float(np.mean(action_norms)) if action_norms else 0.0,
            "max_action_norm": float(np.max(action_norms)) if action_norms else 0.0,
            "mean_residual_norm": float(np.mean(residual_norms)) if residual_norms else 0.0,
            "max_residual_norm": float(np.max(residual_norms)) if residual_norms else 0.0,
            "residual_scale": float(args.residual_scale),
            "task": task,
            "complexity": complexity,
        }
    )
    return summary


def run_episode(args: argparse.Namespace, model: dict, seed: int, task: str, complexity: str, viewer=None) -> dict:
    env, obs = configure_env(args, seed, task, complexity)
    summary = rollout_with_env(args, model, env, obs, seed, task, complexity, viewer)
    print("episode_summary:", summary, flush=True)
    return summary


def main() -> None:
    args = parse_args()
    model_path = args.model or latest_model()
    model = load_residual_model(model_path)
    default_task, default_complexity = infer_task_defaults(model)
    task = args.task or default_task
    complexity = args.complexity or default_complexity

    print(f"model_path: {model_path}", flush=True)
    print(f"model_train_run: {model['metadata']['run_dir']}", flush=True)
    print(f"source_episodes: {model['metadata']['source_episodes']}", flush=True)
    print(f"samples: {model['metadata']['samples']}", flush=True)
    print(f"task: {task}", flush=True)
    print(f"complexity: {complexity}", flush=True)

    summaries = []
    if args.viewer:
        env, obs = configure_env(args, int(args.seed), task, complexity)
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            summary = rollout_with_env(args, model, env, obs, int(args.seed), task, complexity, viewer)
            summaries.append(summary)
            print("episode_summary:", summary, flush=True)
            start = time.time()
            while viewer.is_running():
                viewer.sync()
                if args.duration and time.time() - start > float(args.duration):
                    break
                time.sleep(0.01)
        start_episode = 1
    else:
        start_episode = 0

    for episode in range(start_episode, int(args.episodes)):
        seed = int(args.seed) + episode
        summaries.append(run_episode(args, model, seed, task, complexity))

    successes = sum(1 for item in summaries if item["success"])
    tcp_lifts = sum(1 for item in summaries if item["tcp_grasp_lift_success"])
    print(f"success_rate: {successes}/{len(summaries)} = {successes / max(1, len(summaries)):.3f}", flush=True)
    print(f"tcp_grasp_lift_rate: {tcp_lifts}/{len(summaries)} = {tcp_lifts / max(1, len(summaries)):.3f}", flush=True)


if __name__ == "__main__":
    main()
