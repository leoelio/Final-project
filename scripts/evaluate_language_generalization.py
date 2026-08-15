from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_bc_policy  # noqa: E402
import run_chunk_policy  # noqa: E402
import run_diffusion_policy  # noqa: E402
import run_knn_policy  # noqa: E402
import run_mlp_policy  # noqa: E402
import run_object_action_head  # noqa: E402
import run_peft_action_head  # noqa: E402
import run_phase_action_head  # noqa: E402
import run_structured_waypoint_policy  # noqa: E402
import run_trajectory_prior_residual_policy  # noqa: E402
import run_torch_act_cvae_policy  # noqa: E402
import run_torch_act_policy  # noqa: E402
import run_torch_diffusion_policy  # noqa: E402
import run_trajectory_knn_policy  # noqa: E402
import run_visual_feature_act_policy  # noqa: E402
import run_vision_language_action_head  # noqa: E402
import run_clip_action_head  # noqa: E402
from widowx_env import TASKS, WidowXTabletopEnv  # noqa: E402
from widowx_env.scripted_expert import PickPlaceConfig, PickPlaceExpert  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate language/spatial generalization on MuJoCo tabletop tasks.")
    parser.add_argument("--task", choices=sorted(TASKS), default="move_leftmost_to_bowl")
    parser.add_argument("--complexity", choices=("easy", "medium", "hard", "language"), default="language")
    parser.add_argument("--workspace-profile", choices=("legacy", "core_v2"), default="legacy")
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=2840)
    parser.add_argument("--methods", default="expert,structured_waypoint_policy,linear_bc,knn_bc,mlp_bc,act_lite,trajectory_chunk,trajectory_knn,torch_act,torch_act_cuda,torch_act_cvae,torch_diffusion_policy,visual_feature_act,object_action_head,reward_weighted_action_head,phase_action_head,adapter_action_head,lora_action_head,vision_language_action_head,clip_action_head,multi_task_object_action_head,diffusion_policy")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "language_generalization_summary.csv")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--arm-kp", type=float, default=150.0)
    parser.add_argument("--arm-force", type=float, default=100.0)
    parser.add_argument("--gripper-kp", type=float, default=800.0)
    parser.add_argument("--gripper-force", type=float, default=140.0)
    parser.add_argument("--friction", type=float, default=3.0)
    parser.add_argument("--place-tcp-z", type=float, default=0.055)
    parser.add_argument(
        "--structured-model",
        type=Path,
        default=ROOT / "outputs" / "structured_waypoint_policy" / "structured_waypoint_policy_20260720_065456.npz",
    )
    parser.add_argument("--structured-version", default="structured_waypoint_policy_v1")
    parser.add_argument(
        "--linear-model",
        type=Path,
        default=ROOT / "outputs" / "bc" / "bc_linear_20260702_051909.npz",
    )
    parser.add_argument("--linear-version", default="linear_bc_v1")
    parser.add_argument(
        "--knn-model",
        type=Path,
        default=ROOT / "outputs" / "knn_bc" / "knn_bc_20260702_051907.npz",
    )
    parser.add_argument("--knn-version", default="knn_bc_v1")
    parser.add_argument(
        "--trajectory-knn-model",
        type=Path,
        default=ROOT / "outputs" / "trajectory_knn_bc" / "trajectory_knn_chunk_bc_20260720_053423.npz",
    )
    parser.add_argument("--trajectory-knn-version", default="trajectory_knn_chunk_bc_v1")
    parser.add_argument(
        "--object-action-head-model",
        type=Path,
        default=ROOT / "outputs" / "object_action_head" / "object_action_head_lite_20260720_044703.npz",
    )
    parser.add_argument("--object-action-head-version", default="object_language_action_head_lite_v1")
    parser.add_argument("--object-action-alpha", type=float, default=0.2)
    parser.add_argument("--object-max-arm-delta", type=float, default=0.01)
    parser.add_argument("--object-max-gripper-delta", type=float, default=0.0005)
    parser.add_argument(
        "--clip-core-v2-model",
        type=Path,
        default=ROOT / "outputs" / "clip_action_head" / "clip_core_v2_multitask_v1_20260721_104743.npz",
    )
    parser.add_argument("--clip-core-v2-version", default="clip_core_v2_multitask_v1")
    parser.add_argument("--clip-core-v2-action-alpha", type=float, default=0.2)
    parser.add_argument("--clip-core-v2-max-arm-delta", type=float, default=0.01)
    parser.add_argument("--clip-core-v2-max-gripper-delta", type=float, default=0.0005)
    parser.add_argument("--clip-core-v2-vision-interval", type=int, default=64)
    parser.add_argument(
        "--trajectory-prior-residual-model",
        type=Path,
        default=ROOT / "outputs" / "trajectory_prior_residual_bc" / "trajectory_prior_residual_bc_v1_candidate_20260721_040050.npz",
    )
    parser.add_argument("--trajectory-prior-residual-version", default="trajectory_prior_residual_bc_v1_candidate")
    parser.add_argument("--trajectory-prior-residual-scale", type=float, default=0.25)
    return parser.parse_args()


def base_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        seed=args.seed,
        episodes=args.episodes,
        steps=args.steps,
        viewer=False,
        duration=0.0,
        speed=0.0,
        arm_kp=args.arm_kp,
        arm_force=args.arm_force,
        gripper_kp=args.gripper_kp,
        gripper_force=args.gripper_force,
        friction=args.friction,
        workspace_profile=args.workspace_profile,
        clip_actions=True,
        stop_on_unsafe=True,
        log_every=args.log_every,
    )


def configure_env(args: argparse.Namespace, seed: int, task: str, complexity: str) -> tuple[WidowXTabletopEnv, dict]:
    env = WidowXTabletopEnv(seed=seed, workspace_profile=args.workspace_profile)
    env.set_arm_actuator_strength(kp=args.arm_kp, force_limit=args.arm_force)
    env.set_gripper_actuator_strength(kp=args.gripper_kp, force_limit=args.gripper_force)
    env.set_grasp_contact_friction(sliding=args.friction)
    obs = env.reset(task=task, complexity=complexity, seed=seed)
    return env, obs


def run_expert_episode(args: argparse.Namespace, seed: int) -> dict:
    env, obs = configure_env(args, seed, args.task, args.complexity)
    expert = PickPlaceExpert(env, PickPlaceConfig(place_tcp_z=args.place_tcp_z))
    object_name = obs["target_object"]
    plan = expert.plan(object_name, env.task.target_geom)
    summary = expert.execute(plan, speed=0.0)
    metrics = env.metrics()
    return {
        "seed": seed,
        "task": args.task,
        "complexity": args.complexity,
        "instruction": obs["instruction"],
        "target_object": object_name,
        "active_objects": list(obs["active_objects"]),
        "success": bool(summary["success"]),
        "target_distance": float(summary.get("target_distance", metrics["target_distance"])),
        "object_z": float(summary.get("object_z", metrics["object_z"])),
        "grasp_success": bool(metrics["grasp_success"]),
        "out_of_table": bool(summary["out_of_table"]),
        "steps_taken": int(sum((260, 220, 260, 420, 700, 320, 220, 280, 160))),
        "stop_reason": None,
        "mean_action_norm": None,
        "max_action_norm": None,
        "metrics": metrics,
    }


def run_policy_episode(
    args: argparse.Namespace,
    seed: int,
    load_model: Callable[[Path], dict],
    rollout: Callable,
    model_path: Path,
    policy_args: SimpleNamespace,
) -> dict:
    env, obs = configure_env(args, seed, args.task, args.complexity)
    model = load_model(model_path)
    result = rollout(policy_args, model, env, obs, seed, args.task, args.complexity)
    result["target_object"] = obs["target_object"]
    return result


def rollout_linear(policy_args: SimpleNamespace, model: dict, env: WidowXTabletopEnv, obs: dict, seed: int, task: str, complexity: str) -> dict:
    return run_bc_policy.rollout_with_env(policy_args, model, env, seed, task, complexity, None)


def load_clip_policy(model_path: Path) -> dict:
    cache = getattr(load_clip_policy, "_cache", {})
    key = str(model_path.resolve())
    if key not in cache:
        policy = run_clip_action_head.load_policy(model_path)
        clip_model, processor = run_clip_action_head.load_clip(str(policy["metadata"]["clip_model"]))
        cache[key] = {"policy": policy, "clip_model": clip_model, "processor": processor}
        setattr(load_clip_policy, "_cache", cache)
    return cache[key]


def rollout_clip_action_head(policy_args: SimpleNamespace, model: dict, env: WidowXTabletopEnv, obs: dict, seed: int, task: str, complexity: str) -> dict:
    return run_clip_action_head.rollout_with_env(
        policy_args,
        model["policy"],
        model["clip_model"],
        model["processor"],
        env,
        obs,
        seed,
        task,
        complexity,
    )


def method_specs(args: argparse.Namespace) -> dict[str, dict]:
    common = base_args(args)

    linear_args = SimpleNamespace(**vars(common))
    linear_args.action_alpha = 0.2
    linear_args.max_arm_delta = 0.01
    linear_args.max_gripper_delta = 0.0005

    knn_args = SimpleNamespace(**vars(common))
    knn_args.action_alpha = 1.0
    knn_args.max_arm_delta = 0.05
    knn_args.max_gripper_delta = 0.002
    knn_args.k = 3
    knn_args.phase_window = 0.02
    knn_args.min_candidates = 128
    knn_args.qpos_weight = 0.25
    knn_args.qvel_weight = 0.05
    knn_args.ctrl_weight = 0.25
    knn_args.tcp_weight = 1.0
    knn_args.object_weight = 4.0
    knn_args.target_weight = 1.0
    knn_args.phase_weight = 2.0

    mlp_args = SimpleNamespace(**vars(common))
    mlp_args.action_alpha = 0.2
    mlp_args.max_arm_delta = 0.01
    mlp_args.max_gripper_delta = 0.0005

    act_args = SimpleNamespace(**vars(common))
    act_args.action_alpha = 0.9
    act_args.max_arm_delta = 0.05
    act_args.max_gripper_delta = 0.002
    act_args.replan_interval = 1
    act_args.temporal_ensemble = True
    act_args.ensemble_decay = 0.1

    trajectory_args = SimpleNamespace(**vars(common))
    trajectory_args.action_alpha = 0.25
    trajectory_args.max_arm_delta = 0.012
    trajectory_args.max_gripper_delta = 0.0005
    trajectory_args.replan_interval = 1
    trajectory_args.temporal_ensemble = True
    trajectory_args.ensemble_decay = 0.1

    trajectory_knn_args = SimpleNamespace(**vars(common))
    trajectory_knn_args.action_alpha = 0.85
    trajectory_knn_args.max_arm_delta = 0.04
    trajectory_knn_args.max_gripper_delta = 0.0015
    trajectory_knn_args.k = 3
    trajectory_knn_args.phase_window = 0.03
    trajectory_knn_args.min_candidates = 256
    trajectory_knn_args.history_decay = 0.25
    trajectory_knn_args.replan_interval = 1
    trajectory_knn_args.temporal_ensemble = True
    trajectory_knn_args.ensemble_decay = 0.1

    torch_act_args = SimpleNamespace(**vars(common))
    torch_act_args.action_alpha = 0.25
    torch_act_args.max_arm_delta = 0.012
    torch_act_args.max_gripper_delta = 0.0005
    torch_act_args.replan_interval = 4
    torch_act_args.temporal_ensemble = True
    torch_act_args.ensemble_decay = 0.1

    torch_act_cvae_args = SimpleNamespace(**vars(common))
    torch_act_cvae_args.action_alpha = 0.25
    torch_act_cvae_args.max_arm_delta = 0.012
    torch_act_cvae_args.max_gripper_delta = 0.0005
    torch_act_cvae_args.replan_interval = 4
    torch_act_cvae_args.temporal_ensemble = True
    torch_act_cvae_args.ensemble_decay = 0.1

    torch_diffusion_args = SimpleNamespace(**vars(common))
    torch_diffusion_args.action_alpha = 0.25
    torch_diffusion_args.max_arm_delta = 0.012
    torch_diffusion_args.max_gripper_delta = 0.0005
    torch_diffusion_args.replan_interval = 8
    torch_diffusion_args.temporal_ensemble = True
    torch_diffusion_args.ensemble_decay = 0.1
    torch_diffusion_args.deterministic = True
    torch_diffusion_args.sample_steps = 4

    visual_feature_act_args = SimpleNamespace(**vars(common))
    visual_feature_act_args.action_alpha = 0.25
    visual_feature_act_args.max_arm_delta = 0.012
    visual_feature_act_args.max_gripper_delta = 0.0005
    visual_feature_act_args.replan_interval = 4
    visual_feature_act_args.temporal_ensemble = True
    visual_feature_act_args.ensemble_decay = 0.1

    diffusion_args = SimpleNamespace(**vars(common))
    diffusion_args.action_alpha = 0.8
    diffusion_args.max_arm_delta = 0.03
    diffusion_args.max_gripper_delta = 0.001
    diffusion_args.replan_interval = 1
    diffusion_args.temporal_ensemble = True
    diffusion_args.ensemble_decay = 0.1
    diffusion_args.deterministic = True

    object_args = SimpleNamespace(**vars(common))
    object_args.action_alpha = args.object_action_alpha
    object_args.max_arm_delta = args.object_max_arm_delta
    object_args.max_gripper_delta = args.object_max_gripper_delta

    phase_args = SimpleNamespace(**vars(common))
    phase_args.action_alpha = 0.2
    phase_args.max_arm_delta = 0.01
    phase_args.max_gripper_delta = 0.0005
    phase_args.phase_mode = "progress"

    clip_args = SimpleNamespace(**vars(object_args))
    clip_args.image_size = 224
    clip_args.vision_interval = 64

    clip_core_v2_args = SimpleNamespace(**vars(common))
    clip_core_v2_args.action_alpha = args.clip_core_v2_action_alpha
    clip_core_v2_args.max_arm_delta = args.clip_core_v2_max_arm_delta
    clip_core_v2_args.max_gripper_delta = args.clip_core_v2_max_gripper_delta
    clip_core_v2_args.image_size = 224
    clip_core_v2_args.vision_interval = args.clip_core_v2_vision_interval

    prior_residual_args = SimpleNamespace(**vars(common))
    prior_residual_args.task = args.task
    prior_residual_args.complexity = args.complexity
    prior_residual_args.approach_z = 0.12
    prior_residual_args.grasp_z = 0.008
    prior_residual_args.lift_z = 0.18
    prior_residual_args.place_tcp_z = args.place_tcp_z
    prior_residual_args.residual_scale = args.trajectory_prior_residual_scale
    prior_residual_args.clip_actions = False
    prior_residual_args.action_alpha = 1.0
    prior_residual_args.max_arm_delta = 0.02
    prior_residual_args.max_gripper_delta = 0.0008
    prior_residual_args.require_lift_before_transfer = True
    prior_residual_args.lift_threshold = 0.085
    prior_residual_args.tcp_lift_threshold = 0.12

    return {
        "expert": {"version": "expert_scripted_language_v1", "stage": "language_oracle", "runner": "expert"},
        "linear_bc": {
            "version": args.linear_version,
            "stage": "weak_bc_baseline",
            "model": args.linear_model,
            "load": run_bc_policy.load_model,
            "rollout": rollout_linear,
            "args": linear_args,
        },
        "structured_waypoint_policy": {
            "version": args.structured_version,
            "stage": "structured_control_baseline",
            "model": args.structured_model,
            "load": run_structured_waypoint_policy.load_model,
            "rollout": run_structured_waypoint_policy.rollout_with_env,
            "args": common,
        },
        "knn_bc": {
            "version": args.knn_version,
            "stage": "non_neural_baseline",
            "model": args.knn_model,
            "load": run_knn_policy.load_model,
            "rollout": run_knn_policy.rollout_with_env,
            "args": knn_args,
        },
        "mlp_bc": {
            "version": "mlp_bc_v1",
            "stage": "neural_bc_baseline",
            "model": ROOT / "outputs" / "mlp_bc" / "mlp_bc_20260702_053322.npz",
            "load": run_mlp_policy.load_model,
            "rollout": run_mlp_policy.rollout_with_env,
            "args": mlp_args,
        },
        "act_lite": {
            "version": "act_lite_chunk_bc_v1",
            "stage": "trajectory_conditioned_baseline",
            "model": ROOT / "outputs" / "chunk_bc" / "chunk_bc_20260702_072710.npz",
            "load": run_chunk_policy.load_model,
            "rollout": run_chunk_policy.rollout_with_env,
            "args": act_args,
        },
        "trajectory_chunk": {
            "version": "trajectory_conditioned_chunk_bc_v2",
            "stage": "trajectory_conditioned_baseline",
            "model": ROOT / "outputs" / "chunk_bc" / "trajectory_chunk_bc_20260720_043500.npz",
            "load": run_chunk_policy.load_model,
            "rollout": run_chunk_policy.rollout_with_env,
            "args": trajectory_args,
        },
        "trajectory_knn": {
            "version": args.trajectory_knn_version,
            "stage": "trajectory_memory_baseline",
            "model": args.trajectory_knn_model,
            "load": run_trajectory_knn_policy.load_model,
            "rollout": run_trajectory_knn_policy.rollout_with_env,
            "args": trajectory_knn_args,
        },
        "torch_act": {
            "version": "torch_act_state_chunk_v1",
            "stage": "torch_act_baseline",
            "model": ROOT / "outputs" / "torch_act" / "torch_act_state_chunk_20260720_055409.pt",
            "load": run_torch_act_policy.load_model,
            "rollout": run_torch_act_policy.rollout_with_env,
            "args": torch_act_args,
        },
        "torch_act_cuda": {
            "version": "torch_act_state_chunk_cuda_v1",
            "stage": "torch_act_baseline",
            "model": ROOT / "outputs" / "torch_act" / "torch_act_state_chunk_cuda_20260720_095442.pt",
            "load": run_torch_act_policy.load_model,
            "rollout": run_torch_act_policy.rollout_with_env,
            "args": torch_act_args,
        },
        "torch_act_cvae": {
            "version": "torch_act_cvae_state_chunk_v1",
            "stage": "torch_act_cvae_baseline",
            "model": ROOT / "outputs" / "torch_act_cvae" / "torch_act_cvae_state_chunk_20260720_084842.pt",
            "load": run_torch_act_cvae_policy.load_model,
            "rollout": run_torch_act_cvae_policy.rollout_with_env,
            "args": torch_act_cvae_args,
        },
        "torch_diffusion_policy": {
            "version": "torch_diffusion_policy_state_chunk_v1",
            "stage": "torch_diffusion_policy_baseline",
            "model": ROOT / "outputs" / "torch_diffusion_policy" / "torch_diffusion_policy_state_chunk_20260720_101928.pt",
            "load": run_torch_diffusion_policy.load_model,
            "rollout": run_torch_diffusion_policy.rollout_with_env,
            "args": torch_diffusion_args,
        },
        "visual_feature_act": {
            "version": "visual_feature_act_lite_v1",
            "stage": "visual_feature_act_baseline",
            "model": ROOT / "outputs" / "visual_feature_act" / "visual_feature_act_lite_20260720_091256.pt",
            "load": run_visual_feature_act_policy.load_model,
            "rollout": run_visual_feature_act_policy.rollout_with_env,
            "args": visual_feature_act_args,
        },
        "object_action_head": {
            "version": args.object_action_head_version,
            "stage": "vla_action_head_proxy",
            "model": args.object_action_head_model,
            "load": run_object_action_head.load_model,
            "rollout": run_object_action_head.rollout_with_env,
            "args": object_args,
        },
        "trajectory_prior_residual": {
            "version": args.trajectory_prior_residual_version,
            "stage": "structured_prior_residual_diagnostic",
            "model": args.trajectory_prior_residual_model,
            "load": run_trajectory_prior_residual_policy.load_residual_model,
            "rollout": run_trajectory_prior_residual_policy.rollout_with_env,
            "args": prior_residual_args,
        },
        "reward_weighted_action_head": {
            "version": "reward_weighted_action_head_lite_v1",
            "stage": "reward_weighted_bc_post_training",
            "model": ROOT / "outputs" / "reward_weighted_action_head" / "reward_weighted_action_head_lite_20260720_080912.npz",
            "load": run_object_action_head.load_model,
            "rollout": run_object_action_head.rollout_with_env,
            "args": object_args,
        },
        "phase_action_head": {
            "version": "phase_conditioned_action_head_lite_v1",
            "stage": "phase_conditioned_action_head_proxy",
            "model": ROOT / "outputs" / "phase_action_head" / "phase_conditioned_action_head_lite_20260720_082827.npz",
            "load": run_phase_action_head.load_model,
            "rollout": run_phase_action_head.rollout_with_env,
            "args": phase_args,
        },
        "vision_language_action_head": {
            "version": "vision_language_action_head_lite_v1",
            "stage": "vla_action_head_proxy",
            "model": ROOT / "outputs" / "vision_language_action_head" / "vision_language_action_head_lite_20260720_063123.npz",
            "load": run_vision_language_action_head.load_model,
            "rollout": run_vision_language_action_head.rollout_with_env,
            "args": object_args,
        },
        "clip_action_head": {
            "version": "clip_action_head_lite_v1",
            "stage": "pretrained_vlm_action_head_proxy",
            "model": ROOT / "outputs" / "clip_action_head" / "clip_action_head_lite_20260720_074716.npz",
            "load": load_clip_policy,
            "rollout": rollout_clip_action_head,
            "args": clip_args,
        },
        "clip_core_v2_action_head": {
            "version": args.clip_core_v2_version,
            "stage": "frozen_pretrained_vlm_action_head",
            "model": args.clip_core_v2_model,
            "load": load_clip_policy,
            "rollout": rollout_clip_action_head,
            "args": clip_core_v2_args,
        },
        "adapter_action_head": {
            "version": "adapter_action_head_lite_v1",
            "stage": "peft_action_head_proxy",
            "model": ROOT / "outputs" / "peft_action_head" / "adapter_action_head_lite_20260720_072914.npz",
            "load": run_peft_action_head.load_model,
            "rollout": run_peft_action_head.rollout_with_env,
            "args": object_args,
        },
        "lora_action_head": {
            "version": "lora_action_head_lite_v1",
            "stage": "peft_action_head_proxy",
            "model": ROOT / "outputs" / "peft_action_head" / "lora_action_head_lite_20260720_072913.npz",
            "load": run_peft_action_head.load_model,
            "rollout": run_peft_action_head.rollout_with_env,
            "args": object_args,
        },
        "multi_task_object_action_head": {
            "version": "multi_task_object_action_head_lite_v1",
            "stage": "multi_task_action_head_proxy",
            "model": ROOT / "outputs" / "object_action_head" / "multi_task_object_action_head_lite_20260720_051331.npz",
            "load": run_object_action_head.load_model,
            "rollout": run_object_action_head.rollout_with_env,
            "args": object_args,
        },
        "diffusion_policy": {
            "version": "diffusion_policy_lite_v1",
            "stage": "diffusion_policy_baseline",
            "model": ROOT / "outputs" / "diffusion_policy" / "diffusion_policy_lite_20260720_042215.npz",
            "load": run_diffusion_policy.load_model,
            "rollout": run_diffusion_policy.rollout_with_env,
            "args": diffusion_args,
        },
    }


def summarize(version: str, stage: str, results: list[dict], artifact: str | None) -> dict:
    successes = sum(int(item["success"]) for item in results)
    distances = [float(item["target_distance"]) for item in results if np.isfinite(item["target_distance"])]
    return {
        "version": version,
        "stage": stage,
        "artifact": artifact or "",
        "success": f"{successes}/{len(results)}",
        "success_rate": successes / max(1, len(results)),
        "mean_target_distance": float(np.mean(distances)) if distances else float("nan"),
        "seeds": ",".join(str(item["seed"]) for item in results),
    }


def main() -> None:
    args = parse_args()
    specs = method_specs(args)
    selected = [name.strip() for name in args.methods.split(",") if name.strip()]
    unknown = [name for name in selected if name not in specs]
    if unknown:
        raise KeyError(f"unknown methods: {unknown}; available={sorted(specs)}")

    all_results: dict[str, list[dict]] = {}
    rows: list[dict] = []
    for method in selected:
        spec = specs[method]
        results = []
        for offset in range(args.episodes):
            seed = args.seed + offset
            if spec.get("runner") == "expert":
                result = run_expert_episode(args, seed)
                artifact = "scripts/run_expert.py"
            else:
                artifact_path = Path(spec["model"])
                result = run_policy_episode(args, seed, spec["load"], spec["rollout"], artifact_path, spec["args"])
                artifact = artifact_path.as_posix()
            result["method"] = method
            result["version"] = spec["version"]
            results.append(result)
            print(
                f"method={method} seed={seed} success={result['success']} "
                f"target={result['target_object'] if 'target_object' in result else result.get('active_objects', [''])[0]} "
                f"distance={float(result['target_distance']):.4f}",
                flush=True,
            )
        all_results[method] = results
        row = summarize(spec["version"], spec["stage"], results, artifact)
        row["method_key"] = method
        rows.append(row)
        print(f"method_summary: {method} {row['success']} rate={row['success_rate']:.3f}", flush=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as file:
        fieldnames = ["method_key", "version", "stage", "artifact", "success", "success_rate", "mean_target_distance", "seeds"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    output_json = args.output_json or ROOT / "outputs" / "evaluations" / f"language_generalization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "task": args.task,
                "complexity": args.complexity,
                "protocol": {
                    "workspace_profile": args.workspace_profile,
                    "arm_kp": args.arm_kp,
                    "arm_force": args.arm_force,
                    "gripper_kp": args.gripper_kp,
                    "gripper_force": args.gripper_force,
                    "friction": args.friction,
                    "place_tcp_z": args.place_tcp_z,
                },
                "seed": args.seed,
                "episodes": args.episodes,
                "rows": rows,
                "episodes_by_method": all_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"csv_path: {args.output_csv}", flush=True)
    print(f"json_path: {output_json}", flush=True)


if __name__ == "__main__":
    main()
