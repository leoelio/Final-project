from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import replay_demo  # noqa: E402
import run_bc_policy  # noqa: E402
import run_chunk_policy  # noqa: E402
import run_diffusion_policy  # noqa: E402
import run_knn_policy  # noqa: E402
import run_mlp_policy  # noqa: E402
import run_object_action_head  # noqa: E402
import run_peft_action_head  # noqa: E402
import run_phase_action_head  # noqa: E402
import run_structured_waypoint_policy  # noqa: E402
import run_torch_act_cvae_policy  # noqa: E402
import run_torch_act_policy  # noqa: E402
import run_torch_diffusion_policy  # noqa: E402
import run_trajectory_knn_policy  # noqa: E402
import run_trajectory_phase_template_policy  # noqa: E402
import run_trajectory_prior_residual_policy  # noqa: E402
import run_timing_aware_trajectory_prior_residual_policy  # noqa: E402
import run_grasp_gated_trajectory_knn_policy  # noqa: E402
import run_preference_trajectory_post_training_policy  # noqa: E402
import run_visual_act_cnn_cvae_policy  # noqa: E402
import run_visual_feature_act_policy  # noqa: E402
import run_vision_language_action_head  # noqa: E402
import run_clip_action_head  # noqa: E402
import run_clip_semantic_waypoint  # noqa: E402
import run_contact_stage_subpolicy  # noqa: E402
import run_gripper_timing_probe  # noqa: E402
from widowx_env import TASKS, WidowXTabletopEnv  # noqa: E402
from widowx_env.scripted_expert import PickPlaceConfig, PickPlaceExpert  # noqa: E402


DEFAULTS = {
    "expert": {
        "version": "expert_scripted_v1",
        "model": None,
    },
    "grasp_lift_subpolicy_probe": {
        "version": "grasp_lift_subpolicy_probe_v1_candidate",
        "model": None,
    },
    "contact_stage_subpolicy": {
        "version": "contact_stage_subpolicy_v1_candidate",
        "model": None,
    },
    "gripper_timing_contact_probe": {
        "version": "gripper_timing_contact_probe_v1_candidate",
        "model": None,
    },
    "replay": {
        "version": "replay_demo_v1",
        "model": None,
    },
    "linear_bc": {
        "version": "linear_bc_v1",
        "model": ROOT / "outputs" / "bc" / "bc_linear_20260702_051909.npz",
    },
    "knn_bc": {
        "version": "knn_bc_v1",
        "model": ROOT / "outputs" / "knn_bc" / "knn_bc_20260702_051907.npz",
    },
    "mlp_bc": {
        "version": "mlp_bc_v1",
        "model": ROOT / "outputs" / "mlp_bc" / "mlp_bc_20260702_053322.npz",
    },
    "chunk_bc": {
        "version": "act_lite_chunk_bc_v1",
        "model": ROOT / "outputs" / "chunk_bc" / "chunk_bc_20260702_072710.npz",
    },
    "diffusion_policy": {
        "version": "diffusion_policy_lite_v1",
        "model": ROOT / "outputs" / "diffusion_policy" / "diffusion_policy_lite_20260720_042215.npz",
    },
    "structured_waypoint_policy": {
        "version": "structured_waypoint_policy_v1",
        "model": ROOT / "outputs" / "structured_waypoint_policy" / "structured_waypoint_policy_20260720_065456.npz",
    },
    "trajectory_knn_bc": {
        "version": "trajectory_knn_chunk_bc_v1",
        "model": ROOT / "outputs" / "trajectory_knn_bc" / "trajectory_knn_chunk_bc_20260720_053423.npz",
    },
    "contact_aware_trajectory_knn": {
        "version": "contact_aware_trajectory_knn_v1_candidate",
        "model": ROOT / "outputs" / "trajectory_knn_bc" / "contact_aware_trajectory_knn_20260720_233445.npz",
    },
    "trajectory_phase_template_bc": {
        "version": "trajectory_phase_template_bc_v1_candidate",
        "model": ROOT / "outputs" / "trajectory_phase_template_bc" / "trajectory_phase_template_bc_20260720_160007.npz",
    },
    "trajectory_prior_residual_bc": {
        "version": "trajectory_prior_residual_bc_v1_candidate",
        "model": ROOT / "outputs" / "trajectory_prior_residual_bc" / "trajectory_prior_residual_bc_v1_candidate_20260721_040050.npz",
    },
    "timing_aware_trajectory_prior_residual_bc": {
        "version": "timing_aware_trajectory_prior_residual_bc_v1_candidate",
        "model": None,
    },
    "grasp_gated_trajectory_chunk_bc": {
        "version": "grasp_gated_trajectory_conditioned_chunk_bc_v1_candidate",
        "model": ROOT / "outputs" / "chunk_bc" / "trajectory_chunk_bc_20260720_043500.npz",
    },
    "grasp_gated_trajectory_knn": {
        "version": "grasp_gated_trajectory_knn_v1_candidate",
        "model": ROOT / "outputs" / "trajectory_knn_bc" / "trajectory_knn_chunk_bc_20260720_053423.npz",
    },
    "preference_trajectory_post_training": {
        "version": "preference_trajectory_post_training_v1_candidate",
        "model": ROOT / "outputs" / "preference_post_training" / "preference_trajectory_post_training_20260720_165005.npz",
    },
    "torch_act": {
        "version": "torch_act_state_chunk_v1",
        "model": ROOT / "outputs" / "torch_act" / "torch_act_state_chunk_20260720_055409.pt",
    },
    "grasp_gated_torch_act": {
        "version": "grasp_gated_torch_act_state_chunk_v1_candidate",
        "model": ROOT / "outputs" / "torch_act" / "torch_act_state_chunk_20260720_055409.pt",
    },
    "phase_weighted_torch_act": {
        "version": "phase_weighted_torch_act_v1_candidate",
        "model": ROOT / "outputs" / "torch_act" / "phase_weighted_torch_act_v1_candidate_20260720_225108.pt",
    },
    "torch_act_cvae": {
        "version": "torch_act_cvae_state_chunk_v1",
        "model": ROOT / "outputs" / "torch_act_cvae" / "torch_act_cvae_state_chunk_20260720_084842.pt",
    },
    "torch_diffusion_policy": {
        "version": "torch_diffusion_policy_state_chunk_v1",
        "model": ROOT / "outputs" / "torch_diffusion_policy" / "torch_diffusion_policy_state_chunk_20260720_101928.pt",
    },
    "visual_feature_act": {
        "version": "visual_feature_act_lite_v1",
        "model": ROOT / "outputs" / "visual_feature_act" / "visual_feature_act_lite_20260720_091256.pt",
    },
    "visual_act_cnn_cvae": {
        "version": "visual_act_cnn_cvae_v1",
        "model": ROOT / "outputs" / "visual_act_cnn_cvae" / "visual_act_cnn_cvae_20260720_115104.pt",
    },
    "object_action_head": {
        "version": "object_language_action_head_lite_v1",
        "model": ROOT / "outputs" / "object_action_head" / "object_action_head_lite_20260720_044703.npz",
    },
    "reward_weighted_action_head": {
        "version": "reward_weighted_action_head_lite_v1",
        "model": ROOT / "outputs" / "reward_weighted_action_head" / "reward_weighted_action_head_lite_20260720_080912.npz",
    },
    "phase_action_head": {
        "version": "phase_conditioned_action_head_lite_v1",
        "model": ROOT / "outputs" / "phase_action_head" / "phase_conditioned_action_head_lite_20260720_082827.npz",
    },
    "vision_language_action_head": {
        "version": "vision_language_action_head_lite_v1",
        "model": ROOT / "outputs" / "vision_language_action_head" / "vision_language_action_head_lite_20260720_063123.npz",
    },
    "clip_action_head": {
        "version": "clip_action_head_lite_v1",
        "model": ROOT / "outputs" / "clip_action_head" / "clip_action_head_lite_20260720_074716.npz",
    },
    "clip_semantic_waypoint": {
        "version": "clip_semantic_waypoint_core_v2_v1",
        "model": ROOT / "outputs" / "clip_semantic_waypoint" / "clip_semantic_waypoint_core_v2_v1_20260721_110325.npz",
    },
    "clip_semantic_contact_fusion": {
        "version": "clip_semantic_contact_fusion_low_friction_multitask_v1",
        "model": ROOT / "outputs" / "clip_semantic_waypoint" / "clip_semantic_waypoint_core_v2_v1_20260721_110325.npz",
    },
    "peft_action_head": {
        "version": "adapter_action_head_lite_v1",
        "model": ROOT / "outputs" / "peft_action_head" / "adapter_action_head_lite_20260720_072914.npz",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MuJoCo rollout clips for experiment comparison.")
    parser.add_argument("--method", choices=sorted(DEFAULTS), required=True)
    parser.add_argument("--version", default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--task", choices=sorted(TASKS), default="place_blue_cube_blue_pad")
    parser.add_argument("--instruction", default=None, help="Optional instruction override for VLM/VLA evaluation clips.")
    parser.add_argument("--instruction-normalization", choices=("none", "desktop_alias_v1"), default="none")
    parser.add_argument("--complexity", choices=("easy", "medium", "hard", "language"), default="medium")
    parser.add_argument("--workspace-profile", choices=("legacy", "core_v2"), default="legacy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2840)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data" / "demos" / "place_blue_cube_blue_pad_medium_20260702_051752")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--camera", default="top_rgb")
    parser.add_argument("--arm-kp", type=float, default=150.0)
    parser.add_argument("--arm-force", type=float, default=100.0)
    parser.add_argument("--gripper-kp", type=float, default=800.0)
    parser.add_argument("--gripper-force", type=float, default=140.0)
    parser.add_argument("--friction", type=float, default=3.0)
    parser.add_argument("--place-tcp-z", type=float, default=0.055)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-stride", type=int, default=8)
    parser.add_argument("--variant", choices=sorted(run_gripper_timing_probe.VARIANTS), default="tight_close_hold")
    parser.add_argument("--phase-mode", choices=("progress", "state", "hybrid"), default="progress")
    parser.add_argument("--action-alpha", type=float, default=None)
    parser.add_argument("--max-arm-delta", type=float, default=None)
    parser.add_argument("--max-gripper-delta", type=float, default=None)
    parser.add_argument("--residual-scale", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=0)
    return parser.parse_args()


class FfmpegWriter:
    def __init__(self, path: Path, width: int, height: int, fps: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        code = self.process.wait()
        if code:
            raise RuntimeError(f"ffmpeg failed with exit code {code}")


class FrameRecorder:
    def __init__(self, env: WidowXTabletopEnv, output: Path, width: int, height: int, fps: int, frame_stride: int, camera: str) -> None:
        self.env = env
        self.frame_stride = max(1, int(frame_stride))
        self.step_count = 0
        self.frame_count = 0
        self.renderer = mujoco.Renderer(env.model, height=height, width=width)
        self.camera = camera
        self.writer = FfmpegWriter(output, width, height, fps)

    def capture(self) -> None:
        self.renderer.update_scene(self.env.data, camera=self.camera)
        self.writer.write(self.renderer.render())
        self.frame_count += 1

    def sync(self) -> None:
        self.step_count += 1
        if self.step_count % self.frame_stride == 0:
            self.capture()

    def close(self) -> None:
        self.capture()
        self.renderer.close()
        self.writer.close()


def common_policy_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        seed=args.seed,
        episodes=1,
        steps=args.max_steps or args.steps,
        viewer=False,
        duration=0.0,
        speed=0.0,
        arm_kp=args.arm_kp,
        arm_force=args.arm_force,
        gripper_kp=args.gripper_kp,
        gripper_force=args.gripper_force,
        friction=args.friction,
        clip_actions=True,
        stop_on_unsafe=True,
        log_every=args.log_every,
        grasp_gate=False,
        close_phase=0.22,
        release_phase=0.78,
        near_threshold=0.11,
        release_distance=0.095,
        open_gripper=0.037,
        close_gripper=0.015,
    )


def configure_env(args: argparse.Namespace) -> tuple[WidowXTabletopEnv, dict]:
    env = WidowXTabletopEnv(
        seed=args.seed,
        image_size=(args.height, args.width),
        camera=args.camera,
        workspace_profile=args.workspace_profile,
    )
    obs = env.reset(task=args.task, complexity=args.complexity, seed=args.seed)
    return env, obs


def apply_domain_parameters(env: WidowXTabletopEnv, args: argparse.Namespace) -> None:
    env.set_arm_actuator_strength(kp=args.arm_kp, force_limit=args.arm_force)
    env.set_gripper_actuator_strength(kp=args.gripper_kp, force_limit=args.gripper_force)
    env.set_grasp_contact_friction(sliding=args.friction)


def run_expert_clip(args: argparse.Namespace, recorder: FrameRecorder) -> dict:
    env = recorder.env
    obs = env.observation(render=False)
    expert = PickPlaceExpert(env, PickPlaceConfig(place_tcp_z=args.place_tcp_z))
    plan = expert.plan(obs["target_object"], env.task.target_geom)
    summary = expert.execute(plan, viewer=recorder, speed=0.0)
    return {"summary": summary, "instruction": obs["instruction"], "active_objects": list(obs["active_objects"])}


def run_grasp_lift_subpolicy_probe_clip(args: argparse.Namespace, recorder: FrameRecorder) -> dict:
    env = recorder.env
    obs = env.observation(render=False)
    expert = PickPlaceExpert(env, PickPlaceConfig())
    lift_threshold = 0.085
    tcp_lift_threshold = 0.12
    trace = {
        "steps_taken": 0,
        "max_object_z": 0.0,
        "max_contact_count": 0.0,
        "min_tcp_object_distance": None,
        "min_tcp_object_distance_while_lifted": None,
        "ever_grasp_success": False,
        "ever_tcp_lift_success": False,
        "first_grasp_step": None,
        "first_lift_step": None,
        "first_tcp_lift_step": None,
    }

    def record_step(_action: np.ndarray, current_env: WidowXTabletopEnv) -> None:
        metrics = current_env.metrics()
        object_position = current_env.object_position(current_env.episode_target_object)
        tcp_object_distance = float(np.linalg.norm(current_env.tcp_position() - object_position))
        trace["steps_taken"] += 1
        trace["max_object_z"] = max(float(trace["max_object_z"]), float(metrics["object_z"]))
        trace["max_contact_count"] = max(float(trace["max_contact_count"]), float(metrics["contact_count"]))
        if trace["min_tcp_object_distance"] is None or tcp_object_distance < float(trace["min_tcp_object_distance"]):
            trace["min_tcp_object_distance"] = tcp_object_distance
        if bool(metrics["grasp_success"]) and trace["first_grasp_step"] is None:
            trace["first_grasp_step"] = int(trace["steps_taken"])
        if bool(metrics["grasp_success"]):
            trace["ever_grasp_success"] = True
        lifted = float(metrics["object_z"]) >= lift_threshold
        if lifted:
            if trace["first_lift_step"] is None:
                trace["first_lift_step"] = int(trace["steps_taken"])
            if trace["min_tcp_object_distance_while_lifted"] is None or tcp_object_distance < float(trace["min_tcp_object_distance_while_lifted"]):
                trace["min_tcp_object_distance_while_lifted"] = tcp_object_distance
            if tcp_object_distance < tcp_lift_threshold:
                trace["ever_tcp_lift_success"] = True
                if trace["first_tcp_lift_step"] is None:
                    trace["first_tcp_lift_step"] = int(trace["steps_taken"])

    summary: dict[str, object] = {"success": False, "attempts": 0}
    for attempt in range(1, 4):
        plan = expert.plan(obs["target_object"], env.task.target_geom)
        summary = expert.execute(plan, viewer=recorder, record_step=record_step, speed=0.0)
        summary["attempts"] = attempt
        if bool(summary["success"]):
            break

    metrics = env.metrics()
    max_object_z = float(trace["max_object_z"])
    summary.update(
        {
            "target_distance": float(metrics["target_distance"]),
            "object_z": float(metrics["object_z"]),
            "grasp_success": bool(metrics["grasp_success"]),
            "ever_grasp_success": bool(trace["ever_grasp_success"]),
            "ever_tcp_lift_success": bool(trace["ever_tcp_lift_success"]),
            "height_threshold_hit": bool(max_object_z >= lift_threshold),
            "strict_grasp_lift_success": bool(trace["ever_grasp_success"] and max_object_z >= lift_threshold),
            "tcp_grasp_lift_success": bool(trace["ever_tcp_lift_success"] and max_object_z >= lift_threshold),
            "out_of_table": bool(metrics["out_of_table"]),
            "contact_count": float(metrics["contact_count"]),
            "max_contact_count": float(trace["max_contact_count"]),
            "max_object_z": max_object_z,
            "min_tcp_object_distance": trace["min_tcp_object_distance"],
            "min_tcp_object_distance_while_lifted": trace["min_tcp_object_distance_while_lifted"],
            "steps_taken": int(trace["steps_taken"]),
            "first_grasp_step": trace["first_grasp_step"],
            "first_lift_step": trace["first_lift_step"],
            "first_tcp_lift_step": trace["first_tcp_lift_step"],
        }
    )
    return {"summary": summary, "instruction": obs["instruction"], "active_objects": list(obs["active_objects"])}


def run_contact_stage_subpolicy_clip(args: argparse.Namespace, recorder: FrameRecorder) -> dict:
    run_args = argparse.Namespace(
        task=args.task,
        complexity=args.complexity,
        seed=args.seed,
        episodes=1,
        viewer=False,
        duration=0.0,
        speed=0.0,
        arm_kp=args.arm_kp,
        arm_force=args.arm_force,
        gripper_kp=args.gripper_kp,
        gripper_force=args.gripper_force,
        friction=args.friction,
        approach_z=0.12,
        grasp_z=0.008,
        lift_z=0.18,
        retries=2,
        lift_threshold=0.085,
        tcp_lift_threshold=0.12,
        log_every=args.log_every,
    )
    obs = recorder.env.observation(render=False)
    obs = dict(obs)
    obs["seed"] = int(args.seed)
    summary = run_contact_stage_subpolicy.rollout_with_env(run_args, recorder.env, obs, recorder)
    return {"summary": summary, "instruction": obs["instruction"], "active_objects": list(obs["active_objects"])}


def run_gripper_timing_contact_probe_clip(args: argparse.Namespace, recorder: FrameRecorder) -> dict:
    run_args = argparse.Namespace(
        task=args.task,
        complexity=args.complexity,
        seed=args.seed,
        episodes=1,
        variant=args.variant,
        viewer=False,
        duration=0.0,
        speed=0.0,
        arm_kp=args.arm_kp,
        arm_force=args.arm_force,
        gripper_kp=args.gripper_kp,
        gripper_force=args.gripper_force,
        friction=args.friction,
        approach_z=0.12,
        grasp_z=0.008,
        lift_z=0.18,
        retries=1,
        lift_threshold=0.085,
        tcp_lift_threshold=0.12,
        log_every=args.log_every,
    )
    obs = recorder.env.observation(render=False)
    obs = dict(obs)
    obs["seed"] = int(args.seed)
    summary = run_gripper_timing_probe.rollout_with_env(run_args, recorder.env, obs, recorder)
    return {"summary": summary, "instruction": obs["instruction"], "active_objects": list(obs["active_objects"])}


def run_replay_clip(args: argparse.Namespace, recorder: FrameRecorder) -> dict:
    metadata = replay_demo.load_metadata(args.run_dir, args.episode_index)
    trajectory_path = args.run_dir / metadata["trajectory_file"]
    with np.load(trajectory_path) as recorded:
        step_limit = args.max_steps
        stats = replay_demo.replay(recorder.env, recorded["actions"], recorded, viewer=recorder, speed=0.0, max_steps=step_limit)
    return {"summary": stats, "instruction": metadata["instruction"], "active_objects": metadata["active_objects"]}


def run_linear_clip(args: argparse.Namespace, recorder: FrameRecorder, model_path: Path) -> dict:
    run_args = common_policy_args(args)
    run_args.action_alpha = 0.2
    run_args.max_arm_delta = 0.01
    run_args.max_gripper_delta = 0.0005
    model = run_bc_policy.load_model(model_path)
    summary = run_bc_policy.rollout_with_env(run_args, model, recorder.env, args.seed, args.task, args.complexity, recorder)
    return {"summary": summary, "instruction": summary["instruction"], "active_objects": summary["active_objects"]}


def run_knn_clip(args: argparse.Namespace, recorder: FrameRecorder, model_path: Path) -> dict:
    run_args = common_policy_args(args)
    run_args.action_alpha = 1.0
    run_args.max_arm_delta = 0.05
    run_args.max_gripper_delta = 0.002
    run_args.k = 3
    run_args.phase_window = 0.02
    run_args.min_candidates = 128
    run_args.qpos_weight = 0.25
    run_args.qvel_weight = 0.05
    run_args.ctrl_weight = 0.25
    run_args.tcp_weight = 1.0
    run_args.object_weight = 4.0
    run_args.target_weight = 1.0
    run_args.phase_weight = 2.0
    model = run_knn_policy.load_model(model_path)
    obs = recorder.env.observation(render=False)
    summary = run_knn_policy.rollout_with_env(run_args, model, recorder.env, obs, args.seed, args.task, args.complexity, recorder)
    return {"summary": summary, "instruction": summary["instruction"], "active_objects": summary["active_objects"]}


def run_mlp_clip(args: argparse.Namespace, recorder: FrameRecorder, model_path: Path) -> dict:
    run_args = common_policy_args(args)
    run_args.action_alpha = 0.2
    run_args.max_arm_delta = 0.01
    run_args.max_gripper_delta = 0.0005
    model = run_mlp_policy.load_model(model_path)
    obs = recorder.env.observation(render=False)
    summary = run_mlp_policy.rollout_with_env(run_args, model, recorder.env, obs, args.seed, args.task, args.complexity, recorder)
    return {"summary": summary, "instruction": summary["instruction"], "active_objects": summary["active_objects"]}


def run_chunk_clip(args: argparse.Namespace, recorder: FrameRecorder, model_path: Path) -> dict:
    run_args = common_policy_args(args)
    model = run_chunk_policy.load_model(model_path)
    history = int(model["metadata"].get("history", 1))
    if history > 1:
        run_args.action_alpha = 0.25
        run_args.max_arm_delta = 0.012
        run_args.max_gripper_delta = 0.0005
    else:
        run_args.action_alpha = 0.9
        run_args.max_arm_delta = 0.05
        run_args.max_gripper_delta = 0.002
    run_args.replan_interval = 1
    run_args.temporal_ensemble = True
    run_args.ensemble_decay = 0.1
    obs = recorder.env.observation(render=False)
    summary = run_chunk_policy.rollout_with_env(run_args, model, recorder.env, obs, args.seed, args.task, args.complexity, recorder)
    return {"summary": summary, "instruction": summary["instruction"], "active_objects": summary["active_objects"]}


def run_diffusion_clip(args: argparse.Namespace, recorder: FrameRecorder, model_path: Path) -> dict:
    run_args = common_policy_args(args)
    run_args.action_alpha = 0.8
    run_args.max_arm_delta = 0.03
    run_args.max_gripper_delta = 0.001
    run_args.replan_interval = 1
    run_args.temporal_ensemble = True
    run_args.ensemble_decay = 0.1
    run_args.deterministic = True
    model = run_diffusion_policy.load_model(model_path)
    obs = recorder.env.observation(render=False)
    summary = run_diffusion_policy.rollout_with_env(run_args, model, recorder.env, obs, args.seed, args.task, args.complexity, recorder)
    return {"summary": summary, "instruction": summary["instruction"], "active_objects": summary["active_objects"]}


def run_structured_waypoint_clip(args: argparse.Namespace, recorder: FrameRecorder, model_path: Path) -> dict:
    run_args = common_policy_args(args)
    model = run_structured_waypoint_policy.load_model(model_path)
    obs = recorder.env.observation(render=False)
    summary = run_structured_waypoint_policy.rollout_with_env(run_args, model, recorder.env, obs, args.seed, args.task, args.complexity, recorder)
    return {"summary": summary, "instruction": summary["instruction"], "active_objects": summary["active_objects"]}


def run_trajectory_knn_clip(args: argparse.Namespace, recorder: FrameRecorder, model_path: Path) -> dict:
    run_args = common_policy_args(args)
    run_args.action_alpha = 0.85
    run_args.max_arm_delta = 0.04
    run_args.max_gripper_delta = 0.0015
    run_args.k = 3
    run_args.phase_window = 0.03
    run_args.min_candidates = 256
    run_args.history_decay = 0.25
    run_args.replan_interval = 1
    run_args.temporal_ensemble = True
    run_args.ensemble_decay = 0.1
    model = run_trajectory_knn_policy.load_model(model_path)
    obs = recorder.env.observation(render=False)
    summary = run_trajectory_knn_policy.rollout_with_env(run_args, model, recorder.env, obs, args.seed, args.task, args.complexity, recorder)
    return {"summary": summary, "instruction": summary["instruction"], "active_objects": summary["active_objects"]}


def run_contact_aware_trajectory_knn_clip(args: argparse.Namespace, recorder: FrameRecorder, model_path: Path) -> dict:
    run_args = common_policy_args(args)
    run_args.action_alpha = 0.25
    run_args.max_arm_delta = 0.012
    run_args.max_gripper_delta = 0.0005
    run_args.k = 3
    run_args.phase_window = 0.03
    run_args.min_candidates = 256
    run_args.history_decay = 0.25
    run_args.replan_interval = 1
    run_args.temporal_ensemble = True
    run_args.ensemble_decay = 0.1
    model = run_trajectory_knn_policy.load_model(model_path)
    obs = recorder.env.observation(render=False)
    summary = run_trajectory_knn_policy.rollout_with_env(run_args, model, recorder.env, obs, args.seed, args.task, args.complexity, recorder)
    return {"summary": summary, "instruction": summary["instruction"], "active_objects": summary["active_objects"]}


def run_trajectory_phase_template_clip(args: argparse.Namespace, recorder: FrameRecorder, model_path: Path) -> dict:
    run_args = common_policy_args(args)
    run_args.action_alpha = 0.35
    run_args.max_arm_delta = 0.018
    run_args.max_gripper_delta = 0.0008
    model = run_trajectory_phase_template_policy.load_model(model_path)
    obs = recorder.env.observation(render=False)
    summary = run_trajectory_phase_template_policy.rollout_with_env(
        run_args,
        model,
        recorder.env,
        obs,
        args.seed,
        args.task,
        args.complexity,
        recorder,
    )
    return {"summary": summary, "instruction": summary["instruction"], "active_objects": summary["active_objects"]}


def run_trajectory_prior_residual_clip(args: argparse.Namespace, recorder: FrameRecorder, model_path: Path) -> dict:
    run_args = common_policy_args(args)
    run_args.task = args.task
    run_args.complexity = args.complexity
    run_args.action_alpha = args.action_alpha if args.action_alpha is not None else 1.0
    run_args.max_arm_delta = args.max_arm_delta if args.max_arm_delta is not None else 0.02
    run_args.max_gripper_delta = args.max_gripper_delta if args.max_gripper_delta is not None else 0.0008
    run_args.approach_z = 0.12
    run_args.grasp_z = 0.008
    run_args.lift_z = 0.18
    run_args.place_tcp_z = args.place_tcp_z
    run_args.residual_scale = args.residual_scale if args.residual_scale is not None else 0.25
    run_args.clip_actions = False
    run_args.require_lift_before_transfer = True
    run_args.lift_threshold = 0.085
    run_args.tcp_lift_threshold = 0.12
    model = run_trajectory_prior_residual_policy.load_residual_model(model_path)
    obs = recorder.env.observation(render=False)
    summary = run_trajectory_prior_residual_policy.rollout_with_env(
        run_args,
        model,
        recorder.env,
        obs,
        args.seed,
        args.task,
        args.complexity,
        recorder,
    )
    return {"summary": summary, "instruction": summary["instruction"], "active_objects": summary["active_objects"]}


def run_timing_aware_trajectory_prior_residual_clip(args: argparse.Namespace, recorder: FrameRecorder, model_path: Path) -> dict:
    run_args = common_policy_args(args)
    run_args.task = args.task
    run_args.complexity = args.complexity
    run_args.action_alpha = args.action_alpha if args.action_alpha is not None else 1.0
    run_args.max_arm_delta = args.max_arm_delta if args.max_arm_delta is not None else 0.02
    run_args.max_gripper_delta = args.max_gripper_delta if args.max_gripper_delta is not None else 0.0008
    run_args.approach_z = 0.12
    run_args.grasp_z = 0.008
    run_args.lift_z = 0.18
    run_args.residual_scale = args.residual_scale if args.residual_scale is not None else 0.02
    run_args.require_lift_before_transfer = True
    run_args.lift_threshold = 0.085
    run_args.tcp_lift_threshold = 0.12
    model = run_timing_aware_trajectory_prior_residual_policy.load_residual_model(model_path)
    obs = recorder.env.observation(render=False)
    summary = run_timing_aware_trajectory_prior_residual_policy.rollout_with_env(
        run_args,
        model,
        recorder.env,
        obs,
        args.seed,
        args.task,
        args.complexity,
        recorder,
    )
    return {"summary": summary, "instruction": summary["instruction"], "active_objects": summary["active_objects"]}


def run_grasp_gated_trajectory_knn_clip(args: argparse.Namespace, recorder: FrameRecorder, model_path: Path) -> dict:
    run_args = common_policy_args(args)
    run_args.action_alpha = 0.85
    run_args.max_arm_delta = 0.04
    run_args.max_gripper_delta = 0.002
    run_args.k = 3
    run_args.phase_window = 0.03
    run_args.min_candidates = 256
    run_args.history_decay = 0.25
    run_args.replan_interval = 1
    run_args.temporal_ensemble = True
    run_args.ensemble_decay = 0.1
    run_args.close_phase = 0.22
    run_args.release_phase = 0.78
    run_args.near_threshold = 0.11
    run_args.open_gripper = 0.037
    run_args.close_gripper = 0.015
    model = run_grasp_gated_trajectory_knn_policy.load_model(model_path)
    obs = recorder.env.observation(render=False)
    summary = run_grasp_gated_trajectory_knn_policy.rollout_with_env(
        run_args,
        model,
        recorder.env,
        obs,
        args.seed,
        args.task,
        args.complexity,
        recorder,
    )
    return {"summary": summary, "instruction": summary["instruction"], "active_objects": summary["active_objects"]}


def run_grasp_gated_trajectory_chunk_clip(args: argparse.Namespace, recorder: FrameRecorder, model_path: Path) -> dict:
    run_args = common_policy_args(args)
    run_args.action_alpha = 0.12
    run_args.max_arm_delta = 0.006
    run_args.max_gripper_delta = 0.00025
    run_args.replan_interval = 1
    run_args.temporal_ensemble = True
    run_args.ensemble_decay = 0.1
    run_args.grasp_gate = True
    model = run_chunk_policy.load_model(model_path)
    obs = recorder.env.observation(render=False)
    summary = run_chunk_policy.rollout_with_env(run_args, model, recorder.env, obs, args.seed, args.task, args.complexity, recorder)
    return {"summary": summary, "instruction": summary["instruction"], "active_objects": summary["active_objects"]}


def run_preference_trajectory_post_training_clip(args: argparse.Namespace, recorder: FrameRecorder, model_path: Path) -> dict:
    run_args = common_policy_args(args)
    run_args.action_alpha = 0.85
    run_args.max_arm_delta = 0.04
    run_args.max_gripper_delta = 0.0015
    run_args.k = 3
    run_args.phase_window = 0.03
    run_args.min_candidates = 256
    run_args.history_decay = 0.25
    run_args.preference_power = 1.0
    run_args.distance_epsilon = 1e-6
    run_args.replan_interval = 1
    run_args.temporal_ensemble = True
    run_args.ensemble_decay = 0.1
    model = run_preference_trajectory_post_training_policy.load_model(model_path)
    obs = recorder.env.observation(render=False)
    summary = run_preference_trajectory_post_training_policy.rollout_with_env(
        run_args,
        model,
        recorder.env,
        obs,
        args.seed,
        args.task,
        args.complexity,
        recorder,
    )
    return {"summary": summary, "instruction": summary["instruction"], "active_objects": summary["active_objects"]}


def run_torch_act_clip(args: argparse.Namespace, recorder: FrameRecorder, model_path: Path) -> dict:
    run_args = common_policy_args(args)
    run_args.action_alpha = 0.25
    run_args.max_arm_delta = 0.012
    run_args.max_gripper_delta = 0.0005
    run_args.replan_interval = 4
    run_args.temporal_ensemble = True
    run_args.ensemble_decay = 0.1
    model = run_torch_act_policy.load_model(model_path)
    obs = recorder.env.observation(render=False)
    summary = run_torch_act_policy.rollout_with_env(run_args, model, recorder.env, obs, args.seed, args.task, args.complexity, recorder)
    return {"summary": summary, "instruction": summary["instruction"], "active_objects": summary["active_objects"]}


def run_grasp_gated_torch_act_clip(args: argparse.Namespace, recorder: FrameRecorder, model_path: Path) -> dict:
    run_args = common_policy_args(args)
    run_args.action_alpha = 0.12
    run_args.max_arm_delta = 0.006
    run_args.max_gripper_delta = 0.00025
    run_args.replan_interval = 4
    run_args.temporal_ensemble = True
    run_args.ensemble_decay = 0.1
    run_args.grasp_gate = True
    model = run_torch_act_policy.load_model(model_path)
    obs = recorder.env.observation(render=False)
    summary = run_torch_act_policy.rollout_with_env(run_args, model, recorder.env, obs, args.seed, args.task, args.complexity, recorder)
    return {"summary": summary, "instruction": summary["instruction"], "active_objects": summary["active_objects"]}


def run_torch_act_cvae_clip(args: argparse.Namespace, recorder: FrameRecorder, model_path: Path) -> dict:
    run_args = common_policy_args(args)
    run_args.action_alpha = 0.25
    run_args.max_arm_delta = 0.012
    run_args.max_gripper_delta = 0.0005
    run_args.replan_interval = 4
    run_args.temporal_ensemble = True
    run_args.ensemble_decay = 0.1
    model = run_torch_act_cvae_policy.load_model(model_path)
    obs = recorder.env.observation(render=False)
    summary = run_torch_act_cvae_policy.rollout_with_env(run_args, model, recorder.env, obs, args.seed, args.task, args.complexity, recorder)
    return {"summary": summary, "instruction": summary["instruction"], "active_objects": summary["active_objects"]}


def run_torch_diffusion_clip(args: argparse.Namespace, recorder: FrameRecorder, model_path: Path) -> dict:
    run_args = common_policy_args(args)
    run_args.action_alpha = 0.25
    run_args.max_arm_delta = 0.012
    run_args.max_gripper_delta = 0.0005
    run_args.replan_interval = 8
    run_args.temporal_ensemble = True
    run_args.ensemble_decay = 0.1
    run_args.deterministic = True
    run_args.sample_steps = 4
    model = run_torch_diffusion_policy.load_model(model_path)
    obs = recorder.env.observation(render=False)
    summary = run_torch_diffusion_policy.rollout_with_env(run_args, model, recorder.env, obs, args.seed, args.task, args.complexity, recorder)
    return {"summary": summary, "instruction": summary["instruction"], "active_objects": summary["active_objects"]}


def run_visual_feature_act_clip(args: argparse.Namespace, recorder: FrameRecorder, model_path: Path) -> dict:
    run_args = common_policy_args(args)
    run_args.action_alpha = 0.25
    run_args.max_arm_delta = 0.012
    run_args.max_gripper_delta = 0.0005
    run_args.replan_interval = 4
    run_args.temporal_ensemble = True
    run_args.ensemble_decay = 0.1
    model = run_visual_feature_act_policy.load_model(model_path)
    obs = recorder.env.observation(render=False)
    summary = run_visual_feature_act_policy.rollout_with_env(run_args, model, recorder.env, obs, args.seed, args.task, args.complexity, recorder)
    return {"summary": summary, "instruction": summary["instruction"], "active_objects": summary["active_objects"]}


def run_visual_act_cnn_cvae_clip(args: argparse.Namespace, recorder: FrameRecorder, model_path: Path) -> dict:
    run_args = common_policy_args(args)
    run_args.action_alpha = 0.25
    run_args.max_arm_delta = 0.012
    run_args.max_gripper_delta = 0.0005
    run_args.replan_interval = 4
    run_args.temporal_ensemble = True
    run_args.ensemble_decay = 0.1
    model = run_visual_act_cnn_cvae_policy.load_model(model_path)
    obs = recorder.env.observation(render=False)
    summary = run_visual_act_cnn_cvae_policy.rollout_with_env(run_args, model, recorder.env, obs, args.seed, args.task, args.complexity, recorder)
    return {"summary": summary, "instruction": summary["instruction"], "active_objects": summary["active_objects"]}


def run_object_action_head_clip(args: argparse.Namespace, recorder: FrameRecorder, model_path: Path) -> dict:
    run_args = common_policy_args(args)
    run_args.action_alpha = 0.2
    run_args.max_arm_delta = 0.01
    run_args.max_gripper_delta = 0.0005
    model = run_object_action_head.load_model(model_path)
    obs = recorder.env.observation(render=False)
    summary = run_object_action_head.rollout_with_env(run_args, model, recorder.env, obs, args.seed, args.task, args.complexity, recorder)
    return {"summary": summary, "instruction": summary["instruction"], "active_objects": summary["active_objects"]}


def run_phase_action_head_clip(args: argparse.Namespace, recorder: FrameRecorder, model_path: Path) -> dict:
    run_args = common_policy_args(args)
    run_args.action_alpha = 0.12 if args.action_alpha is None else float(args.action_alpha)
    run_args.max_arm_delta = 0.006 if args.max_arm_delta is None else float(args.max_arm_delta)
    run_args.max_gripper_delta = 0.0003 if args.max_gripper_delta is None else float(args.max_gripper_delta)
    run_args.phase_mode = args.phase_mode
    model = run_phase_action_head.load_model(model_path)
    obs = recorder.env.observation(render=False)
    summary = run_phase_action_head.rollout_with_env(run_args, model, recorder.env, obs, args.seed, args.task, args.complexity, recorder)
    return {"summary": summary, "instruction": summary["instruction"], "active_objects": summary["active_objects"]}


def run_vision_language_action_head_clip(args: argparse.Namespace, recorder: FrameRecorder, model_path: Path) -> dict:
    run_args = common_policy_args(args)
    run_args.action_alpha = 0.2
    run_args.max_arm_delta = 0.01
    run_args.max_gripper_delta = 0.0005
    model = run_vision_language_action_head.load_model(model_path)
    obs = recorder.env.observation(render=False)
    summary = run_vision_language_action_head.rollout_with_env(run_args, model, recorder.env, obs, args.seed, args.task, args.complexity, recorder)
    return {"summary": summary, "instruction": summary["instruction"], "active_objects": summary["active_objects"]}


def run_clip_action_head_clip(args: argparse.Namespace, recorder: FrameRecorder, model_path: Path) -> dict:
    run_args = common_policy_args(args)
    run_args.action_alpha = 0.2
    run_args.max_arm_delta = 0.01
    run_args.max_gripper_delta = 0.0005
    run_args.image_size = 224
    run_args.vision_interval = 64
    policy = run_clip_action_head.load_policy(model_path)
    clip_model, processor = run_clip_action_head.load_clip(str(policy["metadata"]["clip_model"]))
    obs = recorder.env.observation(render=False)
    summary = run_clip_action_head.rollout_with_env(run_args, policy, clip_model, processor, recorder.env, obs, args.seed, args.task, args.complexity, recorder)
    return {"summary": summary, "instruction": summary["instruction"], "active_objects": summary["active_objects"]}


def run_clip_semantic_waypoint_clip(args: argparse.Namespace, recorder: FrameRecorder, model_path: Path) -> dict:
    run_args = SimpleNamespace(**vars(args))
    run_args.image_size = 224
    run_args.speed = 0.0
    run_args.executor = "contact_fusion" if args.method == "clip_semantic_contact_fusion" else "standard"
    policy = run_clip_semantic_waypoint.load_policy(model_path)
    clip_model, processor = run_clip_action_head.load_clip(str(policy["metadata"]["clip_model"]))
    obs = recorder.env.observation(render=False)
    summary = run_clip_semantic_waypoint.rollout_episode(
        run_args,
        policy,
        clip_model,
        processor,
        args.seed,
        recorder,
        recorder.env,
        obs,
    )
    return {"summary": summary, "instruction": summary["instruction"], "active_objects": list(obs["active_objects"])}


def run_peft_action_head_clip(args: argparse.Namespace, recorder: FrameRecorder, model_path: Path) -> dict:
    run_args = common_policy_args(args)
    run_args.action_alpha = 0.2
    run_args.max_arm_delta = 0.01
    run_args.max_gripper_delta = 0.0005
    model = run_peft_action_head.load_model(model_path)
    obs = recorder.env.observation(render=False)
    summary = run_peft_action_head.rollout_with_env(run_args, model, recorder.env, obs, args.seed, args.task, args.complexity, recorder)
    return {"summary": summary, "instruction": summary["instruction"], "active_objects": summary["active_objects"]}


def output_path(args: argparse.Namespace, version: str) -> Path:
    if args.output:
        return args.output
    return ROOT / "outputs" / "videos" / f"{version}_seed{args.seed}.mp4"


def main() -> None:
    args = parse_args()
    version = args.version or DEFAULTS[args.method]["version"]
    model_path = args.model or DEFAULTS[args.method]["model"]
    output = output_path(args, version)

    if args.method == "replay":
        metadata = replay_demo.load_metadata(args.run_dir, args.episode_index)
        args.seed = int(metadata["seed"])
        args.task = metadata["task"]
        args.complexity = metadata["complexity"]
        env, obs = configure_env(args)
        seed = int(metadata["seed"])
    else:
        env, obs = configure_env(args)
        seed = args.seed

    recorder = FrameRecorder(env, output, args.width, args.height, args.fps, args.frame_stride, args.camera)
    apply_domain_parameters(env, args)
    recorder.capture()
    try:
        if args.method == "expert":
            result = run_expert_clip(args, recorder)
        elif args.method == "grasp_lift_subpolicy_probe":
            result = run_grasp_lift_subpolicy_probe_clip(args, recorder)
        elif args.method == "contact_stage_subpolicy":
            result = run_contact_stage_subpolicy_clip(args, recorder)
        elif args.method == "gripper_timing_contact_probe":
            result = run_gripper_timing_contact_probe_clip(args, recorder)
        elif args.method == "replay":
            result = run_replay_clip(args, recorder)
        elif args.method == "linear_bc":
            result = run_linear_clip(args, recorder, Path(model_path))
        elif args.method == "knn_bc":
            result = run_knn_clip(args, recorder, Path(model_path))
        elif args.method == "mlp_bc":
            result = run_mlp_clip(args, recorder, Path(model_path))
        elif args.method == "chunk_bc":
            result = run_chunk_clip(args, recorder, Path(model_path))
        elif args.method == "diffusion_policy":
            result = run_diffusion_clip(args, recorder, Path(model_path))
        elif args.method == "structured_waypoint_policy":
            result = run_structured_waypoint_clip(args, recorder, Path(model_path))
        elif args.method == "trajectory_knn_bc":
            result = run_trajectory_knn_clip(args, recorder, Path(model_path))
        elif args.method == "contact_aware_trajectory_knn":
            result = run_contact_aware_trajectory_knn_clip(args, recorder, Path(model_path))
        elif args.method == "trajectory_phase_template_bc":
            result = run_trajectory_phase_template_clip(args, recorder, Path(model_path))
        elif args.method == "trajectory_prior_residual_bc":
            result = run_trajectory_prior_residual_clip(args, recorder, Path(model_path))
        elif args.method == "timing_aware_trajectory_prior_residual_bc":
            if model_path is None:
                model_path = run_timing_aware_trajectory_prior_residual_policy.latest_model()
            result = run_timing_aware_trajectory_prior_residual_clip(args, recorder, Path(model_path))
        elif args.method == "grasp_gated_trajectory_chunk_bc":
            result = run_grasp_gated_trajectory_chunk_clip(args, recorder, Path(model_path))
        elif args.method == "grasp_gated_trajectory_knn":
            result = run_grasp_gated_trajectory_knn_clip(args, recorder, Path(model_path))
        elif args.method == "preference_trajectory_post_training":
            result = run_preference_trajectory_post_training_clip(args, recorder, Path(model_path))
        elif args.method == "torch_act":
            result = run_torch_act_clip(args, recorder, Path(model_path))
        elif args.method == "grasp_gated_torch_act":
            result = run_grasp_gated_torch_act_clip(args, recorder, Path(model_path))
        elif args.method == "phase_weighted_torch_act":
            result = run_torch_act_clip(args, recorder, Path(model_path))
        elif args.method == "torch_act_cvae":
            result = run_torch_act_cvae_clip(args, recorder, Path(model_path))
        elif args.method == "torch_diffusion_policy":
            result = run_torch_diffusion_clip(args, recorder, Path(model_path))
        elif args.method == "visual_feature_act":
            result = run_visual_feature_act_clip(args, recorder, Path(model_path))
        elif args.method == "visual_act_cnn_cvae":
            result = run_visual_act_cnn_cvae_clip(args, recorder, Path(model_path))
        elif args.method == "object_action_head":
            result = run_object_action_head_clip(args, recorder, Path(model_path))
        elif args.method == "reward_weighted_action_head":
            result = run_object_action_head_clip(args, recorder, Path(model_path))
        elif args.method == "phase_action_head":
            result = run_phase_action_head_clip(args, recorder, Path(model_path))
        elif args.method == "vision_language_action_head":
            result = run_vision_language_action_head_clip(args, recorder, Path(model_path))
        elif args.method == "clip_action_head":
            result = run_clip_action_head_clip(args, recorder, Path(model_path))
        elif args.method == "clip_semantic_waypoint":
            result = run_clip_semantic_waypoint_clip(args, recorder, Path(model_path))
        elif args.method == "clip_semantic_contact_fusion":
            result = run_clip_semantic_waypoint_clip(args, recorder, Path(model_path))
        elif args.method == "peft_action_head":
            result = run_peft_action_head_clip(args, recorder, Path(model_path))
        else:
            raise ValueError(f"unknown method: {args.method}")
    finally:
        recorder.close()

    metadata = {
        "version": version,
        "method": args.method,
        "seed": seed,
        "task": obs["task"],
        "complexity": args.complexity,
        "model": str(model_path) if model_path else None,
        "output": str(output),
        "frames": recorder.frame_count,
        "frame_stride": args.frame_stride,
        "fps": args.fps,
        "domain_parameters": {
            "arm_kp": float(args.arm_kp),
            "arm_force": float(args.arm_force),
            "gripper_kp": float(args.gripper_kp),
            "gripper_force": float(args.gripper_force),
            "friction": float(args.friction),
        },
        **result,
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"video_path: {output}", flush=True)
    print(f"metadata_path: {metadata_path}", flush=True)
    print(f"frames: {recorder.frame_count}", flush=True)
    print(f"summary: {result['summary']}", flush=True)


if __name__ == "__main__":
    main()
