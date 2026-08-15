from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from widowx_env import TASKS, WidowXTabletopEnv  # noqa: E402
from widowx_env.rollout_video import Mp4FrameRecorder  # noqa: E402
from widowx_env.scripted_expert import PickPlaceConfig, PickPlaceExpert  # noqa: E402
from widowx_env.vision_grounding import (  # noqa: E402
    load_calibration,
    locate_color_near,
    locate_leftmost_cube,
    locate_object,
    relocate_known_object,
)


STATIC_TARGETS = {
    "target_blue_pad": np.array([0.50, 0.16, 0.002], dtype=float),
    "target_red_pad": np.array([0.50, -0.16, 0.002], dtype=float),
    "target_bowl": np.array([0.33, 0.25, 0.006], dtype=float),
}
TARGET_COLORS = {"target_blue_pad": "blue", "target_red_pad": "red"}
INTENTS = {
    "place_blue_cube_blue_pad": ("blue_cube", "target_blue_pad"),
    "place_blue_cube_red_pad": ("blue_cube", "target_red_pad"),
    "place_red_cube_red_pad": ("red_cube", "target_red_pad"),
    "move_leftmost_cube_to_bowl": ("leftmost_cube", "target_bowl"),
}


def normalize_instruction(instruction: str) -> str:
    normalized = instruction.lower()
    replacements = (
        (r"furthest to the left|left-most|westernmost", "leftmost"),
        (r"azure|cobalt|navy|cerulean", "blue"),
        (r"crimson|scarlet|ruby|vermilion", "red"),
        (r"cuboid|block|object", "cube"),
        (r"disk|circle|mat|platform", "pad"),
        (r"container|vessel|receptacle", "bowl"),
    )
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized)
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RGB feedback recovery after a CLIP semantic waypoint attempt.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--calibration", type=Path, default=ROOT / "outputs" / "rgb_grounding" / "top_rgb_core_v2_calibration_v1.json")
    parser.add_argument("--task", choices=sorted(TASKS), default="place_blue_cube_blue_pad")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--instruction-normalization", choices=("none", "desktop_alias_v1"), default="none")
    parser.add_argument("--intent-source", choices=("clip", "task_registry"), default="clip")
    parser.add_argument("--complexity", choices=("easy", "medium", "hard", "language"), default="medium")
    parser.add_argument("--workspace-profile", choices=("legacy", "core_v2"), default="core_v2")
    parser.add_argument("--seed", type=int, default=720)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--feedback-attempts", type=int, choices=(0, 1, 2), default=1, help="0 is RGB open loop; positive values permit that many RGB-triggered retries.")
    parser.add_argument("--recovery-search", choices=("source", "table"), default="source", help="RGB source search region used only before the permitted retry.")
    parser.add_argument("--recovery-profile", choices=("standard", "deep_tight_slow"), default="standard", help="Trajectory used only for the RGB-triggered retry.")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-viewer", dest="viewer", action="store_false")
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--speed", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--camera", default="top_rgb")
    parser.add_argument("--arm-kp", type=float, default=150.0)
    parser.add_argument("--arm-force", type=float, default=100.0)
    parser.add_argument("--gripper-kp", type=float, default=1200.0)
    parser.add_argument("--gripper-force", type=float, default=200.0)
    parser.add_argument("--friction", type=float, default=5.0)
    parser.add_argument("--place-tcp-z", type=float, default=0.041)
    parser.add_argument("--video-path", type=Path, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--frame-stride", type=int, default=12)
    parser.set_defaults(viewer=True)
    return parser.parse_args()


def configure_env(args: argparse.Namespace, seed: int) -> tuple[WidowXTabletopEnv, dict]:
    env = WidowXTabletopEnv(seed=seed, image_size=(args.image_size, args.image_size), camera=args.camera, workspace_profile=args.workspace_profile)
    env.set_arm_actuator_strength(kp=args.arm_kp, force_limit=args.arm_force)
    env.set_gripper_actuator_strength(kp=args.gripper_kp, force_limit=args.gripper_force)
    env.set_grasp_contact_friction(sliding=args.friction)
    return env, env.reset(task=args.task, complexity=args.complexity, seed=seed)


def render_top_rgb(env: WidowXTabletopEnv, image_size: int, camera: str) -> np.ndarray:
    renderer = mujoco.Renderer(env.model, height=image_size, width=image_size)
    try:
        renderer.update_scene(env.data, camera=camera)
        return renderer.render().copy()
    finally:
        renderer.close()


def locate_initial_source(image: np.ndarray, calibration, source_name: str) -> tuple[str, np.ndarray, object]:
    if source_name == "leftmost_cube":
        return locate_leftmost_cube(image, calibration)
    position, detection = locate_object(image, calibration, source_name)
    return source_name, position, detection


def visual_target_status(image: np.ndarray, calibration, source_name: str, target_geom: str) -> dict:
    """Only visual evidence is used here; same-color pad placements remain explicitly ambiguous."""
    target_xy = STATIC_TARGETS[target_geom][:2]
    source_color = source_name.split("_", 1)[0]
    target_color = TARGET_COLORS.get(target_geom)
    if target_color == source_color:
        return {"verifiable": False, "complete": False, "reason": "same_color_object_and_pad"}
    try:
        position, _ = locate_color_near(image, calibration, source_color, target_xy, radius=0.065)
        return {
            "verifiable": True,
            "complete": True,
            "reason": "colored_object_at_target",
            "position": position.round(5).tolist(),
        }
    except LookupError:
        return {"verifiable": True, "complete": False, "reason": "object_not_visually_at_target"}


def attempt_config(args: argparse.Namespace, attempt_index: int) -> PickPlaceConfig:
    base = PickPlaceConfig(place_tcp_z=args.place_tcp_z)
    if attempt_index >= 1 and args.recovery_profile == "deep_tight_slow":
        return replace(base, grasp_z_offset=0.0, close_gripper=0.007, close_steps=420, descend_steps=300, lift_steps=500, transfer_steps=800)
    return base


def rollout(args, policy: dict, clip_model, processor, calibration, seed: int, viewer=None, recorder=None, env=None, obs=None, intent_predictor=None) -> dict:
    if env is None or obs is None:
        env, obs = configure_env(args, seed)
    instruction = str(getattr(args, "instruction", None) or obs["instruction"])
    normalized = normalize_instruction(instruction) if getattr(args, "instruction_normalization", "none") == "desktop_alias_v1" else instruction
    if args.intent_source == "task_registry":
        intent = args.task
        logits = []
        source_name = str(obs["target_object"])
        target_geom = str(env.task.target_geom)
    else:
        intent, logits = intent_predictor(env, {**obs, "instruction": normalized}, policy, clip_model, processor, args.image_size, args.camera)
        source_name, target_geom = INTENTS[intent]
    expected_intent = "move_leftmost_cube_to_bowl" if args.task == "move_leftmost_to_bowl" else args.task
    semantic_correct = intent == expected_intent and target_geom == env.task.target_geom
    target_position = STATIC_TARGETS[target_geom]
    image = render_top_rgb(env, args.image_size, args.camera)
    try:
        selected_name, source_position, detection = locate_initial_source(image, calibration, source_name)
    except LookupError as error:
        metrics = env.metrics()
        return {
            "seed": seed,
            "task": args.task,
            "complexity": args.complexity,
            "instruction": instruction,
            "normalized_instruction": normalized,
            "predicted_intent": intent,
            "semantic_correct": semantic_correct,
            "selected_object": source_name,
            "oracle_target_object": obs["target_object"],
            "visual_selection_correct": False,
            "initial_source_position_error_m": float("nan"),
            "feedback_attempts_allowed": int(args.feedback_attempts),
            "recovery_profile": args.recovery_profile,
            "attempt_count": 0,
            "recovery_triggered": False,
            "recovery_reason": f"initial_rgb_grounding_failed: {error}",
            "initial_detection_area_px": None,
            "final_source_position_error_m": float("nan"),
            "runtime_position_source": "top_rgb + offline plane calibration; MuJoCo object/target state is not used for retry decisions or trajectory planning",
            "target_source": "fixed scene target configuration",
            "attempt_logs": [],
            "logits": logits,
            "strict_grasp_success": False,
            "task_success": False,
            "target_distance": float(metrics["target_distance"]),
            "out_of_table": bool(metrics["out_of_table"]),
        }
    initial_detection_area = int(detection.area)
    initial_source_error = float(np.linalg.norm(source_position[:2] - env.object_position(obs["target_object"])[:2]))
    attempt_logs: list[dict] = []
    steps_taken = 0
    recovery_reason = "not_requested" if args.feedback_attempts == 0 else "not_needed"
    final_summary: dict = {}
    current_name, current_position = selected_name, source_position
    for attempt_index in range(args.feedback_attempts + 1):
        source_error = float(np.linalg.norm(current_position[:2] - env.object_position(obs["target_object"])[:2]))
        expert = PickPlaceExpert(env, attempt_config(args, attempt_index))
        steps_taken += sum(
            int(getattr(expert.config, name))
            for name in ("approach_steps", "descend_steps", "close_steps", "lift_steps", "transfer_steps", "place_descend_steps", "open_steps", "retreat_steps", "hold_steps")
        )
        plan = expert.plan_from_positions(current_position, target_position, target_geom=target_geom)
        final_summary = expert.execute(
            plan,
            viewer=viewer,
            record_step=(lambda _action, _env: recorder.sync()) if recorder is not None else None,
            speed=args.speed if viewer is not None else 0.0,
        )
        after_image = render_top_rgb(env, args.image_size, args.camera)
        target_status = visual_target_status(after_image, calibration, current_name, target_geom)
        attempt_logs.append(
            {
                "attempt": attempt_index + 1,
                "source_name": current_name,
                "source_position": current_position.round(5).tolist(),
                "source_position_error_m": source_error,
                "detection_area_px": int(detection.area),
                "visual_target_status": target_status,
                "evaluation_strict_grasp_success": bool(final_summary.get("strict_grasp_success", False)),
                "evaluation_target_distance_m": float(final_summary.get("target_distance", float("nan"))),
            }
        )
        if target_status["complete"]:
            recovery_reason = "visual_target_confirmed"
            break
        if attempt_index >= args.feedback_attempts:
            if args.feedback_attempts:
                recovery_reason = "retry_budget_exhausted"
            elif not target_status["verifiable"]:
                recovery_reason = "target_visual_status_ambiguous_no_retry"
            break
        try:
            current_position, detection = relocate_known_object(
                after_image,
                calibration,
                current_name,
                current_position[:2],
                search_scope=getattr(args, "recovery_search", "source"),
            )
        except LookupError:
            scope = getattr(args, "recovery_search", "source")
            recovery_reason = f"{scope}_object_not_visually_recoverable" if target_status["verifiable"] else f"same_color_target_ambiguous_{scope}_object_not_found"
            break
        scope = getattr(args, "recovery_search", "source")
        recovery_reason = f"{scope}_object_visually_relocalized" if target_status["verifiable"] else f"same_color_target_ambiguous_{scope}_object_relocalized"
    metrics = env.metrics()
    final_source_error = float(attempt_logs[-1]["source_position_error_m"])
    return {
        "seed": seed,
        "task": args.task,
        "complexity": args.complexity,
        "instruction": instruction,
        "normalized_instruction": normalized,
        "predicted_intent": intent,
        "semantic_correct": semantic_correct,
        "selected_object": selected_name,
        "oracle_target_object": obs["target_object"],
        "visual_selection_correct": bool(initial_source_error <= 0.04),
        "initial_source_position_error_m": initial_source_error,
        "feedback_attempts_allowed": int(args.feedback_attempts),
        "recovery_profile": args.recovery_profile,
        "attempt_count": len(attempt_logs),
        "recovery_triggered": len(attempt_logs) > 1,
        "recovery_reason": recovery_reason,
        "initial_detection_area_px": initial_detection_area,
        "final_source_position_error_m": final_source_error,
        "runtime_position_source": "top_rgb + offline plane calibration; MuJoCo object/target state is not used for retry decisions or trajectory planning",
        "target_source": "fixed scene target configuration",
        "attempt_logs": attempt_logs,
        "logits": logits,
        "success": bool(final_summary.get("success", False)),
        "task_success": bool(final_summary.get("success", False) and semantic_correct and selected_name == obs["target_object"]),
        "strict_grasp_success": bool(final_summary.get("strict_grasp_success", False)),
        "steps_taken": steps_taken,
        "target_distance": float(metrics["target_distance"]),
        "out_of_table": bool(metrics["out_of_table"]),
    }


def main() -> None:
    args = parse_args()
    calibration = load_calibration(args.calibration)
    if args.intent_source == "clip":
        if args.model is None:
            raise ValueError("--model is required when --intent-source=clip")
        from torch_runtime import ensure_torch_path
        from vlm_runtime import ensure_vlm_path

        ensure_torch_path()
        ensure_vlm_path()
        from run_clip_action_head import load_clip
        from run_clip_semantic_waypoint import load_policy, predict_intent

        policy = load_policy(args.model)
        clip_model, processor = load_clip(str(policy["metadata"]["clip_model"]))
        intent_predictor = predict_intent
    else:
        policy, clip_model, processor, intent_predictor = {}, None, None, None
    summaries: list[dict] = []
    if args.viewer:
        env, obs = configure_env(args, args.seed)
        recorder = Mp4FrameRecorder(env, args.video_path, args.width, args.height, args.fps, args.frame_stride, args.camera) if args.video_path else None
        if recorder is not None:
            recorder.capture()
        try:
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                summary = rollout(args, policy, clip_model, processor, calibration, args.seed, viewer=viewer, recorder=recorder, env=env, obs=obs, intent_predictor=intent_predictor)
                summaries.append(summary)
                print("episode_summary:", json.dumps(summary, ensure_ascii=False), flush=True)
                started = time.time()
                while viewer.is_running() and time.time() - started < args.duration:
                    viewer.sync()
                    time.sleep(0.01)
        finally:
            if recorder is not None:
                recorder.close()
        start = 1
    else:
        start = 0
    for offset in range(start, args.episodes):
        env, obs = configure_env(args, args.seed + offset)
        recorder = None
        if args.video_path and offset == 0:
            recorder = Mp4FrameRecorder(env, args.video_path, args.width, args.height, args.fps, args.frame_stride, args.camera)
            recorder.capture()
        try:
            summary = rollout(args, policy, clip_model, processor, calibration, args.seed + offset, recorder=recorder, env=env, obs=obs, intent_predictor=intent_predictor)
        finally:
            if recorder is not None:
                recorder.close()
        summaries.append(summary)
        print("episode_summary:", json.dumps(summary, ensure_ascii=False), flush=True)
    if args.video_path:
        args.video_path.with_suffix(".json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"video_path: {args.video_path}", flush=True)
    print(f"task_success_rate: {sum(int(row['task_success']) for row in summaries)}/{len(summaries)}", flush=True)


if __name__ == "__main__":
    main()
