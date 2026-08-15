from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from torch_runtime import ensure_torch_path  # noqa: E402
from vlm_runtime import ensure_vlm_path  # noqa: E402

ensure_torch_path()
ensure_vlm_path()

from run_clip_action_head import load_clip  # noqa: E402
from run_clip_semantic_waypoint import INTENTS, load_policy, normalize_instruction, predict_intent  # noqa: E402
from widowx_env import TASKS, WidowXTabletopEnv  # noqa: E402
from widowx_env.scripted_expert import PickPlaceConfig, PickPlaceExpert  # noqa: E402
from widowx_env.vision_grounding import load_calibration, locate_leftmost_cube, locate_object  # noqa: E402


STATIC_TARGETS = {
    "target_blue_pad": np.array([0.50, 0.16, 0.002], dtype=float),
    "target_red_pad": np.array([0.50, -0.16, 0.002], dtype=float),
    "target_bowl": np.array([0.33, 0.25, 0.006], dtype=float),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CLIP intent selection with RGB-grounded object localization and a structured executor.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=ROOT / "outputs" / "rgb_grounding" / "top_rgb_core_v2_calibration_v1.json")
    parser.add_argument("--task", choices=sorted(TASKS), default="place_blue_cube_blue_pad")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--instruction-normalization", choices=("none", "desktop_alias_v1"), default="none")
    parser.add_argument("--complexity", choices=("easy", "medium", "hard", "language"), default="medium")
    parser.add_argument("--workspace-profile", choices=("legacy", "core_v2"), default="core_v2")
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--episodes", type=int, default=1)
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
    parser.set_defaults(viewer=True)
    return parser.parse_args()


def configure_env(args: argparse.Namespace, seed: int) -> tuple[WidowXTabletopEnv, dict]:
    env = WidowXTabletopEnv(seed=seed, image_size=(args.image_size, args.image_size), camera=args.camera, workspace_profile=args.workspace_profile)
    env.set_arm_actuator_strength(kp=args.arm_kp, force_limit=args.arm_force)
    env.set_gripper_actuator_strength(kp=args.gripper_kp, force_limit=args.gripper_force)
    env.set_grasp_contact_friction(sliding=args.friction)
    return env, env.reset(task=args.task, complexity=args.complexity, seed=seed)


def locate_source_from_rgb(env: WidowXTabletopEnv, intent: str, calibration, image_size: int, camera: str):
    renderer = mujoco.Renderer(env.model, height=image_size, width=image_size)
    try:
        renderer.update_scene(env.data, camera=camera)
        image = renderer.render().copy()
    finally:
        renderer.close()
    source_name, _ = INTENTS[intent]
    if source_name == "leftmost_cube":
        return locate_leftmost_cube(image, calibration)
    position, detection = locate_object(image, calibration, source_name)
    return source_name, position, detection


def rollout(args: argparse.Namespace, policy: dict, clip_model, processor, calibration, seed: int, viewer=None, env=None, obs=None) -> dict:
    if env is None or obs is None:
        env, obs = configure_env(args, seed)
    instruction = str(args.instruction or obs["instruction"])
    normalized = normalize_instruction(instruction) if args.instruction_normalization == "desktop_alias_v1" else instruction
    prediction_obs = {**obs, "instruction": normalized}
    intent, logits = predict_intent(env, prediction_obs, policy, clip_model, processor, args.image_size, args.camera)
    selected_name, source_position, detection = locate_source_from_rgb(env, intent, calibration, args.image_size, args.camera)
    _, target_geom = INTENTS[intent]
    target_position = STATIC_TARGETS[target_geom]
    source_position_error_m = float(np.linalg.norm(source_position[:2] - env.object_position(obs["target_object"])[:2]))
    expert = PickPlaceExpert(env, PickPlaceConfig(place_tcp_z=args.place_tcp_z))
    plan = expert.plan_from_positions(source_position, target_position, target_geom=target_geom)
    summary = expert.execute(plan, viewer=viewer, speed=args.speed if viewer is not None else 0.0)
    metrics = env.metrics()
    return {
        "seed": seed,
        "task": args.task,
        "instruction": instruction,
        "normalized_instruction": normalized,
        "predicted_intent": intent,
        "semantic_correct": intent == args.task,
        "selected_object": selected_name,
        "oracle_target_object": obs["target_object"],
        "visual_selection_correct": selected_name == obs["target_object"],
        "visual_source_position": source_position.round(5).tolist(),
        "source_position_error_m": source_position_error_m,
        "detection_area_px": int(detection.area),
        "detection_fill_ratio": float(detection.fill_ratio),
        "runtime_position_source": "top_rgb + offline plane calibration; no MuJoCo object/target position used in planning",
        "target_source": "fixed scene target configuration, not dynamic MuJoCo state",
        "logits": logits,
        "success": bool(summary["success"]),
        "task_success": bool(summary["success"] and intent == args.task and selected_name == obs["target_object"]),
        "strict_grasp_success": bool(summary["strict_grasp_success"]),
        "target_distance": float(metrics["target_distance"]),
        "out_of_table": bool(metrics["out_of_table"]),
    }


def main() -> None:
    args = parse_args()
    policy = load_policy(args.model)
    calibration = load_calibration(args.calibration)
    clip_model, processor = load_clip(str(policy["metadata"]["clip_model"]))
    summaries: list[dict] = []
    if args.viewer:
        env, obs = configure_env(args, args.seed)
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            summary = rollout(args, policy, clip_model, processor, calibration, args.seed, viewer=viewer, env=env, obs=obs)
            summaries.append(summary)
            print("episode_summary:", json.dumps(summary, ensure_ascii=False), flush=True)
            started = time.time()
            while viewer.is_running() and time.time() - started < args.duration:
                viewer.sync()
                time.sleep(0.01)
        start = 1
    else:
        start = 0
    for offset in range(start, args.episodes):
        summary = rollout(args, policy, clip_model, processor, calibration, args.seed + offset)
        summaries.append(summary)
        print("episode_summary:", json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"task_success_rate: {sum(int(row['task_success']) for row in summaries)}/{len(summaries)}", flush=True)


if __name__ == "__main__":
    main()
