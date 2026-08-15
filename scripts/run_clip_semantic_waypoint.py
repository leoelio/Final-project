from __future__ import annotations

import argparse
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

from torch_runtime import ensure_torch_path  # noqa: E402
from vlm_runtime import ensure_vlm_path  # noqa: E402

ensure_torch_path()
ensure_vlm_path()

from run_clip_action_head import encode_clip, load_clip  # noqa: E402
from train_clip_semantic_waypoint import TASK_LABELS  # noqa: E402
from widowx_env import TASKS, WidowXTabletopEnv  # noqa: E402
from widowx_env.scripted_expert import ContactFusionPickPlaceExpert, ContactFusionConfig, PickPlaceConfig, PickPlaceExpert  # noqa: E402


INTENTS = {
    "place_blue_cube_blue_pad": ("blue_cube", "target_blue_pad"),
    "place_blue_cube_red_pad": ("blue_cube", "target_red_pad"),
    "place_red_cube_red_pad": ("red_cube", "target_red_pad"),
    "move_leftmost_cube_to_bowl": ("leftmost_cube", "target_bowl"),
}
CUBES = ("red_cube", "blue_cube", "green_cube", "yellow_cube")


def normalize_instruction(instruction: str) -> str:
    """Map the tabletop task's closed vocabulary to the training vocabulary."""
    normalized = instruction.lower()
    replacements = (
        (r"furthest to the left", "leftmost"),
        (r"left-most", "leftmost"),
        (r"westernmost", "leftmost"),
        (r"azure|cobalt|navy|cerulean", "blue"),
        (r"crimson|scarlet|ruby|vermilion", "red"),
        (r"cuboid|block|object", "cube"),
        (r"disk|circle|mat|platform", "pad"),
        (r"container|vessel|receptacle", "bowl"),
    )
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized)
    if "blue" in normalized:
        normalized = normalized.replace("matching pad", "blue pad")
    elif "red" in normalized:
        normalized = normalized.replace("matching pad", "red pad")
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen CLIP intent selection with a structured Core V2 waypoint executor.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(TASKS), default="place_blue_cube_blue_pad")
    parser.add_argument("--instruction", default=None, help="Optional evaluation-time instruction override.")
    parser.add_argument("--instruction-normalization", choices=("none", "desktop_alias_v1"), default="none")
    parser.add_argument("--executor", choices=("standard", "contact_fusion"), default="standard")
    parser.add_argument("--complexity", choices=("easy", "medium", "hard", "language"), default="medium")
    parser.add_argument("--workspace-profile", choices=("legacy", "core_v2"), default="core_v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-viewer", dest="viewer", action="store_false")
    parser.add_argument("--duration", type=float, default=0.0)
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


def load_policy(path: Path) -> dict:
    with np.load(path) as data:
        policy = {
            "x_mean": data["x_mean"].astype(np.float32),
            "x_std": data["x_std"].astype(np.float32),
            "metadata": json.loads(data["metadata"].item()),
        }
        if "weights" in data:
            policy["weights"] = data["weights"].astype(np.float32)
            policy["bias"] = data["bias"].astype(np.float32)
        else:
            policy["down_weight"] = data["down_weight"].astype(np.float32)
            policy["down_bias"] = data["down_bias"].astype(np.float32)
            policy["up_weight"] = data["up_weight"].astype(np.float32)
            policy["up_bias"] = data["up_bias"].astype(np.float32)
        return policy


def configure_env(args: argparse.Namespace, seed: int) -> tuple[WidowXTabletopEnv, dict]:
    env = WidowXTabletopEnv(seed=seed, image_size=(args.image_size, args.image_size), camera=args.camera, workspace_profile=args.workspace_profile)
    env.set_arm_actuator_strength(kp=args.arm_kp, force_limit=args.arm_force)
    env.set_gripper_actuator_strength(kp=args.gripper_kp, force_limit=args.gripper_force)
    env.set_grasp_contact_friction(sliding=args.friction)
    return env, env.reset(task=args.task, complexity=args.complexity, seed=seed)


def predict_intent(env: WidowXTabletopEnv, obs: dict, policy: dict, clip_model, processor, image_size: int, camera: str) -> tuple[str, list[float]]:
    renderer = mujoco.Renderer(env.model, height=image_size, width=image_size)
    try:
        renderer.update_scene(env.data, camera=camera)
        feature = encode_clip(clip_model, processor, renderer.render(), str(obs["instruction"]))
    finally:
        renderer.close()
    x = (feature - policy["x_mean"]) / policy["x_std"]
    if "weights" in policy:
        logits = x @ policy["weights"] + policy["bias"]
    else:
        hidden = np.maximum(0.0, x @ policy["down_weight"] + policy["down_bias"])
        logits = hidden @ policy["up_weight"] + policy["up_bias"]
    return TASK_LABELS[int(np.argmax(logits))], [float(value) for value in logits]


def resolve_target(env: WidowXTabletopEnv, obs: dict, intent: str) -> tuple[str, str]:
    object_name, target_geom = INTENTS[intent]
    if object_name != "leftmost_cube":
        return object_name, target_geom
    active = set(obs["active_objects"])
    candidates = [name for name in CUBES if name in active]
    return min(candidates, key=lambda name: float(env.data.geom_xpos[env.model.geom(f"{name}_geom").id][0])), target_geom


def rollout_episode(args: argparse.Namespace, policy: dict, clip_model, processor, seed: int, viewer=None, env=None, obs=None) -> dict:
    if env is None or obs is None:
        env, obs = configure_env(args, seed)
    instruction = getattr(args, "instruction", None)
    if instruction:
        obs = {**obs, "instruction": instruction}
    normalized_instruction = str(obs["instruction"])
    if getattr(args, "instruction_normalization", "none") == "desktop_alias_v1":
        normalized_instruction = normalize_instruction(normalized_instruction)
    prediction_obs = {**obs, "instruction": normalized_instruction}
    intent, logits = predict_intent(env, prediction_obs, policy, clip_model, processor, args.image_size, args.camera)
    object_name, target_geom = resolve_target(env, obs, intent)
    if args.executor == "contact_fusion":
        expert = ContactFusionPickPlaceExpert(env, ContactFusionConfig(place_tcp_z=args.place_tcp_z))
    else:
        expert = PickPlaceExpert(env, PickPlaceConfig(place_tcp_z=args.place_tcp_z))
    summary = expert.execute(expert.plan(object_name, target_geom), viewer=viewer, speed=args.speed if viewer is not None else 0.0)
    metrics = env.metrics()
    return {
        "seed": seed,
        "task": args.task,
        "complexity": args.complexity,
        "instruction": obs["instruction"],
        "normalized_instruction": normalized_instruction,
        "executor": args.executor,
        "predicted_intent": intent,
        "semantic_correct": intent == args.task,
        "selected_object": object_name,
        "target_geom": target_geom,
        "logits": logits,
        "success": bool(summary["success"]),
        "task_success": bool(summary["success"] and intent == args.task),
        "placed": bool(summary.get("placed", summary["success"])),
        "strict_grasp_success": bool(summary.get("strict_grasp_success", False)),
        "contact_regrasp_attempts": int(summary.get("contact_regrasp_attempts", 0)),
        "transport_hold_confirmed": bool(summary.get("transport_hold_confirmed", False)),
        "contact_recovery_reason": str(summary.get("contact_recovery_reason", "not_applicable")),
        "max_object_z": float(summary.get("max_object_z", metrics["object_z"])),
        "lifted_steps_near_tcp": int(summary.get("lifted_steps_near_tcp", 0)),
        "target_distance": float(metrics["target_distance"]),
        # This final-state value is false after a successful release; use strict_grasp_success above.
        "grasp_success": bool(metrics["grasp_success"]),
        "out_of_table": bool(metrics["out_of_table"]),
    }


def main() -> None:
    args = parse_args()
    policy = load_policy(args.model)
    clip_model, processor = load_clip(str(policy["metadata"]["clip_model"]))
    print(f"model_path: {args.model}", flush=True)
    print(f"task: {args.task}", flush=True)
    print(f"executor: {args.executor}", flush=True)
    summaries = []
    if args.viewer:
        env, obs = configure_env(args, args.seed)
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            summary = rollout_episode(args, policy, clip_model, processor, args.seed, viewer, env, obs)
            summaries.append(summary)
            print("episode_summary:", summary, flush=True)
            started = time.time()
            while viewer.is_running() and (not args.duration or time.time() - started < args.duration):
                viewer.sync()
                time.sleep(0.01)
        start = 1
    else:
        start = 0
    for offset in range(start, args.episodes):
        summary = rollout_episode(args, policy, clip_model, processor, args.seed + offset)
        summaries.append(summary)
        print("episode_summary:", summary, flush=True)
    print(f"success_rate: {sum(int(row['success']) for row in summaries)}/{len(summaries)}", flush=True)


if __name__ == "__main__":
    main()
