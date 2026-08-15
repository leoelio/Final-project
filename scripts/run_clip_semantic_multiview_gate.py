from __future__ import annotations

import argparse
from dataclasses import replace
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
from run_clip_action_head import clip_tensor  # noqa: E402
from run_clip_semantic_waypoint import INTENTS, load_policy, predict_intent  # noqa: E402
from widowx_env import TASKS, WidowXTabletopEnv  # noqa: E402
from widowx_env.multiview_features import extract_multiview_features  # noqa: E402
from widowx_env.rollout_video import Mp4FrameRecorder  # noqa: E402
from widowx_env.scripted_expert import PickPlaceConfig, PickPlaceExpert  # noqa: E402
from widowx_env.vision_grounding import load_calibration, locate_leftmost_cube, locate_object, relocate_known_object  # noqa: E402


STATIC_TARGETS = {
    "target_blue_pad": np.asarray([0.50, 0.16, 0.002], dtype=float),
    "target_red_pad": np.asarray([0.50, -0.16, 0.002], dtype=float),
    "target_bowl": np.asarray([0.33, 0.25, 0.006], dtype=float),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen CLIP semantics with a compact visual terminal/recovery gate.")
    parser.add_argument("--model", type=Path, required=True, help="Frozen CLIP intent-adapter model.")
    parser.add_argument("--terminal-head", type=Path, required=True)
    parser.add_argument("--recovery-head", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=ROOT / "outputs" / "rgb_grounding" / "top_rgb_core_v2_calibration_v1.json")
    parser.add_argument("--task", choices=sorted(INTENTS), default="move_leftmost_cube_to_bowl")
    parser.add_argument("--complexity", choices=("easy", "medium", "hard", "language"), default="language")
    parser.add_argument("--seed", type=int, default=750)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--workspace-profile", choices=("legacy", "core_v2"), default="core_v2")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--arm-kp", type=float, default=150.0)
    parser.add_argument("--arm-force", type=float, default=100.0)
    parser.add_argument("--gripper-kp", type=float, default=1200.0)
    parser.add_argument("--gripper-force", type=float, default=200.0)
    parser.add_argument("--friction", type=float, default=5.0)
    parser.add_argument("--place-tcp-z", type=float, default=0.041)
    parser.add_argument("--recovery-mode", choices=("rule", "head", "clip_head"), default="rule", help="rule is RGB re-localization; head is the sparse hand-feature ablation; clip_head is the frozen-CLIP recovery adapter.")
    parser.add_argument("--recovery-profile", choices=("standard", "deep_tight_slow"), default="standard", help="Trajectory used only for a visually re-localized second attempt.")
    parser.add_argument("--clip-recovery-head", type=Path, default=None)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-viewer", dest="viewer", action="store_false")
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--speed", type=float, default=0.25)
    parser.add_argument("--video-path", type=Path, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--frame-stride", type=int, default=12)
    parser.set_defaults(viewer=True)
    return parser.parse_args()


def configure_env(args: argparse.Namespace, seed: int) -> tuple[WidowXTabletopEnv, dict]:
    env = WidowXTabletopEnv(seed=seed, image_size=(args.image_size, args.image_size), camera="top_rgb", workspace_profile=args.workspace_profile)
    env.set_arm_actuator_strength(kp=args.arm_kp, force_limit=args.arm_force)
    env.set_gripper_actuator_strength(kp=args.gripper_kp, force_limit=args.gripper_force)
    env.set_grasp_contact_friction(sliding=args.friction)
    return env, env.reset(task=args.task, complexity=args.complexity, seed=seed)


def render_rgb(env: WidowXTabletopEnv, camera: str, image_size: int) -> np.ndarray:
    renderer = mujoco.Renderer(env.model, height=image_size, width=image_size)
    try:
        renderer.update_scene(env.data, camera=camera)
        return renderer.render().copy()
    finally:
        renderer.close()


def initial_source(top_rgb: np.ndarray, calibration, intent: str) -> tuple[str, np.ndarray, object]:
    source_kind, _ = INTENTS[intent]
    if source_kind == "leftmost_cube":
        return locate_leftmost_cube(top_rgb, calibration)
    position, detection = locate_object(top_rgb, calibration, source_kind)
    return source_kind, position, detection


def load_head(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {
            "w0": data["w0"].astype(np.float32),
            "b0": data["b0"].astype(np.float32),
            "w1": data["w1"].astype(np.float32),
            "b1": data["b1"].astype(np.float32),
            "x_mean": data["x_mean"].astype(np.float32),
            "x_std": data["x_std"].astype(np.float32),
            "metadata": json.loads(data["metadata"].item()),
        }


def predict_head(head: dict, full_features: np.ndarray) -> dict:
    indices = np.asarray(head["metadata"]["feature_indices"], dtype=int)
    x = ((full_features[indices] - head["x_mean"]) / head["x_std"])[None, :]
    hidden = np.maximum(0.0, x @ head["w0"] + head["b0"])
    logits = (hidden @ head["w1"] + head["b1"])[0]
    logits -= logits.max()
    probabilities = np.exp(logits) / np.exp(logits).sum()
    labels = list(head["metadata"]["class_names"])
    index = int(probabilities.argmax())
    return {"label": labels[index], "probabilities": {label: float(probabilities[i]) for i, label in enumerate(labels)}}


def load_clip_recovery_head(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {
            "down_weight": data["down_weight"].astype(np.float32),
            "down_bias": data["down_bias"].astype(np.float32),
            "up_weight": data["up_weight"].astype(np.float32),
            "up_bias": data["up_bias"].astype(np.float32),
            "x_mean": data["x_mean"].astype(np.float32),
            "x_std": data["x_std"].astype(np.float32),
            "metadata": json.loads(data["metadata"].item()),
        }


def recovery_config(args: argparse.Namespace) -> PickPlaceConfig:
    base = PickPlaceConfig(place_tcp_z=args.place_tcp_z)
    if args.recovery_profile == "deep_tight_slow":
        return replace(base, grasp_z_offset=0.0, close_gripper=0.007, close_steps=420, descend_steps=300, lift_steps=500, transfer_steps=800)
    return base


def predict_clip_recovery_head(head: dict, clip_model, processor, top_rgb: np.ndarray, front_rgb: np.ndarray, instruction: str) -> dict:
    from PIL import Image
    import torch

    images = [Image.fromarray(top_rgb.astype(np.uint8), mode="RGB")]
    if head["metadata"]["view"] == "top_front":
        images.append(Image.fromarray(front_rgb.astype(np.uint8), mode="RGB"))
    inputs = processor(images=images, return_tensors="pt")
    device = next(clip_model.parameters()).device
    with torch.no_grad():
        features = clip_tensor(clip_model.get_image_features(pixel_values=inputs["pixel_values"].to(device)))
        features = torch.nn.functional.normalize(features, dim=-1).cpu().numpy().astype(np.float32).reshape(-1)
        if head["metadata"].get("uses_instruction", False):
            text_inputs = processor(text=[instruction], padding=True, return_tensors="pt")
            text_features = clip_tensor(
                clip_model.get_text_features(
                    input_ids=text_inputs["input_ids"].to(device),
                    attention_mask=text_inputs["attention_mask"].to(device),
                )
            )
            text_features = torch.nn.functional.normalize(text_features, dim=-1).cpu().numpy().astype(np.float32).reshape(-1)
            features = np.concatenate([features, text_features], axis=0)
    x = (features - head["x_mean"]) / head["x_std"]
    hidden = np.maximum(0.0, x @ head["down_weight"] + head["down_bias"])
    logits = hidden @ head["up_weight"] + head["up_bias"]
    logits -= logits.max()
    probabilities = np.exp(logits) / np.exp(logits).sum()
    labels = list(head["metadata"]["class_names"])
    index = int(probabilities.argmax())
    return {"label": labels[index], "probabilities": {label: float(probabilities[i]) for i, label in enumerate(labels)}}


def rollout(args: argparse.Namespace, policy: dict, clip_model, processor, calibration, terminal_head: dict, recovery_head: dict, seed: int, viewer=None, recorder=None, env=None, obs=None, clip_recovery_head: dict | None = None) -> dict:
    if env is None or obs is None:
        env, obs = configure_env(args, seed)
    intent, logits = predict_intent(env, obs, policy, clip_model, processor, args.image_size, "top_rgb")
    source_name, source_position, detection = initial_source(render_rgb(env, "top_rgb", args.image_size), calibration, intent)
    _, target_name = INTENTS[intent]
    target_position = STATIC_TARGETS[target_name]
    expert = PickPlaceExpert(env, PickPlaceConfig(place_tcp_z=args.place_tcp_z))
    first = expert.execute(
        expert.plan_from_positions(source_position, target_position, target_geom=target_name),
        viewer=viewer,
        record_step=(lambda _action, _env: recorder.sync()) if recorder is not None else None,
        speed=args.speed if viewer is not None else 0.0,
    )
    top_after = render_rgb(env, "top_rgb", args.image_size)
    front_after = render_rgb(env, "front_rgb", args.image_size)
    features = extract_multiview_features(top_after, front_after, intent, source_name, target_name, target_position[:2], calibration, pool_grid=4)
    terminal_prediction = predict_head(terminal_head, features)
    recovery_prediction = predict_head(recovery_head, features)
    clip_recovery_prediction = None
    if args.recovery_mode == "clip_head":
        if clip_recovery_head is None:
            raise ValueError("--clip-recovery-head is required when --recovery-mode clip_head")
        clip_recovery_prediction = predict_clip_recovery_head(clip_recovery_head, clip_model, processor, top_after, front_after, obs["instruction"])
    active_recovery_prediction = clip_recovery_prediction or recovery_prediction
    decision = "accept_terminal" if terminal_prediction["label"] == "complete" else f"stop_{active_recovery_prediction['label']}"
    retry = None
    should_try = terminal_prediction["label"] != "complete" and (args.recovery_mode == "rule" or active_recovery_prediction["label"] == "retry")
    if should_try:
        try:
            retry_position, retry_detection = relocate_known_object(top_after, calibration, source_name, source_position[:2])
            retry_expert = PickPlaceExpert(env, recovery_config(args))
            retry = retry_expert.execute(
                retry_expert.plan_from_positions(retry_position, target_position, target_geom=target_name),
                viewer=viewer,
                record_step=(lambda _action, _env: recorder.sync()) if recorder is not None else None,
                speed=args.speed if viewer is not None else 0.0,
            )
            decision = f"{args.recovery_mode}_retry_executed"
            retry_area = int(retry_detection.area)
        except (LookupError, ValueError) as error:
            decision = f"stop_{args.recovery_mode}_retry_source_not_visual"
            retry_area = None
            retry = {"error": str(error), "success": False}
    else:
        retry_area = None
    final = retry if retry is not None and "error" not in retry else first
    metrics = env.metrics()
    return {
        "seed": seed,
        "task": args.task,
        "complexity": args.complexity,
        "instruction": obs["instruction"],
        "predicted_intent": intent,
        "semantic_correct_evaluation": bool(intent == args.task),
        "selected_object": source_name,
        "visual_selection_correct_evaluation": bool(source_name == obs["target_object"]),
        "initial_detection_area_px": int(detection.area),
        "first_execution_evaluation_success": bool(first["success"]),
        "terminal_prediction": terminal_prediction,
        "recovery_prediction": active_recovery_prediction,
        "hand_feature_recovery_prediction": recovery_prediction,
        "clip_recovery_prediction": clip_recovery_prediction,
        "gate_decision": decision,
        "recovery_mode": args.recovery_mode,
        "recovery_profile": args.recovery_profile,
        "retry_executed": decision.endswith("retry_executed"),
        "retry_detection_area_px": retry_area,
        "retry_execution_evaluation_success": None if retry is None else bool(retry.get("success", False)),
        "runtime_input_boundary": "Gate decisions use only top_rgb, front_rgb, static task configuration, and frozen semantic intent. MuJoCo object state is not used for decisions or action planning.",
        "task_success": bool(final.get("success", False) and intent == args.task and source_name == obs["target_object"]),
        "strict_grasp_success": bool(final.get("strict_grasp_success", False)),
        "target_distance_m": float(metrics["target_distance"]),
        "out_of_table": bool(metrics["out_of_table"]),
        "semantic_logits": logits,
    }


def main() -> None:
    args = parse_args()
    policy = load_policy(args.model)
    calibration = load_calibration(args.calibration)
    clip_model, processor = load_clip(str(policy["metadata"]["clip_model"]))
    terminal_head = load_head(args.terminal_head)
    recovery_head = load_head(args.recovery_head)
    clip_recovery_head = load_clip_recovery_head(args.clip_recovery_head) if args.clip_recovery_head else None
    summaries = []
    if args.viewer:
        env, obs = configure_env(args, args.seed)
        recorder = Mp4FrameRecorder(env, args.video_path, args.width, args.height, args.fps, args.frame_stride, "top_rgb") if args.video_path else None
        if recorder is not None:
            recorder.capture()
        try:
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                summary = rollout(args, policy, clip_model, processor, calibration, terminal_head, recovery_head, args.seed, viewer, recorder, env, obs, clip_recovery_head)
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
        summaries.append(rollout(args, policy, clip_model, processor, calibration, terminal_head, recovery_head, args.seed + offset, env=env, obs=obs, clip_recovery_head=clip_recovery_head))
        print("episode_summary:", json.dumps(summaries[-1], ensure_ascii=False), flush=True)
    if args.video_path:
        args.video_path.with_suffix(".json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"video_path: {args.video_path}", flush=True)
    print(f"task_success_rate: {sum(int(item['task_success']) for item in summaries)}/{len(summaries)}", flush=True)


if __name__ == "__main__":
    main()
