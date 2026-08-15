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

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from run_clip_action_head import load_clip  # noqa: E402
from clip_lora_utils import inject_vision_lora, load_lora_state_dict  # noqa: E402
from train_clip_patch_pointer import PatchPointerHead  # noqa: E402
from train_clip_semantic_waypoint import TASK_LABELS  # noqa: E402
from widowx_env import TASKS, WidowXTabletopEnv  # noqa: E402
from widowx_env.rollout_video import Mp4FrameRecorder  # noqa: E402
from widowx_env.scripted_expert import PickPlaceConfig, PickPlaceExpert  # noqa: E402


STATIC_TARGETS = {
    "target_blue_pad": np.array([0.50, 0.16, 0.002], dtype=float),
    "target_red_pad": np.array([0.50, -0.16, 0.002], dtype=float),
    "target_bowl": np.array([0.33, 0.25, 0.006], dtype=float),
}
INTENTS = {
    "place_blue_cube_blue_pad": "target_blue_pad",
    "place_blue_cube_red_pad": "target_red_pad",
    "place_red_cube_red_pad": "target_red_pad",
    "move_leftmost_cube_to_bowl": "target_bowl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen-CLIP patch-token pointer prediction with a structured MuJoCo executor.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--complexity", choices=("easy", "medium", "hard", "language"), required=True)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-viewer", dest="viewer", action="store_false")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--speed", type=float, default=0.2)
    parser.add_argument("--workspace-profile", choices=("legacy", "core_v2"), default="core_v2")
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


def load_policy(path: Path) -> tuple[PatchPointerHead, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    model = PatchPointerHead(int(metadata["patch_dim"]), int(metadata["text_dim"]), int(metadata["hidden_size"]), len(metadata["task_labels"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def load_clip_for_policy(payload: dict):
    clip_model, processor = load_clip(str(payload["metadata"]["clip_model"]))
    config = payload["metadata"].get("lora_config")
    if config is not None:
        inject_vision_lora(clip_model, config["layer_indices"], int(config["rank"]), float(config["alpha"]))
        load_lora_state_dict(clip_model, payload["lora_state_dict"])
    clip_model.eval()
    return clip_model, processor


def configure_env(args: argparse.Namespace, seed: int) -> tuple[WidowXTabletopEnv, dict]:
    env = WidowXTabletopEnv(seed=seed, image_size=(224, 224), camera="top_rgb", workspace_profile=args.workspace_profile)
    env.set_arm_actuator_strength(kp=args.arm_kp, force_limit=args.arm_force)
    env.set_gripper_actuator_strength(kp=args.gripper_kp, force_limit=args.gripper_force)
    env.set_grasp_contact_friction(sliding=args.friction)
    return env, env.reset(task=args.task, complexity=args.complexity, seed=seed)


def predict(policy: PatchPointerHead, payload: dict, clip_model, processor, image: np.ndarray, instruction: str) -> tuple[str, np.ndarray, np.ndarray, list[float]]:
    inputs = processor(text=[instruction], images=[Image.fromarray(image.astype(np.uint8), mode="RGB")], return_tensors="pt", padding=True)
    device = next(clip_model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        patches = clip_model.vision_model(pixel_values=inputs["pixel_values"]).last_hidden_state[:, 1:, :].cpu()
        text = clip_model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        if hasattr(text, "pooler_output"):
            text = text.pooler_output
        text = torch.nn.functional.normalize(text, dim=-1).cpu()
        logits, uv, _ = policy(patches, text, int(payload["metadata"]["patch_grid_size"]))
    pixel_uv = uv[0].numpy() * float(payload["metadata"]["image_size"] - 1)
    matrix = payload["calibration_matrix"]
    world_xy = matrix[:, :2] @ pixel_uv + matrix[:, 2]
    intent = payload["metadata"]["task_labels"][int(logits.argmax(dim=1).item())]
    return intent, pixel_uv.astype(float), world_xy.astype(float), [float(value) for value in logits[0].numpy()]


def rollout_episode(args: argparse.Namespace, policy: PatchPointerHead, payload: dict, clip_model, processor, seed: int, viewer=None, recorder=None, env=None, obs=None) -> dict:
    if env is None or obs is None:
        env, obs = configure_env(args, seed)
    instruction = str(args.instruction or obs["instruction"])
    renderer = mujoco.Renderer(env.model, height=224, width=224)
    try:
        renderer.update_scene(env.data, camera="top_rgb")
        intent, pixel_uv, source_xy, logits = predict(policy, payload, clip_model, processor, renderer.render(), instruction)
    finally:
        renderer.close()
    initial_source_xy = env.object_position(obs["target_object"])[:2].copy()
    target_geom = INTENTS[intent]
    expert = PickPlaceExpert(env, PickPlaceConfig(place_tcp_z=args.place_tcp_z))
    outcome = expert.execute(
        expert.plan_from_positions(np.array([source_xy[0], source_xy[1], 0.026], dtype=float), STATIC_TARGETS[target_geom], target_geom=target_geom),
        viewer=viewer,
        record_step=(lambda _action, _env: recorder.sync()) if recorder is not None else None,
        speed=args.speed if viewer is not None else 0.0,
    )
    metrics = env.metrics()
    return {
        "seed": seed,
        "task": args.task,
        "complexity": args.complexity,
        "instruction": instruction,
        "predicted_intent": intent,
        "semantic_correct": intent == args.task,
        "predicted_pixel_uv": pixel_uv.round(3).tolist(),
        "predicted_source_xy_m": source_xy.round(5).tolist(),
        "offline_source_error_m": float(np.linalg.norm(source_xy - initial_source_xy)),
        "runtime_input_boundary": "top RGB, instruction, fixed camera-plane calibration, and model predictions only; the pre-action MuJoCo object position is retained solely for offline scoring.",
        "strict_grasp_success": bool(outcome["strict_grasp_success"]),
        "task_success": bool(outcome["success"] and intent == args.task),
        "target_distance_m": float(metrics["target_distance"]),
        "out_of_table": bool(metrics["out_of_table"]),
        "logits": logits,
    }


def main() -> None:
    args = parse_args()
    policy, payload = load_policy(args.model)
    clip_model, processor = load_clip_for_policy(payload)
    rows = []
    if args.viewer:
        env, obs = configure_env(args, args.seed)
        recorder = Mp4FrameRecorder(env, args.video_path, args.width, args.height, args.fps, args.frame_stride, "top_rgb") if args.video_path else None
        if recorder is not None:
            recorder.capture()
        try:
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                row = rollout_episode(args, policy, payload, clip_model, processor, args.seed, viewer, recorder, env, obs)
                rows.append(row)
                print("episode_summary:", json.dumps(row, ensure_ascii=False), flush=True)
                started = time.time()
                while viewer.is_running() and time.time() - started < args.duration:
                    viewer.sync()
                    time.sleep(0.01)
        finally:
            if recorder is not None:
                recorder.close()
        start_episode = 1
    else:
        start_episode = 0
    for offset in range(start_episode, args.episodes):
        row = rollout_episode(args, policy, payload, clip_model, processor, args.seed + offset)
        rows.append(row)
        print("episode_summary:", json.dumps(row, ensure_ascii=False), flush=True)
    if args.video_path:
        args.video_path.with_suffix(".json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"task_success_rate: {sum(int(row['task_success']) for row in rows)}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
