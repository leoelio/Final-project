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
from transformers import CLIPModel, CLIPProcessor  # noqa: E402

from run_bc_policy import postprocess_action, print_step, unsafe_reason  # noqa: E402
from run_mlp_policy import forward  # noqa: E402
from widowx_env import TASKS, WidowXTabletopEnv  # noqa: E402
from widowx_env.demo_dataset import phase_features, read_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a frozen CLIP image/text encoder + lightweight action head policy.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--task", choices=sorted(TASKS), default=None)
    parser.add_argument("--complexity", choices=("easy", "medium", "hard", "language"), default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2840)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-viewer", dest="viewer", action="store_false")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--speed", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--camera", default="top_rgb")
    parser.add_argument("--workspace-profile", choices=("legacy", "core_v2"), default="legacy")
    parser.add_argument("--vision-interval", type=int, default=32)
    parser.add_argument("--arm-kp", type=float, default=150.0)
    parser.add_argument("--arm-force", type=float, default=100.0)
    parser.add_argument("--gripper-kp", type=float, default=800.0)
    parser.add_argument("--gripper-force", type=float, default=140.0)
    parser.add_argument("--friction", type=float, default=3.0)
    parser.add_argument("--clip-actions", action="store_true")
    parser.add_argument("--no-clip-actions", dest="clip_actions", action="store_false")
    parser.add_argument("--action-alpha", type=float, default=0.2)
    parser.add_argument("--max-arm-delta", type=float, default=0.01)
    parser.add_argument("--max-gripper-delta", type=float, default=0.0005)
    parser.add_argument("--stop-on-unsafe", action="store_true")
    parser.add_argument("--no-stop-on-unsafe", dest="stop_on_unsafe", action="store_false")
    parser.add_argument("--log-every", type=int, default=500)
    parser.set_defaults(clip_actions=True)
    parser.set_defaults(stop_on_unsafe=True)
    parser.set_defaults(viewer=True)
    return parser.parse_args()


def latest_model() -> Path:
    candidates = sorted((ROOT / "outputs" / "clip_action_head").glob("clip_action_head_lite_*.npz"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no CLIP action-head models found under {ROOT / 'outputs' / 'clip_action_head'}")
    return candidates[-1]


def load_policy(path: Path) -> dict:
    with np.load(path) as data:
        metadata = json.loads(data["metadata"].item())
        layers = []
        for index in range(len(metadata["hidden_sizes"]) + 1):
            layers.append({"w": data[f"w{index}"].astype(np.float32), "b": data[f"b{index}"].astype(np.float32)})
        return {
            "layers": layers,
            "x_mean": data["x_mean"].astype(np.float32),
            "x_std": data["x_std"].astype(np.float32),
            "y_mean": data["y_mean"].astype(np.float32),
            "y_std": data["y_std"].astype(np.float32),
            "action_min": data["action_min"].astype(np.float32),
            "action_max": data["action_max"].astype(np.float32),
            "metadata": metadata,
        }


def load_clip(name: str) -> tuple[CLIPModel, CLIPProcessor]:
    processor = CLIPProcessor.from_pretrained(name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(name).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, processor


def clip_tensor(output):
    return output.pooler_output if hasattr(output, "pooler_output") else output


def infer_task_defaults(model: dict) -> tuple[str, str]:
    first = read_metadata(Path(model["metadata"]["run_dir"]))[0]
    return str(first["task"]), str(first["complexity"])


def encode_clip(model: CLIPModel, processor: CLIPProcessor, rgb: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        image_features = clip_tensor(model.get_image_features(pixel_values=inputs["pixel_values"]))
        text_features = clip_tensor(model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]))
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        return torch.cat([image_features, text_features], dim=-1)[0].cpu().numpy().astype(np.float32)


def make_feature(clip_feature: np.ndarray, env: WidowXTabletopEnv, phase: float) -> np.ndarray:
    proprio = np.concatenate(
        [
            env.data.qpos[: env.robot_nq].astype(np.float32),
            env.data.qvel[: env.robot_nv].astype(np.float32),
            env.data.ctrl.astype(np.float32),
            phase_features(phase),
        ]
    ).astype(np.float32)
    return np.concatenate([clip_feature, proprio], axis=0).astype(np.float32)


def predict_action(policy: dict, feature: np.ndarray) -> np.ndarray:
    x = ((feature - policy["x_mean"]) / policy["x_std"])[None, :].astype(np.float32)
    y = forward(policy["layers"], x)[0].astype(np.float32)
    return y * policy["y_std"] + policy["y_mean"]


def configure_env(args: argparse.Namespace, seed: int, task: str, complexity: str) -> tuple[WidowXTabletopEnv, dict]:
    env = WidowXTabletopEnv(
        seed=seed,
        image_size=(args.image_size, args.image_size),
        camera=args.camera,
        workspace_profile=args.workspace_profile,
    )
    env.set_arm_actuator_strength(kp=args.arm_kp, force_limit=args.arm_force)
    env.set_gripper_actuator_strength(kp=args.gripper_kp, force_limit=args.gripper_force)
    env.set_grasp_contact_friction(sliding=args.friction)
    obs = env.reset(task=task, complexity=complexity, seed=seed)
    return env, obs


def rollout_with_env(
    args: argparse.Namespace,
    policy: dict,
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    env: WidowXTabletopEnv,
    obs: dict,
    seed: int,
    task: str,
    complexity: str,
    viewer=None,
) -> dict:
    ctrl_min = env.model.actuator_ctrlrange[:, 0]
    ctrl_max = env.model.actuator_ctrlrange[:, 1]
    dt = float(env.model.opt.timestep)
    previous_action = env.data.ctrl.copy()
    action_norms = []
    stop_reason = None
    steps_taken = 0
    phase_denom = max(1, args.steps - 1)
    vision_interval = max(1, int(args.vision_interval))
    renderer = mujoco.Renderer(env.model, height=args.image_size, width=args.image_size)
    clip_feature = None
    instruction = str(obs["instruction"])
    try:
        for step in range(args.steps):
            phase = step / phase_denom
            if clip_feature is None or step % vision_interval == 0:
                renderer.update_scene(env.data, camera=env.camera)
                clip_feature = encode_clip(clip_model, processor, renderer.render(), instruction)
            raw_action = predict_action(policy, make_feature(clip_feature, env, phase))
            action = postprocess_action(args, policy, raw_action, previous_action, ctrl_min, ctrl_max)
            delta = action - previous_action
            env.step(action)
            previous_action = action
            action_norms.append(float(np.linalg.norm(action)))
            steps_taken = step + 1
            metrics = env.metrics()
            if args.log_every > 0 and (step == 0 or (step + 1) % args.log_every == 0):
                print_step(step + 1, metrics, action, delta)
            stop_reason = unsafe_reason(metrics) if args.stop_on_unsafe else None
            if viewer is not None:
                viewer.sync()
                if args.speed > 0:
                    time.sleep(dt / args.speed)
            if stop_reason:
                print(f"stopped_early: step={step + 1}, reason={stop_reason}", flush=True)
                break
    finally:
        renderer.close()

    metrics = env.metrics()
    return {
        "seed": seed,
        "task": task,
        "complexity": complexity,
        "instruction": obs["instruction"],
        "active_objects": list(obs["active_objects"]),
        "success": bool(metrics["success"]),
        "target_distance": float(metrics["target_distance"]),
        "object_z": float(metrics["object_z"]),
        "grasp_success": bool(metrics["grasp_success"]),
        "out_of_table": bool(metrics["out_of_table"]),
        "steps_taken": steps_taken,
        "stop_reason": stop_reason,
        "mean_action_norm": float(np.mean(action_norms)),
        "max_action_norm": float(np.max(action_norms)),
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    model_path = args.model or latest_model()
    policy = load_policy(model_path)
    clip_model, processor = load_clip(str(policy["metadata"]["clip_model"]))
    default_task, default_complexity = infer_task_defaults(policy)
    task = args.task or default_task
    complexity = args.complexity or default_complexity

    print(f"model_path: {model_path}", flush=True)
    print(f"clip_model: {policy['metadata']['clip_model']}", flush=True)
    print(f"feature_dim: {policy['metadata']['feature_dim']}", flush=True)
    print(f"frozen_encoder_params: {policy['metadata'].get('frozen_encoder_params')}", flush=True)
    print(f"task: {task}", flush=True)
    print(f"complexity: {complexity}", flush=True)

    summaries = []
    if args.viewer:
        env, obs = configure_env(args, args.seed, task, complexity)
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            summary = rollout_with_env(args, policy, clip_model, processor, env, obs, args.seed, task, complexity, viewer)
            summaries.append(summary)
            print("episode_summary:", summary, flush=True)
            start = time.time()
            while viewer.is_running():
                viewer.sync()
                if args.duration and time.time() - start > args.duration:
                    break
                time.sleep(0.01)
        start_episode = 1
    else:
        start_episode = 0

    for episode in range(start_episode, args.episodes):
        seed = args.seed + episode
        env, obs = configure_env(args, seed, task, complexity)
        summary = rollout_with_env(args, policy, clip_model, processor, env, obs, seed, task, complexity)
        summaries.append(summary)
        print("episode_summary:", summary, flush=True)

    successes = sum(int(item["success"]) for item in summaries)
    print(f"success_rate: {successes}/{len(summaries)} = {successes / max(1, len(summaries)):.3f}", flush=True)


if __name__ == "__main__":
    main()
