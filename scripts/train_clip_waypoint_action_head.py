from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from torch_runtime import ensure_torch_path  # noqa: E402
from vlm_runtime import ensure_vlm_path  # noqa: E402

ensure_torch_path()
ensure_vlm_path()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from run_clip_action_head import load_clip  # noqa: E402
from train_clip_action_head import clip_encode_batches, reset_to_saved_state  # noqa: E402
from train_clip_semantic_waypoint import LANGUAGE_AUGMENTATIONS, TASK_LABELS  # noqa: E402
from train_vision_language_action_head import attempt_start_index, selected_attempts  # noqa: E402
from widowx_env import WidowXTabletopEnv  # noqa: E402
from widowx_env.demo_dataset import read_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a frozen CLIP image-language encoder with a lightweight intent and 2D waypoint action head.")
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "clip_waypoint_action_head")
    parser.add_argument("--model-prefix", default="clip_waypoint_action_head_core_v2_v1")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--workspace-profile", choices=("legacy", "core_v2"), default="core_v2")
    parser.add_argument("--max-episodes-per-run", type=int, default=0, help="Use the first N successful episodes per task source; 0 keeps all.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--camera", default="top_rgb")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--language-augmentation", choices=("none", "semantic_alias_v1"), default="semantic_alias_v1")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--clip-batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


class WaypointActionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, task_count: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(input_dim, hidden_size), nn.ReLU())
        self.intent = nn.Linear(hidden_size, task_count)
        self.waypoint = nn.Linear(hidden_size, 2)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        return self.intent(hidden), self.waypoint(hidden)


def resolve_device(value: str) -> torch.device:
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def restore_initial_state(env: WidowXTabletopEnv, trajectory_path: Path, metadata: dict) -> None:
    with np.load(trajectory_path) as data:
        attempts = selected_attempts(data, metadata, include_failures=False)
        start = attempt_start_index(data, int(attempts[0]))
        reset_to_saved_state(env, data["attempt_start_qpos"][start], data["attempt_start_qvel"][start], data["attempt_start_ctrl"][start])


def collect_features(args: argparse.Namespace, clip_model, processor) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    env = WidowXTabletopEnv(seed=args.seed, image_size=(args.image_size, args.image_size), camera=args.camera, workspace_profile=args.workspace_profile)
    renderer = mujoco.Renderer(env.model, height=args.image_size, width=args.image_size)
    images: list[np.ndarray] = []
    texts: list[str] = []
    labels: list[int] = []
    waypoints: list[np.ndarray] = []
    episodes: list[int] = []
    sources: list[dict] = []
    episode_offset = 0
    try:
        for source_id, run_dir in enumerate(args.run_dirs):
            rows = [row for row in read_metadata(run_dir) if row.get("success") and str(row["task"]) in TASK_LABELS]
            rows.sort(key=lambda row: int(row["episode_index"]))
            if args.max_episodes_per_run > 0:
                rows = rows[: args.max_episodes_per_run]
            source_samples = 0
            for metadata in rows:
                task = str(metadata["task"])
                env.reset(task=task, complexity=str(metadata["complexity"]), seed=int(metadata["seed"]))
                restore_initial_state(env, run_dir / metadata["trajectory_file"], metadata)
                renderer.update_scene(env.data, camera=args.camera)
                image = renderer.render().copy()
                target = np.asarray(metadata["initial_objects"][str(metadata["target_object"])][:2], dtype=np.float32)
                instructions = [str(metadata["instruction"])]
                if args.language_augmentation == "semantic_alias_v1":
                    instructions.extend(LANGUAGE_AUGMENTATIONS[task])
                episode_id = episode_offset + int(metadata["episode_index"])
                for instruction in instructions:
                    images.append(image)
                    texts.append(instruction)
                    labels.append(TASK_LABELS.index(task))
                    waypoints.append(target)
                    episodes.append(episode_id)
                    source_samples += 1
            sources.append({"run_dir": str(run_dir), "successful_episodes": len(rows), "samples": source_samples, "source_id": source_id})
            episode_offset += max((int(row["episode_index"]) for row in rows), default=-1) + 1
    finally:
        renderer.close()
    if not images:
        raise ValueError("no eligible successful demonstrations found")
    features = clip_encode_batches(clip_model, processor, images, texts, args.clip_batch_size)
    return features, np.asarray(labels, dtype=np.int64), np.stack(waypoints).astype(np.float32), np.asarray(episodes, dtype=np.int32), sources


def split_by_task_episode(labels: np.ndarray, episodes: np.ndarray, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    validation = np.zeros(len(labels), dtype=bool)
    for task_label in np.unique(labels):
        task_episodes = np.unique(episodes[labels == task_label])
        count = max(1, int(round(len(task_episodes) * val_fraction)))
        held_out = set(task_episodes[-count:].tolist())
        validation |= (labels == task_label) & np.isin(episodes, list(held_out))
    return ~validation, validation


def score(model: WaypointActionHead, features: torch.Tensor, labels: torch.Tensor, waypoints: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits, prediction = model(features)
        prediction_m = prediction * y_std + y_mean
        target_m = waypoints * y_std + y_mean
        return {
            "intent_accuracy": float((logits.argmax(dim=1) == labels).float().mean().item()),
            "waypoint_mae_m": float(torch.abs(prediction_m - target_m).mean().item()),
            "waypoint_rmse_m": float(torch.sqrt(torch.mean((prediction_m - target_m) ** 2)).item()),
        }


def main() -> None:
    args = parse_args()
    started = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    clip_model, processor = load_clip(args.clip_model)
    clip_model = clip_model.to(device)
    frozen_encoder_params = int(sum(parameter.numel() for parameter in clip_model.parameters()))
    features, labels, waypoints, episodes, sources = collect_features(args, clip_model, processor)
    train_mask, validation_mask = split_by_task_episode(labels, episodes, args.val_fraction)
    x_mean = features[train_mask].mean(axis=0)
    x_std = features[train_mask].std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    y_mean = waypoints[train_mask].mean(axis=0)
    y_std = waypoints[train_mask].std(axis=0)
    y_std[y_std < 1e-6] = 1.0
    x = ((features - x_mean) / x_std).astype(np.float32)
    y = ((waypoints - y_mean) / y_std).astype(np.float32)
    model = WaypointActionHead(x.shape[1], args.hidden_size, len(TASK_LABELS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x[train_mask]), torch.from_numpy(labels[train_mask]), torch.from_numpy(y[train_mask])), batch_size=args.batch_size, shuffle=True)
    train_x = torch.from_numpy(x[train_mask]).to(device)
    train_labels = torch.from_numpy(labels[train_mask]).to(device)
    train_y = torch.from_numpy(y[train_mask]).to(device)
    val_x = torch.from_numpy(x[validation_mask]).to(device)
    val_labels = torch.from_numpy(labels[validation_mask]).to(device)
    val_y = torch.from_numpy(y[validation_mask]).to(device)
    y_mean_tensor = torch.from_numpy(y_mean).to(device)
    y_std_tensor = torch.from_numpy(y_std).to(device)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_labels, batch_y in train_loader:
            logits, prediction = model(batch_x.to(device))
            loss = torch.nn.functional.cross_entropy(logits, batch_labels.to(device)) + torch.nn.functional.mse_loss(prediction, batch_y.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        if epoch in {1, args.epochs} or epoch % 50 == 0:
            train_metrics = score(model, train_x, train_labels, train_y, y_mean_tensor, y_std_tensor)
            val_metrics = score(model, val_x, val_labels, val_y, y_mean_tensor, y_std_tensor)
            history.append({"epoch": epoch, "loss": float(np.mean(losses)), "train": train_metrics, "validation": val_metrics})
            print(f"epoch={epoch} loss={np.mean(losses):.5f} val_intent={val_metrics['intent_accuracy']:.3f} val_waypoint_rmse_m={val_metrics['waypoint_rmse_m']:.4f}", flush=True)
    train_metrics = score(model, train_x, train_labels, train_y, y_mean_tensor, y_std_tensor)
    validation_metrics = score(model, val_x, val_labels, val_y, y_mean_tensor, y_std_tensor)
    metadata = {
        "version": "clip_waypoint_action_head_core_v2_v1",
        "method": "frozen CLIP image-language encoder + lightweight joint intent/2D waypoint action head + structured executor",
        "method_boundary": "Runtime uses top RGB and instruction to predict task intent and source XY waypoint. The structured executor performs continuous contact control. This is a lightweight VLM action-parameter head, not end-to-end VLA, OpenVLA, or LoRA.",
        "clip_model": args.clip_model,
        "frozen_encoder_params": frozen_encoder_params,
        "trainable_head_params": int(sum(parameter.numel() for parameter in model.parameters())),
        "task_labels": list(TASK_LABELS),
        "samples": int(len(labels)),
        "train_samples": int(train_mask.sum()),
        "validation_samples": int(validation_mask.sum()),
        "workspace_profile": args.workspace_profile,
        "max_episodes_per_run": args.max_episodes_per_run,
        "image_size": args.image_size,
        "camera": args.camera,
        "language_augmentation": args.language_augmentation,
        "hidden_size": args.hidden_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "sources": sources,
        "train_time_seconds": time.time() - started,
        "peak_vram_mb": float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / f"{args.model_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save({"state_dict": model.state_dict(), "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std, "metadata": metadata}, model_path)
    metrics_path = model_path.with_suffix(".json")
    metrics_path.write_text(json.dumps({"model": str(model_path), "history": history, "metadata": metadata}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"model_path: {model_path}", flush=True)
    print(f"metrics_path: {metrics_path}", flush=True)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
