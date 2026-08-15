from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from torch_runtime import ensure_torch_path  # noqa: E402

ensure_torch_path()

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from train_torch_act import StateACTPolicy, make_loader, weighted_mse  # noqa: E402
from train_vision_language_action_head import (  # noqa: E402
    attempt_start_index,
    feature_from_env_state,
    pre_step_array,
    selected_attempts,
)
from widowx_env import WidowXTabletopEnv  # noqa: E402
from widowx_env.demo_dataset import read_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a rendered-visual-feature ACT-lite action-chunk policy.")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data" / "demos" / "place_blue_cube_blue_pad_medium_20260702_051752")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "visual_feature_act")
    parser.add_argument("--model-prefix", default="visual_feature_act_lite")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--sample-stride", type=int, default=32)
    parser.add_argument("--max-train-chunks", type=int, default=5000)
    parser.add_argument("--max-val-chunks", type=int, default=1200)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--camera", default="top_rgb")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gripper-loss-weight", type=float, default=4.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--all-attempts", action="store_true")
    parser.add_argument("--log-every-episodes", type=int, default=10)
    return parser.parse_args()


def split_metadata(rows: list[dict], val_fraction: float) -> tuple[list[dict], list[dict]]:
    rows = sorted(rows, key=lambda item: int(item["episode_index"]))
    if len(rows) <= 1:
        return rows, rows
    val_count = max(1, int(round(len(rows) * val_fraction)))
    return rows[:-val_count], rows[-val_count:]


def sample_starts(length: int, history: int, horizon: int, stride: int) -> list[int]:
    if length < horizon + history - 1:
        return []
    return list(range(history - 1, length - horizon + 1, max(1, stride)))


def render_chunk_samples(
    env: WidowXTabletopEnv,
    renderer: mujoco.Renderer,
    trajectory_path: Path,
    metadata: dict,
    args: argparse.Namespace,
    max_chunks: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    with np.load(trajectory_path) as data:
        attempts = selected_attempts(data, metadata, include_failures=args.include_failures or args.all_attempts)
        for attempt_id in attempts:
            indices = np.flatnonzero(data["attempt_ids"] == attempt_id)
            starts = sample_starts(len(indices), args.history, args.horizon, args.sample_stride)
            if not starts:
                continue
            start_index = attempt_start_index(data, attempt_id)
            qpos = pre_step_array(data["qpos"], data["attempt_start_qpos"][start_index], indices)
            qvel = pre_step_array(data["qvel"], data["attempt_start_qvel"][start_index], indices)
            ctrl = pre_step_array(data["ctrl"], data["attempt_start_ctrl"][start_index], indices)
            local_phase = np.linspace(0.0, 1.0, len(indices), dtype=np.float32)

            for start in starts:
                history_features = []
                for local in range(start - args.history + 1, start + 1):
                    env.data.qpos[:] = qpos[local]
                    env.data.qvel[:] = qvel[local]
                    env.data.ctrl[:] = ctrl[local]
                    mujoco.mj_forward(env.model, env.data)
                    history_features.append(feature_from_env_state(env, renderer, metadata, float(local_phase[local]), args.grid_size))
                xs.append(np.stack(history_features).astype(np.float32))
                ys.append(data["actions"][indices[start : start + args.horizon]].astype(np.float32))
                if max_chunks > 0 and len(xs) >= max_chunks:
                    return xs, ys
    return xs, ys


def build_arrays_for_rows(args: argparse.Namespace, rows: list[dict], max_chunks: int, split_name: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    env = WidowXTabletopEnv(seed=args.seed, image_size=(args.image_size, args.image_size), camera=args.camera)
    renderer = mujoco.Renderer(env.model, height=args.image_size, width=args.image_size)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    sources: list[dict] = []
    started = time.time()
    try:
        for row_number, metadata in enumerate(rows, start=1):
            remaining = max_chunks - len(xs) if max_chunks > 0 else 0
            if max_chunks > 0 and remaining <= 0:
                break
            env.reset(task=str(metadata["task"]), complexity=str(metadata["complexity"]), seed=int(metadata["seed"]))
            chunk_xs, chunk_ys = render_chunk_samples(
                env,
                renderer,
                args.run_dir / metadata["trajectory_file"],
                metadata,
                args,
                remaining,
            )
            if chunk_ys:
                xs.extend(chunk_xs)
                ys.extend(chunk_ys)
                sources.append(
                    {
                        "episode_index": int(metadata["episode_index"]),
                        "seed": int(metadata["seed"]),
                        "chunks": int(len(chunk_ys)),
                        "split": split_name,
                    }
                )
            if args.log_every_episodes > 0 and row_number % args.log_every_episodes == 0:
                print(
                    f"{split_name}: rendered_episodes={row_number} chunks={len(ys)} elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
    finally:
        renderer.close()

    if not xs:
        raise ValueError(f"no {split_name} chunks were built; reduce --history/--horizon or increase data")
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32), sources


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, weights: torch.Tensor, device: torch.device) -> float:
    model.eval()
    losses = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        losses.append(float(weighted_mse(model(x), y, weights).cpu()))
    return float(np.mean(losses))


def main() -> None:
    args = parse_args()
    started = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = [row for row in read_metadata(args.run_dir) if args.include_failures or bool(row["success"])]
    train_rows, val_rows = split_metadata(rows, args.val_fraction)
    x_train, y_train, train_sources = build_arrays_for_rows(args, train_rows, args.max_train_chunks, "train")
    x_val, y_val, val_sources = build_arrays_for_rows(args, val_rows, args.max_val_chunks, "val")

    x_mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    x_std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    y_mean = y_train.reshape(-1, y_train.shape[-1]).mean(axis=0)
    y_std = y_train.reshape(-1, y_train.shape[-1]).std(axis=0)
    y_std[y_std < 1e-6] = 1.0

    x_train_norm = ((x_train - x_mean[None, None, :]) / x_std[None, None, :]).astype(np.float32)
    x_val_norm = ((x_val - x_mean[None, None, :]) / x_std[None, None, :]).astype(np.float32)
    y_train_norm = ((y_train - y_mean[None, None, :]) / y_std[None, None, :]).astype(np.float32)
    y_val_norm = ((y_val - y_mean[None, None, :]) / y_std[None, None, :]).astype(np.float32)

    train_loader = make_loader(x_train_norm, y_train_norm, args.batch_size, shuffle=True)
    val_loader = make_loader(x_val_norm, y_val_norm, args.batch_size, shuffle=False)
    model = StateACTPolicy(
        observation_dim=x_train.shape[-1],
        action_dim=y_train.shape[-1],
        history=args.history,
        horizon=args.horizon,
        d_model=args.d_model,
        nhead=args.nhead,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    action_weights = torch.ones(y_train.shape[-1], dtype=torch.float32, device=device)
    action_weights[-1] = max(1.0, float(args.gripper_loss_weight))
    action_weights = action_weights / action_weights.mean()

    best_state = None
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_mse(model(x), y, action_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        val_loss = evaluate(model, val_loader, action_weights, device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"epoch={epoch} train_mse_norm={np.mean(train_losses):.8f} val_mse_norm={val_loss:.8f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    train_mse_norm = evaluate(model, train_loader, action_weights, device)
    val_mse_norm = evaluate(model, val_loader, action_weights, device)
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    train_time_seconds = time.time() - started

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / f"{args.model_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    metadata = {
        "method": "visual_feature_transformer_act_lite",
        "run_dir": str(args.run_dir),
        "source_episodes": int(len(rows)),
        "train_chunks": int(len(x_train)),
        "val_chunks": int(len(x_val)),
        "observation_dim": int(x_train.shape[-1]),
        "action_dim": int(y_train.shape[-1]),
        "horizon": int(args.horizon),
        "history": int(args.history),
        "sample_stride": int(args.sample_stride),
        "max_train_chunks": int(args.max_train_chunks),
        "max_val_chunks": int(args.max_val_chunks),
        "image_size": int(args.image_size),
        "grid_size": int(args.grid_size),
        "camera": str(args.camera),
        "d_model": int(args.d_model),
        "nhead": int(args.nhead),
        "encoder_layers": int(args.encoder_layers),
        "decoder_layers": int(args.decoder_layers),
        "dim_feedforward": int(args.dim_feedforward),
        "dropout": float(args.dropout),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "gripper_loss_weight": float(args.gripper_loss_weight),
        "train_mse_norm": float(train_mse_norm),
        "val_mse_norm": float(val_mse_norm),
        "train_time_seconds": float(train_time_seconds),
        "peak_vram_mb": 0 if device.type == "cpu" else None,
        "trainable_params": int(trainable_params),
        "torch_version": str(torch.__version__),
        "device": str(device),
        "successful_only": not args.include_failures,
        "successful_attempt_only": not args.all_attempts,
        "train_sources": train_sources,
        "val_sources": val_sources,
        "note": "Rendered MuJoCo RGB pooled features plus language/proprioception Transformer ACT-lite; not a full CNN visual ACT.",
    }
    checkpoint = {
        "model_state": model.state_dict(),
        "metadata": metadata,
        "x_mean": torch.from_numpy(x_mean.astype(np.float32)),
        "x_std": torch.from_numpy(x_std.astype(np.float32)),
        "y_mean": torch.from_numpy(y_mean.astype(np.float32)),
        "y_std": torch.from_numpy(y_std.astype(np.float32)),
        "action_min": torch.from_numpy(y_train.reshape(-1, y_train.shape[-1]).min(axis=0).astype(np.float32)),
        "action_max": torch.from_numpy(y_train.reshape(-1, y_train.shape[-1]).max(axis=0).astype(np.float32)),
    }
    torch.save(checkpoint, model_path)

    print(f"run_dir: {args.run_dir}", flush=True)
    print(f"model_path: {model_path}", flush=True)
    print(f"source_episodes: {metadata['source_episodes']}", flush=True)
    print(f"train_chunks: {metadata['train_chunks']}", flush=True)
    print(f"val_chunks: {metadata['val_chunks']}", flush=True)
    print(f"observation_dim: {metadata['observation_dim']}", flush=True)
    print(f"action_dim: {metadata['action_dim']}", flush=True)
    print(f"horizon: {metadata['horizon']}", flush=True)
    print(f"history: {metadata['history']}", flush=True)
    print(f"trainable_params: {trainable_params}", flush=True)
    print(f"train_time_seconds: {train_time_seconds:.2f}", flush=True)
    print(f"train_mse_norm: {train_mse_norm:.8f}", flush=True)
    print(f"val_mse_norm: {val_mse_norm:.8f}", flush=True)


if __name__ == "__main__":
    main()
