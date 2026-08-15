from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from torch_runtime import ensure_torch_path  # noqa: E402

ensure_torch_path()

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from train_mlp_bc import split_by_episode  # noqa: E402
from train_torch_act import build_samples  # noqa: E402
from widowx_env.demo_dataset import latest_run_dir, load_demo_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight PyTorch state Diffusion Policy action-chunk baseline.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "torch_diffusion_policy")
    parser.add_argument("--model-prefix", default="torch_diffusion_policy_state_chunk")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--sample-stride", type=int, default=16)
    parser.add_argument("--diffusion-steps", type=int, default=16)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
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
    return parser.parse_args()


class StateDiffusionPolicy(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        history: int,
        horizon: int,
        diffusion_steps: int,
        d_model: int,
        nhead: int,
        encoder_layers: int,
        decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.history = int(history)
        self.horizon = int(horizon)
        self.diffusion_steps = int(diffusion_steps)
        self.obs_proj = nn.Linear(observation_dim, d_model)
        self.history_pos = nn.Parameter(torch.zeros(1, history, d_model))
        self.action_proj = nn.Linear(action_dim, d_model)
        self.action_pos = nn.Parameter(torch.zeros(1, horizon, d_model))
        self.time_proj = nn.Sequential(nn.Linear(3, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.noise_head = nn.Linear(d_model, action_dim)

    def timestep_features(self, timesteps: torch.Tensor) -> torch.Tensor:
        t = timesteps.float() / max(1.0, float(self.diffusion_steps - 1))
        return torch.stack((t, torch.sin(torch.pi * t), torch.cos(torch.pi * t)), dim=1)

    def forward(self, observations: torch.Tensor, noisy_actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        memory = self.obs_proj(observations) + self.history_pos
        memory = self.encoder(memory)
        time_token = self.time_proj(self.timestep_features(timesteps)).unsqueeze(1)
        target = self.action_proj(noisy_actions) + self.action_pos + time_token
        decoded = self.decoder(target, memory)
        return self.noise_head(decoded)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def diffusion_schedule(steps: int, beta_start: float, beta_end: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    betas = torch.linspace(float(beta_start), float(beta_end), int(steps), device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def weighted_noise_mse(prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.mean((prediction - target).pow(2) * weights.view(1, 1, -1))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    alpha_bars: torch.Tensor,
    weights: torch.Tensor,
    device: torch.device,
    diffusion_steps: int,
) -> float:
    model.eval()
    losses = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        timesteps = torch.randint(0, diffusion_steps, (len(x),), device=device)
        noise = torch.randn_like(y)
        sqrt_ab = torch.sqrt(alpha_bars[timesteps]).view(-1, 1, 1)
        sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bars[timesteps]).view(-1, 1, 1)
        noisy = sqrt_ab * y + sqrt_one_minus_ab * noise
        prediction = model(x, noisy, timesteps)
        losses.append(float(weighted_noise_mse(prediction, noise, weights).cpu()))
    return float(np.mean(losses))


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    run_dir = args.run_dir or latest_run_dir()
    dataset = load_demo_dataset(
        run_dir,
        successful_only=not args.include_failures,
        successful_attempt_only=not args.all_attempts,
    )
    train_mask, val_mask = split_by_episode(dataset, args.val_fraction)
    train_episodes = set(int(item) for item in np.unique(dataset.episode_indices[train_mask]))
    val_episodes = set(int(item) for item in np.unique(dataset.episode_indices[val_mask]))

    horizon = max(2, int(args.horizon))
    history = max(1, int(args.history))
    sample_stride = max(1, int(args.sample_stride))
    x_train, y_train = build_samples(dataset.observations, dataset.actions, dataset.segments, train_episodes, horizon, history, sample_stride)
    x_val, y_val = build_samples(dataset.observations, dataset.actions, dataset.segments, val_episodes, horizon, history, sample_stride)

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
    model = StateDiffusionPolicy(
        observation_dim=x_train.shape[-1],
        action_dim=y_train.shape[-1],
        history=history,
        horizon=horizon,
        diffusion_steps=int(args.diffusion_steps),
        d_model=args.d_model,
        nhead=args.nhead,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    betas, alphas, alpha_bars = diffusion_schedule(args.diffusion_steps, args.beta_start, args.beta_end, device)
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
            timesteps = torch.randint(0, int(args.diffusion_steps), (len(x),), device=device)
            noise = torch.randn_like(y)
            sqrt_ab = torch.sqrt(alpha_bars[timesteps]).view(-1, 1, 1)
            sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bars[timesteps]).view(-1, 1, 1)
            noisy = sqrt_ab * y + sqrt_one_minus_ab * noise

            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, noisy, timesteps)
            loss = weighted_noise_mse(prediction, noise, action_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_loss = evaluate(model, val_loader, alpha_bars, action_weights, device, int(args.diffusion_steps))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"epoch={epoch} train_noise_mse_norm={np.mean(train_losses):.8f} val_noise_mse_norm={val_loss:.8f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    train_noise_mse_norm = evaluate(model, train_loader, alpha_bars, action_weights, device, int(args.diffusion_steps))
    val_noise_mse_norm = evaluate(model, val_loader, alpha_bars, action_weights, device, int(args.diffusion_steps))
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
    train_time_seconds = time.perf_counter() - start_time
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / f"{args.model_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    metadata = {
        "method": "torch_state_diffusion_policy",
        "run_dir": str(run_dir),
        "source_samples": int(len(dataset.actions)),
        "train_chunks": int(len(x_train)),
        "val_chunks": int(len(x_val)),
        "observation_dim": int(x_train.shape[-1]),
        "action_dim": int(y_train.shape[-1]),
        "horizon": horizon,
        "history": history,
        "sample_stride": sample_stride,
        "diffusion_steps": int(args.diffusion_steps),
        "beta_start": float(args.beta_start),
        "beta_end": float(args.beta_end),
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
        "train_noise_mse_norm": float(train_noise_mse_norm),
        "val_noise_mse_norm": float(val_noise_mse_norm),
        "train_time_seconds": float(train_time_seconds),
        "peak_vram_mb": float(peak_vram_mb),
        "trainable_params": int(trainable_params),
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": str(getattr(torch.version, "cuda", None)),
        "device": str(device),
        "successful_only": not args.include_failures,
        "successful_attempt_only": not args.all_attempts,
    }
    checkpoint = {
        "model_state": model.state_dict(),
        "metadata": metadata,
        "x_mean": torch.from_numpy(x_mean.astype(np.float32)),
        "x_std": torch.from_numpy(x_std.astype(np.float32)),
        "y_mean": torch.from_numpy(y_mean.astype(np.float32)),
        "y_std": torch.from_numpy(y_std.astype(np.float32)),
        "action_min": torch.from_numpy(dataset.actions.min(axis=0).astype(np.float32)),
        "action_max": torch.from_numpy(dataset.actions.max(axis=0).astype(np.float32)),
        "betas": betas.detach().cpu(),
        "alphas": alphas.detach().cpu(),
        "alpha_bars": alpha_bars.detach().cpu(),
    }
    torch.save(checkpoint, model_path)

    print(f"run_dir: {run_dir}", flush=True)
    print(f"model_path: {model_path}", flush=True)
    print(f"source_samples: {metadata['source_samples']}", flush=True)
    print(f"train_chunks: {metadata['train_chunks']}", flush=True)
    print(f"val_chunks: {metadata['val_chunks']}", flush=True)
    print(f"observation_dim: {metadata['observation_dim']}", flush=True)
    print(f"action_dim: {metadata['action_dim']}", flush=True)
    print(f"horizon: {horizon}", flush=True)
    print(f"history: {history}", flush=True)
    print(f"diffusion_steps: {metadata['diffusion_steps']}", flush=True)
    print(f"trainable_params: {trainable_params}", flush=True)
    print(f"train_time_seconds: {train_time_seconds:.2f}", flush=True)
    print(f"train_noise_mse_norm: {train_noise_mse_norm:.8f}", flush=True)
    print(f"val_noise_mse_norm: {val_noise_mse_norm:.8f}", flush=True)


if __name__ == "__main__":
    main()
