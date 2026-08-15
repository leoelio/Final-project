from __future__ import annotations

import argparse
from datetime import datetime
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
from torch.utils.data import DataLoader  # noqa: E402

from train_mlp_bc import split_by_episode  # noqa: E402
from train_torch_act import build_samples, make_loader, weighted_mse  # noqa: E402
from widowx_env.demo_dataset import latest_run_dir, load_demo_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight state ACT-CVAE action-chunk policy.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "torch_act_cvae")
    parser.add_argument("--model-prefix", default="torch_act_cvae_state_chunk")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--sample-stride", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gripper-loss-weight", type=float, default=4.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--all-attempts", action="store_true")
    return parser.parse_args()


class StateACTCVAEPolicy(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        history: int,
        horizon: int,
        d_model: int,
        nhead: int,
        encoder_layers: int,
        decoder_layers: int,
        dim_feedforward: int,
        latent_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.history = int(history)
        self.horizon = int(horizon)
        self.latent_dim = int(latent_dim)
        self.obs_proj = nn.Linear(observation_dim, d_model)
        self.history_pos = nn.Parameter(torch.zeros(1, history, d_model))
        self.action_queries = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)
        self.latent_proj = nn.Linear(latent_dim, d_model)

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
        posterior_input_dim = observation_dim + horizon * action_dim
        self.posterior = nn.Sequential(
            nn.Linear(posterior_input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(d_model, latent_dim)
        self.logvar_head = nn.Linear(d_model, latent_dim)
        self.action_head = nn.Linear(d_model, action_dim)

    def encode_latent(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        posterior_input = torch.cat([observations[:, -1], actions.reshape(actions.shape[0], -1)], dim=1)
        hidden = self.posterior(posterior_input)
        mu = self.mu_head(hidden)
        logvar = torch.clamp(self.logvar_head(hidden), min=-8.0, max=4.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, observations: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        if z is None:
            z = torch.zeros(observations.shape[0], self.latent_dim, device=observations.device, dtype=observations.dtype)
        memory = self.obs_proj(observations) + self.history_pos
        memory = self.encoder(memory)
        latent_bias = self.latent_proj(z).unsqueeze(1)
        queries = self.action_queries.expand(observations.shape[0], -1, -1) + latent_bias
        decoded = self.decoder(queries, memory)
        return self.action_head(decoded)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_latent(observations, actions)
        z = self.reparameterize(mu, logvar)
        return self.decode(observations, z), mu, logvar


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.sum(-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()), dim=1))


def cvae_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon = weighted_mse(prediction, target, weights)
    kl = kl_divergence(mu, logvar)
    return recon + float(kl_weight) * kl, recon, kl


@torch.no_grad()
def evaluate(model: StateACTCVAEPolicy, loader: DataLoader, weights: torch.Tensor, kl_weight: float, device: torch.device) -> dict[str, float]:
    model.eval()
    losses = []
    recons = []
    kls = []
    zero_recons = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        prediction, mu, logvar = model(x, y)
        loss, recon, kl = cvae_loss(prediction, y, weights, mu, logvar, kl_weight)
        zero_prediction = model.decode(x)
        zero_recon = weighted_mse(zero_prediction, y, weights)
        losses.append(float(loss.cpu()))
        recons.append(float(recon.cpu()))
        kls.append(float(kl.cpu()))
        zero_recons.append(float(zero_recon.cpu()))
    return {
        "loss": float(np.mean(losses)),
        "recon": float(np.mean(recons)),
        "kl": float(np.mean(kls)),
        "zero_recon": float(np.mean(zero_recons)),
    }


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    model = StateACTCVAEPolicy(
        observation_dim=x_train.shape[-1],
        action_dim=y_train.shape[-1],
        history=history,
        horizon=horizon,
        d_model=args.d_model,
        nhead=args.nhead,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        dim_feedforward=args.dim_feedforward,
        latent_dim=args.latent_dim,
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
        train_recons = []
        train_kls = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, mu, logvar = model(x, y)
            loss, recon, kl = cvae_loss(prediction, y, action_weights, mu, logvar, args.kl_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            train_recons.append(float(recon.detach().cpu()))
            train_kls.append(float(kl.detach().cpu()))
        val = evaluate(model, val_loader, action_weights, args.kl_weight, device)
        if val["zero_recon"] < best_val:
            best_val = val["zero_recon"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"epoch={epoch} train_loss={np.mean(train_losses):.8f} "
            f"train_recon={np.mean(train_recons):.8f} train_kl={np.mean(train_kls):.8f} "
            f"val_recon={val['recon']:.8f} val_zero_recon={val['zero_recon']:.8f} val_kl={val['kl']:.8f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = evaluate(model, train_loader, action_weights, args.kl_weight, device)
    val_metrics = evaluate(model, val_loader, action_weights, args.kl_weight, device)
    train_time_seconds = time.perf_counter() - start_time
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / f"{args.model_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    metadata = {
        "method": "torch_state_transformer_act_cvae",
        "run_dir": str(run_dir),
        "source_samples": int(len(dataset.actions)),
        "train_chunks": int(len(x_train)),
        "val_chunks": int(len(x_val)),
        "observation_dim": int(x_train.shape[-1]),
        "action_dim": int(y_train.shape[-1]),
        "horizon": horizon,
        "history": history,
        "sample_stride": sample_stride,
        "d_model": int(args.d_model),
        "nhead": int(args.nhead),
        "encoder_layers": int(args.encoder_layers),
        "decoder_layers": int(args.decoder_layers),
        "dim_feedforward": int(args.dim_feedforward),
        "latent_dim": int(args.latent_dim),
        "kl_weight": float(args.kl_weight),
        "dropout": float(args.dropout),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "gripper_loss_weight": float(args.gripper_loss_weight),
        "train_mse_norm": float(train_metrics["zero_recon"]),
        "val_mse_norm": float(val_metrics["zero_recon"]),
        "train_posterior_recon_norm": float(train_metrics["recon"]),
        "val_posterior_recon_norm": float(val_metrics["recon"]),
        "train_kl": float(train_metrics["kl"]),
        "val_kl": float(val_metrics["kl"]),
        "train_time_seconds": float(train_time_seconds),
        "peak_vram_mb": 0 if device.type == "cpu" else None,
        "trainable_params": int(trainable_params),
        "torch_version": str(torch.__version__),
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
    print(f"latent_dim: {metadata['latent_dim']}", flush=True)
    print(f"kl_weight: {metadata['kl_weight']}", flush=True)
    print(f"trainable_params: {trainable_params}", flush=True)
    print(f"train_time_seconds: {train_time_seconds:.2f}", flush=True)
    print(f"train_mse_norm: {train_metrics['zero_recon']:.8f}", flush=True)
    print(f"val_mse_norm: {val_metrics['zero_recon']:.8f}", flush=True)
    print(f"train_posterior_recon_norm: {train_metrics['recon']:.8f}", flush=True)
    print(f"val_posterior_recon_norm: {val_metrics['recon']:.8f}", flush=True)
    print(f"train_kl: {train_metrics['kl']:.8f}", flush=True)
    print(f"val_kl: {val_metrics['kl']:.8f}", flush=True)


if __name__ == "__main__":
    main()
