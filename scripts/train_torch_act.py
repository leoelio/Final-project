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

from train_chunk_bc import augment_relative_features, observation_layout  # noqa: E402
from train_mlp_bc import split_by_episode  # noqa: E402
from widowx_env.demo_dataset import latest_run_dir, load_demo_dataset  # noqa: E402


PHASE_NAMES = ("approach", "grasp", "lift", "transfer", "place_release")
PHASE_THRESHOLDS = np.asarray([0.17, 0.26, 0.41, 0.66], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight PyTorch Transformer ACT-style action-chunk policy.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "torch_act")
    parser.add_argument("--model-prefix", default="torch_act_state_chunk")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--sample-stride", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gripper-loss-weight", type=float, default=4.0)
    parser.add_argument(
        "--phase-loss-weights",
        default="",
        help="Optional comma list such as grasp:3,lift:3,place_release:2. Defaults to uniform phase weights.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--augment-relative", action="store_true")
    parser.add_argument("--no-augment-relative", dest="augment_relative", action="store_false")
    parser.add_argument("--phase-one-hot", action="store_true")
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--all-attempts", action="store_true")
    parser.set_defaults(augment_relative=False)
    return parser.parse_args()


class StateACTPolicy(nn.Module):
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
        dropout: float,
    ) -> None:
        super().__init__()
        self.history = int(history)
        self.horizon = int(horizon)
        self.obs_proj = nn.Linear(observation_dim, d_model)
        self.history_pos = nn.Parameter(torch.zeros(1, history, d_model))
        self.action_queries = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)

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
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        memory = self.obs_proj(observations) + self.history_pos
        memory = self.encoder(memory)
        queries = self.action_queries.expand(observations.shape[0], -1, -1)
        decoded = self.decoder(queries, memory)
        return self.action_head(decoded)


def append_phase_one_hot(observations: np.ndarray, phase_index: int = -3) -> np.ndarray:
    phase = observations[..., phase_index].astype(np.float32)
    phase_ids = np.digitize(phase, PHASE_THRESHOLDS).astype(np.int64)
    one_hot = np.eye(len(PHASE_NAMES), dtype=np.float32)[phase_ids]
    return np.concatenate([observations.astype(np.float32), one_hot], axis=-1)


def build_samples(
    observations: np.ndarray,
    actions: np.ndarray,
    segments: list[dict],
    allowed_episodes: set[int],
    horizon: int,
    history: int,
    sample_stride: int,
    phase_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    phases: list[float] = []
    offset = 0
    for segment in segments:
        length = int(segment["steps"])
        segment_slice = slice(offset, offset + length)
        offset += length
        if int(segment["episode_index"]) not in allowed_episodes:
            continue
        if length < horizon + history - 1:
            continue

        segment_observations = observations[segment_slice]
        segment_actions = actions[segment_slice]
        for start in range(history - 1, length - horizon + 1, sample_stride):
            xs.append(segment_observations[start - history + 1: start + 1])
            ys.append(segment_actions[start: start + horizon])
            phases.append(float(segment_observations[start, phase_index]))

    if not xs:
        raise ValueError("no ACT samples were built; reduce --history, --horizon, or --sample-stride")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), np.asarray(phases, dtype=np.float32)


def make_loader(x: np.ndarray, y: np.ndarray, sample_weights: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(sample_weights))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    action_weights: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    loss = (prediction - target).pow(2) * action_weights.view(1, 1, -1)
    return torch.mean(loss * sample_weights.view(-1, 1, 1))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, action_weights: torch.Tensor, device: torch.device) -> float:
    model.eval()
    losses = []
    for x, y, sample_weights in loader:
        x = x.to(device)
        y = y.to(device)
        sample_weights = sample_weights.to(device)
        losses.append(float(weighted_mse(model(x), y, action_weights, sample_weights).cpu()))
    return float(np.mean(losses))


def parse_phase_loss_weights(spec: str) -> np.ndarray:
    weights = np.ones(len(PHASE_NAMES), dtype=np.float32)
    if not spec.strip():
        return weights
    index = {name: offset for offset, name in enumerate(PHASE_NAMES)}
    for item in spec.split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError(f"phase loss weight must be name:value, got {item!r}")
        name, value = item.split(":", 1)
        name = name.strip()
        if name not in index:
            raise ValueError(f"unknown phase name {name!r}; choose from {', '.join(PHASE_NAMES)}")
        weights[index[name]] = max(0.0, float(value))
    if float(weights.mean()) <= 0:
        raise ValueError("phase loss weights must have positive mean")
    return weights


def make_sample_weights(phases: np.ndarray, phase_weights: np.ndarray) -> np.ndarray:
    phase_ids = np.digitize(phases.astype(np.float32), PHASE_THRESHOLDS).astype(np.int64)
    sample_weights = phase_weights[phase_ids].astype(np.float32)
    sample_weights /= float(sample_weights.mean())
    return sample_weights.astype(np.float32)


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
    raw_observation_dim = int(dataset.observations.shape[-1])
    layout = observation_layout()
    observations = dataset.observations.astype(np.float32)
    if args.augment_relative:
        observations = augment_relative_features(observations, layout)
    if args.phase_one_hot:
        observations = append_phase_one_hot(observations, phase_index=raw_observation_dim - 3)
    train_mask, val_mask = split_by_episode(dataset, args.val_fraction)
    train_episodes = set(int(item) for item in np.unique(dataset.episode_indices[train_mask]))
    val_episodes = set(int(item) for item in np.unique(dataset.episode_indices[val_mask]))

    horizon = max(2, int(args.horizon))
    history = max(1, int(args.history))
    sample_stride = max(1, int(args.sample_stride))
    phase_index = raw_observation_dim - 3
    x_train, y_train, train_phases = build_samples(
        observations,
        dataset.actions,
        dataset.segments,
        train_episodes,
        horizon,
        history,
        sample_stride,
        phase_index,
    )
    x_val, y_val, val_phases = build_samples(
        observations,
        dataset.actions,
        dataset.segments,
        val_episodes,
        horizon,
        history,
        sample_stride,
        phase_index,
    )
    phase_loss_weights = parse_phase_loss_weights(args.phase_loss_weights)
    train_sample_weights = make_sample_weights(train_phases, phase_loss_weights)
    val_sample_weights = make_sample_weights(val_phases, phase_loss_weights)

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

    train_loader = make_loader(x_train_norm, y_train_norm, train_sample_weights, args.batch_size, shuffle=True)
    val_loader = make_loader(x_val_norm, y_val_norm, val_sample_weights, args.batch_size, shuffle=False)
    model = StateACTPolicy(
        observation_dim=x_train.shape[-1],
        action_dim=y_train.shape[-1],
        history=history,
        horizon=horizon,
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
        for x, y, sample_weights in train_loader:
            x = x.to(device)
            y = y.to(device)
            sample_weights = sample_weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_mse(model(x), y, action_weights, sample_weights)
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
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
    train_time_seconds = time.perf_counter() - start_time
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / f"{args.model_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    if args.phase_one_hot and args.augment_relative:
        method_name = "contact_aware_phase_conditioned_torch_state_transformer_act"
    elif args.phase_one_hot:
        method_name = "phase_conditioned_torch_state_transformer_act"
    elif args.augment_relative:
        method_name = "contact_aware_torch_state_transformer_act"
    else:
        method_name = "torch_state_transformer_act"
    metadata = {
        "method": method_name,
        "run_dir": str(run_dir),
        "source_samples": int(len(dataset.actions)),
        "train_chunks": int(len(x_train)),
        "val_chunks": int(len(x_val)),
        "observation_dim": int(x_train.shape[-1]),
        "raw_observation_dim": raw_observation_dim,
        "action_dim": int(y_train.shape[-1]),
        "horizon": horizon,
        "history": history,
        "sample_stride": sample_stride,
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
        "phase_loss_weights": {name: float(weight) for name, weight in zip(PHASE_NAMES, phase_loss_weights)},
        "train_sample_weight_min": float(train_sample_weights.min()),
        "train_sample_weight_max": float(train_sample_weights.max()),
        "val_sample_weight_min": float(val_sample_weights.min()),
        "val_sample_weight_max": float(val_sample_weights.max()),
        "train_mse_norm": float(train_mse_norm),
        "val_mse_norm": float(val_mse_norm),
        "train_time_seconds": float(train_time_seconds),
        "peak_vram_mb": float(peak_vram_mb),
        "trainable_params": int(trainable_params),
        "augment_relative": bool(args.augment_relative),
        "layout": {key: int(value) for key, value in layout.items()},
        "phase_one_hot": bool(args.phase_one_hot),
        "phase_names": list(PHASE_NAMES) if args.phase_one_hot else [],
        "phase_thresholds": [float(item) for item in PHASE_THRESHOLDS] if args.phase_one_hot else [],
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
    print(f"trainable_params: {trainable_params}", flush=True)
    print(f"train_time_seconds: {train_time_seconds:.2f}", flush=True)
    print(f"train_mse_norm: {train_mse_norm:.8f}", flush=True)
    print(f"val_mse_norm: {val_mse_norm:.8f}", flush=True)


if __name__ == "__main__":
    main()
