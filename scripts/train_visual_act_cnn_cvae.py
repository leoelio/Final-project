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
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from train_object_action_head import TARGET_GEOMS, one_hot  # noqa: E402
from train_torch_act import weighted_mse  # noqa: E402
from train_torch_act_cvae import kl_divergence  # noqa: E402
from train_vision_language_action_head import TASK_KINDS, attempt_start_index, pre_step_array, selected_attempts  # noqa: E402
from widowx_env import TASKS, WidowXTabletopEnv  # noqa: E402
from widowx_env.demo_dataset import phase_features, read_metadata  # noqa: E402
from widowx_env.tabletop_env import OBJECTS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight CNN visual ACT-CVAE baseline.")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data" / "demos" / "place_blue_cube_blue_pad_medium_20260702_051752")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "visual_act_cnn_cvae")
    parser.add_argument("--model-prefix", default="visual_act_cnn_cvae")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--sample-stride", type=int, default=64)
    parser.add_argument("--max-train-chunks", type=int, default=1600)
    parser.add_argument("--max-val-chunks", type=int, default=400)
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--camera", default="top_rgb")
    parser.add_argument("--visual-dim", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gripper-loss-weight", type=float, default=4.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--all-attempts", action="store_true")
    parser.add_argument("--log-every-episodes", type=int, default=10)
    return parser.parse_args()


def language_features(metadata: dict) -> np.ndarray:
    task = TASKS[str(metadata["task"])]
    target_object = str(metadata["target_object"])
    target_geom = metadata.get("target_geom")
    active_objects = set(metadata.get("active_objects", []))
    active_mask = np.asarray([name in active_objects for name in OBJECTS], dtype=np.float32)
    task_kind = one_hot(task.kind, TASK_KINDS)
    relation = np.asarray([1.0 if task.relation == "leftmost" else 0.0], dtype=np.float32)
    return np.concatenate(
        [
            one_hot(target_object, OBJECTS),
            one_hot(target_geom, TARGET_GEOMS),
            active_mask,
            task_kind,
            relation,
        ]
    ).astype(np.float32)


def aux_from_env_state(env: WidowXTabletopEnv, metadata: dict, phase: float) -> np.ndarray:
    return np.concatenate(
        [
            env.data.qpos[: env.robot_nq].astype(np.float32),
            env.data.qvel[: env.robot_nv].astype(np.float32),
            env.data.ctrl.astype(np.float32),
            language_features(metadata),
            phase_features(phase),
        ]
    ).astype(np.float32)


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


def render_rgb_chw(renderer: mujoco.Renderer, env: WidowXTabletopEnv) -> np.ndarray:
    renderer.update_scene(env.data, camera=env.camera)
    rgb = renderer.render()
    return np.transpose(rgb, (2, 0, 1)).astype(np.uint8)


def render_chunk_samples(
    env: WidowXTabletopEnv,
    renderer: mujoco.Renderer,
    trajectory_path: Path,
    metadata: dict,
    args: argparse.Namespace,
    max_chunks: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    images: list[np.ndarray] = []
    aux: list[np.ndarray] = []
    actions: list[np.ndarray] = []
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
                image_history = []
                aux_history = []
                for local in range(start - args.history + 1, start + 1):
                    env.data.qpos[:] = qpos[local]
                    env.data.qvel[:] = qvel[local]
                    env.data.ctrl[:] = ctrl[local]
                    mujoco.mj_forward(env.model, env.data)
                    image_history.append(render_rgb_chw(renderer, env))
                    aux_history.append(aux_from_env_state(env, metadata, float(local_phase[local])))
                images.append(np.stack(image_history).astype(np.uint8))
                aux.append(np.stack(aux_history).astype(np.float32))
                actions.append(data["actions"][indices[start: start + args.horizon]].astype(np.float32))
                if max_chunks > 0 and len(actions) >= max_chunks:
                    return images, aux, actions
    return images, aux, actions


def build_arrays_for_rows(args: argparse.Namespace, rows: list[dict], max_chunks: int, split_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    env = WidowXTabletopEnv(seed=args.seed, image_size=(args.image_size, args.image_size), camera=args.camera)
    renderer = mujoco.Renderer(env.model, height=args.image_size, width=args.image_size)
    images: list[np.ndarray] = []
    aux: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    sources: list[dict] = []
    started = time.time()
    try:
        for row_number, metadata in enumerate(rows, start=1):
            remaining = max_chunks - len(actions) if max_chunks > 0 else 0
            if max_chunks > 0 and remaining <= 0:
                break
            env.reset(task=str(metadata["task"]), complexity=str(metadata["complexity"]), seed=int(metadata["seed"]))
            chunk_images, chunk_aux, chunk_actions = render_chunk_samples(
                env,
                renderer,
                args.run_dir / metadata["trajectory_file"],
                metadata,
                args,
                remaining,
            )
            if chunk_actions:
                images.extend(chunk_images)
                aux.extend(chunk_aux)
                actions.extend(chunk_actions)
                sources.append(
                    {
                        "episode_index": int(metadata["episode_index"]),
                        "seed": int(metadata["seed"]),
                        "chunks": int(len(chunk_actions)),
                        "split": split_name,
                    }
                )
            if args.log_every_episodes > 0 and row_number % args.log_every_episodes == 0:
                print(f"{split_name}: rendered_episodes={row_number} chunks={len(actions)} elapsed={time.time() - started:.1f}s", flush=True)
    finally:
        renderer.close()

    if not actions:
        raise ValueError(f"no {split_name} chunks were built")
    return np.stack(images).astype(np.uint8), np.stack(aux).astype(np.float32), np.stack(actions).astype(np.float32), sources


class VisualACTCNNCVAEPolicy(nn.Module):
    def __init__(
        self,
        aux_dim: int,
        action_dim: int,
        history: int,
        horizon: int,
        visual_dim: int,
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
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, visual_dim),
            nn.ReLU(),
        )
        self.obs_proj = nn.Linear(visual_dim + aux_dim, d_model)
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
        self.posterior = nn.Sequential(
            nn.Linear(d_model + horizon * action_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(d_model, latent_dim)
        self.logvar_head = nn.Linear(d_model, latent_dim)
        self.action_head = nn.Linear(d_model, action_dim)

    def encode_memory(self, images: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        batch, history, channels, height, width = images.shape
        image_features = self.cnn(images.float().reshape(batch * history, channels, height, width) / 255.0)
        image_features = image_features.reshape(batch, history, -1)
        observation = torch.cat([image_features, aux], dim=-1)
        return self.encoder(self.obs_proj(observation) + self.history_pos)

    def encode_latent(self, memory: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        posterior_input = torch.cat([memory[:, -1], actions.reshape(actions.shape[0], -1)], dim=1)
        hidden = self.posterior(posterior_input)
        mu = self.mu_head(hidden)
        logvar = torch.clamp(self.logvar_head(hidden), min=-8.0, max=4.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, images: torch.Tensor, aux: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        if z is None:
            z = torch.zeros(images.shape[0], self.latent_dim, device=images.device, dtype=aux.dtype)
        memory = self.encode_memory(images, aux)
        queries = self.action_queries.expand(images.shape[0], -1, -1) + self.latent_proj(z).unsqueeze(1)
        decoded = self.decoder(queries, memory)
        return self.action_head(decoded)

    def forward(self, images: torch.Tensor, aux: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        memory = self.encode_memory(images, aux)
        mu, logvar = self.encode_latent(memory, actions)
        z = self.reparameterize(mu, logvar)
        queries = self.action_queries.expand(images.shape[0], -1, -1) + self.latent_proj(z).unsqueeze(1)
        decoded = self.decoder(queries, memory)
        return self.action_head(decoded), mu, logvar


def make_loader(images: np.ndarray, aux: np.ndarray, actions: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(images), torch.from_numpy(aux), torch.from_numpy(actions))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def cvae_loss(prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, kl_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon = weighted_mse(prediction, target, weights)
    kl = kl_divergence(mu, logvar)
    return recon + float(kl_weight) * kl, recon, kl


@torch.no_grad()
def evaluate(model: VisualACTCNNCVAEPolicy, loader: DataLoader, weights: torch.Tensor, kl_weight: float, device: torch.device) -> dict[str, float]:
    model.eval()
    losses = []
    recons = []
    kls = []
    zero_recons = []
    for images, aux, actions in loader:
        images = images.to(device)
        aux = aux.to(device)
        actions = actions.to(device)
        prediction, mu, logvar = model(images, aux, actions)
        loss, recon, kl = cvae_loss(prediction, actions, weights, mu, logvar, kl_weight)
        zero_prediction = model.decode(images, aux)
        zero_recon = weighted_mse(zero_prediction, actions, weights)
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
    started = time.perf_counter()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rows = [row for row in read_metadata(args.run_dir) if args.include_failures or bool(row["success"])]
    train_rows, val_rows = split_metadata(rows, args.val_fraction)
    image_train, aux_train, y_train, train_sources = build_arrays_for_rows(args, train_rows, args.max_train_chunks, "train")
    image_val, aux_val, y_val, val_sources = build_arrays_for_rows(args, val_rows, args.max_val_chunks, "val")

    aux_mean = aux_train.reshape(-1, aux_train.shape[-1]).mean(axis=0)
    aux_std = aux_train.reshape(-1, aux_train.shape[-1]).std(axis=0)
    aux_std[aux_std < 1e-6] = 1.0
    y_mean = y_train.reshape(-1, y_train.shape[-1]).mean(axis=0)
    y_std = y_train.reshape(-1, y_train.shape[-1]).std(axis=0)
    y_std[y_std < 1e-6] = 1.0

    aux_train_norm = ((aux_train - aux_mean[None, None, :]) / aux_std[None, None, :]).astype(np.float32)
    aux_val_norm = ((aux_val - aux_mean[None, None, :]) / aux_std[None, None, :]).astype(np.float32)
    y_train_norm = ((y_train - y_mean[None, None, :]) / y_std[None, None, :]).astype(np.float32)
    y_val_norm = ((y_val - y_mean[None, None, :]) / y_std[None, None, :]).astype(np.float32)

    train_loader = make_loader(image_train, aux_train_norm, y_train_norm, args.batch_size, shuffle=True)
    val_loader = make_loader(image_val, aux_val_norm, y_val_norm, args.batch_size, shuffle=False)
    model = VisualACTCNNCVAEPolicy(
        aux_dim=aux_train.shape[-1],
        action_dim=y_train.shape[-1],
        history=args.history,
        horizon=args.horizon,
        visual_dim=args.visual_dim,
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
        for images, aux, actions in train_loader:
            images = images.to(device)
            aux = aux.to(device)
            actions = actions.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, mu, logvar = model(images, aux, actions)
            loss, recon, kl = cvae_loss(prediction, actions, action_weights, mu, logvar, args.kl_weight)
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
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
    train_time_seconds = time.perf_counter() - started
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / f"{args.model_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    metadata = {
        "method": "visual_act_cnn_cvae",
        "run_dir": str(args.run_dir),
        "source_episodes": int(len(rows)),
        "train_chunks": int(len(image_train)),
        "val_chunks": int(len(image_val)),
        "image_size": int(args.image_size),
        "camera": str(args.camera),
        "aux_dim": int(aux_train.shape[-1]),
        "observation_dim": int(args.visual_dim + aux_train.shape[-1]),
        "action_dim": int(y_train.shape[-1]),
        "horizon": int(args.horizon),
        "history": int(args.history),
        "sample_stride": int(args.sample_stride),
        "max_train_chunks": int(args.max_train_chunks),
        "max_val_chunks": int(args.max_val_chunks),
        "visual_dim": int(args.visual_dim),
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
        "peak_vram_mb": float(peak_vram_mb),
        "trainable_params": int(trainable_params),
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": str(getattr(torch.version, "cuda", None)),
        "device": str(device),
        "successful_only": not args.include_failures,
        "successful_attempt_only": not args.all_attempts,
        "train_sources": train_sources,
        "val_sources": val_sources,
        "note": "Small CNN RGB encoder plus proprioception/language tokens and ACT-CVAE action chunks; a visual ACT baseline, not official ACT.",
    }
    checkpoint = {
        "model_state": model.state_dict(),
        "metadata": metadata,
        "aux_mean": torch.from_numpy(aux_mean.astype(np.float32)),
        "aux_std": torch.from_numpy(aux_std.astype(np.float32)),
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
    print(f"image_size: {metadata['image_size']}", flush=True)
    print(f"aux_dim: {metadata['aux_dim']}", flush=True)
    print(f"action_dim: {metadata['action_dim']}", flush=True)
    print(f"horizon: {metadata['horizon']}", flush=True)
    print(f"history: {metadata['history']}", flush=True)
    print(f"latent_dim: {metadata['latent_dim']}", flush=True)
    print(f"trainable_params: {trainable_params}", flush=True)
    print(f"train_time_seconds: {train_time_seconds:.2f}", flush=True)
    print(f"peak_vram_mb: {peak_vram_mb:.2f}", flush=True)
    print(f"train_mse_norm: {train_metrics['zero_recon']:.8f}", flush=True)
    print(f"val_mse_norm: {val_metrics['zero_recon']:.8f}", flush=True)
    print(f"train_posterior_recon_norm: {train_metrics['recon']:.8f}", flush=True)
    print(f"val_posterior_recon_norm: {val_metrics['recon']:.8f}", flush=True)


if __name__ == "__main__":
    main()
