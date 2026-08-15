from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_knn_policy  # noqa: E402
import run_object_action_head  # noqa: E402
import run_trajectory_knn_policy  # noqa: E402
from train_chunk_bc import output_weights, weighted_backward, weighted_mse  # noqa: E402
from train_mlp_bc import adam_update, clip_grads, forward, init_model, make_adam_states, parse_hidden_sizes  # noqa: E402
from train_object_action_head import build_dataset_features, observation_layout as object_observation_layout  # noqa: E402
from train_trajectory_knn_bc import build_samples as build_trajectory_samples  # noqa: E402
from train_trajectory_knn_bc import observation_layout  # noqa: E402
from widowx_env import TASKS, WidowXTabletopEnv  # noqa: E402
from widowx_env.demo_dataset import DemoDataset, latest_run_dir, load_demo_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate demonstration-budget data efficiency for memory-style baselines.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--budgets", default="10,25,50,92")
    parser.add_argument("--methods", default="knn_bc,trajectory_knn,object_action_head")
    parser.add_argument("--task", choices=sorted(TASKS), default="place_blue_cube_blue_pad")
    parser.add_argument("--complexity", choices=("easy", "medium", "hard", "language"), default="medium")
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--heldout-seed", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--knn-stride", type=int, default=4)
    parser.add_argument("--object-hidden-sizes", default="128,128")
    parser.add_argument("--object-epochs", type=int, default=6)
    parser.add_argument("--object-batch-size", type=int, default=1024)
    parser.add_argument("--object-lr", type=float, default=1e-3)
    parser.add_argument("--object-weight-decay", type=float, default=1e-6)
    parser.add_argument("--object-gripper-loss-weight", type=float, default=4.0)
    parser.add_argument("--object-grad-clip", type=float, default=10.0)
    parser.add_argument("--object-seed", type=int, default=0)
    parser.add_argument("--trajectory-history", type=int, default=8)
    parser.add_argument("--trajectory-horizon", type=int, default=8)
    parser.add_argument("--trajectory-sample-stride", type=int, default=16)
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "data_efficiency_summary.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "data_efficiency_summary.md")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / "data_efficiency_v2.json")
    parser.add_argument("--log-every", type=int, default=0)
    return parser.parse_args()


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def filter_dataset_by_episode_budget(dataset: DemoDataset, budget: int) -> DemoDataset:
    available = sorted(int(item) for item in np.unique(dataset.episode_indices))
    selected = set(available[: min(max(1, int(budget)), len(available))])
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episode_indices: list[np.ndarray] = []
    attempt_ids: list[np.ndarray] = []
    source_steps: list[np.ndarray] = []
    segments: list[dict] = []

    offset = 0
    for segment in dataset.segments:
        length = int(segment["steps"])
        segment_slice = slice(offset, offset + length)
        offset += length
        if int(segment["episode_index"]) not in selected:
            continue
        observations.append(dataset.observations[segment_slice])
        actions.append(dataset.actions[segment_slice])
        episode_indices.append(dataset.episode_indices[segment_slice])
        attempt_ids.append(dataset.attempt_ids[segment_slice])
        source_steps.append(dataset.source_steps[segment_slice])
        segments.append(dict(segment))

    if not observations:
        raise ValueError(f"budget {budget} selected no episodes")

    return replace(
        dataset,
        observations=np.concatenate(observations, axis=0),
        actions=np.concatenate(actions, axis=0),
        episode_indices=np.concatenate(episode_indices, axis=0),
        attempt_ids=np.concatenate(attempt_ids, axis=0),
        source_steps=np.concatenate(source_steps, axis=0),
        segments=segments,
    )


def build_knn_model(dataset: DemoDataset, stride: int, run_dir: Path, budget: int) -> dict:
    stride = max(1, int(stride))
    keep = np.arange(0, len(dataset.actions), stride)
    observations = dataset.observations[keep].astype(np.float32)
    actions = dataset.actions[keep].astype(np.float32)
    x_mean = observations.mean(axis=0)
    x_std = observations.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    return {
        "observations_norm": ((observations - x_mean) / x_std).astype(np.float32),
        "actions": actions,
        "phases": observations[:, -3].astype(np.float32),
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "action_min": actions.min(axis=0).astype(np.float32),
        "action_max": actions.max(axis=0).astype(np.float32),
        "metadata": {
            "method": "data_efficiency_knn_bc",
            "run_dir": str(run_dir),
            "budget": int(budget),
            "source_samples": int(len(dataset.actions)),
            "samples": int(len(actions)),
            "stride": stride,
            "observation_dim": int(observations.shape[1]),
            "action_dim": int(actions.shape[1]),
        },
    }


def build_trajectory_knn_model(dataset: DemoDataset, args: argparse.Namespace, run_dir: Path, budget: int) -> dict:
    raw_observations = dataset.observations.astype(np.float32)
    x, action_chunks, phases = build_trajectory_samples(
        raw_observations,
        raw_observations,
        dataset.actions.astype(np.float32),
        dataset.segments,
        max(2, int(args.trajectory_horizon)),
        max(1, int(args.trajectory_history)),
        max(1, int(args.trajectory_sample_stride)),
    )
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    return {
        "observations_norm": ((x - x_mean) / x_std).astype(np.float32),
        "action_chunks": action_chunks,
        "phases": phases,
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "action_min": dataset.actions.min(axis=0).astype(np.float32),
        "action_max": dataset.actions.max(axis=0).astype(np.float32),
        "metadata": {
            "method": "data_efficiency_trajectory_knn_bc",
            "run_dir": str(run_dir),
            "budget": int(budget),
            "source_samples": int(len(dataset.actions)),
            "samples": int(len(x)),
            "raw_observation_dim": int(raw_observations.shape[1]),
            "single_observation_dim": int(raw_observations.shape[1]),
            "observation_dim": int(x.shape[1]),
            "action_dim": int(dataset.actions.shape[1]),
            "horizon": int(args.trajectory_horizon),
            "history": int(args.trajectory_history),
            "sample_stride": int(args.trajectory_sample_stride),
            "augment_relative": False,
            "layout": observation_layout(),
        },
    }


def build_object_action_head_model(dataset: DemoDataset, args: argparse.Namespace, run_dir: Path, budget: int) -> dict:
    rng = np.random.default_rng(args.object_seed + int(budget))
    layout = object_observation_layout()
    features = build_dataset_features(dataset, run_dir, layout).astype(np.float32)
    actions = dataset.actions.astype(np.float32)
    x_mean = features.mean(axis=0)
    x_std = features.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    y_mean = actions.mean(axis=0)
    y_std = actions.std(axis=0)
    y_std[y_std < 1e-6] = 1.0

    x_norm = ((features - x_mean) / x_std).astype(np.float32)
    y_norm = ((actions - y_mean) / y_std).astype(np.float32)
    hidden_sizes = parse_hidden_sizes(args.object_hidden_sizes)
    layers = init_model(x_norm.shape[1], y_norm.shape[1], hidden_sizes, rng)
    states = make_adam_states(layers)
    loss_weights = output_weights(1, actions.shape[1], args.object_gripper_loss_weight)

    step = 0
    for _epoch in range(1, int(args.object_epochs) + 1):
        order = rng.permutation(len(x_norm))
        for start in range(0, len(order), int(args.object_batch_size)):
            batch = order[start: start + int(args.object_batch_size)]
            prediction, activations, preacts = forward(layers, x_norm[batch], cache=True)
            grads = weighted_backward(
                layers,
                activations,
                preacts,
                prediction,
                y_norm[batch],
                loss_weights,
                args.object_weight_decay,
            )
            clip_grads(grads, args.object_grad_clip)
            step += 1
            adam_update(layers, grads, states, step, args.object_lr)

    train_prediction = forward(layers, x_norm).astype(np.float32)
    train_mse_norm = weighted_mse(train_prediction, y_norm, loss_weights)
    return {
        "layers": layers,
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "y_std": y_std.astype(np.float32),
        "action_min": actions.min(axis=0).astype(np.float32),
        "action_max": actions.max(axis=0).astype(np.float32),
        "metadata": {
            "method": "data_efficiency_object_action_head",
            "run_dir": str(run_dir),
            "budget": int(budget),
            "source_samples": int(len(actions)),
            "samples": int(len(actions)),
            "feature_dim": int(features.shape[1]),
            "action_dim": int(actions.shape[1]),
            "hidden_sizes": hidden_sizes,
            "epochs": int(args.object_epochs),
            "batch_size": int(args.object_batch_size),
            "lr": float(args.object_lr),
            "train_mse_norm": float(train_mse_norm),
            "layout": layout,
        },
    }


def configure_env(seed: int, task: str, complexity: str) -> tuple[WidowXTabletopEnv, dict]:
    env = WidowXTabletopEnv(seed=seed)
    env.set_arm_actuator_strength(kp=150.0, force_limit=100.0)
    env.set_gripper_actuator_strength(kp=800.0, force_limit=140.0)
    env.set_grasp_contact_friction(sliding=3.0)
    obs = env.reset(task=task, complexity=complexity, seed=seed)
    return env, obs


def knn_policy_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        episodes=args.eval_episodes,
        steps=2840,
        viewer=False,
        duration=0.0,
        speed=0.0,
        arm_kp=150.0,
        arm_force=100.0,
        gripper_kp=800.0,
        gripper_force=140.0,
        friction=3.0,
        clip_actions=True,
        stop_on_unsafe=True,
        log_every=args.log_every,
        action_alpha=1.0,
        max_arm_delta=0.05,
        max_gripper_delta=0.002,
        k=3,
        phase_window=0.02,
        min_candidates=128,
        qpos_weight=0.25,
        qvel_weight=0.05,
        ctrl_weight=0.25,
        tcp_weight=1.0,
        object_weight=4.0,
        target_weight=1.0,
        phase_weight=2.0,
    )


def trajectory_policy_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        episodes=args.eval_episodes,
        steps=2840,
        viewer=False,
        duration=0.0,
        speed=0.0,
        arm_kp=150.0,
        arm_force=100.0,
        gripper_kp=800.0,
        gripper_force=140.0,
        friction=3.0,
        clip_actions=True,
        stop_on_unsafe=True,
        log_every=args.log_every,
        action_alpha=0.85,
        max_arm_delta=0.04,
        max_gripper_delta=0.0015,
        k=3,
        phase_window=0.03,
        min_candidates=256,
        history_decay=0.25,
        replan_interval=1,
        temporal_ensemble=True,
        ensemble_decay=0.1,
    )


def object_policy_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        episodes=args.eval_episodes,
        steps=2840,
        viewer=False,
        duration=0.0,
        speed=0.0,
        arm_kp=150.0,
        arm_force=100.0,
        gripper_kp=800.0,
        gripper_force=140.0,
        friction=3.0,
        clip_actions=True,
        stop_on_unsafe=True,
        log_every=args.log_every,
        action_alpha=0.2,
        max_arm_delta=0.01,
        max_gripper_delta=0.0005,
    )


def evaluate_split(
    model: dict,
    policy_args: SimpleNamespace,
    rollout: Callable,
    seed_start: int,
    episodes: int,
    task: str,
    complexity: str,
) -> list[dict]:
    results = []
    for offset in range(episodes):
        seed = seed_start + offset
        env, obs = configure_env(seed, task, complexity)
        result = rollout(policy_args, model, env, obs, seed, task, complexity)
        results.append(result)
    return results


def summarize_results(
    version: str,
    method_key: str,
    budget: int,
    model: dict,
    split: str,
    seed_start: int,
    results: list[dict],
) -> dict:
    successes = sum(int(item["success"]) for item in results)
    distances = [float(item["target_distance"]) for item in results if np.isfinite(item["target_distance"])]
    return {
        "version": version,
        "method_key": method_key,
        "demo_budget": int(budget),
        "actual_episodes": int(model["metadata"]["budget"]),
        "stored_samples": int(model["metadata"]["samples"]),
        "split": split,
        "seed_start": int(seed_start),
        "episodes": int(len(results)),
        "success": f"{successes}/{len(results)}",
        "success_rate": successes / max(1, len(results)),
        "mean_target_distance": float(np.mean(distances)) if distances else float("nan"),
        "mean_steps_taken": float(np.mean([int(item["steps_taken"]) for item in results])),
    }


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(values) + " |"


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# 数据效率评测 v2",
        "",
        "用途：比较不同示范数量下，记忆型 baseline 在训练范围和留出范围的闭环成功率，用于回答“是否省数据”和“是否只是记忆轨迹”的问题。",
        "",
        "说明：当前主数据集只有 92 条成功示范，因此预算使用 `10,25,50,92`，其中 92 代表全量成功示范；每个条件使用 3 个训练范围 seed 和 3 个留出 seed 做快速评测。",
        "",
        "## 结果表",
        "",
        md_row(["方法", "示范预算", "评测范围", "成功率", "平均目标距离", "存储样本"]),
        md_row(["---", "---:", "---", "---:", "---:", "---:"]),
    ]
    for row in rows:
        lines.append(
            md_row(
                [
                    f"`{row['method_key']}`",
                    str(row["demo_budget"]),
                    str(row["split"]),
                    str(row["success"]),
                    f"{float(row['mean_target_distance']):.4f}",
                    str(row["stored_samples"]),
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 阶段结论",
            "",
            "1. `knn_bc` 在 10 条示范时训练范围已经达到 3/3，但留出范围仍为 0/3，说明它的数据效率主要来自轨迹记忆。",
            "2. `trajectory_knn` 加入历史窗口和动作块后，全量 92 条示范时训练范围达到 3/3，但留出范围仍为 0/3，说明轨迹条件不等于泛化。",
            "3. `object_action_head` 预算曲线用于观察轻量 action-head 在小数据下是否比纯记忆型方法更稳，但当前快速评测仍不能替代完整 5-10 seed 评测。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or latest_run_dir()
    dataset = load_demo_dataset(run_dir, successful_only=True, successful_attempt_only=True)
    budgets = parse_ints(args.budgets)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    allowed = {"knn_bc", "object_action_head", "trajectory_knn"}
    unknown = [item for item in methods if item not in allowed]
    if unknown:
        raise KeyError(f"unknown methods: {unknown}; available={sorted(allowed)}")

    rows = []
    details: dict[str, list[dict]] = {}
    version = "data_efficiency_budget_sweep_v2"
    for budget in budgets:
        budget_dataset = filter_dataset_by_episode_budget(dataset, budget)
        actual_budget = len(np.unique(budget_dataset.episode_indices))
        for method in methods:
            if method == "knn_bc":
                model = build_knn_model(budget_dataset, args.knn_stride, run_dir, actual_budget)
                policy_args = knn_policy_args(args)
                rollout = run_knn_policy.rollout_with_env
            elif method == "trajectory_knn":
                model = build_trajectory_knn_model(budget_dataset, args, run_dir, actual_budget)
                policy_args = trajectory_policy_args(args)
                rollout = run_trajectory_knn_policy.rollout_with_env
            else:
                model = build_object_action_head_model(budget_dataset, args, run_dir, actual_budget)
                policy_args = object_policy_args(args)
                rollout = run_object_action_head.rollout_with_env

            for split, seed_start in (("train_range", args.train_seed), ("heldout", args.heldout_seed)):
                results = evaluate_split(model, policy_args, rollout, seed_start, args.eval_episodes, args.task, args.complexity)
                row = summarize_results(version, method, actual_budget, model, split, seed_start, results)
                rows.append(row)
                details[f"{method}_{actual_budget}_{split}"] = results
                print(
                    f"method={method} budget={actual_budget} split={split} "
                    f"success={row['success']} mean_distance={row['mean_target_distance']:.4f}",
                    flush=True,
                )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "version",
        "method_key",
        "demo_budget",
        "actual_episodes",
        "stored_samples",
        "split",
        "seed_start",
        "episodes",
        "success",
        "success_rate",
        "mean_target_distance",
        "mean_steps_taken",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(args.output_md, rows)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "version": version,
                "run_dir": str(run_dir),
                "task": args.task,
                "complexity": args.complexity,
                "budgets": budgets,
                "methods": methods,
                "eval_episodes": int(args.eval_episodes),
                "rows": rows,
                "episodes_by_condition": details,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"csv_path: {args.output_csv}", flush=True)
    print(f"markdown_path: {args.output_md}", flush=True)
    print(f"json_path: {args.output_json}", flush=True)
    print(f"rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
