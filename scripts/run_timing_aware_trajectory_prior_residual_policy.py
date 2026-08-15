from __future__ import annotations

from pathlib import Path
import sys
import time

import mujoco.viewer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_trajectory_prior_residual_policy as base_runner  # noqa: E402
from timing_aware_trajectory_prior_residual_common import (  # noqa: E402
    VERSION,
    build_plan,
    build_segments,
    load_residual_model,
    make_config,
    predict_residual,
    prior_action_for_step,
    residual_feature,
    segment_for_step,
    total_steps,
)


def patch_base_runner() -> None:
    base_runner.VERSION = VERSION
    base_runner.build_plan = build_plan
    base_runner.build_segments = build_segments
    base_runner.load_residual_model = load_residual_model
    base_runner.make_config = make_config
    base_runner.predict_residual = predict_residual
    base_runner.prior_action_for_step = prior_action_for_step
    base_runner.residual_feature = residual_feature
    base_runner.segment_for_step = segment_for_step
    base_runner.total_steps = total_steps


def parse_args():
    return base_runner.parse_args()


def latest_model() -> Path:
    candidates = sorted(
        (ROOT / "outputs" / "timing_aware_trajectory_prior_residual_bc").glob(f"{VERSION}_*.npz"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError("no timing-aware trajectory-prior residual BC model found")
    return candidates[-1]


def rollout_with_env(args, model: dict, env, obs: dict, seed: int, task: str, complexity: str, viewer=None) -> dict:
    patch_base_runner()
    return base_runner.rollout_with_env(args, model, env, obs, seed, task, complexity, viewer)


def run_episode(args, model: dict, seed: int, task: str, complexity: str, viewer=None) -> dict:
    patch_base_runner()
    return base_runner.run_episode(args, model, seed, task, complexity, viewer)


def main() -> None:
    patch_base_runner()
    args = parse_args()
    model_path = args.model or latest_model()
    model = load_residual_model(model_path)
    default_task, default_complexity = base_runner.infer_task_defaults(model)
    task = args.task or default_task
    complexity = args.complexity or default_complexity

    print(f"model_path: {model_path}", flush=True)
    print(f"model_train_run: {model['metadata']['run_dir']}", flush=True)
    print(f"source_episodes: {model['metadata']['source_episodes']}", flush=True)
    print(f"samples: {model['metadata']['samples']}", flush=True)
    print(f"task: {task}", flush=True)
    print(f"complexity: {complexity}", flush=True)
    print(f"version: {VERSION}", flush=True)

    summaries = []
    if args.viewer:
        env, obs = base_runner.configure_env(args, int(args.seed), task, complexity)
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            summary = rollout_with_env(args, model, env, obs, int(args.seed), task, complexity, viewer)
            summaries.append(summary)
            print("episode_summary:", summary, flush=True)
            start = time.time()
            while viewer.is_running():
                viewer.sync()
                if args.duration and time.time() - start > float(args.duration):
                    break
                time.sleep(0.01)
        start_episode = 1
    else:
        start_episode = 0

    for episode in range(start_episode, int(args.episodes)):
        seed = int(args.seed) + episode
        summaries.append(run_episode(args, model, seed, task, complexity))

    successes = sum(1 for item in summaries if item["success"])
    tcp_lifts = sum(1 for item in summaries if item["tcp_grasp_lift_success"])
    print(f"success_rate: {successes}/{len(summaries)} = {successes / max(1, len(summaries)):.3f}", flush=True)
    print(f"tcp_grasp_lift_rate: {tcp_lifts}/{len(summaries)} = {tcp_lifts / max(1, len(summaries)):.3f}", flush=True)


if __name__ == "__main__":
    main()
