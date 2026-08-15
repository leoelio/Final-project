from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from widowx_env import WidowXTabletopEnv  # noqa: E402
from widowx_env.multiview_features import TASK_NAMES, extract_multiview_features, feature_spec  # noqa: E402
from widowx_env.vision_grounding import load_calibration  # noqa: E402


CLASS_NAMES = ("not_complete", "complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect balanced MuJoCo terminal scenes for top-only versus top+front completion verification.")
    parser.add_argument("--calibration", type=Path, default=ROOT / "outputs" / "rgb_grounding" / "top_rgb_core_v2_calibration_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "multiview_recovery" / "widowx_multiview_terminal_v1.npz")
    parser.add_argument("--records", type=Path, default=ROOT / "data" / "multiview_recovery" / "widowx_multiview_terminal_v1.jsonl")
    parser.add_argument("--summary", type=Path, default=ROOT / "data" / "multiview_recovery" / "widowx_multiview_terminal_v1_summary.json")
    parser.add_argument("--train-scenes", type=int, default=40, help="Scenes per task, balanced by intended terminal state.")
    parser.add_argument("--test-scenes", type=int, default=20, help="Seed-disjoint scenes per task, balanced by intended terminal state.")
    parser.add_argument("--train-seed", type=int, default=10000)
    parser.add_argument("--test-seed", type=int, default=20000)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--pool-grid", type=int, default=4)
    parser.add_argument("--settle-steps", type=int, default=40)
    parser.add_argument("--log-every", type=int, default=40)
    return parser.parse_args()


def render_rgb(env: WidowXTabletopEnv, camera: str, image_size: int) -> np.ndarray:
    renderer = mujoco.Renderer(env.model, height=image_size, width=image_size)
    try:
        renderer.update_scene(env.data, camera=camera)
        return renderer.render().copy()
    finally:
        renderer.close()


def sampled_position(target: np.ndarray, complete: bool, rng: np.random.Generator) -> np.ndarray:
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    radius = float(rng.uniform(0.0, 0.040) if complete else rng.uniform(0.085, 0.155))
    return np.asarray([target[0] + radius * np.cos(angle), target[1] + radius * np.sin(angle), 0.026], dtype=float)


def collect_one(args: argparse.Namespace, calibration, split: str, task: str, seed: int, intended_complete: bool) -> tuple[np.ndarray, int, dict]:
    env = WidowXTabletopEnv(seed=seed, image_size=(args.image_size, args.image_size), camera="top_rgb", workspace_profile="core_v2")
    complexity = "language" if task == "move_leftmost_cube_to_bowl" else "medium"
    obs = env.reset(task=task, complexity=complexity, seed=seed)
    target_name = str(env.task.target_geom)
    source_name = str(obs["target_object"])
    target = env.target_position(target_name).copy()
    rng = np.random.default_rng(seed + 7919)
    env._set_free_object_pose(source_name, sampled_position(target, intended_complete, rng))
    for _ in range(args.settle_steps):
        mujoco.mj_step(env.model, env.data)
    mujoco.mj_forward(env.model, env.data)
    top_rgb = render_rgb(env, "top_rgb", args.image_size)
    front_rgb = render_rgb(env, "front_rgb", args.image_size)
    terminal_complete = bool(env.metrics()["success"])
    feature = extract_multiview_features(
        top_rgb,
        front_rgb,
        task,
        source_name,
        target_name,
        target[:2],
        calibration,
        pool_grid=args.pool_grid,
    )
    record = {
        "split": split,
        "task": task,
        "complexity": complexity,
        "seed": seed,
        "source_name": source_name,
        "target_name": target_name,
        "intended_complete": intended_complete,
        "terminal_complete_label": terminal_complete,
        "target_distance_m": float(env.metrics()["target_distance"]),
        "label": CLASS_NAMES[int(terminal_complete)],
        "label_id": int(terminal_complete),
        "runtime_input_boundary": "top_rgb + front_rgb + task configuration only; MuJoCo state is used only for offline terminal labels.",
    }
    return feature, int(terminal_complete), record


def main() -> None:
    args = parse_args()
    if args.train_scenes < 2 or args.test_scenes < 2:
        raise ValueError("train/test scene counts must both be at least two to include both terminal labels")
    calibration = load_calibration(args.calibration)
    features: list[np.ndarray] = []
    labels: list[int] = []
    records: list[dict] = []
    for split, count, base in (("train", args.train_scenes, args.train_seed), ("test", args.test_scenes, args.test_seed)):
        for task_index, task in enumerate(TASK_NAMES):
            for offset in range(count):
                seed = int(base + task_index * 1000 + offset)
                x, y, record = collect_one(args, calibration, split, task, seed, intended_complete=bool(offset % 2))
                features.append(x)
                labels.append(y)
                records.append(record)
                if args.log_every and len(records) % args.log_every == 0:
                    print(json.dumps(record, ensure_ascii=False), flush=True)
    x_array = np.stack(features).astype(np.float32)
    y_array = np.asarray(labels, dtype=np.int64)
    split_array = np.asarray([0 if row["split"] == "train" else 1 for row in records], dtype=np.int8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=x_array,
        labels=y_array,
        splits=split_array,
        seeds=np.asarray([row["seed"] for row in records], dtype=np.int32),
        task_names=np.asarray([row["task"] for row in records]),
        metadata=json.dumps(
            {
                "version": "widowx_multiview_terminal_v1",
                "method": "balanced terminal-scene supervision for fixed MuJoCo cameras",
                "class_names": CLASS_NAMES,
                "feature_spec": feature_spec(args.pool_grid),
                "feature_dim": int(x_array.shape[1]),
                "tasks": list(TASK_NAMES),
                "runtime_boundary": "Runtime consumes only two RGB views and static task configuration. Simulator state is offline supervision only.",
            },
            ensure_ascii=False,
        ),
    )
    args.records.parent.mkdir(parents=True, exist_ok=True)
    args.records.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    counts = {split: {name: int(sum(row["split"] == split and row["label"] == name for row in records)) for name in CLASS_NAMES} for split in ("train", "test")}
    summary = {
        "version": "widowx_multiview_terminal_v1",
        "samples": len(records),
        "feature_dim": int(x_array.shape[1]),
        "split_class_counts": counts,
        "dataset": str(args.output),
        "records": str(args.records),
        "scope": "These are controlled MuJoCo terminal scenes for visual-verification supervision. Real rollout terminal evaluation remains a separate held-out protocol.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
