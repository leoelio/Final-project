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
from widowx_env.multiview_features import extract_multiview_features, feature_spec  # noqa: E402
from widowx_env.scripted_expert import PickPlaceConfig, PickPlaceExpert  # noqa: E402
from widowx_env.vision_grounding import load_calibration, locate_leftmost_cube, locate_object, relocate_known_object  # noqa: E402


TASK_SPECS = (
    ("place_blue_cube_blue_pad", "medium", "blue_cube", "target_blue_pad"),
    ("place_blue_cube_red_pad", "medium", "blue_cube", "target_red_pad"),
    ("place_red_cube_red_pad", "medium", "red_cube", "target_red_pad"),
    ("move_leftmost_cube_to_bowl", "language", "leftmost_cube", "target_bowl"),
)
STATIC_TARGETS = {
    "target_blue_pad": np.asarray([0.50, 0.16, 0.002], dtype=float),
    "target_red_pad": np.asarray([0.50, -0.16, 0.002], dtype=float),
    "target_bowl": np.asarray([0.33, 0.25, 0.006], dtype=float),
}
DOMAINS = {
    "nominal": {"arm_kp": 150.0, "arm_force": 100.0, "gripper_kp": 1200.0, "gripper_force": 200.0, "friction": 5.0},
    "mild_contact_shift": {"arm_kp": 135.0, "arm_force": 90.0, "gripper_kp": 950.0, "gripper_force": 150.0, "friction": 2.5},
    "low_contact_shift": {"arm_kp": 120.0, "arm_force": 80.0, "gripper_kp": 750.0, "gripper_force": 110.0, "friction": 1.5},
    "severe_contact_shift": {"arm_kp": 105.0, "arm_force": 70.0, "gripper_kp": 550.0, "gripper_force": 75.0, "friction": 0.8},
}
CLASS_NAMES = ("accept", "retry", "stop")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect compact top+front RGB terminal and recovery labels in MuJoCo.")
    parser.add_argument("--calibration", type=Path, default=ROOT / "outputs" / "rgb_grounding" / "top_rgb_core_v2_calibration_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "multiview_recovery" / "widowx_multiview_recovery_v1.npz")
    parser.add_argument("--records", type=Path, default=ROOT / "data" / "multiview_recovery" / "widowx_multiview_recovery_v1.jsonl")
    parser.add_argument("--summary", type=Path, default=ROOT / "data" / "multiview_recovery" / "widowx_multiview_recovery_v1_summary.json")
    parser.add_argument("--train-episodes", type=int, default=6, help="Episodes per task/domain in the seed-disjoint training split.")
    parser.add_argument("--test-episodes", type=int, default=3, help="Episodes per task/domain in the seed-disjoint test split.")
    parser.add_argument("--train-seed", type=int, default=1000)
    parser.add_argument("--test-seed", type=int, default=5000)
    parser.add_argument("--domains", default=",".join(DOMAINS), help="Comma-separated contact domains.")
    parser.add_argument("--tasks", default=",".join(item[0] for item in TASK_SPECS), help="Comma-separated task names.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--place-tcp-z", type=float, default=0.041)
    parser.add_argument("--pool-grid", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=12)
    return parser.parse_args()


def render_rgb(env: WidowXTabletopEnv, camera: str, image_size: int) -> np.ndarray:
    renderer = mujoco.Renderer(env.model, height=image_size, width=image_size)
    try:
        renderer.update_scene(env.data, camera=camera)
        return renderer.render().copy()
    finally:
        renderer.close()


def configure_env(domain: dict, seed: int, image_size: int) -> WidowXTabletopEnv:
    env = WidowXTabletopEnv(seed=seed, image_size=(image_size, image_size), camera="top_rgb", workspace_profile="core_v2")
    env.set_arm_actuator_strength(kp=domain["arm_kp"], force_limit=domain["arm_force"])
    env.set_gripper_actuator_strength(kp=domain["gripper_kp"], force_limit=domain["gripper_force"])
    env.set_grasp_contact_friction(sliding=domain["friction"])
    return env


def initial_source(top_rgb: np.ndarray, calibration, source_kind: str) -> tuple[str, np.ndarray, object]:
    if source_kind == "leftmost_cube":
        return locate_leftmost_cube(top_rgb, calibration)
    position, detection = locate_object(top_rgb, calibration, source_kind)
    return source_kind, position, detection


def episode_seed(base: int, offset: int) -> int:
    """Pair initial scenes across tasks and contact domains; only train/test ranges differ."""
    return int(base + offset)


def select_specs(task_names: str, domain_names: str) -> tuple[list[tuple], list[tuple[str, dict]]]:
    requested_tasks = [item.strip() for item in task_names.split(",") if item.strip()]
    requested_domains = [item.strip() for item in domain_names.split(",") if item.strip()]
    task_lookup = {item[0]: item for item in TASK_SPECS}
    unknown_tasks = [item for item in requested_tasks if item not in task_lookup]
    unknown_domains = [item for item in requested_domains if item not in DOMAINS]
    if unknown_tasks or unknown_domains:
        raise KeyError(f"unknown tasks={unknown_tasks}, domains={unknown_domains}")
    return [task_lookup[item] for item in requested_tasks], [(item, DOMAINS[item]) for item in requested_domains]


def collect_one(args: argparse.Namespace, calibration, split: str, domain_name: str, domain: dict, task_spec: tuple, seed: int) -> tuple[np.ndarray, int, dict]:
    task, complexity, source_kind, target_name = task_spec
    env = configure_env(domain, seed, args.image_size)
    obs = env.reset(task=task, complexity=complexity, seed=seed)
    before_top = render_rgb(env, "top_rgb", args.image_size)
    source_name, source_position, detection = initial_source(before_top, calibration, source_kind)
    target_position = STATIC_TARGETS[target_name]
    expert = PickPlaceExpert(env, PickPlaceConfig(place_tcp_z=args.place_tcp_z))
    first = expert.execute(expert.plan_from_positions(source_position, target_position, target_geom=target_name), speed=0.0)
    after_top = render_rgb(env, "top_rgb", args.image_size)
    after_front = render_rgb(env, "front_rgb", args.image_size)
    features = extract_multiview_features(
        after_top,
        after_front,
        task,
        source_name,
        target_name,
        target_position[:2],
        calibration,
        pool_grid=args.pool_grid,
    )
    source_relocalizable = False
    retry_success = False
    retry_target_distance = None
    retry_error = None
    if not bool(first["success"]):
        try:
            retry_position, retry_detection = relocate_known_object(after_top, calibration, source_name, source_position[:2])
            source_relocalizable = True
            retry = expert.execute(expert.plan_from_positions(retry_position, target_position, target_geom=target_name), speed=0.0)
            retry_success = bool(retry["success"])
            retry_target_distance = float(retry["target_distance"])
            retry_area = int(retry_detection.area)
        except (LookupError, ValueError) as error:
            retry_error = str(error)
            retry_area = None
    else:
        retry_area = None
    label = 0 if bool(first["success"]) else 1 if retry_success else 2
    record = {
        "split": split,
        "domain": domain_name,
        "domain_parameters": domain,
        "task": task,
        "complexity": complexity,
        "seed": seed,
        "instruction": obs["instruction"],
        "source_name": source_name,
        "target_name": target_name,
        "initial_detection_area_px": int(detection.area),
        "first_success_label": bool(first["success"]),
        "first_strict_grasp_label": bool(first["strict_grasp_success"]),
        "first_target_distance_m": float(first["target_distance"]),
        "source_relocalizable_visual": source_relocalizable,
        "retry_success_label": retry_success,
        "retry_target_distance_m": retry_target_distance,
        "retry_detection_area_px": retry_area,
        "retry_error": retry_error,
        "label": CLASS_NAMES[label],
        "label_id": label,
        "runtime_input_boundary": "top_rgb + front_rgb + task configuration only; MuJoCo object state is used only for offline labels and scoring.",
    }
    return features, label, record


def main() -> None:
    args = parse_args()
    if args.train_episodes < 1 or args.test_episodes < 1:
        raise ValueError("train/test episode counts must both be positive")
    calibration = load_calibration(args.calibration)
    task_specs, domain_specs = select_specs(args.tasks, args.domains)
    features: list[np.ndarray] = []
    labels: list[int] = []
    records: list[dict] = []
    for split, count, base in (("train", args.train_episodes, args.train_seed), ("test", args.test_episodes, args.test_seed)):
        for domain_name, domain in domain_specs:
            for task_spec in task_specs:
                for offset in range(count):
                    seed = episode_seed(base, offset)
                    x, y, record = collect_one(args, calibration, split, domain_name, domain, task_spec, seed)
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
        first_success_labels=np.asarray([row["first_success_label"] for row in records], dtype=np.int8),
        source_relocalizable=np.asarray([row["source_relocalizable_visual"] for row in records], dtype=np.int8),
        retry_success_labels=np.asarray([row["retry_success_label"] for row in records], dtype=np.int8),
        task_names=np.asarray([row["task"] for row in records]),
        domain_names=np.asarray([row["domain"] for row in records]),
        metadata=json.dumps(
            {
                "version": "widowx_multiview_recovery_v2",
                "method": "post-attempt top+front RGB recovery-value dataset",
                "class_names": CLASS_NAMES,
                "feature_spec": feature_spec(args.pool_grid),
                "feature_dim": int(x_array.shape[1]),
                "tasks": [item[0] for item in task_specs],
                "domains": {name: value for name, value in domain_specs},
                "train_seed_base": args.train_seed,
                "test_seed_base": args.test_seed,
                "runtime_boundary": "Runtime consumes only two RGB views and static task configuration. Simulator state is offline supervision only.",
            },
            ensure_ascii=False,
        ),
    )
    args.records.parent.mkdir(parents=True, exist_ok=True)
    args.records.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    class_counts = {name: int(sum(label == index for label in labels)) for index, name in enumerate(CLASS_NAMES)}
    split_counts = {
        split: {name: int(sum(row["split"] == split and row["label"] == name for row in records)) for name in CLASS_NAMES}
        for split in ("train", "test")
    }
    summary = {
        "version": "widowx_multiview_recovery_v2",
        "samples": len(records),
        "feature_dim": int(x_array.shape[1]),
        "class_counts": class_counts,
        "split_class_counts": split_counts,
        "dataset": str(args.output),
        "records": str(args.records),
        "warning": "Class balance is reported as observed. The training script must not claim a balanced benchmark when these counts are uneven.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
