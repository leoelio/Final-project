from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from collect_multiview_recovery_dataset import DOMAINS, STATIC_TARGETS, TASK_SPECS, configure_env, initial_source, render_rgb  # noqa: E402
from widowx_env.multiview_features import extract_multiview_features, feature_spec  # noqa: E402
from widowx_env.scripted_expert import PickPlaceConfig, PickPlaceExpert  # noqa: E402
from widowx_env.vision_grounding import load_calibration, relocate_known_object  # noqa: E402


CLASS_NAMES = ("stop", "retry")
DEFAULT_TASK = "move_leftmost_cube_to_bowl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect only real MuJoCo post-failure RGB states that pass visual source re-localization.")
    parser.add_argument("--calibration", type=Path, default=ROOT / "outputs" / "rgb_grounding" / "top_rgb_core_v2_calibration_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "clip_recovery_bank" / "clip_recovery_bank_v1.npz")
    parser.add_argument("--records", type=Path, default=ROOT / "data" / "clip_recovery_bank" / "clip_recovery_bank_v1.jsonl")
    parser.add_argument("--summary", type=Path, default=ROOT / "data" / "clip_recovery_bank" / "clip_recovery_bank_v1_summary.json")
    parser.add_argument("--train-seed", type=int, default=760)
    parser.add_argument("--train-episodes", type=int, default=20, help="Distinct scene seeds per contact domain.")
    parser.add_argument("--test-seed", type=int, default=800)
    parser.add_argument("--test-episodes", type=int, default=10, help="Distinct held-out scene seeds per contact domain.")
    parser.add_argument("--domains", default=",".join(DOMAINS))
    parser.add_argument("--tasks", default=DEFAULT_TASK, help="Comma-separated task names; each task receives the requested seed count per domain.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--pool-grid", type=int, default=4)
    parser.add_argument("--place-tcp-z", type=float, default=0.041)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def selected_domains(value: str) -> list[tuple[str, dict]]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in names if item not in DOMAINS]
    if unknown:
        raise KeyError(f"unknown domains: {unknown}")
    return [(name, DOMAINS[name]) for name in names]


def selected_tasks(value: str) -> list[tuple[str, str, str, str]]:
    by_name = {task: (task, complexity, source, target) for task, complexity, source, target in TASK_SPECS}
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in names if item not in by_name]
    if unknown:
        raise KeyError(f"unknown tasks: {unknown}")
    return [by_name[name] for name in names]


def collect_one(
    args: argparse.Namespace,
    calibration,
    split: str,
    domain_name: str,
    domain: dict,
    task: str,
    complexity: str,
    source_kind: str,
    target_name: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, dict] | None:
    env = configure_env(domain, seed, args.image_size)
    env.reset(task=task, complexity=complexity, seed=seed)
    top_before = render_rgb(env, "top_rgb", args.image_size)
    try:
        source_name, source_position, detection = initial_source(top_before, calibration, source_kind)
    except (LookupError, ValueError):
        return None
    expert = PickPlaceExpert(env, PickPlaceConfig(place_tcp_z=args.place_tcp_z))
    target_position = STATIC_TARGETS[target_name]
    first = expert.execute(expert.plan_from_positions(source_position, target_position, target_geom=target_name), speed=0.0)
    if bool(first["success"]):
        return None
    top_after = render_rgb(env, "top_rgb", args.image_size)
    front_after = render_rgb(env, "front_rgb", args.image_size)
    try:
        retry_position, retry_detection = relocate_known_object(top_after, calibration, source_name, source_position[:2])
    except (LookupError, ValueError):
        return None
    features = extract_multiview_features(
        top_after,
        front_after,
        task,
        source_name,
        target_name,
        target_position[:2],
        calibration,
        pool_grid=args.pool_grid,
    )
    retry = expert.execute(expert.plan_from_positions(retry_position, target_position, target_geom=target_name), speed=0.0)
    label = int(bool(retry["success"]))
    record = {
        "split": split,
        "domain": domain_name,
        "domain_parameters": domain,
        "task": task,
        "complexity": complexity,
        "seed": seed,
        "source_name": source_name,
        "target_name": target_name,
        "initial_detection_area_px": int(detection.area),
        "first_target_distance_m": float(first["target_distance"]),
        "retry_detection_area_px": int(retry_detection.area),
        "retry_success_label": bool(retry["success"]),
        "retry_target_distance_m": float(retry["target_distance"]),
        "label": CLASS_NAMES[label],
        "label_id": label,
        "runtime_input_boundary": "The value head receives only post-first-attempt top_rgb, front_rgb, and static task configuration. MuJoCo state is offline supervision only.",
    }
    return features, top_after, front_after, label, record


def main() -> None:
    args = parse_args()
    if args.train_episodes < 0 or args.test_episodes < 0 or args.train_episodes + args.test_episodes < 1:
        raise ValueError("at least one train or test episode is required; counts cannot be negative")
    calibration = load_calibration(args.calibration)
    features: list[np.ndarray] = []
    top_images: list[np.ndarray] = []
    front_images: list[np.ndarray] = []
    labels: list[int] = []
    records: list[dict] = []
    scanned = {"train": 0, "test": 0}
    for split, count, base_seed in (("train", args.train_episodes, args.train_seed), ("test", args.test_episodes, args.test_seed)):
        for domain_name, domain in selected_domains(args.domains):
            for task, complexity, source_kind, target_name in selected_tasks(args.tasks):
                for offset in range(count):
                    seed = base_seed + offset
                    scanned[split] += 1
                    sample = collect_one(args, calibration, split, domain_name, domain, task, complexity, source_kind, target_name, seed)
                    if sample is None:
                        continue
                    x, top_rgb, front_rgb, label, record = sample
                    features.append(x)
                    top_images.append(top_rgb)
                    front_images.append(front_rgb)
                    labels.append(label)
                    records.append(record)
                    if args.log_every and len(records) % args.log_every == 0:
                        print(json.dumps(record, ensure_ascii=False), flush=True)
    if not features:
        raise RuntimeError("no visually re-localizable failure states were collected")
    x_array = np.stack(features).astype(np.float32)
    y_array = np.asarray(labels, dtype=np.int64)
    split_array = np.asarray([0 if row["split"] == "train" else 1 for row in records], dtype=np.int8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=x_array,
        top_images=np.stack(top_images).astype(np.uint8),
        front_images=np.stack(front_images).astype(np.uint8),
        labels=y_array,
        splits=split_array,
        seeds=np.asarray([row["seed"] for row in records], dtype=np.int32),
        task_names=np.asarray([row["task"] for row in records]),
        domain_names=np.asarray([row["domain"] for row in records]),
        metadata=json.dumps(
            {
                "version": "clip_recovery_bank_v1",
                "method": "actual first-attempt failures filtered by visual source re-localization, with counterfactual retry labels",
                "class_names": CLASS_NAMES,
                "feature_spec": feature_spec(args.pool_grid),
                "feature_dim": int(x_array.shape[1]),
                "runtime_boundary": "Only RGB views and static task configuration are runtime inputs. MuJoCo state is offline supervision only.",
            },
            ensure_ascii=False,
        ),
    )
    args.records.parent.mkdir(parents=True, exist_ok=True)
    args.records.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    summary = {
        "version": "clip_recovery_bank_v1",
        "scanned_primary_episodes": scanned,
        "candidate_samples": len(records),
        "tasks": [task for task, _complexity, _source, _target in selected_tasks(args.tasks)],
        "feature_dim": int(x_array.shape[1]),
        "split_class_counts": {
            split: {name: int(sum(row["split"] == split and row["label"] == name for row in records)) for name in CLASS_NAMES}
            for split in ("train", "test")
        },
        "dataset": str(args.output),
        "records": str(args.records),
        "note": "Only failure states with a visually re-localizable source are stored. Successful first attempts and visually unavailable sources are intentionally excluded from this recovery-value dataset.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
