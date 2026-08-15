from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from collect_multiview_recovery_dataset import DOMAINS, STATIC_TARGETS, TASK_SPECS, configure_env, initial_source, render_rgb  # noqa: E402
from run_clip_semantic_rgb_feedback import visual_target_status  # noqa: E402
from widowx_env.scripted_expert import PickPlaceConfig, PickPlaceExpert, capture_state, restore_state  # noqa: E402
from widowx_env.vision_grounding import load_calibration, relocate_known_object  # noqa: E402


ACTION_LABELS = ("stop", "standard", "deep_tight_slow")
DEFAULT_TASKS = "place_blue_cube_red_pad,move_leftmost_cube_to_bowl"
PROPRIO_SPEC = {
    "qpos_indices": list(range(8)),
    "qvel_indices": list(range(8)),
    "ctrl_indices": list(range(7)),
    "actuator_force_indices": list(range(7)),
    "boundary": "Robot arm and gripper joints/actuators only. Object free-joint state starts at qpos index 8 and is excluded.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect seed-disjoint RGB states with counterfactual standard/deep recovery outcomes.")
    parser.add_argument("--calibration", type=Path, default=ROOT / "outputs" / "rgb_grounding" / "top_rgb_core_v2_calibration_v1.json")
    parser.add_argument("--version", default="action_profile_bank_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "action_profile_bank" / "action_profile_bank_v1.npz")
    parser.add_argument("--records", type=Path, default=ROOT / "data" / "action_profile_bank" / "action_profile_bank_v1.jsonl")
    parser.add_argument("--summary", type=Path, default=ROOT / "data" / "action_profile_bank" / "action_profile_bank_v1_summary.json")
    parser.add_argument("--train-seed", type=int, default=1400)
    parser.add_argument("--train-episodes", type=int, default=40, help="Seeds per task in the training split.")
    parser.add_argument("--test-seed", type=int, default=1500)
    parser.add_argument("--test-episodes", type=int, default=20, help="Seeds per task in the held-out split.")
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--place-tcp-z", type=float, default=0.041)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def selected_tasks(value: str) -> list[tuple[str, str, str, str]]:
    known = {task: (task, complexity, source, target) for task, complexity, source, target in TASK_SPECS}
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in known]
    if unknown:
        raise KeyError(f"unknown tasks: {unknown}")
    return [known[name] for name in names]


def profile_configs(place_tcp_z: float) -> dict[str, PickPlaceConfig]:
    base = PickPlaceConfig(place_tcp_z=place_tcp_z)
    return {
        "standard": base,
        "deep_tight_slow": replace(base, grasp_z_offset=0.0, close_gripper=0.007, close_steps=420, descend_steps=300, lift_steps=500, transfer_steps=800),
    }


def action_label(standard_success: bool, deep_success: bool) -> tuple[int, str]:
    if deep_success and not standard_success:
        return 2, "deep_only_success"
    if standard_success and not deep_success:
        return 1, "standard_only_success"
    if standard_success and deep_success:
        return 1, "both_success_prefer_standard"
    return 0, "both_failed"


def robot_proprioception(env) -> np.ndarray:
    return np.concatenate(
        [
            env.data.qpos[:8],
            env.data.qvel[:8],
            env.data.ctrl[:7],
            env.data.actuator_force[:7],
        ]
    ).astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.train_episodes < 1 or args.test_episodes < 1:
        raise ValueError("train/test episode counts must be positive")
    calibration = load_calibration(args.calibration)
    profiles = profile_configs(args.place_tcp_z)
    top_images: list[np.ndarray] = []
    front_images: list[np.ndarray] = []
    proprioceptions: list[np.ndarray] = []
    labels: list[int] = []
    splits: list[int] = []
    seeds: list[int] = []
    task_names: list[str] = []
    records: list[dict] = []
    counters = {"scanned": 0, "reset_failed": 0, "initial_not_visible": 0, "first_success": 0, "target_complete_or_ambiguous": 0, "source_not_visible_after_failure": 0}
    for split_name, split_id, base_seed, count in (("train", 0, args.train_seed, args.train_episodes), ("test", 1, args.test_seed, args.test_episodes)):
        for task, complexity, source_kind, target_name in selected_tasks(args.tasks):
            for offset in range(count):
                seed = base_seed + offset
                counters["scanned"] += 1
                env = configure_env(DOMAINS["severe_contact_shift"], seed, args.image_size)
                try:
                    env.reset(task=task, complexity=complexity, seed=seed)
                except RuntimeError:
                    counters["reset_failed"] += 1
                    continue
                top_before = render_rgb(env, "top_rgb", args.image_size)
                try:
                    source_name, source_position, detection = initial_source(top_before, calibration, source_kind)
                except (LookupError, ValueError):
                    counters["initial_not_visible"] += 1
                    continue
                target_position = STATIC_TARGETS[target_name]
                first_expert = PickPlaceExpert(env, profiles["standard"])
                first = first_expert.execute(first_expert.plan_from_positions(source_position, target_position, target_geom=target_name), speed=0.0)
                if bool(first["success"]):
                    counters["first_success"] += 1
                    continue
                top_after = render_rgb(env, "top_rgb", args.image_size)
                front_after = render_rgb(env, "front_rgb", args.image_size)
                target_status = visual_target_status(top_after, calibration, source_name, target_name)
                if not bool(target_status["verifiable"]) or bool(target_status["complete"]):
                    counters["target_complete_or_ambiguous"] += 1
                    continue
                try:
                    retry_position, retry_detection = relocate_known_object(top_after, calibration, source_name, source_position[:2])
                except (LookupError, ValueError):
                    counters["source_not_visible_after_failure"] += 1
                    continue
                proprioception = robot_proprioception(env)
                snapshot = capture_state(env)
                outcomes = {}
                for name, config in profiles.items():
                    restore_state(env, snapshot)
                    expert = PickPlaceExpert(env, config)
                    outcome = expert.execute(expert.plan_from_positions(retry_position, target_position, target_geom=target_name), speed=0.0)
                    outcomes[name] = {"success": bool(outcome["success"]), "strict_grasp_success": bool(outcome["strict_grasp_success"]), "target_distance_m": float(outcome["target_distance"])}
                label_id, outcome_type = action_label(outcomes["standard"]["success"], outcomes["deep_tight_slow"]["success"])
                record = {
                    "split": split_name,
                    "seed": seed,
                    "task": task,
                    "domain": "severe_contact_shift",
                    "source_name": source_name,
                    "target_name": target_name,
                    "initial_detection_area_px": int(detection.area),
                    "retry_detection_area_px": int(retry_detection.area),
                    "first_target_distance_m": float(first["target_distance"]),
                    "visual_target_status": target_status,
                    "outcomes": outcomes,
                    "action_label": ACTION_LABELS[label_id],
                    "action_label_id": label_id,
                    "outcome_type": outcome_type,
                    "runtime_input_boundary": "A future selector receives only post-failure RGB and the task instruction. MuJoCo snapshots are used only for offline counterfactual labels.",
                }
                top_images.append(top_after)
                front_images.append(front_after)
                proprioceptions.append(proprioception)
                labels.append(label_id)
                splits.append(split_id)
                seeds.append(seed)
                task_names.append(task)
                records.append(record)
                if args.log_every and len(records) % args.log_every == 0:
                    print(json.dumps(record, ensure_ascii=False), flush=True)
    if not records:
        raise RuntimeError("no RGB-verifiable post-failure states were collected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        top_images=np.stack(top_images).astype(np.uint8),
        front_images=np.stack(front_images).astype(np.uint8),
        proprio=np.stack(proprioceptions).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int8),
        splits=np.asarray(splits, dtype=np.int8),
        seeds=np.asarray(seeds, dtype=np.int32),
        task_names=np.asarray(task_names),
        metadata=json.dumps({"version": args.version, "class_names": ACTION_LABELS, "profiles": list(profiles), "proprio_spec": PROPRIO_SPEC, "runtime_boundary": "Post-failure RGB, instruction, and robot-proprioception only at selector runtime."}, ensure_ascii=False),
    )
    args.records.parent.mkdir(parents=True, exist_ok=True)
    args.records.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    summary = {
        "version": args.version,
        "counters": counters,
        "candidate_samples": len(records),
        "proprio_dim": int(proprioceptions[0].shape[0]),
        "proprio_spec": PROPRIO_SPEC,
        "split_class_counts": {split: {name: int(sum(record["split"] == split and record["action_label"] == name for record in records)) for name in ACTION_LABELS} for split in ("train", "test")},
        "split_outcome_counts": {split: {name: int(sum(record["split"] == split and record["outcome_type"] == name for record in records)) for name in ("both_failed", "both_success_prefer_standard", "standard_only_success", "deep_only_success")} for split in ("train", "test")},
        "dataset": str(args.output),
        "records": str(args.records),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
