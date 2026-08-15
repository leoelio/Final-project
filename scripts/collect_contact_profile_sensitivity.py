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

from collect_multiview_recovery_dataset import DOMAINS, STATIC_TARGETS, configure_env, initial_source, render_rgb  # noqa: E402
from run_clip_semantic_rgb_feedback import visual_target_status  # noqa: E402
from widowx_env.scripted_expert import PickPlaceConfig, PickPlaceExpert, capture_state, restore_state  # noqa: E402
from widowx_env.vision_grounding import load_calibration, relocate_known_object  # noqa: E402


TASK = "move_leftmost_cube_to_bowl"
COMPLEXITY = "language"
SOURCE_KIND = "leftmost_cube"
TARGET_NAME = "target_bowl"
ACTION_NAMES = ("stop", "standard", "low_grasp", "tight_grip", "slow_timing", "gentle_transfer", "deep_tight_slow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect updated severe-contact failure states and same-snapshot recovery-profile outcomes.")
    parser.add_argument("--calibration", type=Path, default=ROOT / "outputs" / "rgb_grounding" / "top_rgb_core_v2_calibration_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "contact_profile_sensitivity" / "leftmost_severe_v1.npz")
    parser.add_argument("--records", type=Path, default=ROOT / "data" / "contact_profile_sensitivity" / "leftmost_severe_v1.jsonl")
    parser.add_argument("--summary", type=Path, default=ROOT / "data" / "contact_profile_sensitivity" / "leftmost_severe_v1_summary.json")
    parser.add_argument("--train-seed", type=int, default=3800)
    parser.add_argument("--train-episodes", type=int, default=80)
    parser.add_argument("--test-seed", type=int, default=4800)
    parser.add_argument("--test-episodes", type=int, default=40)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--place-tcp-z", type=float, default=0.041)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def profile_configs(place_tcp_z: float) -> dict[str, PickPlaceConfig]:
    base = PickPlaceConfig(place_tcp_z=place_tcp_z)
    return {
        "standard": base,
        "low_grasp": replace(base, grasp_z_offset=0.0),
        "tight_grip": replace(base, close_gripper=0.007),
        "slow_timing": replace(base, close_steps=420, descend_steps=300, lift_steps=500, transfer_steps=800),
        "gentle_transfer": replace(base, transfer_steps=1600),
        "deep_tight_slow": replace(base, grasp_z_offset=0.0, close_gripper=0.007, close_steps=420, descend_steps=300, lift_steps=500, transfer_steps=800),
    }


def robot_proprioception(env) -> np.ndarray:
    return np.concatenate([env.data.qpos[:8], env.data.qvel[:8], env.data.ctrl[:7], env.data.actuator_force[:7]]).astype(np.float32)


def preferred_action(outcomes: dict[str, dict] | None) -> str:
    if outcomes is None:
        return "stop"
    for name in ACTION_NAMES[1:]:
        if outcomes[name]["success"]:
            return name
    return "stop"


def main() -> None:
    args = parse_args()
    if args.train_episodes < 1 or args.test_episodes < 1:
        raise ValueError("train/test episode counts must be positive")

    calibration = load_calibration(args.calibration)
    profiles = profile_configs(args.place_tcp_z)
    top_images: list[np.ndarray] = []
    front_images: list[np.ndarray] = []
    proprioceptions: list[np.ndarray] = []
    success_matrix: list[np.ndarray] = []
    strict_matrix: list[np.ndarray] = []
    labels: list[int] = []
    splits: list[int] = []
    seeds: list[int] = []
    records: list[dict] = []
    counters = {
        "scanned": 0,
        "reset_failed": 0,
        "initial_not_visible": 0,
        "first_success": 0,
        "post_failure_states": 0,
        "visual_target_complete_or_ambiguous": 0,
        "source_not_relocalizable": 0,
        "counterfactual_states": 0,
    }

    for split_name, split_id, base_seed, count in (("train", 0, args.train_seed, args.train_episodes), ("test", 1, args.test_seed, args.test_episodes)):
        for offset in range(count):
            seed = base_seed + offset
            counters["scanned"] += 1
            env = configure_env(DOMAINS["severe_contact_shift"], seed, args.image_size)
            try:
                env.reset(task=TASK, complexity=COMPLEXITY, seed=seed)
            except RuntimeError:
                counters["reset_failed"] += 1
                continue
            before_top = render_rgb(env, "top_rgb", args.image_size)
            try:
                source_name, source_position, detection = initial_source(before_top, calibration, SOURCE_KIND)
            except (LookupError, ValueError):
                counters["initial_not_visible"] += 1
                continue

            target_position = STATIC_TARGETS[TARGET_NAME]
            first_expert = PickPlaceExpert(env, profiles["standard"])
            first = first_expert.execute(first_expert.plan_from_positions(source_position, target_position, target_geom=TARGET_NAME), speed=0.0)
            if bool(first["success"]):
                counters["first_success"] += 1
                continue

            counters["post_failure_states"] += 1
            top_after = render_rgb(env, "top_rgb", args.image_size)
            front_after = render_rgb(env, "front_rgb", args.image_size)
            post_failure_proprio = robot_proprioception(env)
            target_status = visual_target_status(top_after, calibration, source_name, TARGET_NAME)
            route = "counterfactual"
            retry_detection = None
            outcomes: dict[str, dict] | None = None
            if not bool(target_status["verifiable"]) or bool(target_status["complete"]):
                route = "visual_target_complete_or_ambiguous"
                counters["visual_target_complete_or_ambiguous"] += 1
            else:
                try:
                    retry_position, retry_detection = relocate_known_object(top_after, calibration, source_name, source_position[:2])
                except (LookupError, ValueError):
                    route = "source_not_relocalizable"
                    counters["source_not_relocalizable"] += 1
                else:
                    counters["counterfactual_states"] += 1
                    snapshot = capture_state(env)
                    outcomes = {}
                    for name, config in profiles.items():
                        restore_state(env, snapshot)
                        expert = PickPlaceExpert(env, config)
                        result = expert.execute(expert.plan_from_positions(retry_position, target_position, target_geom=TARGET_NAME), speed=0.0)
                        outcomes[name] = {
                            "success": bool(result["success"]),
                            "strict_grasp_success": bool(result["strict_grasp_success"]),
                            "target_distance_m": float(result["target_distance"]),
                            "max_object_z_m": float(result["max_object_z"]),
                            "held_after_lift": bool(result["held_after_lift"]),
                            "held_after_transfer": bool(result["held_after_transfer"]),
                            "held_before_release": bool(result["held_before_release"]),
                            "transfer_tcp_object_distance_m": float(result["transfer_tcp_object_distance"]),
                            "place_tcp_object_distance_m": float(result["place_tcp_object_distance"]),
                        }

            action = preferred_action(outcomes)
            record = {
                "split": split_name,
                "seed": seed,
                "task": TASK,
                "domain": "severe_contact_shift",
                "source_name": source_name,
                "target_name": TARGET_NAME,
                "initial_detection_area_px": int(detection.area),
                "first": {
                    "success": bool(first["success"]),
                    "strict_grasp_success": bool(first["strict_grasp_success"]),
                    "target_distance_m": float(first["target_distance"]),
                    "max_object_z_m": float(first["max_object_z"]),
                },
                "visual_target_status": target_status,
                "recovery_route": route,
                "retry_detection_area_px": None if retry_detection is None else int(retry_detection.area),
                "outcomes": outcomes,
                "preferred_action": action,
                "runtime_input_boundary": "A future selector may use only post-failure top/front RGB, the instruction, and robot proprioception. MuJoCo snapshots are offline counterfactual supervision only.",
            }
            top_images.append(top_after)
            front_images.append(front_after)
            proprioceptions.append(post_failure_proprio)
            success_matrix.append(np.asarray([-1 if outcomes is None else int(outcomes[name]["success"]) for name in ACTION_NAMES[1:]], dtype=np.int8))
            strict_matrix.append(np.asarray([-1 if outcomes is None else int(outcomes[name]["strict_grasp_success"]) for name in ACTION_NAMES[1:]], dtype=np.int8))
            labels.append(ACTION_NAMES.index(action))
            splits.append(split_id)
            seeds.append(seed)
            records.append(record)
            if args.log_every and len(records) % args.log_every == 0:
                print(json.dumps(record, ensure_ascii=False), flush=True)

    if not records:
        raise RuntimeError("no post-failure states were collected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        top_images=np.stack(top_images).astype(np.uint8),
        front_images=np.stack(front_images).astype(np.uint8),
        proprio=np.stack(proprioceptions).astype(np.float32),
        profile_success=np.stack(success_matrix),
        profile_strict_grasp=np.stack(strict_matrix),
        preferred_actions=np.asarray(labels, dtype=np.int8),
        splits=np.asarray(splits, dtype=np.int8),
        seeds=np.asarray(seeds, dtype=np.int32),
        metadata=json.dumps({"version": "leftmost_severe_contact_profile_sensitivity_v1", "action_names": ACTION_NAMES, "profiles": list(profiles), "task": TASK, "domain": "severe_contact_shift"}, ensure_ascii=False),
    )
    args.records.parent.mkdir(parents=True, exist_ok=True)
    args.records.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    action_counts = {split: {name: int(sum(record["split"] == split and record["preferred_action"] == name for record in records)) for name in ACTION_NAMES} for split in ("train", "test")}
    outcome_patterns = {split: {} for split in ("train", "test")}
    for record in records:
        if record["outcomes"] is None:
            continue
        pattern = "/".join(name for name in ACTION_NAMES[1:] if record["outcomes"][name]["success"]) or "all_failed"
        outcome_patterns[record["split"]][pattern] = outcome_patterns[record["split"]].get(pattern, 0) + 1
    summary = {
        "version": "leftmost_severe_contact_profile_sensitivity_v1",
        "task_protocol": "Updated visually separable move_leftmost_cube_to_bowl only (minimum leftmost-cube gap 0.03 m).",
        "counters": counters,
        "post_failure_samples": len(records),
        "split_preferred_action_counts": action_counts,
        "split_counterfactual_success_patterns": outcome_patterns,
        "dataset": str(args.output),
        "records": str(args.records),
        "decision_rule": "This collector does not train or deploy a selector. A selector is considered only if seed-disjoint counterfactual data contains enough non-majority action labels.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
