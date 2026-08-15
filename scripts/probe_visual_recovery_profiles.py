from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from collect_multiview_recovery_dataset import DOMAINS, STATIC_TARGETS, TASK_SPECS, configure_env, initial_source, render_rgb  # noqa: E402
from widowx_env.scripted_expert import PickPlaceConfig, PickPlaceExpert, capture_state, restore_state  # noqa: E402
from widowx_env.vision_grounding import load_calibration, relocate_known_object  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe visual recovery action profiles from identical post-failure MuJoCo states.")
    parser.add_argument("--calibration", type=Path, default=ROOT / "outputs" / "rgb_grounding" / "top_rgb_core_v2_calibration_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "probes" / "visual_recovery_profiles_v1.json")
    parser.add_argument("--seed", type=int, default=1200)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--domain", choices=tuple(DOMAINS), default="severe_contact_shift")
    parser.add_argument("--tasks", default="move_leftmost_cube_to_bowl")
    parser.add_argument("--place-tcp-z", type=float, default=0.041)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def selected_tasks(value: str) -> list[tuple[str, str, str, str]]:
    known = {task: (task, complexity, source, target) for task, complexity, source, target in TASK_SPECS}
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in known]
    if unknown:
        raise KeyError(f"unknown tasks: {unknown}")
    return [known[name] for name in names]


def profiles(place_tcp_z: float) -> dict[str, PickPlaceConfig]:
    base = PickPlaceConfig(place_tcp_z=place_tcp_z)
    return {
        "standard": base,
        "tight_grip": replace(base, close_gripper=0.007, close_steps=420),
        "tight_grip_high_lift": replace(base, close_gripper=0.007, close_steps=420, lift_z_offset=0.22, transfer_z_offset=0.22),
        "deep_tight_grip": replace(base, grasp_z_offset=0.0, close_gripper=0.007, close_steps=420),
        "deep_tight_slow": replace(base, grasp_z_offset=0.0, close_gripper=0.007, close_steps=420, descend_steps=300, lift_steps=500, transfer_steps=800),
    }


def main() -> None:
    args = parse_args()
    calibration = load_calibration(args.calibration)
    records: list[dict] = []
    profile_configs = profiles(args.place_tcp_z)
    for task, complexity, source_kind, target_name in selected_tasks(args.tasks):
        for offset in range(args.episodes):
            seed = args.seed + offset
            env = configure_env(DOMAINS[args.domain], seed, args.image_size)
            env.reset(task=task, complexity=complexity, seed=seed)
            top_before = render_rgb(env, "top_rgb", args.image_size)
            try:
                source_name, source_position, _ = initial_source(top_before, calibration, source_kind)
            except (LookupError, ValueError):
                continue
            target_position = STATIC_TARGETS[target_name]
            first_expert = PickPlaceExpert(env, profile_configs["standard"])
            first = first_expert.execute(first_expert.plan_from_positions(source_position, target_position, target_geom=target_name), speed=0.0)
            if bool(first["success"]):
                continue
            snapshot = capture_state(env)
            top_after = render_rgb(env, "top_rgb", args.image_size)
            try:
                retry_position, _ = relocate_known_object(top_after, calibration, source_name, source_position[:2])
            except (LookupError, ValueError):
                continue
            outcomes = {}
            for name, config in profile_configs.items():
                restore_state(env, snapshot)
                expert = PickPlaceExpert(env, config)
                outcome = expert.execute(expert.plan_from_positions(retry_position, target_position, target_geom=target_name), speed=0.0)
                outcomes[name] = {"success": bool(outcome["success"]), "strict_grasp_success": bool(outcome["strict_grasp_success"]), "target_distance_m": float(outcome["target_distance"])}
            record = {"seed": seed, "task": task, "domain": args.domain, "source_name": source_name, "first_target_distance_m": float(first["target_distance"]), "profiles": outcomes}
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    summary = {
        "version": "visual_recovery_profiles_v1",
        "domain": args.domain,
        "tasks": [task for task, _complexity, _source, _target in selected_tasks(args.tasks)],
        "candidate_states": len(records),
        "profile_successes": {name: int(sum(record["profiles"][name]["success"] for record in records)) for name in profile_configs},
        "records": records,
        "runtime_boundary": "Every profile starts from the same post-first-attempt MuJoCo snapshot but plans from RGB re-localization; state is used only to restore counterfactual offline probes.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
