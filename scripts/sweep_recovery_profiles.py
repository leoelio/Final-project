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
from run_clip_semantic_rgb_feedback import visual_target_status  # noqa: E402
from widowx_env.scripted_expert import PickPlaceConfig, PickPlaceExpert, capture_state, restore_state  # noqa: E402
from widowx_env.vision_grounding import load_calibration, relocate_known_object  # noqa: E402


DEFAULT_TASKS = "place_blue_cube_red_pad,move_leftmost_cube_to_bowl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep contact domains for seed-paired standard versus deep recovery trajectories.")
    parser.add_argument("--calibration", type=Path, default=ROOT / "outputs" / "rgb_grounding" / "top_rgb_core_v2_calibration_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "evaluations" / "recovery_profile_contact_sweep_v1.json")
    parser.add_argument("--seed", type=int, default=2100)
    parser.add_argument("--episodes", type=int, default=30, help="Paired scene seeds per task and domain.")
    parser.add_argument("--domains", default="mild_contact_shift,low_contact_shift,severe_contact_shift")
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


def selected_domains(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in DOMAINS]
    if unknown:
        raise KeyError(f"unknown domains: {unknown}")
    return names


def configs(place_tcp_z: float) -> dict[str, PickPlaceConfig]:
    base = PickPlaceConfig(place_tcp_z=place_tcp_z)
    return {
        "standard": base,
        "deep_tight_slow": replace(base, grasp_z_offset=0.0, close_gripper=0.007, close_steps=420, descend_steps=300, lift_steps=500, transfer_steps=800),
    }


def outcome_type(standard: bool, deep: bool) -> str:
    if standard and deep:
        return "both_success"
    if standard:
        return "standard_only_success"
    if deep:
        return "deep_only_success"
    return "both_failed"


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    calibration = load_calibration(args.calibration)
    profiles = configs(args.place_tcp_z)
    records: list[dict] = []
    counters = {"scanned": 0, "reset_failed": 0, "initial_not_visible": 0, "first_success": 0, "target_complete_or_ambiguous": 0, "source_not_visible": 0}
    for domain_name in selected_domains(args.domains):
        for task, complexity, source_kind, target_name in selected_tasks(args.tasks):
            for offset in range(args.episodes):
                seed = args.seed + offset
                counters["scanned"] += 1
                env = configure_env(DOMAINS[domain_name], seed, args.image_size)
                try:
                    env.reset(task=task, complexity=complexity, seed=seed)
                except RuntimeError:
                    counters["reset_failed"] += 1
                    continue
                try:
                    source_name, source_position, _ = initial_source(render_rgb(env, "top_rgb", args.image_size), calibration, source_kind)
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
                target_status = visual_target_status(top_after, calibration, source_name, target_name)
                if not bool(target_status["verifiable"]) or bool(target_status["complete"]):
                    counters["target_complete_or_ambiguous"] += 1
                    continue
                try:
                    retry_position, _ = relocate_known_object(top_after, calibration, source_name, source_position[:2])
                except (LookupError, ValueError):
                    counters["source_not_visible"] += 1
                    continue
                snapshot = capture_state(env)
                outcomes = {}
                for name, config in profiles.items():
                    restore_state(env, snapshot)
                    expert = PickPlaceExpert(env, config)
                    result = expert.execute(expert.plan_from_positions(retry_position, target_position, target_geom=target_name), speed=0.0)
                    outcomes[name] = bool(result["success"])
                record = {"domain": domain_name, "task": task, "seed": seed, "outcomes": outcomes, "outcome_type": outcome_type(outcomes["standard"], outcomes["deep_tight_slow"])}
                records.append(record)
                if args.log_every and len(records) % args.log_every == 0:
                    print(json.dumps(record, ensure_ascii=False), flush=True)
    names = ("both_success", "standard_only_success", "deep_only_success", "both_failed")
    summary = {
        "version": "recovery_profile_contact_sweep_v1",
        "seed_range": f"{args.seed}-{args.seed + args.episodes - 1}",
        "domains": {name: DOMAINS[name] for name in selected_domains(args.domains)},
        "tasks": [task for task, _complexity, _source, _target in selected_tasks(args.tasks)],
        "counters": counters,
        "candidate_states": len(records),
        "by_domain": {
            domain: {
                "candidate_states": sum(record["domain"] == domain for record in records),
                "standard_successes": sum(record["domain"] == domain and record["outcomes"]["standard"] for record in records),
                "deep_successes": sum(record["domain"] == domain and record["outcomes"]["deep_tight_slow"] for record in records),
                "outcome_counts": {name: sum(record["domain"] == domain and record["outcome_type"] == name for record in records) for name in names},
            }
            for domain in selected_domains(args.domains)
        },
        "records": records,
        "runtime_boundary": "This is an offline counterfactual sweep. Each recovery trajectory plans from RGB source re-localization; MuJoCo snapshots only reset the post-failure state for fair candidate comparison.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
