from __future__ import annotations

import argparse
from collections import Counter
from copy import copy
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_rgb_recovery_profiles import DOMAINS, rollout_args  # noqa: E402
from run_clip_action_head import load_clip  # noqa: E402
from run_clip_semantic_rgb_feedback import (  # noqa: E402
    STATIC_TARGETS,
    attempt_config,
    configure_env,
    locate_initial_source,
    render_top_rgb,
    visual_target_status,
)
from run_clip_semantic_waypoint import INTENTS, load_policy, normalize_instruction, predict_intent  # noqa: E402
from widowx_env.scripted_expert import PickPlaceExpert, capture_state, restore_state  # noqa: E402
from widowx_env.vision_grounding import load_calibration, relocate_known_object  # noqa: E402

from evaluate_contact_phase_monitor import TASKS, execute_until_lift, finish_first_attempt, robot_vector  # noqa: E402


VERSION = "counterfactual_intervention_pairs_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect same-lift-state continue-versus-early-regrasp counterfactual labels.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=ROOT / "outputs" / "rgb_grounding" / "top_rgb_core_v2_calibration_v1.json")
    parser.add_argument("--seed", type=int, default=11000)
    parser.add_argument("--episodes", type=int, default=20, help="Seeds per task and contact domain.")
    parser.add_argument("--domains", default="mild_contact_shift,low_contact_shift,severe_contact_shift")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "counterfactual_intervention_pairs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--speed", type=float, default=0.18)
    return parser.parse_args()


def selected(value: str, known: dict, label: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in known]
    if not names or unknown:
        raise KeyError(f"unknown {label}: {unknown}")
    return names


def make_run_dir(args: argparse.Namespace) -> Path:
    name = args.run_name or f"{VERSION}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output / name
    if run_dir.exists():
        raise FileExistsError(run_dir)
    (run_dir / "states").mkdir(parents=True)
    return run_dir


def branch_continue(env, args, calibration, selected_name: str, source_position: np.ndarray, target_position: np.ndarray, target_geom: str, plan: dict, trace: dict, viewer) -> tuple[dict, str]:
    expert = PickPlaceExpert(env, attempt_config(args, 0))
    final = finish_first_attempt(expert, plan, trace, viewer, args.speed if viewer is not None else 0.0)
    after = render_top_rgb(env, args.image_size, args.camera)
    status = visual_target_status(after, calibration, selected_name, target_geom)
    if status["complete"]:
        return final, "continue_visual_target_confirmed"
    try:
        recovered, _ = relocate_known_object(after, calibration, selected_name, source_position[:2], search_scope="table")
    except LookupError:
        return final, "continue_table_object_not_visually_recoverable"
    retry_expert = PickPlaceExpert(env, attempt_config(args, 1))
    retry_plan = retry_expert.plan_from_positions(recovered, target_position, target_geom=target_geom)
    return retry_expert.execute(retry_plan, viewer=viewer, speed=args.speed if viewer is not None else 0.0), "continue_v4_table_retry"


def branch_early_deep(env, args, calibration, selected_name: str, source_position: np.ndarray, target_position: np.ndarray, target_geom: str, viewer) -> tuple[dict, str]:
    after_lift = render_top_rgb(env, args.image_size, args.camera)
    try:
        recovered, _ = relocate_known_object(after_lift, calibration, selected_name, source_position[:2], search_scope="table")
    except LookupError:
        return {"success": False, "strict_grasp_success": False, "target_distance": float("nan")}, "early_table_object_not_visually_recoverable"
    recovery_args = copy(args)
    recovery_args.recovery_profile = "deep_tight_slow"
    expert = PickPlaceExpert(env, attempt_config(recovery_args, 1))
    plan = expert.plan_from_positions(recovered, target_position, target_geom=target_geom)
    return expert.execute(plan, viewer=viewer, speed=args.speed if viewer is not None else 0.0), "early_table_relocalized_deep_retry"


def label(continue_success: bool, early_success: bool) -> str:
    if early_success and not continue_success:
        return "early_better"
    if continue_success and not early_success:
        return "continue_better"
    return "tie"


def collect_one(args, policy, clip_model, processor, calibration, domain_name: str, task: str, seed: int, run_dir: Path, index: int, viewer=None, env=None, obs=None) -> dict:
    config = rollout_args(task, TASKS[task], "standard", DOMAINS[domain_name])
    config.recovery_search = "table"
    config.speed = args.speed
    if env is None or obs is None:
        env, obs = configure_env(config, seed)
    instruction = str(obs["instruction"])
    intent, _ = predict_intent(env, {**obs, "instruction": instruction}, policy, clip_model, processor, config.image_size, config.camera)
    source_name, target_geom = INTENTS[intent]
    initial = render_top_rgb(env, config.image_size, config.camera)
    base = {
        "version": VERSION,
        "index": index,
        "domain": domain_name,
        "task": task,
        "seed": seed,
        "instruction": instruction,
        "predicted_intent": intent,
        "semantic_correct": bool(intent == task),
        "selected_object": source_name,
        "target_geom": target_geom,
        "offline_label_boundary": "Full MuJoCo state is used only in-memory to restore counterfactual branches. Saved policy inputs contain RGB and robot-only proprioception; object truth is offline scoring only.",
    }
    try:
        selected_name, source_position, _ = locate_initial_source(initial, calibration, intent)
    except LookupError as error:
        return {**base, "label": "not_executable", "reason": f"initial_rgb_grounding_failed: {error}"}
    source_error = float(np.linalg.norm(source_position[:2] - env.object_position(obs["target_object"])[:2]))
    if not base["semantic_correct"] or selected_name != obs["target_object"] or source_error > 0.04:
        return {**base, "label": "not_executable", "reason": "semantic_or_initial_selection_incorrect", "initial_source_error_m": source_error}

    target_position = STATIC_TARGETS[target_geom]
    expert = PickPlaceExpert(env, attempt_config(config, 0))
    plan = expert.plan_from_positions(source_position, target_position, target_geom=target_geom)
    trace, close_rgb, lift_rgb, close_robot, lift_robot = execute_until_lift(expert, plan, viewer, args.speed if viewer is not None else 0.0)
    lift_snapshot = capture_state(env)

    continue_final, continue_reason = branch_continue(env, config, calibration, selected_name, source_position, target_position, target_geom, plan, trace, viewer)
    continue_metrics = env.metrics()
    restore_state(env, lift_snapshot)
    early_final, early_reason = branch_early_deep(env, config, calibration, selected_name, source_position, target_position, target_geom, viewer)
    early_metrics = env.metrics()

    continue_success = bool(continue_final.get("success", False))
    early_success = bool(early_final.get("success", False))
    state_file = run_dir / "states" / f"pair_{index:05d}_{domain_name}_{task}_seed_{seed}.npz"
    np.savez_compressed(
        state_file,
        close_rgb=close_rgb,
        lift_rgb=lift_rgb,
        close_robot=close_robot,
        lift_robot=lift_robot,
    )
    return {
        **base,
        "selected_object": selected_name,
        "initial_source_error_m": source_error,
        "label": label(continue_success, early_success),
        "continue_v4_success": continue_success,
        "early_deep_success": early_success,
        "continue_reason": continue_reason,
        "early_reason": early_reason,
        "continue_target_distance_m": float(continue_metrics["target_distance"]),
        "early_target_distance_m": float(early_metrics["target_distance"]),
        "continue_strict_grasp_success": bool(continue_final.get("strict_grasp_success", False)),
        "early_strict_grasp_success": bool(early_final.get("strict_grasp_success", False)),
        "state_file": state_file.relative_to(run_dir).as_posix(),
        "robot_feature_width": int(close_robot.shape[0]),
    }


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    domains = selected(args.domains, DOMAINS, "domains")
    tasks = selected(args.tasks, TASKS, "tasks")
    run_dir = make_run_dir(args)
    policy = load_policy(args.model)
    calibration = load_calibration(args.calibration)
    clip_model, processor = load_clip(str(policy["metadata"]["clip_model"]))
    rows: list[dict] = []
    metadata = run_dir / "metadata.jsonl"
    with metadata.open("w", encoding="utf-8") as file:
        index = 0
        for domain_name in domains:
            for task in tasks:
                for offset in range(args.episodes):
                    seed = args.seed + offset
                    if args.viewer and index == 0:
                        config = rollout_args(task, TASKS[task], "standard", DOMAINS[domain_name])
                        env, obs = configure_env(config, seed)
                        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                            row = collect_one(args, policy, clip_model, processor, calibration, domain_name, task, seed, run_dir, index, viewer, env, obs)
                            started = time.time()
                            while viewer.is_running() and time.time() - started < args.duration:
                                viewer.sync()
                                time.sleep(0.01)
                    else:
                        row = collect_one(args, policy, clip_model, processor, calibration, domain_name, task, seed, run_dir, index)
                    rows.append(row)
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    file.flush()
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    index += 1
    labels = Counter(row["label"] for row in rows)
    summary = {
        "version": VERSION,
        "scenes": len(rows),
        "labels": dict(sorted(labels.items())),
        "domains": domains,
        "tasks": tasks,
        "episodes_per_task_domain": args.episodes,
        "metadata": metadata.name,
        "saved_input_schema": "states/*.npz contains close/lift RGB and robot-only 32D vectors; full MuJoCo branch snapshots are never persisted.",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
