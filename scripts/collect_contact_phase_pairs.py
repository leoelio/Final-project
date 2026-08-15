from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from widowx_env import WidowXTabletopEnv  # noqa: E402
from widowx_env.scripted_expert import PickPlaceConfig, PickPlaceExpert, interpolate_action, new_motion_trace, update_motion_trace  # noqa: E402


VERSION = "contact_phase_pairs_v3"
TASKS = (
    ("place_blue_cube_blue_pad", "medium"),
    ("place_blue_cube_red_pad", "medium"),
    ("place_red_cube_red_pad", "medium"),
    ("move_leftmost_cube_to_bowl", "language"),
)
STAGES = (
    ("approach", "approach", "approach_steps"),
    ("descend", "grasp_open", "descend_steps"),
    ("close", "grasp_closed", "close_steps"),
    ("lift", "lift_closed", "lift_steps"),
    ("transfer", "transfer_closed", "transfer_steps"),
    ("place_descend", "place_closed", "place_descend_steps"),
    ("release", "place_open", "open_steps"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect paired success/stress MuJoCo contact-stage states.")
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--pairs-per-task", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "contact_phase_pairs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-viewer", dest="viewer", action="store_false")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--speed", type=float, default=0.18)
    parser.add_argument("--nominal-arm-kp", type=float, default=105.0)
    parser.add_argument("--nominal-arm-force", type=float, default=70.0)
    parser.add_argument("--nominal-gripper-kp", type=float, default=550.0)
    parser.add_argument("--nominal-gripper-force", type=float, default=75.0)
    parser.add_argument("--nominal-friction", type=float, default=0.8)
    parser.add_argument("--stress-arm-kp", type=float, default=70.0)
    parser.add_argument("--stress-arm-force", type=float, default=35.0)
    parser.add_argument("--stress-gripper-kp", type=float, default=80.0)
    parser.add_argument("--stress-gripper-force", type=float, default=5.0)
    parser.add_argument("--stress-friction", type=float, default=0.08)
    parser.set_defaults(viewer=False)
    return parser.parse_args()


def make_run_dir(args: argparse.Namespace) -> Path:
    name = args.run_name or f"{VERSION}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output / name
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    (run_dir / "states").mkdir(parents=True)
    return run_dir


def profile(args: argparse.Namespace, name: str) -> dict[str, float]:
    prefix = f"{name}_"
    return {
        "arm_kp": float(getattr(args, f"{prefix}arm_kp")),
        "arm_force": float(getattr(args, f"{prefix}arm_force")),
        "gripper_kp": float(getattr(args, f"{prefix}gripper_kp")),
        "gripper_force": float(getattr(args, f"{prefix}gripper_force")),
        "friction": float(getattr(args, f"{prefix}friction")),
    }


def setup_env(args: argparse.Namespace, task: str, complexity: str, seed: int, settings: dict[str, float]) -> tuple[WidowXTabletopEnv, dict]:
    env = WidowXTabletopEnv(seed=seed, image_size=(args.image_size, args.image_size), camera="top_rgb", workspace_profile="core_v2")
    env.set_arm_actuator_strength(kp=settings["arm_kp"], force_limit=settings["arm_force"])
    env.set_gripper_actuator_strength(kp=settings["gripper_kp"], force_limit=settings["gripper_force"])
    env.set_grasp_contact_friction(sliding=settings["friction"])
    return env, env.reset(task=task, complexity=complexity, seed=seed)


def capture(env: WidowXTabletopEnv, label: str, images: list[np.ndarray], states: list[dict]) -> None:
    images.append(env.render_rgb().astype(np.uint8))
    states.append(
        {
            "label": label,
            "qpos": env.data.qpos[: env.robot_nq].astype(np.float32).copy(),
            "qvel": env.data.qvel[: env.robot_nv].astype(np.float32).copy(),
            "ctrl": env.data.ctrl.astype(np.float32).copy(),
            "actuator_force": env.data.actuator_force.astype(np.float32).copy(),
            "tcp": env.tcp_position().astype(np.float32).copy(),
            "time": float(env.data.time),
        }
    )


def execute_stage(env: WidowXTabletopEnv, target_action: np.ndarray, steps: int, trace: dict, viewer, speed: float) -> None:
    start_action = env.data.ctrl.copy()
    dt = float(env.model.opt.timestep)
    for index in range(int(steps)):
        env.step(interpolate_action(start_action, target_action, index, int(steps)))
        update_motion_trace(env, trace)
        if viewer is not None:
            viewer.sync()
            if speed > 0:
                time.sleep(dt / speed)


def collect_trial(
    args: argparse.Namespace,
    run_dir: Path,
    trial_index: int,
    pair_id: str,
    task: str,
    complexity: str,
    seed: int,
    profile_name: str,
    viewer=None,
    env: WidowXTabletopEnv | None = None,
    obs: dict | None = None,
) -> dict:
    settings = profile(args, profile_name)
    if env is None or obs is None:
        env, obs = setup_env(args, task, complexity, seed, settings)
    expert = PickPlaceExpert(env, PickPlaceConfig(place_tcp_z=0.041))
    object_name = str(obs["target_object"])
    plan = expert.plan(object_name, str(env.task.target_geom))
    trace = new_motion_trace(env)
    images: list[np.ndarray] = []
    snapshots: list[dict] = []
    capture(env, "approach_pre", images, snapshots)
    for stage_name, action_key, steps_attr in STAGES:
        if stage_name == "close":
            capture(env, "close_pre", images, snapshots)
        execute_stage(env, plan["actions"][action_key], int(getattr(expert.config, steps_attr)), trace, viewer, args.speed)
        capture(env, f"{stage_name}_post", images, snapshots)
    metrics = env.metrics()
    final_position = env.object_position(object_name)
    target_position = env.target_position(str(env.task.target_geom))
    target_distance = float(np.linalg.norm(final_position[:2] - target_position[:2]))
    strict_grasp = bool(trace["strict_grasp_success"])
    task_success = bool(metrics["success"] and strict_grasp)
    failure_stage = "complete"
    if not task_success:
        failure_stage = "close_or_lift" if not strict_grasp else "transfer_or_place"
    state_file = run_dir / "states" / f"trial_{trial_index:05d}_{profile_name}_{task}_seed_{seed}.npz"
    np.savez_compressed(
        state_file,
        images=np.stack(images),
        snapshot_labels=np.asarray([item["label"] for item in snapshots]),
        qpos=np.stack([item["qpos"] for item in snapshots]),
        qvel=np.stack([item["qvel"] for item in snapshots]),
        ctrl=np.stack([item["ctrl"] for item in snapshots]),
        actuator_force=np.stack([item["actuator_force"] for item in snapshots]),
        tcp=np.stack([item["tcp"] for item in snapshots]),
        times=np.asarray([item["time"] for item in snapshots], dtype=np.float32),
    )
    return {
        "version": VERSION,
        "trial_index": trial_index,
        "pair_id": pair_id,
        "profile": profile_name,
        "task": task,
        "complexity": complexity,
        "seed": seed,
        "instruction": obs["instruction"],
        "target_object": object_name,
        "target_geom": env.task.target_geom,
        "contact_settings": settings,
        "task_success": task_success,
        "strict_grasp_success": strict_grasp,
        "failure_stage": failure_stage,
        "target_distance_m": target_distance,
        "object_z": float(final_position[2]),
        "out_of_table": bool(metrics["out_of_table"]),
        "state_file": state_file.relative_to(run_dir).as_posix(),
        "snapshot_labels": [item["label"] for item in snapshots],
        "offline_label_boundary": "Object pose, strict grasp, target distance and success are offline labels only; policy inputs are the stored RGB, robot state and action history.",
    }


def main() -> None:
    args = parse_args()
    if args.pairs_per_task < 1:
        raise ValueError("pairs-per-task must be positive")
    run_dir = make_run_dir(args)
    metadata_path = run_dir / "metadata.jsonl"
    rows: list[dict] = []
    with metadata_path.open("w", encoding="utf-8") as file:
        trial_index = 0
        for task_index, (task, complexity) in enumerate(TASKS):
            for offset in range(args.pairs_per_task):
                seed = args.seed + task_index * 100 + offset
                pair_id = f"{task}_seed_{seed}"
                for profile_name in ("nominal", "stress"):
                    viewer_enabled = bool(args.viewer and trial_index == 0)
                    if viewer_enabled:
                        env, obs = setup_env(args, task, complexity, seed, profile(args, profile_name))
                        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                            row = collect_trial(
                                args,
                                run_dir,
                                trial_index,
                                pair_id,
                                task,
                                complexity,
                                seed,
                                profile_name,
                                viewer,
                                env,
                                obs,
                            )
                            started = time.time()
                            while viewer.is_running() and time.time() - started < args.duration:
                                viewer.sync()
                                time.sleep(0.01)
                    else:
                        row = collect_trial(args, run_dir, trial_index, pair_id, task, complexity, seed, profile_name)
                    rows.append(row)
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    file.flush()
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    trial_index += 1
    summary = {
        "version": VERSION,
        "pairs_per_task": args.pairs_per_task,
        "trials": len(rows),
        "task_successes": sum(row["task_success"] for row in rows),
        "failures": sum(not row["task_success"] for row in rows),
        "by_profile": {
            name: {
                "trials": sum(row["profile"] == name for row in rows),
                "successes": sum(row["profile"] == name and row["task_success"] for row in rows),
                "failures": sum(row["profile"] == name and not row["task_success"] for row in rows),
            }
            for name in ("nominal", "stress")
        },
        "metadata": metadata_path.name,
        "schema": "states/*.npz contains RGB stage frames, robot-only qpos/qvel, ctrl, actuator_force, TCP and phase labels.",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"run_dir: {run_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
