from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from widowx_env import WidowXTabletopEnv
from widowx_env.tabletop_env import OBJECTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a saved demonstration trajectory.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Demo run folder. Defaults to the newest data/demos run.")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-viewer", dest="viewer", action="store_false")
    parser.add_argument("--duration", type=float, default=0.0, help="Viewer hold time after replay. 0 means keep open.")
    parser.add_argument("--speed", type=float, default=1.0, help="Viewer playback speed multiplier.")
    parser.add_argument("--arm-kp", type=float, default=150.0)
    parser.add_argument("--arm-force", type=float, default=100.0)
    parser.add_argument("--gripper-kp", type=float, default=800.0)
    parser.add_argument("--gripper-force", type=float, default=140.0)
    parser.add_argument("--friction", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.set_defaults(viewer=True)
    return parser.parse_args()


def latest_run_dir() -> Path:
    demos_dir = ROOT / "data" / "demos"
    candidates = [path for path in demos_dir.glob("*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no demo run folders found under {demos_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_metadata(run_dir: Path, episode_index: int) -> dict:
    metadata_path = run_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata file not found: {metadata_path}")
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        metadata = json.loads(line)
        if int(metadata["episode_index"]) == episode_index:
            return metadata
    raise ValueError(f"episode_index {episode_index} not found in {metadata_path}")


def object_positions(env: WidowXTabletopEnv) -> np.ndarray:
    return np.stack([env.object_position(name) for name in OBJECTS])


def dict_object_error(env: WidowXTabletopEnv, expected: dict[str, list[float]]) -> float:
    errors = [np.linalg.norm(env.object_position(name) - np.asarray(expected[name], dtype=float)) for name in OBJECTS]
    return float(max(errors))


def restore_attempt_state(env: WidowXTabletopEnv, recorded: np.lib.npyio.NpzFile, attempt_id: int) -> bool:
    if "attempt_start_qpos" not in recorded.files:
        return False
    matches = np.flatnonzero(recorded["attempt_start_ids"] == attempt_id)
    if len(matches) == 0:
        return False
    index = int(matches[0])
    env.data.qpos[:] = recorded["attempt_start_qpos"][index]
    env.data.qvel[:] = recorded["attempt_start_qvel"][index]
    env.data.ctrl[:] = recorded["attempt_start_ctrl"][index]
    mujoco.mj_forward(env.model, env.data)
    return True


def replay(
    env: WidowXTabletopEnv,
    actions: np.ndarray,
    recorded: np.lib.npyio.NpzFile,
    viewer=None,
    speed: float = 1.0,
    max_steps: int | None = None,
) -> dict[str, float]:
    qpos_max_error = 0.0
    tcp_max_error = 0.0
    object_max_error = 0.0
    dt = float(env.model.opt.timestep)
    step_count = len(actions) if max_steps is None else min(len(actions), max_steps)
    current_attempt = None

    for step in range(step_count):
        attempt_id = int(recorded["attempt_ids"][step]) if "attempt_ids" in recorded.files else 1
        if attempt_id != current_attempt:
            restore_attempt_state(env, recorded, attempt_id)
            current_attempt = attempt_id
        env.step(actions[step])
        qpos_max_error = max(qpos_max_error, float(np.linalg.norm(env.data.qpos - recorded["qpos"][step])))
        tcp_max_error = max(tcp_max_error, float(np.linalg.norm(env.tcp_position() - recorded["tcp"][step])))
        object_max_error = max(object_max_error, float(np.linalg.norm(object_positions(env) - recorded["object_positions"][step])))
        if viewer is not None:
            viewer.sync()
            if speed > 0:
                time.sleep(dt / speed)

    return {
        "steps_replayed": step_count,
        "qpos_max_error": qpos_max_error,
        "tcp_max_error": tcp_max_error,
        "object_max_error": object_max_error,
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or latest_run_dir()
    metadata = load_metadata(run_dir, args.episode_index)
    trajectory_path = run_dir / metadata["trajectory_file"]

    env = WidowXTabletopEnv(seed=int(metadata["seed"]))
    env.set_arm_actuator_strength(kp=args.arm_kp, force_limit=args.arm_force)
    env.set_gripper_actuator_strength(kp=args.gripper_kp, force_limit=args.gripper_force)
    env.set_grasp_contact_friction(sliding=args.friction)
    obs = env.reset(task=metadata["task"], complexity=metadata["complexity"], seed=int(metadata["seed"]))
    initial_error = dict_object_error(env, metadata["initial_objects"])

    with np.load(trajectory_path) as recorded:
        actions = recorded["actions"]
        if args.viewer:
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                stats = replay(env, actions, recorded, viewer=viewer, speed=args.speed, max_steps=args.max_steps)
                start = time.time()
                while viewer.is_running():
                    viewer.sync()
                    if args.duration and time.time() - start > args.duration:
                        break
                    time.sleep(0.01)
        else:
            stats = replay(env, actions, recorded, max_steps=args.max_steps)

    metrics = env.metrics()
    final_error = dict_object_error(env, metadata["final_objects"]) if args.max_steps is None else float("nan")
    print(f"run_dir: {run_dir}", flush=True)
    print(f"trajectory_file: {trajectory_path}", flush=True)
    print(f"episode_index: {metadata['episode_index']}", flush=True)
    print(f"seed: {metadata['seed']}", flush=True)
    print(f"instruction: {metadata['instruction']}", flush=True)
    print(f"active_objects: {', '.join(obs['active_objects'])}", flush=True)
    print(f"recorded_success: {metadata['success']}", flush=True)
    print(f"replay_success: {bool(metrics['success'])}", flush=True)
    print(f"initial_object_max_error: {initial_error:.8f}", flush=True)
    print(f"final_object_max_error: {final_error:.8f}", flush=True)
    print(
        "trajectory_error: "
        f"qpos={stats['qpos_max_error']:.8f}, "
        f"tcp={stats['tcp_max_error']:.8f}, "
        f"objects={stats['object_max_error']:.8f}",
        flush=True,
    )
    print(f"metrics: {metrics}", flush=True)

    if args.max_steps is None and bool(metrics["success"]) != bool(metadata["success"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
