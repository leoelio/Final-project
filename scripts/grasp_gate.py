from __future__ import annotations

import argparse

import numpy as np


def add_grasp_gate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--grasp-gate", action="store_true")
    parser.add_argument("--no-grasp-gate", dest="grasp_gate", action="store_false")
    parser.add_argument("--close-phase", type=float, default=0.22)
    parser.add_argument("--release-phase", type=float, default=0.78)
    parser.add_argument("--near-threshold", type=float, default=0.11)
    parser.add_argument("--release-distance", type=float, default=0.095)
    parser.add_argument("--open-gripper", type=float, default=0.037)
    parser.add_argument("--close-gripper", type=float, default=0.015)
    parser.set_defaults(grasp_gate=False)


def make_gate_stats() -> dict[str, int]:
    return {"gate_open_steps": 0, "gate_closed_steps": 0, "gate_policy_steps": 0}


def apply_grasp_gate(args: argparse.Namespace, env, raw_action: np.ndarray, phase: float) -> tuple[np.ndarray, str]:
    if not bool(getattr(args, "grasp_gate", False)):
        return raw_action.astype(np.float32), "policy"

    action = raw_action.copy().astype(np.float32)
    metrics = env.metrics()
    near_object = float(metrics["ee_object_distance"]) <= float(args.near_threshold)
    release_ready = (
        bool(env.task.target_geom)
        and np.isfinite(metrics["target_distance"])
        and float(metrics["target_distance"]) < float(args.release_distance)
    )

    if phase >= float(args.release_phase) and release_ready:
        action[6] = float(args.open_gripper)
        return action, "open"
    if phase >= float(args.close_phase) or near_object:
        action[6] = float(args.close_gripper)
        return action, "closed"

    action[6] = float(args.open_gripper)
    return action, "open"


def update_gate_stats(stats: dict[str, int], gate_state: str) -> None:
    if gate_state == "open":
        stats["gate_open_steps"] += 1
    elif gate_state == "closed":
        stats["gate_closed_steps"] += 1
    else:
        stats["gate_policy_steps"] += 1
