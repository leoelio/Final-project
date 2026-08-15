from __future__ import annotations

import argparse
from copy import copy
import json
from math import comb
from pathlib import Path
import sys
import time

import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from torch_runtime import ensure_torch_path  # noqa: E402

ensure_torch_path()

import torch  # noqa: E402

from evaluate_rgb_recovery_profiles import DOMAINS, rollout_args  # noqa: E402
from run_clip_action_head import encode_clip, load_clip  # noqa: E402
from run_clip_semantic_rgb_feedback import (  # noqa: E402
    STATIC_TARGETS,
    attempt_config,
    configure_env,
    locate_initial_source,
    render_top_rgb,
    rollout as v4_rollout,
    visual_target_status,
)
from run_clip_semantic_waypoint import INTENTS, load_policy, normalize_instruction, predict_intent  # noqa: E402
from widowx_env.rollout_video import Mp4FrameRecorder  # noqa: E402
from widowx_env.scripted_expert import PickPlaceExpert, new_motion_trace  # noqa: E402
from widowx_env.vision_grounding import load_calibration, relocate_known_object  # noqa: E402


TASKS = {
    "place_blue_cube_blue_pad": "medium",
    "place_blue_cube_red_pad": "medium",
    "place_red_cube_red_pad": "medium",
    "move_leftmost_cube_to_bowl": "language",
}
VARIANTS = ("v4_standard", "fixed_deep_tight_slow", "monitor_early_deep")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired held-out evaluation of a frozen CLIP contact-stage monitor.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=ROOT / "outputs" / "rgb_grounding" / "top_rgb_core_v2_calibration_v1.json")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--domains", default="mild_contact_shift,low_contact_shift,severe_contact_shift")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--version", default="contact_phase_monitor_heldout_v1")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / "contact_phase_monitor_heldout_v1.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "contact_phase_monitor_heldout_v1.md")
    parser.add_argument("--log-every", type=int, default=12)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--speed", type=float, default=0.2)
    parser.add_argument("--video-path", type=Path, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--frame-stride", type=int, default=12)
    return parser.parse_args()


def selected(value: str, known: dict | tuple[str, ...], label: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in known]
    if not names or unknown:
        raise KeyError(f"unknown {label}: {unknown}")
    return names


def exact_two_sided(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if not discordant:
        return 1.0
    tail = sum(comb(discordant, value) for value in range(min(improved, regressed) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


class ContactMonitor:
    def __init__(self, path: Path) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            payload = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=self.device)
        self.metadata = payload["metadata"]
        hidden = int(payload["state_dict"]["0.weight"].shape[0])
        input_size = int(payload["state_dict"]["0.weight"].shape[1])
        self.net = torch.nn.Sequential(torch.nn.Linear(input_size, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)).to(self.device)
        self.net.load_state_dict(payload["state_dict"])
        self.net.eval()
        self.mean = payload["x_mean"].astype(np.float32)
        self.std = payload["x_std"].astype(np.float32)

    def probability(self, feature: np.ndarray) -> float:
        normalized = ((feature.astype(np.float32) - self.mean) / self.std)[None, :]
        if normalized.shape[1] != self.mean.shape[0]:
            raise ValueError(f"monitor feature width {normalized.shape[1]} does not match {self.mean.shape[0]}")
        with torch.no_grad():
            return float(torch.sigmoid(self.net(torch.from_numpy(normalized).to(self.device)))[0, 0].cpu())


def robot_vector(env) -> np.ndarray:
    return np.concatenate(
        [
            env.data.qpos[: env.robot_nq],
            env.data.qvel[: env.robot_nv],
            env.data.ctrl,
            env.data.actuator_force,
            env.tcp_position(),
        ]
    ).astype(np.float32)


def monitor_feature(clip_model, processor, close_image: np.ndarray, lift_image: np.ndarray, instruction: str, close: np.ndarray, lift: np.ndarray) -> np.ndarray:
    close_encoded = encode_clip(clip_model, processor, close_image, instruction)
    lift_encoded = encode_clip(clip_model, processor, lift_image, instruction)
    visual_width = close_encoded.shape[0] // 2
    return np.concatenate([close_encoded[:visual_width], lift_encoded[:visual_width], close_encoded[visual_width:], close, lift, lift - close]).astype(np.float32)


def execute_until_lift(expert: PickPlaceExpert, plan: dict, viewer, speed: float, recorder=None) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = expert.config
    actions = plan["actions"]
    trace = new_motion_trace(expert.env)
    record_step = (lambda _action, _env: recorder.sync()) if recorder is not None else None
    expert._track(actions["approach"], cfg.approach_steps, viewer, record_step, speed, trace)
    expert._track(actions["grasp_open"], cfg.descend_steps, viewer, record_step, speed, trace)
    expert._track(actions["grasp_closed"], cfg.close_steps, viewer, record_step, speed, trace)
    close_image = render_top_rgb(expert.env, 224, "top_rgb")
    close = robot_vector(expert.env)
    expert._track(actions["lift_closed"], cfg.lift_steps, viewer, record_step, speed, trace)
    lift_image = render_top_rgb(expert.env, 224, "top_rgb")
    lift = robot_vector(expert.env)
    return trace, close_image, lift_image, close, lift


def finish_first_attempt(expert: PickPlaceExpert, plan: dict, trace: dict, viewer, speed: float, recorder=None) -> dict:
    cfg = expert.config
    actions = plan["actions"]
    record_step = (lambda _action, _env: recorder.sync()) if recorder is not None else None
    expert._track(actions["transfer_closed"], cfg.transfer_steps, viewer, record_step, speed, trace)
    expert._track(actions["place_closed"], cfg.place_descend_steps, viewer, record_step, speed, trace)
    expert._track(actions["place_open"], cfg.open_steps, viewer, record_step, speed, trace)
    expert._track(actions["retreat_open"], cfg.retreat_steps, viewer, record_step, speed, trace)
    expert._track(actions["retreat_open"], cfg.hold_steps, viewer, record_step, speed, trace)
    object_position = expert.env.object_position(expert.env.episode_target_object)
    target_distance = float(np.linalg.norm(object_position[:2] - plan["target_position"][:2]))
    return {
        "success": bool(target_distance < 0.065 and object_position[2] < 0.08 and trace["strict_grasp_success"]),
        "strict_grasp_success": bool(trace["strict_grasp_success"]),
        "target_distance": target_distance,
    }


def candidate_rollout(args, policy: dict, clip_model, processor, calibration, monitor: ContactMonitor, seed: int, viewer=None, recorder=None, env=None, obs=None) -> dict:
    if env is None or obs is None:
        env, obs = configure_env(args, seed)
    instruction = str(args.instruction or obs["instruction"])
    normalized = normalize_instruction(instruction) if args.instruction_normalization == "desktop_alias_v1" else instruction
    intent, _ = predict_intent(env, {**obs, "instruction": normalized}, policy, clip_model, processor, args.image_size, args.camera)
    source_name, target_geom = INTENTS[intent]
    target_position = STATIC_TARGETS[target_geom]
    initial = render_top_rgb(env, args.image_size, args.camera)
    try:
        selected_name, source_position, _ = locate_initial_source(initial, calibration, intent)
    except LookupError as error:
        return {"semantic_correct": intent == args.task, "visual_selection_correct": False, "first_attempt_success": False, "task_success": False, "monitor_probability": None, "early_retry": False, "recovery_reason": f"initial_rgb_grounding_failed: {error}"}

    initial_error = float(np.linalg.norm(source_position[:2] - env.object_position(obs["target_object"])[:2]))
    expert = PickPlaceExpert(env, attempt_config(args, 0))
    plan = expert.plan_from_positions(source_position, target_position, target_geom=target_geom)
    trace, close_image, lift_image, close, lift = execute_until_lift(expert, plan, viewer, args.speed if viewer is not None else 0.0, recorder)
    probability = monitor.probability(monitor_feature(clip_model, processor, close_image, lift_image, instruction, close, lift))
    early_retry = probability < 0.5
    first_success = False
    final = {"success": False, "strict_grasp_success": bool(trace["strict_grasp_success"]), "target_distance": float("nan")}
    recovery_reason = "monitor_stable_continue"

    if early_retry:
        after_lift = render_top_rgb(env, args.image_size, args.camera)
        try:
            recovered_position, _ = relocate_known_object(after_lift, calibration, selected_name, source_position[:2], search_scope="table")
            recovery_args = copy(args)
            recovery_args.recovery_profile = "deep_tight_slow"
            recovery_expert = PickPlaceExpert(env, attempt_config(recovery_args, 1))
            recovery_plan = recovery_expert.plan_from_positions(recovered_position, target_position, target_geom=target_geom)
            final = recovery_expert.execute(recovery_plan, viewer=viewer, record_step=(lambda _action, _env: recorder.sync()) if recorder is not None else None, speed=args.speed if viewer is not None else 0.0)
            recovery_reason = "monitor_early_table_relocalized_deep_retry"
        except LookupError:
            recovery_reason = "monitor_early_table_object_not_visually_recoverable"
    else:
        final = finish_first_attempt(expert, plan, trace, viewer, args.speed if viewer is not None else 0.0, recorder)
        first_success = bool(final["success"])
        after_first = render_top_rgb(env, args.image_size, args.camera)
        status = visual_target_status(after_first, calibration, selected_name, target_geom)
        if not status["complete"]:
            try:
                recovered_position, _ = relocate_known_object(after_first, calibration, selected_name, source_position[:2], search_scope="table")
                retry_expert = PickPlaceExpert(env, attempt_config(args, 1))
                retry_plan = retry_expert.plan_from_positions(recovered_position, target_position, target_geom=target_geom)
                final = retry_expert.execute(retry_plan, viewer=viewer, record_step=(lambda _action, _env: recorder.sync()) if recorder is not None else None, speed=args.speed if viewer is not None else 0.0)
                recovery_reason = "v4_table_retry_after_monitor_pass"
            except LookupError:
                recovery_reason = "v4_table_object_not_visually_recoverable_after_monitor_pass"

    metrics = env.metrics()
    return {
        "semantic_correct": intent == args.task,
        "visual_selection_correct": bool(initial_error <= 0.04),
        "first_attempt_success": first_success,
        "task_success": bool(final["success"] and intent == args.task and selected_name == obs["target_object"]),
        "strict_grasp_success": bool(final.get("strict_grasp_success", False)),
        "target_distance_m": float(metrics["target_distance"]),
        "monitor_probability": probability,
        "early_retry": early_retry,
        "recovery_reason": recovery_reason,
    }


def fixed_rollout(args, policy, clip_model, processor, calibration, seed: int, variant: str, viewer=None, env=None, obs=None) -> dict:
    args.recovery_profile = "deep_tight_slow" if variant == "fixed_deep_tight_slow" else "standard"
    result = v4_rollout(args, policy, clip_model, processor, calibration, seed, viewer=viewer, env=env, obs=obs)
    first = result["attempt_logs"][0] if result["attempt_logs"] else {}
    return {
        "semantic_correct": bool(result["semantic_correct"]),
        "visual_selection_correct": bool(result["visual_selection_correct"]),
        "first_attempt_success": bool(first.get("evaluation_strict_grasp_success", False) and first.get("evaluation_target_distance_m", 1.0) < 0.065),
        "task_success": bool(result["task_success"]),
        "strict_grasp_success": bool(result["strict_grasp_success"]),
        "target_distance_m": float(result["target_distance"]),
        "monitor_probability": None,
        "early_retry": False,
        "recovery_reason": result["recovery_reason"],
    }


def summarize(rows: list[dict], variants: list[str]) -> dict:
    summary = {}
    for variant in variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        summary[variant] = {
            "episodes": len(variant_rows),
            "semantic_correct": sum(row["semantic_correct"] for row in variant_rows),
            "visual_selection_correct": sum(row["visual_selection_correct"] for row in variant_rows),
            "first_attempt_success": sum(row["first_attempt_success"] for row in variant_rows),
            "task_success": sum(row["task_success"] for row in variant_rows),
            "early_retry": sum(row["early_retry"] for row in variant_rows),
        }
    baseline = {(row["domain"], row["task"], row["seed"]): row for row in rows if row["variant"] == "v4_standard"}
    candidate = {(row["domain"], row["task"], row["seed"]): row for row in rows if row["variant"] == "monitor_early_deep"}
    improved = sum(not baseline[key]["task_success"] and candidate[key]["task_success"] for key in baseline if key in candidate)
    regressed = sum(baseline[key]["task_success"] and not candidate[key]["task_success"] for key in baseline if key in candidate)
    return {"by_variant": summary, "paired_v4_vs_monitor": {"improved": improved, "regressed": regressed, "discordant": improved + regressed, "exact_two_sided_p": exact_two_sided(improved, regressed)}}


def main() -> None:
    args = parse_args()
    domains = selected(args.domains, DOMAINS, "domains")
    tasks = selected(args.tasks, TASKS, "tasks")
    variants = selected(args.variants, VARIANTS, "variants")
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    policy = load_policy(args.model)
    calibration = load_calibration(args.calibration)
    clip_model, processor = load_clip(str(policy["metadata"]["clip_model"]))
    monitor = ContactMonitor(args.monitor)
    rows: list[dict] = []
    scene_index = 0
    for domain_name in domains:
        for task in tasks:
            for offset in range(args.episodes):
                seed = args.seed + offset
                for variant in variants:
                    config = rollout_args(task, TASKS[task], "standard", DOMAINS[domain_name])
                    config.recovery_search = "table"
                    use_viewer = bool(args.viewer and scene_index == 0 and variant == variants[0])
                    env, obs = configure_env(config, seed)
                    if use_viewer:
                        config.speed = args.speed
                        recorder = Mp4FrameRecorder(env, args.video_path, args.width, args.height, args.fps, args.frame_stride, config.camera) if args.video_path else None
                        if recorder is not None:
                            recorder.capture()
                        try:
                            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                                result = candidate_rollout(config, policy, clip_model, processor, calibration, monitor, seed, viewer, recorder, env, obs) if variant == "monitor_early_deep" else fixed_rollout(config, policy, clip_model, processor, calibration, seed, variant, viewer, env, obs)
                                started = time.time()
                                while viewer.is_running() and time.time() - started < args.duration:
                                    viewer.sync()
                                    time.sleep(0.01)
                        finally:
                            if recorder is not None:
                                recorder.close()
                    else:
                        recorder = Mp4FrameRecorder(env, args.video_path, args.width, args.height, args.fps, args.frame_stride, config.camera) if args.video_path and scene_index == 0 and variant == variants[0] else None
                        if recorder is not None:
                            recorder.capture()
                        try:
                            result = candidate_rollout(config, policy, clip_model, processor, calibration, monitor, seed, recorder=recorder, env=env, obs=obs) if variant == "monitor_early_deep" else fixed_rollout(config, policy, clip_model, processor, calibration, seed, variant, env=env, obs=obs)
                        finally:
                            if recorder is not None:
                                recorder.close()
                    rows.append({"variant": variant, "domain": domain_name, "task": task, "seed": seed, **result})
                scene_index += 1
                if args.log_every and scene_index % args.log_every == 0:
                    print(json.dumps(rows[-len(variants):], ensure_ascii=False), flush=True)
    summary = summarize(rows, variants)
    output = {
        "version": args.version,
        "method": "Frozen CLIP task semantics and contact monitor; RGB-only grounding and table-bounded re-localization; structured trajectory execution.",
        "monitor": str(args.monitor),
        "seed_range": f"{args.seed}-{args.seed + args.episodes - 1}",
        "domains": {name: DOMAINS[name] for name in domains},
        "tasks": tasks,
        "variants": variants,
        **summary,
        "rows": rows,
        "runtime_boundary": "Runtime inputs are frozen CLIP features, RGB, fixed scene configuration, robot-only proprioception and actions. MuJoCo object truth is offline scoring only.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 接触阶段监测器：独立闭环评测 V1",
        "",
        f"全新 seed `{output['seed_range']}`；{len(domains)} 个接触域、{len(tasks)} 项任务、每个场景三种严格配对变体。",
        "",
        "| 变体 | 场景 | 语义正确 | 对象选择正确 | 首轮成功 | 最终成功 | 提前重抓 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(f"| {name} | {item['episodes']} | {item['semantic_correct']} | {item['visual_selection_correct']} | {item['first_attempt_success']} | {item['task_success']} | {item['early_retry']} |" for name, item in summary["by_variant"].items())
    paired = summary["paired_v4_vs_monitor"]
    lines.extend([
        "",
        f"V4 标准对照与监测候选的配对结果：改进 {paired['improved']}，回退 {paired['regressed']}，精确双侧 p={paired['exact_two_sided_p']:.4f}。",
        "",
        "该结果只在满足预注册门槛时才允许提升候选状态；否则 V4 保持为当前最优可复现方案。",
    ])
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
