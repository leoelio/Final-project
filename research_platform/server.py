from __future__ import annotations

import argparse
import ast
import csv
import ctypes
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import mimetypes
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse
import uuid
import webbrowser
import zlib

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
CALIBRATION_PATH = ROOT / "runtime_assets" / "top_rgb_core_v2_calibration_v1.json"
FINAL_MODEL_PATH = ROOT / "runtime_assets" / "clip_semantic_waypoint_core_v2_v1_20260721_110325.npz"
PLATFORM_OUTPUT = ROOT / "outputs" / "platform_research"
LEDGER_PATH = PLATFORM_OUTPUT / "experiment_ledger.jsonl"
STUDY_REGISTRY_PATH = PLATFORM_OUTPUT / "study_registry.jsonl"
RUN_ARTIFACT_ROOT = PLATFORM_OUTPUT / "runs"
RELEASE_REGISTRY_PATH = PLATFORM_OUTPUT / "release_registry.jsonl"
RELEASE_ROOT = PLATFORM_OUTPUT / "releases"
FINAL_AUDIT_PATH = ROOT / "runtime_assets" / "final_closure_audit_v1.json"
EXPERIMENT_VERSIONS_PATH = ROOT / "docs" / "experiment_versions.json"
MODEL_RESOURCE_PATH = ROOT / "docs" / "model_resource_summary.csv"
VIDEO_AUDIT_PATH = ROOT / "docs" / "video_quality_audit.csv"
ADAPTATION_ROOT = PLATFORM_OUTPUT / "adaptation"
ADAPTATION_TASK_REGISTRY = PLATFORM_OUTPUT / "adaptation_tasks.json"
ADAPTATION_MODEL_ROOT = ROOT / "outputs" / "platform_training" / "adaptation"
OBJECT_ACTION_HEAD_PATH = ROOT / "outputs" / "object_action_head" / "object_action_head_lite_20260720_044703.npz"

sys.path.insert(0, str(ROOT))

from widowx_env import WidowXTabletopEnv  # noqa: E402
from widowx_env.scripted_expert import PickPlaceConfig, PickPlaceExpert  # noqa: E402
from widowx_env.tabletop_env import (  # noqa: E402
    CUSTOM_TASK_SOURCES,
    CUSTOM_TASK_TARGETS,
    TASKS as ENV_TASKS,
    register_custom_task,
)
from widowx_env.vision_grounding import load_calibration, locate_leftmost_cube, locate_object  # noqa: E402


TASKS = {
    "place_blue_cube_blue_pad": {
        "instruction": "place the blue cube on the blue target",
        "instruction_zh": "把蓝色方块放到蓝色目标区",
        "source": "blue_cube",
        "target": "target_blue_pad",
        "complexity": "medium",
    },
    "place_blue_cube_red_pad": {
        "instruction": "place the blue cube on the red target",
        "instruction_zh": "把蓝色方块放到红色目标区",
        "source": "blue_cube",
        "target": "target_red_pad",
        "complexity": "medium",
    },
    "place_red_cube_red_pad": {
        "instruction": "place the red cube on the red target",
        "instruction_zh": "把红色方块放到红色目标区",
        "source": "red_cube",
        "target": "target_red_pad",
        "complexity": "medium",
    },
    "move_leftmost_cube_to_bowl": {
        "instruction": "move the leftmost cube to the white bowl",
        "instruction_zh": "把最左边的方块放到白色碗里",
        "source": "leftmost_cube",
        "target": "target_bowl",
        "complexity": "language",
    },
}

DATASETS = {
    "blue_blue_100": ROOT / "data" / "demos" / "place_blue_cube_blue_pad_medium_20260702_051752",
    "blue_red_100": ROOT / "data" / "demos" / "place_blue_cube_red_pad_medium_20260702_050525",
    "red_red_80": ROOT / "data" / "demos" / "kaggle_scale_place_red_cube_red_pad_medium_80_v1",
    "leftmost_50": ROOT / "data" / "demos" / "move_leftmost_to_bowl_language_multitask_v2",
}

TRAINERS = {
    "mlp_bc": {
        "script": ROOT / "scripts" / "train_mlp_bc.py",
        "args": ["--hidden-sizes", "128,128", "--batch-size", "1024", "--lr", "0.001"],
        "output": ROOT / "outputs" / "platform_training" / "mlp_bc",
    },
    "action_chunk": {
        "script": ROOT / "scripts" / "train_chunk_bc.py",
        "args": [
            "--horizon", "8", "--history", "4", "--sample-stride", "8",
            "--hidden-sizes", "128,128", "--batch-size", "512", "--lr", "0.001",
            "--gripper-loss-weight", "4", "--no-augment-relative",
        ],
        "output": ROOT / "outputs" / "platform_training" / "action_chunk",
    },
    "diffusion_lite": {
        "script": ROOT / "scripts" / "train_diffusion_policy.py",
        "args": [
            "--horizon", "8", "--sample-stride", "8", "--diffusion-steps", "8",
            "--hidden-sizes", "128,128", "--batch-size", "512", "--lr", "0.001",
        ],
        "output": ROOT / "outputs" / "platform_training" / "diffusion_lite",
    },
}

STATIC_TARGETS = {
    "target_blue_pad": np.array([0.50, 0.16, 0.002], dtype=float),
    "target_red_pad": np.array([0.50, -0.16, 0.002], dtype=float),
    "target_bowl": np.array([0.33, 0.25, 0.006], dtype=float),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def system_memory_bytes() -> tuple[int, int]:
    if sys.platform == "win32":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return int(status.total_physical), int(status.available_physical)
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    total = page_size * int(os.sysconf("SC_PHYS_PAGES"))
    available = page_size * int(os.sysconf("SC_AVPHYS_PAGES"))
    return total, available


def process_rss_bytes(pid: int) -> int:
    if sys.platform == "win32":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("page_fault_count", ctypes.c_ulong),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
            ]

        handle = ctypes.windll.kernel32.OpenProcess(0x1000 | 0x0010, False, pid)
        if not handle:
            return 0
        try:
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if not ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return 0
            return int(counters.working_set_size)
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    statm = Path(f"/proc/{pid}/statm")
    if not statm.is_file():
        return 0
    return int(statm.read_text(encoding="ascii").split()[1]) * int(os.sysconf("SC_PAGE_SIZE"))


def process_tree_rss_bytes(root_pid: int) -> int:
    pids = {root_pid}
    if sys.platform == "win32":
        class ProcessEntry(ctypes.Structure):
            _fields_ = [
                ("dwSize", ctypes.c_ulong),
                ("cntUsage", ctypes.c_ulong),
                ("th32ProcessID", ctypes.c_ulong),
                ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", ctypes.c_ulong),
                ("cntThreads", ctypes.c_ulong),
                ("th32ParentProcessID", ctypes.c_ulong),
                ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", ctypes.c_ulong),
                ("szExeFile", ctypes.c_wchar * 260),
            ]

        kernel32 = ctypes.windll.kernel32
        kernel32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_ulong, ctypes.c_ulong]
        kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
        snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
        if snapshot == ctypes.c_void_p(-1).value:
            return process_rss_bytes(root_pid)
        parents: dict[int, int] = {}
        try:
            entry = ProcessEntry()
            entry.dwSize = ctypes.sizeof(entry)
            has_entry = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
            while has_entry:
                parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
                has_entry = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
        finally:
            kernel32.CloseHandle(snapshot)
    else:
        parents = {}
        for stat_path in Path("/proc").glob("[0-9]*/stat"):
            try:
                stat = stat_path.read_text(encoding="ascii").split()
                parents[int(stat[0])] = int(stat[3])
            except (OSError, ValueError, IndexError):
                continue

    changed = True
    while changed:
        before = len(pids)
        pids.update(pid for pid, parent in parents.items() if parent in pids)
        changed = len(pids) != before
    return sum(process_rss_bytes(pid) for pid in pids)


def artifact_url(path: Path) -> str:
    return "/platform_artifacts/" + path.resolve().relative_to(PLATFORM_OUTPUT.resolve()).as_posix()


def wilson_interval(successes: int, episodes: int, z: float = 1.96) -> tuple[float, float]:
    if episodes <= 0:
        return 0.0, 0.0
    rate = successes / episodes
    denominator = 1.0 + z * z / episodes
    centre = (rate + z * z / (2.0 * episodes)) / denominator
    margin = z * np.sqrt((rate * (1.0 - rate) + z * z / (4.0 * episodes)) / episodes) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


class ExperimentLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.records: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            run_id = event.get("run_id")
            if not run_id:
                continue
            if event.get("event") == "start":
                self.records[run_id] = {key: value for key, value in event.items() if key != "event"}
            elif event.get("event") == "finish" and run_id in self.records:
                self.records[run_id].update({key: value for key, value in event.items() if key not in {"event", "run_id"}})
        stale = [run_id for run_id, record in self.records.items() if record.get("status") == "running" and record.get("finished_at") is None]
        for run_id in stale:
            self.finish(run_id, "interrupted", metrics={"reason": "platform restarted before completion"})

    def _append(self, event: dict) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

    def start(self, kind: str, config: dict, command: str = "", parent_id: str | None = None) -> str:
        run_id = f"{kind[:3]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        record = {
            "run_id": run_id,
            "kind": kind,
            "status": "running",
            "started_at": utc_now(),
            "finished_at": None,
            "config": config,
            "metrics": {},
            "artifact": None,
            "assets": {},
            "command": command,
            "parent_id": parent_id,
        }
        with self.lock:
            self.records[run_id] = record
            self._append({"event": "start", **record})
        return run_id

    def finish(
        self,
        run_id: str | None,
        status: str,
        metrics: dict | None = None,
        artifact: str | None = None,
        assets: dict | None = None,
    ) -> None:
        if not run_id:
            return
        with self.lock:
            record = self.records.get(run_id)
            if record is None or record.get("finished_at") is not None:
                return
            update = {
                "status": status,
                "finished_at": utc_now(),
                "metrics": metrics or {},
                "artifact": artifact,
                "assets": assets or {},
            }
            record.update(update)
            self._append({"event": "finish", "run_id": run_id, **update})

    def get(self, run_id: str) -> dict | None:
        with self.lock:
            record = self.records.get(run_id)
            return None if record is None else json.loads(json.dumps(record))

    def children(self, run_id: str) -> list[dict]:
        with self.lock:
            rows = [row for row in self.records.values() if row.get("parent_id") == run_id]
        rows.sort(key=lambda row: row.get("started_at") or "")
        return json.loads(json.dumps(rows))

    def list(self, limit: int = 200, kind: str | None = None, status: str | None = None) -> list[dict]:
        with self.lock:
            rows = list(self.records.values())
        if kind:
            rows = [row for row in rows if row.get("kind") == kind]
        if status:
            rows = [row for row in rows if row.get("status") == status]
        rows.sort(key=lambda row: row.get("started_at") or "", reverse=True)
        return rows[:max(1, min(limit, 1000))]

    def analytics(self) -> dict:
        rows = self.list(limit=1000)
        simulations = [row for row in rows if row.get("kind") == "simulation" and row.get("status") == "completed"]
        trainings = [row for row in rows if row.get("kind") == "training" and row.get("status") == "completed"]
        optimizations = [row for row in rows if row.get("kind") == "optimization"]
        adaptation_candidates = [row for row in rows if row.get("kind") == "adaptation_candidate"]
        policy_groups: dict[str, dict] = {}
        for row in simulations:
            policy = row.get("config", {}).get("policy", "unknown")
            group = policy_groups.setdefault(policy, {"episodes": 0, "successes": 0, "errors": []})
            group["episodes"] += 1
            if row.get("metrics", {}).get("success") is True:
                group["successes"] += 1
            error = row.get("metrics", {}).get("target_distance")
            if isinstance(error, (int, float)):
                group["errors"].append(error)
        policies = []
        for policy, group in policy_groups.items():
            policies.append({
                "policy": policy,
                "episodes": group["episodes"],
                "successes": group["successes"],
                "success_rate": group["successes"] / group["episodes"] if group["episodes"] else 0.0,
                "mean_target_error": sum(group["errors"]) / len(group["errors"]) if group["errors"] else None,
            })
        latest_training: dict[str, dict] = {}
        for row in trainings:
            method = row.get("config", {}).get("method", "unknown")
            latest_training.setdefault(method, row)
        return {
            "total_runs": len(rows),
            "simulations": len([row for row in rows if row.get("kind") == "simulation"]),
            "trainings": len([row for row in rows if row.get("kind") == "training"]),
            "adaptations": len([row for row in rows if row.get("kind") == "adaptation"]),
            "optimizations": len(optimizations),
            "completed_optimizations": len([row for row in optimizations if row.get("status") == "completed"]),
            "adaptation_candidates": len(adaptation_candidates),
            "benchmarks": len([row for row in rows if row.get("kind") == "benchmark"]),
            "policies": policies,
            "latest_training": [
                {
                    "method": method,
                    "train_loss": row.get("metrics", {}).get("train_loss"),
                    "val_loss": row.get("metrics", {}).get("val_loss"),
                    "artifact": row.get("artifact"),
                }
                for method, row in latest_training.items()
            ],
        }

    def csv_bytes(self) -> bytes:
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(["run_id", "kind", "status", "started_at", "finished_at", "task_or_method", "success", "target_error_m", "artifact", "parent_id"])
        for row in self.list(limit=1000):
            config = row.get("config", {})
            metrics = row.get("metrics", {})
            writer.writerow([
                row.get("run_id"), row.get("kind"), row.get("status"), row.get("started_at"), row.get("finished_at"),
                config.get("task") or config.get("method") or "batch", metrics.get("success"), metrics.get("target_distance"),
                row.get("artifact"), row.get("parent_id"),
            ])
        return output.getvalue().encode("utf-8-sig")


class SimulationCancelled(RuntimeError):
    pass


def png_bytes(image: np.ndarray) -> bytes:
    image = np.ascontiguousarray(image, dtype=np.uint8)
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError("RGB frame expected")
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(name: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + name + payload + struct.pack(">I", zlib.crc32(name + payload) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b"")


def parse_instruction(text: str) -> tuple[str, str] | None:
    value = text.strip().lower()
    if not value:
        return None
    controls = {
        "pause": ("pause", "pause"), "暂停": ("pause", "pause"),
        "resume": ("resume", "resume"), "continue": ("resume", "resume"), "继续": ("resume", "resume"),
        "stop": ("stop", "stop"), "停止": ("stop", "stop"),
        "reset": ("reset", "reset"), "重置": ("reset", "reset"),
    }
    if value in controls:
        return controls[value]
    if ("leftmost" in value or "left most" in value or "最左" in value) and ("bowl" in value or "碗" in value):
        return "task", "move_leftmost_cube_to_bowl"
    blue = "blue" in value or "蓝" in value
    red = "red" in value or "红" in value
    if blue and ("blue target" in value or "blue pad" in value or "蓝色目标" in value or "蓝框" in value or "蓝盘" in value):
        return "task", "place_blue_cube_blue_pad"
    if blue and ("red target" in value or "red pad" in value or "红色目标" in value or "红框" in value or "红盘" in value):
        return "task", "place_blue_cube_red_pad"
    if red and ("red target" in value or "red pad" in value or "红色目标" in value or "红框" in value or "红盘" in value):
        return "task", "place_red_cube_red_pad"
    return None


@dataclass
class SimulationState:
    run_id: str | None = None
    status: str = "idle"
    task: str = "place_blue_cube_blue_pad"
    instruction: str = TASKS["place_blue_cube_blue_pad"]["instruction"]
    instruction_zh: str = TASKS["place_blue_cube_blue_pad"]["instruction_zh"]
    policy: str = "rgb_grounded"
    complexity: str = "medium"
    seed: int = 0
    speed: float = 1.0
    phase: str = "ready"
    progress: float = 0.0
    step: int = 0
    total_steps: int = 2840
    frame_seq: int = 0
    started_at: float | None = None
    elapsed: float = 0.0
    fps: float = 0.0
    source_name: str | None = None
    source_position: list[float] | None = None
    position_source: str = "not started"
    rgb_grounding_error: float | None = None
    initial_top_url: str | None = None
    final_front_url: str | None = None
    success: bool | None = None
    target_distance: float | None = None
    object_z: float | None = None
    contact_count: float = 0.0
    error: str | None = None


class SimulationManager:
    def __init__(self, ledger: ExperimentLedger) -> None:
        self.ledger = ledger
        self.lock = threading.RLock()
        self.state = SimulationState()
        self.frame = b""
        self.thread: threading.Thread | None = None
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.stop_event = threading.Event()
        self.last_request: dict = {
            "task": self.state.task,
            "policy": self.state.policy,
            "complexity": self.state.complexity,
            "seed": self.state.seed,
            "speed": self.state.speed,
        }
        self.native_processes: list[subprocess.Popen] = []
        self.active_run_id: str | None = None
        self._prime_frame()

    def _prime_frame(self) -> None:
        try:
            env = WidowXTabletopEnv(seed=0, image_size=(480, 640), camera="front_rgb", workspace_profile="core_v2")
            env.reset(task=self.state.task, complexity="medium", seed=0)
            renderer = mujoco.Renderer(env.model, height=480, width=640)
            try:
                renderer.update_scene(env.data, camera="front_rgb")
                self.frame = png_bytes(renderer.render())
                self.state.frame_seq = 1
            finally:
                renderer.close()
        except Exception as error:  # pragma: no cover - hardware-specific startup failure
            self.state.error = f"initial render failed: {error}"

    def snapshot(self) -> dict:
        with self.lock:
            state = asdict(self.state)
            state["command_examples"] = [
                "Place the blue cube on the blue target",
                "把蓝色方块放到红色目标区",
                "Move the leftmost cube to the bowl",
                "暂停 / resume / stop / reset",
            ]
            return state

    def frame_snapshot(self) -> tuple[bytes, int]:
        with self.lock:
            return self.frame, self.state.frame_seq

    def start(self, request: dict, parent_id: str | None = None) -> dict:
        task = str(request.get("task", "place_blue_cube_blue_pad"))
        policy = str(request.get("policy", "rgb_grounded"))
        complexity = str(request.get("complexity", TASKS.get(task, {}).get("complexity", "medium")))
        seed = int(request.get("seed", 0))
        speed = float(request.get("speed", 1.0))
        if task not in TASKS:
            raise ValueError("unsupported task")
        if policy not in {"rgb_grounded", "structured_state"}:
            raise ValueError("unsupported policy")
        if complexity not in {"medium", "hard", "language"}:
            raise ValueError("unsupported complexity")
        if not 0.25 <= speed <= 3.0:
            raise ValueError("speed must be between 0.25 and 3.0")
        self.stop(wait=True)
        self.stop_event.clear()
        self.pause_event.set()
        self.last_request = {"task": task, "policy": policy, "complexity": complexity, "seed": seed, "speed": speed}
        spec = TASKS[task]
        command = self.reproduction_command(self.last_request)
        run_id = self.ledger.start("simulation", self.last_request.copy(), command=command, parent_id=parent_id)
        with self.lock:
            self.state = SimulationState(
                run_id=run_id, status="starting", task=task, instruction=spec["instruction"], instruction_zh=spec["instruction_zh"],
                policy=policy, complexity=complexity, seed=seed, speed=speed, phase="initialising",
            )
            self.active_run_id = run_id
        self.thread = threading.Thread(target=self._run, args=(self.last_request.copy(),), daemon=True, name="mujoco-live-session")
        self.thread.start()
        return self.snapshot()

    def pause(self) -> dict:
        with self.lock:
            if self.state.status == "running":
                self.pause_event.clear()
                self.state.status = "paused"
        return self.snapshot()

    def resume(self) -> dict:
        with self.lock:
            if self.state.status == "paused":
                self.pause_event.set()
                self.state.status = "running"
        return self.snapshot()

    def stop(self, wait: bool = False) -> dict:
        thread = self.thread
        if thread and thread.is_alive():
            self.stop_event.set()
            self.pause_event.set()
            if wait and threading.current_thread() is not thread:
                thread.join(timeout=5.0)
        with self.lock:
            if self.state.status in {"running", "paused", "starting"}:
                self.state.status = "stopped"
                self.state.phase = "stopped"
        return self.snapshot()

    def reset(self) -> dict:
        return self.start(self.last_request.copy())

    def open_native_viewer(self, request: dict) -> dict:
        task = str(request.get("task", self.state.task))
        policy = str(request.get("policy", self.state.policy))
        seed = int(request.get("seed", self.state.seed))
        complexity = str(request.get("complexity", TASKS.get(task, {}).get("complexity", "medium")))
        if task not in TASKS:
            raise ValueError("unsupported task")
        if policy == "rgb_grounded":
            command = [
                sys.executable, str(ROOT / "scripts" / "run_clip_semantic_rgb_feedback.py"),
                "--model", str(FINAL_MODEL_PATH), "--calibration", str(CALIBRATION_PATH),
                "--task", task, "--complexity", complexity, "--workspace-profile", "core_v2",
                "--seed", str(seed), "--feedback-attempts", "1", "--recovery-search", "table",
                "--viewer", "--duration", "45", "--speed", "0.5",
                "--arm-kp", "105", "--arm-force", "70", "--gripper-kp", "550",
                "--gripper-force", "75", "--friction", "0.8",
            ]
        else:
            command = [
                sys.executable, str(ROOT / "scripts" / "run_expert.py"),
                "--task", task, "--complexity", complexity, "--workspace-profile", "core_v2",
                "--seed", str(seed), "--episodes", "1", "--viewer", "--duration", "45",
                "--speed", "0.5", "--retries", "1",
            ]
        flags = subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
        process = subprocess.Popen(command, cwd=ROOT, creationflags=flags)
        self.native_processes = [item for item in self.native_processes if item.poll() is None]
        self.native_processes.append(process)
        return {"started": True, "pid": process.pid, "command": subprocess.list2cmdline(command)}

    def reproduction_command(self, request: dict) -> str:
        task = request["task"]
        complexity = request["complexity"]
        seed = request["seed"]
        if request["policy"] == "rgb_grounded":
            command = [
                str(ROOT / ".venv" / "Scripts" / "python.exe"), str(ROOT / "scripts" / "run_clip_semantic_rgb_feedback.py"),
                "--model", str(FINAL_MODEL_PATH), "--calibration", str(CALIBRATION_PATH),
                "--task", task, "--complexity", complexity, "--workspace-profile", "core_v2",
                "--seed", str(seed), "--feedback-attempts", "1", "--recovery-search", "table",
                "--viewer", "--duration", "45", "--speed", "0.5",
                "--arm-kp", "105", "--arm-force", "70", "--gripper-kp", "550",
                "--gripper-force", "75", "--friction", "0.8",
            ]
        else:
            command = [
                str(ROOT / ".venv" / "Scripts" / "python.exe"), str(ROOT / "scripts" / "run_expert.py"),
                "--task", task, "--complexity", complexity, "--workspace-profile", "core_v2",
                "--seed", str(seed), "--episodes", "1", "--viewer", "--duration", "45",
                "--speed", "0.5", "--retries", "1",
            ]
        return subprocess.list2cmdline(command)

    def _phase_for_step(self, step: int) -> str:
        phases = [
            (260, "approach"), (480, "descend"), (740, "grasp"), (1160, "lift"),
            (1860, "transfer"), (2180, "place"), (2400, "release"), (2680, "retreat"),
            (2840, "hold"),
        ]
        for boundary, name in phases:
            if step <= boundary:
                return name
        return "complete"

    def _run(self, request: dict) -> None:
        started = time.monotonic()
        renderer: mujoco.Renderer | None = None
        run_id = self.active_run_id
        run_dir = RUN_ARTIFACT_ROOT / str(run_id)
        initial_top_path = run_dir / "initial_top.png"
        final_front_path = run_dir / "final_front.png"
        summary_path = run_dir / "run.json"
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            task = request["task"]
            spec = TASKS[task]
            env = WidowXTabletopEnv(seed=request["seed"], image_size=(224, 224), camera="top_rgb", workspace_profile="core_v2")
            env.set_arm_actuator_strength(kp=150.0, force_limit=100.0)
            env.set_gripper_actuator_strength(kp=1200.0, force_limit=200.0)
            env.set_grasp_contact_friction(sliding=5.0)
            obs = env.reset(task=task, complexity=request["complexity"], seed=request["seed"])
            renderer = mujoco.Renderer(env.model, height=480, width=640)
            vision_renderer = mujoco.Renderer(env.model, height=224, width=224)
            try:
                vision_renderer.update_scene(env.data, camera="top_rgb")
                vision_image = vision_renderer.render().copy()
            finally:
                vision_renderer.close()
            initial_top_path.write_bytes(png_bytes(vision_image))

            if request["policy"] == "rgb_grounded":
                calibration = load_calibration(CALIBRATION_PATH)
                if spec["source"] == "leftmost_cube":
                    source_name, source_position, _ = locate_leftmost_cube(vision_image, calibration)
                else:
                    source_name = spec["source"]
                    source_position, _ = locate_object(vision_image, calibration, source_name)
                position_source = "top-view RGB + calibrated table plane"
                ground_truth_position = env.object_position(source_name)
                rgb_grounding_error = float(np.linalg.norm(np.asarray(source_position[:2]) - ground_truth_position[:2]))
            else:
                source_name = obs["target_object"]
                source_position = env.object_position(source_name).copy()
                position_source = "MuJoCo state reference"
                rgb_grounding_error = None

            target_position = STATIC_TARGETS[spec["target"]]
            expert = PickPlaceExpert(env, PickPlaceConfig(place_tcp_z=0.041))
            plan = expert.plan_from_positions(source_position, target_position, target_geom=spec["target"])
            frame_counter = 0
            render_times: list[float] = []

            def publish(force: bool = False) -> None:
                nonlocal frame_counter
                frame_counter += 1
                if not force and frame_counter % 8:
                    return
                renderer.update_scene(env.data, camera="front_rgb")
                frame = png_bytes(renderer.render())
                now = time.monotonic()
                render_times.append(now)
                if len(render_times) > 16:
                    render_times.pop(0)
                fps = 0.0 if len(render_times) < 2 else (len(render_times) - 1) / max(1e-6, render_times[-1] - render_times[0])
                metrics = env.metrics()
                with self.lock:
                    self.frame = frame
                    self.state.frame_seq += 1
                    self.state.elapsed = now - started
                    self.state.fps = fps
                    self.state.target_distance = None if not np.isfinite(metrics["target_distance"]) else float(metrics["target_distance"])
                    self.state.object_z = float(metrics["object_z"])
                    self.state.contact_count = float(metrics["contact_count"])

            with self.lock:
                self.state.status = "running"
                self.state.phase = "approach"
                self.state.started_at = time.time()
                self.state.source_name = source_name
                self.state.source_position = np.asarray(source_position).round(5).tolist()
                self.state.position_source = position_source
                self.state.rgb_grounding_error = rgb_grounding_error
                self.state.initial_top_url = artifact_url(initial_top_path)
            publish(force=True)

            def record_step(_action: np.ndarray, _env: WidowXTabletopEnv) -> None:
                if self.stop_event.is_set():
                    raise SimulationCancelled("session stopped")
                while not self.pause_event.wait(timeout=0.1):
                    if self.stop_event.is_set():
                        raise SimulationCancelled("session stopped")
                with self.lock:
                    self.state.step += 1
                    self.state.phase = self._phase_for_step(self.state.step)
                    self.state.progress = min(1.0, self.state.step / self.state.total_steps)
                publish()
                time.sleep(float(env.model.opt.timestep) / request["speed"])

            summary = expert.execute(plan, record_step=record_step, speed=0.0)
            publish(force=True)
            frame, _ = self.frame_snapshot()
            final_front_path.write_bytes(frame)
            with self.lock:
                self.state.status = "completed"
                self.state.phase = "complete"
                self.state.progress = 1.0
                self.state.success = bool(summary["success"])
                self.state.target_distance = float(summary["target_distance"])
                self.state.elapsed = time.monotonic() - started
                self.state.final_front_url = artifact_url(final_front_path)
        except SimulationCancelled:
            with self.lock:
                self.state.status = "stopped"
                self.state.phase = "stopped"
        except Exception as error:
            with self.lock:
                self.state.status = "failed"
                self.state.phase = "error"
                self.state.error = str(error)
                self.state.elapsed = time.monotonic() - started
        finally:
            if renderer is not None:
                renderer.close()
            snapshot = self.snapshot()
            assets = {}
            if initial_top_path.exists():
                assets["initial_top"] = artifact_url(initial_top_path)
            if final_front_path.exists():
                assets["final_front"] = artifact_url(final_front_path)
            summary_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "kind": "simulation",
                        "config": request,
                        "state": snapshot,
                        "assets": assets,
                        "runtime_boundary": "MuJoCo state is used only for offline diagnostics and scoring in rgb_grounded runs.",
                    },
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            self.ledger.finish(
                run_id,
                snapshot["status"],
                metrics={
                    "success": snapshot["success"],
                    "target_distance": snapshot["target_distance"],
                    "elapsed": snapshot["elapsed"],
                    "object_z": snapshot["object_z"],
                    "contact_count": snapshot["contact_count"],
                    "source_name": snapshot["source_name"],
                    "position_source": snapshot["position_source"],
                    "rgb_grounding_error": snapshot["rgb_grounding_error"],
                },
                artifact=str(summary_path.relative_to(ROOT)),
                assets=assets,
            )


@dataclass
class TrainingState:
    status: str = "idle"
    method: str = "mlp_bc"
    dataset: str = "blue_blue_100"
    epochs: int = 12
    current_epoch: int = 0
    progress: float = 0.0
    started_at: float | None = None
    elapsed: float = 0.0
    train_loss: float | None = None
    val_loss: float | None = None
    model_path: str | None = None
    error: str | None = None


class _TrainingManagerCore:
    metric_pattern = re.compile(
        r"epoch=(?P<epoch>\d+).*?train_[a-z_]+=(?P<train>[0-9.eE+-]+).*?val_[a-z_]+=(?P<val>[0-9.eE+-]+)",
        re.IGNORECASE,
    )

    def __init__(self, ledger: ExperimentLedger) -> None:
        self.ledger = ledger
        self.lock = threading.RLock()
        self.state = TrainingState()
        self.metrics: list[dict] = []
        self.logs: list[str] = []
        self.process: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None
        self.active_run_id: str | None = None

    def snapshot(self) -> dict:
        with self.lock:
            result = asdict(self.state)
            result["metrics"] = self.metrics[-200:]
            result["logs"] = self.logs[-120:]
            return result

    def start(self, request: dict) -> dict:
        method = str(request.get("method", "mlp_bc"))
        dataset = str(request.get("dataset", "blue_blue_100"))
        epochs = int(request.get("epochs", 12))
        if method not in TRAINERS:
            raise ValueError("unsupported training method")
        if dataset not in DATASETS or not DATASETS[dataset].exists():
            raise ValueError("unsupported or missing dataset")
        if not 1 <= epochs <= 50:
            raise ValueError("epochs must be between 1 and 50")
        if self.process and self.process.poll() is None:
            raise RuntimeError("a training job is already running")
        trainer = TRAINERS[method]
        Path(trainer["output"]).mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, "-u", str(trainer["script"]), "--run-dir", str(DATASETS[dataset]),
            "--output", str(trainer["output"]), "--epochs", str(epochs), *trainer["args"],
        ]
        with self.lock:
            self.state = TrainingState(status="starting", method=method, dataset=dataset, epochs=epochs, started_at=time.time())
            self.metrics = []
            self.logs = [subprocess.list2cmdline(command)]
            self.active_run_id = self.ledger.start(
                "training",
                {"method": method, "dataset": dataset, "epochs": epochs},
                command=subprocess.list2cmdline(command),
            )
        self.thread = threading.Thread(target=self._run, args=(command,), daemon=True, name="training-job")
        self.thread.start()
        return self.snapshot()

class PairedArenaOperations:
    """Shared-data candidate optimization mixed into the adaptation manager."""

    def start_arena(self, request: dict) -> dict:
        estimate = self.estimate_arena(request)
        if not estimate["gate"]["passed"]:
            raise ValueError("resource gate failed: " + "; ".join(estimate["gate"]["reasons"]))
        if self.thread and self.thread.is_alive():
            raise RuntimeError("an adaptation job is already running")
        if self.training.snapshot()["status"] in {"starting", "running"}:
            raise RuntimeError("the baseline trainer is already using local resources")

        task_id = estimate["task_id"]
        seed = int(request.get("seed", 0))
        viewer = bool(request.get("viewer", True))
        run_tag = f"arena_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        cached_path = Path(estimate["cached_dataset_path"]) if estimate["cached_dataset_path"] else None
        dataset_path = cached_path or (ROOT / "data" / "demos" / run_tag)
        collect_command = None if cached_path else self._collect_command(task_id, seed, estimate["episodes"], run_tag, viewer)
        config = {
            "framework": self.ARENA_FRAMEWORK,
            "task_id": task_id,
            "methods": estimate["methods"],
            "profile": estimate["resolved_profile"],
            "episodes": estimate["episodes"],
            "epochs": estimate["epochs"],
            "evaluation_episodes": estimate["evaluation_episodes"],
            "evaluation_seeds": estimate["evaluation_seeds"],
            "seed": seed,
            "viewer": viewer,
            "reuse_dataset": estimate["reuse_dataset"],
            "dataset_fingerprint": estimate["dataset_fingerprint"],
            "resource_estimate": estimate,
        }
        command_text = (
            f"CACHE {dataset_path}" if collect_command is None else subprocess.list2cmdline(collect_command)
        ) + "\nPAIR-OPT candidates: " + ", ".join(estimate["methods"])
        run_id = self.ledger.start("optimization", config, command=command_text)
        self.stop_event.clear()
        with self.lock:
            self.state = AdaptationState(
                run_id=run_id,
                status="starting",
                stage="validate",
                task_id=task_id,
                method=estimate["methods"][0],
                requested_profile=estimate["requested_profile"],
                resolved_profile=estimate["resolved_profile"],
                episodes=estimate["episodes"],
                epochs=estimate["epochs"],
                evaluation_episodes=estimate["evaluation_episodes"],
                seed=seed,
                viewer=viewer,
                mode="arena",
                arena_framework=self.ARENA_FRAMEWORK,
                candidate_methods=estimate["methods"],
                started_at=time.time(),
                dataset_path=str(dataset_path),
                dataset_fingerprint=estimate["dataset_fingerprint"],
                dataset_cache_hit=estimate["dataset_cache_hit"],
                collection_runs_saved=estimate["collection_runs_saved"],
                estimated=estimate,
            )
            self.events = []
            self.logs = [command_text]
            self._append_event("gate", f"paired resource gate passed: {estimate['resolved_profile']}")
            self._append_event("protocol", f"shared dataset {estimate['dataset_fingerprint'][:12]}, seeds={estimate['evaluation_seeds']}")
        self.thread = threading.Thread(
            target=self._run_arena,
            args=(collect_command, dataset_path, estimate),
            daemon=True,
            name="paired-candidate-optimizer",
        )
        self.thread.start()
        return self.snapshot()

    def _reset_candidate_state(self, method: str, index: int) -> None:
        with self.lock:
            self.state.method = method
            self.state.current_candidate_index = index
            self.state.train_loss = None
            self.state.val_loss = None
            self.state.model_path = None
            self.state.trainable_params = None
            self.state.evaluation_seed = None
            self.state.evaluation_successes = 0
            self.state.evaluation_success_rate = None
            self.state.evaluation_success = None
            self.state.evaluation_target_error = None
            self.state.evaluation_mean_target_error = None
            self.state.evaluation_steps = 0
            self.state.evaluation_rows = []
            self.state.candidate_peak_rss_mb = 0.0
            self._append_event("candidate", f"{index}/{len(self.state.candidate_methods)} {method}")

    @staticmethod
    def _exact_mcnemar(a_only: int, b_only: int) -> float:
        disagreements = a_only + b_only
        if disagreements == 0:
            return 1.0
        tail = sum(math.comb(disagreements, value) for value in range(min(a_only, b_only) + 1)) / (2 ** disagreements)
        return min(1.0, 2.0 * tail)

    def _paired_candidate_summary(self, results: list[dict], expected_seeds: list[int]) -> dict:
        completed = [row for row in results if row["status"] == "completed"]

        def sort_key(row: dict) -> tuple:
            error = row["mean_target_error"] if row["mean_target_error"] is not None else float("inf")
            return (-row["success_rate"], error, row["trainable_params"], row["peak_rss_mb"], row["elapsed"])

        ranked = sorted(completed, key=sort_key)

        def dominates(left: dict, right: dict) -> bool:
            left_error = left["mean_target_error"] if left["mean_target_error"] is not None else float("inf")
            right_error = right["mean_target_error"] if right["mean_target_error"] is not None else float("inf")
            left_values = (left["success_rate"], -left_error, -left["trainable_params"], -left["peak_rss_mb"], -left["elapsed"])
            right_values = (right["success_rate"], -right_error, -right["trainable_params"], -right["peak_rss_mb"], -right["elapsed"])
            return all(a >= b for a, b in zip(left_values, right_values)) and any(a > b for a, b in zip(left_values, right_values))

        pareto = [row["method"] for row in completed if not any(dominates(other, row) for other in completed if other is not row)]
        comparisons = []
        for index, left in enumerate(completed):
            left_by_seed = {row["seed"]: row for row in left["evaluation_rows"]}
            for right in completed[index + 1:]:
                right_by_seed = {row["seed"]: row for row in right["evaluation_rows"]}
                both_success = left_only = right_only = both_fail = 0
                for seed in expected_seeds:
                    left_success = bool(left_by_seed.get(seed, {}).get("success"))
                    right_success = bool(right_by_seed.get(seed, {}).get("success"))
                    both_success += int(left_success and right_success)
                    left_only += int(left_success and not right_success)
                    right_only += int(right_success and not left_success)
                    both_fail += int(not left_success and not right_success)
                comparisons.append({
                    "method_a": left["method"],
                    "method_b": right["method"],
                    "both_success": both_success,
                    "a_only": left_only,
                    "b_only": right_only,
                    "both_fail": both_fail,
                    "mcnemar_exact_p": round(self._exact_mcnemar(left_only, right_only), 6),
                })
        champion = ranked[0] if ranked else None
        promotion = "rejected"
        if champion:
            enough = champion["evaluation_episodes"] >= 3
            quality = champion["success_rate"] >= 2 / 3 and champion["mean_target_error"] is not None and champion["mean_target_error"] <= 0.03
            promotion = "promoted" if enough and quality else "needs_evidence" if quality else "rejected"
        reference = ranked[1] if len(ranked) > 1 else None
        paired_improvement = None
        if champion and reference and champion["mean_target_error"] is not None and reference["mean_target_error"] is not None:
            paired_improvement = {
                "reference_method": reference["method"],
                "evaluation_seeds": expected_seeds,
                "success_rate_points": round((champion["success_rate"] - reference["success_rate"]) * 100, 1),
                "target_error_reduction_mm": round((reference["mean_target_error"] - champion["mean_target_error"]) * 1000, 1),
            }
        seed_sets = [sorted(row["evaluation_seeds"]) for row in completed]
        return {
            "framework": self.ARENA_FRAMEWORK,
            "paired": True,
            "expected_seeds": expected_seeds,
            "matched_seed_sets": bool(completed) and all(seeds == expected_seeds for seeds in seed_sets),
            "completed_candidates": len(completed),
            "failed_candidates": len(results) - len(completed),
            "selection_rule": "success rate desc, target error asc, trainable parameters asc, peak RAM asc, elapsed time asc",
            "champion_method": None if champion is None else champion["method"],
            "pareto_methods": pareto,
            "promotion": promotion,
            "paired_improvement": paired_improvement,
            "comparisons": comparisons,
        }

    def _run_arena(self, collect_command: list[str] | None, dataset_path: Path, estimate: dict) -> None:
        run_id = self.state.run_id
        results: list[dict] = []
        try:
            with self.lock:
                self.state.status = "running"
                self.state.progress = 0.03
            if collect_command is None:
                summary = json.loads((dataset_path / "summary.json").read_text(encoding="utf-8"))
                with self.lock:
                    self.state.collection_successes = int(summary["successes"])
                    self.state.progress = 0.20
                    self._append_event("cache", f"reused verified demonstrations: {dataset_path.name}")
            else:
                self._set_stage("collect", self.state.episodes)
                if self._run_process(collect_command, "collect") != 0:
                    if self.stop_event.is_set():
                        return
                    raise RuntimeError("shared demonstration collection failed")
                summary = json.loads((dataset_path / "summary.json").read_text(encoding="utf-8"))
                if int(summary["successes"]) < 2:
                    raise RuntimeError("fewer than two successful shared demonstrations")
                (dataset_path / "shared_demo_protocol.json").write_text(
                    json.dumps({**estimate["dataset_protocol"], "fingerprint": estimate["dataset_fingerprint"]}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                with self.lock:
                    self.state.collection_successes = int(summary["successes"])
                    self.state.progress = 0.20
                    self._append_event("dataset", f"shared demonstrations {summary['successes']}/{summary['episodes']}")

            profile = self.PROFILES[estimate["resolved_profile"]]
            evaluation_seed = estimate["evaluation_seeds"][0]
            arena_model_root = ADAPTATION_MODEL_ROOT / str(self.state.task_id) / str(run_id)
            for index, method in enumerate(estimate["methods"], start=1):
                if self.stop_event.is_set():
                    break
                self._reset_candidate_state(method, index)
                candidate_started = time.time()
                model_output = arena_model_root / method
                train_command = self._train_command(method, profile, estimate["epochs"], dataset_path, model_output, estimate["task_id"])
                child_config = {
                    "framework": self.ARENA_FRAMEWORK,
                    "task_id": estimate["task_id"],
                    "method": method,
                    "profile": estimate["resolved_profile"],
                    "episodes": estimate["episodes"],
                    "epochs": estimate["epochs"],
                    "evaluation_episodes": estimate["evaluation_episodes"],
                    "evaluation_seeds": estimate["evaluation_seeds"],
                    "seed": self.state.seed,
                    "viewer": self.state.viewer,
                    "shared_dataset": str(dataset_path),
                    "dataset_fingerprint": estimate["dataset_fingerprint"],
                }
                child_id = self.ledger.start("adaptation_candidate", child_config, subprocess.list2cmdline(train_command), parent_id=run_id)
                candidate_error = None
                try:
                    self._set_stage("train", 1 if method == "registry_rgb_skill" else estimate["epochs"])
                    with self.lock:
                        self.logs.append(subprocess.list2cmdline(train_command))
                    if self._run_process(train_command, "train") != 0:
                        raise RuntimeError("candidate training failed")
                    model_path = Path(str(self.state.model_path or ""))
                    if not model_path.is_file():
                        raise RuntimeError("candidate completed without a model artifact")
                    evaluation_command = self._evaluation_command(
                        method,
                        model_path,
                        estimate["task_id"],
                        evaluation_seed,
                        estimate["evaluation_episodes"],
                        self.state.viewer,
                    )
                    with self.lock:
                        self.state.evaluation_seed = evaluation_seed
                        self.logs.append(subprocess.list2cmdline(evaluation_command))
                    self._set_stage("evaluate", estimate["evaluation_episodes"])
                    if self._run_process(evaluation_command, "evaluate") != 0:
                        raise RuntimeError("candidate holdout evaluation failed")
                    if len(self.state.evaluation_rows) != estimate["evaluation_episodes"]:
                        raise RuntimeError("candidate produced an incomplete holdout seed set")
                except Exception as error:
                    candidate_error = str(error)
                    self._append_event("candidate_error", f"{method}: {candidate_error}")

                with self.lock:
                    rows = json.loads(json.dumps(self.state.evaluation_rows))
                    successes = sum(int(row["success"]) for row in rows)
                    finite_errors = [float(row["target_error"]) for row in rows if isinstance(row.get("target_error"), (int, float)) and np.isfinite(row["target_error"])]
                    result = {
                        "run_id": child_id,
                        "method": method,
                        "label": self.METHODS[method]["label"],
                        "family": self.METHODS[method]["family"],
                        "status": "failed" if candidate_error else "completed",
                        "error": candidate_error,
                        "evaluation_episodes": len(rows),
                        "evaluation_seeds": [int(row["seed"]) for row in rows],
                        "successes": successes,
                        "success_rate": successes / len(rows) if rows else 0.0,
                        "mean_target_error": float(np.mean(finite_errors)) if finite_errors else None,
                        "trainable_params": int(self.state.trainable_params or 0),
                        "peak_rss_mb": round(self.state.candidate_peak_rss_mb, 3),
                        "elapsed": time.time() - candidate_started,
                        "train_loss": self.state.train_loss,
                        "val_loss": self.state.val_loss,
                        "artifact": self.state.model_path,
                        "evaluation_rows": rows,
                        "dataset_fingerprint": estimate["dataset_fingerprint"],
                    }
                    results.append(result)
                    self.state.candidate_results = json.loads(json.dumps(results))
                    self._append_event("candidate_result", f"{method}: {successes}/{len(rows)}, {result['mean_target_error'] * 1000:.1f} mm" if result["mean_target_error"] is not None else f"{method}: failed")
                candidate_metrics = {key: value for key, value in result.items() if key not in {"run_id", "label", "family", "artifact", "error"}}
                candidate_metrics.update({
                    "evaluation_successes": result["successes"],
                    "evaluation_success_rate": result["success_rate"],
                    "evaluation_success": result["successes"] == result["evaluation_episodes"] and result["evaluation_episodes"] > 0,
                    "evaluation_mean_target_error": result["mean_target_error"],
                    "evaluation_target_error": result["mean_target_error"],
                })
                self.ledger.finish(
                    child_id,
                    result["status"],
                    metrics=candidate_metrics,
                    artifact=result["artifact"],
                    assets={"shared_dataset": str(dataset_path.relative_to(ROOT))},
                )

            if self.stop_event.is_set():
                return
            paired = self._paired_candidate_summary(results, estimate["evaluation_seeds"])
            completed = [row for row in results if row["status"] == "completed"]
            if not completed:
                raise RuntimeError("all paired candidates failed")
            champion = next(row for row in completed if row["method"] == paired["champion_method"])
            with self.lock:
                self.state.paired_summary = paired
                self.state.method = champion["method"]
                self.state.model_path = champion["artifact"]
                self.state.trainable_params = champion["trainable_params"]
                self.state.evaluation_rows = champion["evaluation_rows"]
                self.state.evaluation_successes = champion["successes"]
                self.state.evaluation_success_rate = champion["success_rate"]
                self.state.evaluation_success = champion["successes"] == champion["evaluation_episodes"]
                self.state.evaluation_mean_target_error = champion["mean_target_error"]
                self.state.evaluation_target_error = champion["mean_target_error"]
                self.state.status = "completed"
                self.state.stage = "complete"
                self.state.progress = 1.0
                self.state.stage_progress = 1.0
                self._append_event("promotion", f"paired champion: {champion['method']} ({champion['successes']}/{champion['evaluation_episodes']})")
                self._append_event("complete", "paired candidate evidence package is ready")
        except Exception as error:
            with self.lock:
                self.state.status = "failed"
                self.state.stage = "failed"
                self.state.error = str(error)
                self._append_event("error", str(error))
        finally:
            with self.lock:
                if self.stop_event.is_set() and self.state.status not in {"completed", "failed"}:
                    self.state.status = "stopped"
                    self.state.stage = "stopped"
                if self.state.started_at is not None:
                    self.state.elapsed = time.time() - self.state.started_at
            snapshot = self.snapshot()
            run_dir = ADAPTATION_ROOT / str(run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            summary_path = run_dir / "paired_optimization.json"
            summary_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
            self.ledger.finish(
                run_id,
                snapshot["status"],
                metrics={
                    "framework": self.ARENA_FRAMEWORK,
                    "task_id": snapshot["task_id"],
                    "candidate_methods": snapshot["candidate_methods"],
                    "candidate_results": snapshot["candidate_results"],
                    "paired_summary": snapshot["paired_summary"],
                    "dataset_fingerprint": snapshot["dataset_fingerprint"],
                    "dataset_cache_hit": snapshot["dataset_cache_hit"],
                    "collection_runs_saved": snapshot["collection_runs_saved"],
                    "peak_rss_mb": snapshot["peak_rss_mb"],
                    "elapsed": snapshot["elapsed"],
                },
                artifact=str(summary_path.relative_to(ROOT)),
                assets={
                    "paired_optimization": artifact_url(summary_path),
                    "shared_dataset": str(dataset_path.relative_to(ROOT)),
                },
            )

class TrainingManager(_TrainingManagerCore):
    def stop(self) -> dict:
        process = self.process
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            with self.lock:
                self.state.status = "stopped"
        return self.snapshot()

    def _run(self, command: list[str]) -> None:
        started = time.monotonic()
        run_id = self.active_run_id
        try:
            self.process = subprocess.Popen(
                command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
            with self.lock:
                self.state.status = "running"
            assert self.process.stdout is not None
            for raw_line in self.process.stdout:
                line = raw_line.rstrip()
                if not line:
                    continue
                match = self.metric_pattern.search(line)
                with self.lock:
                    self.logs.append(line)
                    if len(self.logs) > 400:
                        self.logs = self.logs[-400:]
                    self.state.elapsed = time.monotonic() - started
                    if match:
                        row = {
                            "epoch": int(match.group("epoch")),
                            "train": float(match.group("train")),
                            "val": float(match.group("val")),
                        }
                        self.metrics.append(row)
                        self.state.current_epoch = row["epoch"]
                        self.state.progress = min(1.0, row["epoch"] / self.state.epochs)
                        self.state.train_loss = row["train"]
                        self.state.val_loss = row["val"]
                    if line.startswith("model_path:"):
                        self.state.model_path = line.split(":", 1)[1].strip()
            return_code = self.process.wait()
            with self.lock:
                self.state.elapsed = time.monotonic() - started
                if self.state.status != "stopped":
                    self.state.status = "completed" if return_code == 0 else "failed"
                    self.state.progress = 1.0 if return_code == 0 else self.state.progress
                    if return_code != 0:
                        self.state.error = f"trainer exited with code {return_code}"
        except Exception as error:
            with self.lock:
                self.state.status = "failed"
                self.state.error = str(error)
                self.state.elapsed = time.monotonic() - started
        finally:
            snapshot = self.snapshot()
            self.ledger.finish(
                run_id,
                snapshot["status"],
                metrics={
                    "epochs": snapshot["current_epoch"],
                    "train_loss": snapshot["train_loss"],
                    "val_loss": snapshot["val_loss"],
                    "elapsed": snapshot["elapsed"],
                },
                artifact=snapshot["model_path"],
            )


@dataclass
class AdaptationState:
    run_id: str | None = None
    status: str = "idle"
    stage: str = "ready"
    task_id: str | None = None
    method: str = "local_lora"
    requested_profile: str = "auto"
    resolved_profile: str = "eco"
    episodes: int = 3
    epochs: int = 4
    seed: int = 0
    viewer: bool = True
    mode: str = "single"
    arena_framework: str = "PAIR-OPT-1.0"
    candidate_methods: list[str] = field(default_factory=list)
    current_candidate_index: int = 0
    candidate_results: list[dict] = field(default_factory=list)
    paired_summary: dict = field(default_factory=dict)
    dataset_fingerprint: str | None = None
    dataset_cache_hit: bool = False
    collection_runs_saved: int = 0
    progress: float = 0.0
    stage_progress: float = 0.0
    current_item: int = 0
    total_items: int = 0
    started_at: float | None = None
    elapsed: float = 0.0
    train_loss: float | None = None
    val_loss: float | None = None
    collection_successes: int = 0
    dataset_path: str | None = None
    model_path: str | None = None
    trainable_params: int | None = None
    evaluation_seed: int | None = None
    evaluation_episodes: int = 3
    evaluation_successes: int = 0
    evaluation_success_rate: float | None = None
    evaluation_success: bool | None = None
    evaluation_target_error: float | None = None
    evaluation_mean_target_error: float | None = None
    evaluation_steps: int = 0
    evaluation_rows: list[dict] = field(default_factory=list)
    process_rss_mb: float = 0.0
    peak_rss_mb: float = 0.0
    candidate_peak_rss_mb: float = 0.0
    estimated: dict = field(default_factory=dict)
    error: str | None = None


class AdaptationManager(PairedArenaOperations):
    """Resource-gated collection and lightweight task adaptation for local machines."""

    FRAMEWORK = "LOCAL-ADAPT-1.2"
    ARENA_FRAMEWORK = "PAIR-OPT-1.0"
    METHODS = {
        "local_lora": {
            "label": "LoRA-style residual",
            "family": "frozen object-language action head",
            "script": ROOT / "scripts" / "train_peft_action_head.py",
            "truth_boundary": "Local PEFT proxy; not OpenVLA or pretrained VLA LoRA.",
        },
        "local_adapter": {
            "label": "Bottleneck adapter",
            "family": "frozen object-language action head",
            "script": ROOT / "scripts" / "train_peft_action_head.py",
            "truth_boundary": "Local PEFT proxy; not OpenVLA or pretrained VLA Adapter.",
        },
        "micro_head": {
            "label": "Micro action head",
            "family": "small object-language MLP",
            "script": ROOT / "scripts" / "train_object_action_head.py",
            "truth_boundary": "Small supervised action head trained from scratch.",
        },
        "registry_rgb_skill": {
            "label": "Registry RGB skill adapter",
            "family": "task registry + RGB geometry + structured execution",
            "script": ROOT / "scripts" / "compile_registry_rgb_skill_adapter.py",
            "truth_boundary": "Zero-gradient task-registry adapter with calibrated RGB runtime localisation and structured execution; MuJoCo state is reserved for offline scoring, and this is not a learned VLA or OpenVLA fine-tune.",
        },
    }
    PROFILES = {
        "eco": {
            "label": "Eco / 4 GB class",
            "rank": 2,
            "hidden": 32,
            "batch_size": 256,
            "max_episodes": 6,
            "max_epochs": 8,
            "memory_budget_mb": 512,
        },
        "balanced": {
            "label": "Balanced / 8 GB class",
            "rank": 4,
            "hidden": 64,
            "batch_size": 512,
            "max_episodes": 12,
            "max_epochs": 16,
            "memory_budget_mb": 1024,
        },
        "research": {
            "label": "Research / 16 GB class",
            "rank": 8,
            "hidden": 128,
            "batch_size": 1024,
            "max_episodes": 25,
            "max_epochs": 30,
            "memory_budget_mb": 2048,
        },
    }
    collection_pattern = re.compile(r"episode\s+(?P<episode>\d+):.*?success=(?P<success>True|False)", re.IGNORECASE)
    metric_pattern = TrainingManager.metric_pattern

    def __init__(self, ledger: ExperimentLedger, training: TrainingManager) -> None:
        self.ledger = ledger
        self.training = training
        self.lock = threading.RLock()
        self.state = AdaptationState()
        self.events: list[dict] = []
        self.logs: list[str] = []
        self.process: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.native_processes: list[subprocess.Popen] = []
        ADAPTATION_ROOT.mkdir(parents=True, exist_ok=True)
        ADAPTATION_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        self._restore_custom_tasks()

    @staticmethod
    def _hardware() -> dict:
        total_memory, available_memory = system_memory_bytes()
        disk = shutil.disk_usage(ROOT)
        total_gb = total_memory / (1024 ** 3)
        available_gb = available_memory / (1024 ** 3)
        cpu_count = os.cpu_count() or 1
        if total_gb < 6 or available_gb < 1.5 or cpu_count <= 2:
            recommended = "eco"
        elif total_gb < 12 or available_gb < 3 or cpu_count <= 4:
            recommended = "balanced"
        else:
            recommended = "research"
        return {
            "cpu_logical": cpu_count,
            "ram_total_gb": round(total_gb, 2),
            "ram_available_gb": round(available_gb, 2),
            "disk_free_gb": round(disk.free / (1024 ** 3), 2),
            "gpu_required": False,
            "recommended_profile": recommended,
        }

    def _restore_custom_tasks(self) -> None:
        if not ADAPTATION_TASK_REGISTRY.is_file():
            return
        payload = json.loads(ADAPTATION_TASK_REGISTRY.read_text(encoding="utf-8"))
        for row in payload.get("tasks", []):
            spec = register_custom_task(row, persist=False)
            TASKS[spec.name] = {
                "instruction": spec.instruction,
                "instruction_zh": str(row.get("instruction_zh", "")),
                "source": str(spec.target_object),
                "target": str(spec.target_geom),
                "complexity": str(row.get("complexity", "medium")),
                "custom": True,
            }

    def task_catalogue(self) -> list[dict]:
        custom_rows = []
        if ADAPTATION_TASK_REGISTRY.is_file():
            custom_rows = json.loads(ADAPTATION_TASK_REGISTRY.read_text(encoding="utf-8")).get("tasks", [])
        return [
            {
                "task_id": task_id,
                "instruction": spec["instruction"],
                "instruction_zh": spec.get("instruction_zh", ""),
                "source": spec["source"],
                "target": spec["target"],
                "complexity": spec["complexity"],
                "custom": bool(spec.get("custom")),
            }
            for task_id, spec in TASKS.items()
            if task_id in {row.get("task_id") for row in custom_rows}
        ]

    def performance_portfolio(self, task_id: str | None = None) -> dict:
        task_id = task_id or self.state.task_id
        if not task_id:
            tasks = self.task_catalogue()
            task_id = tasks[0]["task_id"] if tasks else None
        latest_by_method: dict[str, dict] = {}
        if task_id:
            for record in self.ledger.list(limit=1000, status="completed"):
                config = record.get("config", {})
                metrics = record.get("metrics", {})
                method = str(config.get("method", ""))
                if record.get("kind") not in {"adaptation", "adaptation_candidate"} or config.get("task_id") != task_id or method not in self.METHODS or method in latest_by_method:
                    continue
                evaluated = int(metrics.get("evaluation_episodes") or (1 if metrics.get("evaluation_success") is not None else 0))
                successes = int(metrics.get("evaluation_successes") if metrics.get("evaluation_successes") is not None else int(metrics.get("evaluation_success") is True))
                rate = float(metrics.get("evaluation_success_rate") if metrics.get("evaluation_success_rate") is not None else successes / max(1, evaluated))
                mean_error = metrics.get("evaluation_mean_target_error", metrics.get("evaluation_target_error"))
                latest_by_method[method] = {
                    "run_id": record["run_id"],
                    "method": method,
                    "label": self.METHODS[method]["label"],
                    "family": self.METHODS[method]["family"],
                    "evaluation_episodes": evaluated,
                    "successes": successes,
                    "success_rate": rate,
                    "mean_target_error": float(mean_error) if isinstance(mean_error, (int, float)) and np.isfinite(mean_error) else None,
                    "trainable_params": int(metrics.get("trainable_params") or 0),
                    "peak_rss_mb": float(metrics.get("peak_rss_mb") or 0.0),
                    "elapsed": float(metrics.get("elapsed") or 0.0),
                    "evaluation_seeds": sorted(
                        int(row["seed"])
                        for row in metrics.get("evaluation_rows", [])
                        if isinstance(row, dict) and isinstance(row.get("seed"), int)
                    ) or ([int(metrics["evaluation_seed"])] if metrics.get("evaluation_seed") is not None else []),
                    "artifact": record.get("artifact"),
                    "started_at": record.get("started_at"),
                }

        candidates = list(latest_by_method.values())

        def dominates(left: dict, right: dict) -> bool:
            left_error = left["mean_target_error"] if left["mean_target_error"] is not None else float("inf")
            right_error = right["mean_target_error"] if right["mean_target_error"] is not None else float("inf")
            left_values = (left["success_rate"], -left_error, -left["trainable_params"], -left["peak_rss_mb"], -left["elapsed"])
            right_values = (right["success_rate"], -right_error, -right["trainable_params"], -right["peak_rss_mb"], -right["elapsed"])
            return all(a >= b for a, b in zip(left_values, right_values)) and any(a > b for a, b in zip(left_values, right_values))

        pareto = [candidate["method"] for candidate in candidates if not any(dominates(other, candidate) for other in candidates if other is not candidate)]
        ranked = sorted(
            candidates,
            key=lambda item: (
                -item["success_rate"],
                item["mean_target_error"] if item["mean_target_error"] is not None else float("inf"),
                item["trainable_params"],
                item["peak_rss_mb"],
                item["elapsed"],
            ),
        )
        champion = ranked[0] if ranked else None
        promotion = "no_evidence"
        if champion:
            enough_evidence = champion["evaluation_episodes"] >= 3
            meets_quality = champion["success_rate"] >= 2 / 3 and champion["mean_target_error"] is not None and champion["mean_target_error"] <= 0.03
            promotion = "promoted" if enough_evidence and meets_quality else "needs_evidence" if meets_quality else "rejected"
        learned = next((item for item in ranked if item["method"] != "registry_rgb_skill"), None)
        improvement = None
        paired = bool(champion and learned and champion["evaluation_seeds"] == learned["evaluation_seeds"] and champion["evaluation_seeds"])
        if paired and champion["mean_target_error"] is not None and learned["mean_target_error"] is not None:
            improvement = {
                "reference_method": learned["method"],
                "evaluation_seeds": champion["evaluation_seeds"],
                "success_rate_points": round((champion["success_rate"] - learned["success_rate"]) * 100, 1),
                "target_error_reduction_mm": round((learned["mean_target_error"] - champion["mean_target_error"]) * 1000, 1),
            }
        return {
            "framework": "RESOURCE-PARETO-1.0",
            "task_id": task_id,
            "selection_rule": "success rate desc, target error asc, trainable parameters asc, peak RAM asc, elapsed time asc",
            "comparison_boundary": "Candidate promotion uses each run's own thresholds. Cross-method improvement is reported only for identical held-out seed sets.",
            "promotion_thresholds": {"minimum_episodes": 3, "minimum_success_rate": round(2 / 3, 4), "maximum_target_error_m": 0.03},
            "promotion": promotion,
            "champion_method": None if champion is None else champion["method"],
            "pareto_methods": pareto,
            "improvement": improvement,
            "candidates": ranked,
        }

    def arena_efficiency(self, task_id: str | None = None) -> dict | None:
        groups: dict[tuple, dict[bool, dict]] = {}
        for record in self.ledger.list(limit=1000, kind="optimization", status="completed"):
            config = record.get("config", {})
            metrics = record.get("metrics", {})
            if task_id and config.get("task_id") != task_id:
                continue
            elapsed = metrics.get("elapsed")
            fingerprint = metrics.get("dataset_fingerprint")
            if not isinstance(elapsed, (int, float)) or not fingerprint:
                continue
            key = (
                config.get("task_id"), tuple(config.get("methods", [])), config.get("profile"),
                config.get("episodes"), config.get("epochs"), config.get("evaluation_episodes"),
                config.get("seed"), fingerprint,
            )
            cached = bool(metrics.get("dataset_cache_hit"))
            groups.setdefault(key, {}).setdefault(cached, record)
        matched = next((pair for pair in groups.values() if False in pair and True in pair), None)
        if matched is None:
            return None
        fresh = matched[False]
        cached = matched[True]
        fresh_seconds = float(fresh["metrics"]["elapsed"])
        cached_seconds = float(cached["metrics"]["elapsed"])
        saved_seconds = fresh_seconds - cached_seconds
        return {
            "framework": self.ARENA_FRAMEWORK,
            "fresh_run_id": fresh["run_id"],
            "cached_run_id": cached["run_id"],
            "fresh_seconds": round(fresh_seconds, 3),
            "cached_seconds": round(cached_seconds, 3),
            "saved_seconds": round(saved_seconds, 3),
            "reduction_percent": round(saved_seconds / fresh_seconds * 100, 1) if fresh_seconds > 0 else 0.0,
            "cached_collection_runs_saved": int(cached["metrics"].get("collection_runs_saved") or 0),
            "boundary": "One matched fresh/cache run pair; this is an observed local measurement, not a population estimate.",
        }

    def latest_arena(self, task_id: str | None = None) -> dict | None:
        for record in self.ledger.list(limit=1000, kind="optimization", status="completed"):
            if task_id and record.get("config", {}).get("task_id") != task_id:
                continue
            return {
                "run_id": record["run_id"],
                "started_at": record.get("started_at"),
                "config": record.get("config", {}),
                "metrics": record.get("metrics", {}),
                "artifact": record.get("artifact"),
                "assets": record.get("assets", {}),
            }
        return None

    def create_task(self, request: dict) -> dict:
        source = str(request.get("source", "")).strip()
        target = str(request.get("target", "")).strip()
        instruction = str(request.get("instruction", "")).strip()
        instruction_zh = str(request.get("instruction_zh", "")).strip()
        complexity = str(request.get("complexity", "medium"))
        if source not in CUSTOM_TASK_SOURCES:
            raise ValueError("source must be one of the available colored cubes")
        if target not in CUSTOM_TASK_TARGETS:
            raise ValueError("target must be an existing pad or bowl")
        if complexity not in {"medium", "hard"}:
            raise ValueError("custom adaptation tasks support medium or hard layouts")
        suffix = target.removeprefix("target_")
        task_id = f"place_{source}_{suffix}"
        if task_id in TASKS or task_id in ENV_TASKS:
            raise ValueError("this source-target task already exists")
        record = {
            "task_id": task_id,
            "instruction": instruction,
            "instruction_zh": instruction_zh,
            "source": source,
            "target": target,
            "complexity": complexity,
            "created_at": utc_now(),
        }
        spec = register_custom_task(record, persist=True)
        TASKS[task_id] = {
            "instruction": spec.instruction,
            "instruction_zh": instruction_zh,
            "source": source,
            "target": target,
            "complexity": complexity,
            "custom": True,
        }
        self._append_event("task", f"registered {task_id}")
        return {**record, "preview_url": f"/api/adaptation/tasks/{task_id}/preview.png?seed=0"}

    def estimate(self, request: dict) -> dict:
        task_id = str(request.get("task_id", ""))
        method = str(request.get("method", "local_lora"))
        requested_profile = str(request.get("profile", "auto"))
        episodes = int(request.get("episodes", 3))
        epochs = int(request.get("epochs", 4))
        evaluation_episodes = int(request.get("evaluation_episodes", 3))
        viewer = bool(request.get("viewer", True))
        if task_id not in TASKS or not TASKS[task_id].get("custom"):
            raise ValueError("select a registered custom task")
        if method not in self.METHODS:
            raise ValueError("unsupported adaptation method")
        hardware = self._hardware()
        profile_name = hardware["recommended_profile"] if requested_profile == "auto" else requested_profile
        if profile_name not in self.PROFILES:
            raise ValueError("unsupported resource profile")
        if not 2 <= episodes <= 25:
            raise ValueError("episodes must be between 2 and 25")
        if not 1 <= epochs <= 30:
            raise ValueError("epochs must be between 1 and 30")
        if not 1 <= evaluation_episodes <= 5:
            raise ValueError("evaluation_episodes must be between 1 and 5")
        profile = self.PROFILES[profile_name]
        rank = int(profile["rank"])
        hidden = int(profile["hidden"])
        if method == "local_lora":
            trainable_params = rank * (256 + 7)
        elif method == "local_adapter":
            trainable_params = 256 * rank + rank + rank * 7 + 7
        elif method == "micro_head":
            trainable_params = 152 * hidden + hidden + hidden * hidden + hidden + hidden * 7 + 7
        else:
            trainable_params = 0
        estimated_samples = episodes * 2840
        data_mb = estimated_samples * (152 + 7) * 4 / (1024 ** 2)
        runtime_base_mb = 420.0 if method == "registry_rgb_skill" else 150.0
        training_peak_mb = round(runtime_base_mb + data_mb * (5.0 if method == "micro_head" else 3.5), 1)
        viewer_overhead_mb = 520.0 if viewer else 0.0
        static_peak_mb = round(training_peak_mb + viewer_overhead_mb, 1)
        observed_peaks = [
            float(record.get("metrics", {}).get("peak_rss_mb") or 0.0)
            for record in self.ledger.list(limit=1000, status="completed")
            if record.get("kind") in {"adaptation", "adaptation_candidate"}
            and record.get("config", {}).get("method") == method
            and bool(record.get("config", {}).get("viewer", True)) == viewer
        ]
        observed_peaks = [value for value in observed_peaks if value > 0]
        calibrated_peak_mb = round(max(observed_peaks) * 1.12, 1) if observed_peaks else 0.0
        estimated_peak_mb = max(static_peak_mb, calibrated_peak_mb)
        collection_seconds = episodes * 2.1
        training_factor = {"local_lora": 0.09, "local_adapter": 0.10, "micro_head": 0.14, "registry_rgb_skill": 0.0}[method]
        evaluation_seconds = (4.0 if method == "registry_rgb_skill" else 1.2) * evaluation_episodes
        estimated_seconds = round(collection_seconds + max(0.2, episodes * epochs * training_factor) + evaluation_seconds + (12.0 if viewer else 0.0), 1)
        reasons = []
        if episodes > int(profile["max_episodes"]):
            reasons.append(f"episodes exceed the {profile_name} profile limit")
        if epochs > int(profile["max_epochs"]):
            reasons.append(f"epochs exceed the {profile_name} profile limit")
        if training_peak_mb > float(profile["memory_budget_mb"]):
            reasons.append("estimated peak memory exceeds the profile budget")
        if hardware["ram_available_gb"] * 1024 < estimated_peak_mb * 1.25:
            reasons.append("available system memory is below the safety margin")
        if hardware["disk_free_gb"] < 0.5:
            reasons.append("less than 0.5 GB disk space is available")
        if method in {"local_lora", "local_adapter"} and not OBJECT_ACTION_HEAD_PATH.is_file():
            reasons.append("frozen base action head is missing")
        if method == "registry_rgb_skill" and not CALIBRATION_PATH.is_file():
            reasons.append("RGB table-plane calibration is missing")
        return {
            "framework": self.FRAMEWORK,
            "task_id": task_id,
            "method": method,
            "method_label": self.METHODS[method]["label"],
            "truth_boundary": self.METHODS[method]["truth_boundary"],
            "requested_profile": requested_profile,
            "resolved_profile": profile_name,
            "profile": profile,
            "episodes": episodes,
            "epochs": epochs,
            "evaluation_episodes": evaluation_episodes,
            "viewer": viewer,
            "estimated_samples": estimated_samples,
            "trainable_params": trainable_params,
            "updated_parameter_mb": round(trainable_params * 4 / (1024 ** 2), 4),
            "estimated_peak_ram_mb": estimated_peak_mb,
            "estimated_training_ram_mb": training_peak_mb,
            "viewer_overhead_mb": viewer_overhead_mb,
            "resource_calibration": {
                "estimate_source": "ledger_calibrated" if calibrated_peak_mb > static_peak_mb else "analytical_floor",
                "evidence_runs": len(observed_peaks),
                "static_peak_ram_mb": static_peak_mb,
                "observed_max_peak_ram_mb": round(max(observed_peaks), 1) if observed_peaks else None,
                "safety_factor": 1.12,
            },
            "estimated_wall_seconds": estimated_seconds,
            "gpu_required": False,
            "hardware": hardware,
            "gate": {"passed": not reasons, "reasons": reasons},
        }

    @staticmethod
    def _dataset_protocol(task_id: str, complexity: str, seed: int, episodes: int) -> dict:
        return {
            "schema": "shared-demo-protocol-v1",
            "task_id": task_id,
            "complexity": complexity,
            "seeds": list(range(seed, seed + episodes)),
            "expert": "scripted-pick-place-core-v2",
            "arm_kp": 150,
            "arm_force": 100,
            "gripper_kp": 1200,
            "gripper_force": 200,
            "friction": 5.0,
            "retries": 1,
        }

    @staticmethod
    def _fingerprint(payload: dict) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _cached_dataset(self, protocol: dict) -> Path | None:
        expected_seeds = protocol["seeds"]
        task_id = protocol["task_id"]
        candidates = sorted((ROOT / "data" / "demos").glob(f"*{task_id}*"), key=lambda path: path.stat().st_mtime, reverse=True)
        for path in candidates:
            summary_path = path / "summary.json"
            metadata_path = path / "metadata.jsonl"
            if not summary_path.is_file() or not metadata_path.is_file():
                continue
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            except (OSError, json.JSONDecodeError):
                continue
            if int(summary.get("episodes", 0)) != len(expected_seeds) or int(summary.get("successes", 0)) != len(expected_seeds):
                continue
            if [int(row.get("seed", -1)) for row in rows] != expected_seeds:
                continue
            if not all(row.get("task") == task_id and row.get("complexity") == protocol["complexity"] and row.get("success") is True for row in rows):
                continue
            if not all((path / str(row.get("trajectory_file", ""))).is_file() for row in rows):
                continue
            return path
        return None

    def estimate_arena(self, request: dict) -> dict:
        methods = list(dict.fromkeys(str(item) for item in request.get("methods", [])))
        if not 2 <= len(methods) <= len(self.METHODS):
            raise ValueError("select between two and four unique candidate methods")
        unsupported = [method for method in methods if method not in self.METHODS]
        if unsupported:
            raise ValueError("unsupported adaptation method: " + ", ".join(unsupported))
        estimates = [self.estimate({**request, "method": method}) for method in methods]
        task_id = estimates[0]["task_id"]
        episodes = estimates[0]["episodes"]
        seed = int(request.get("seed", 0))
        protocol = self._dataset_protocol(task_id, TASKS[task_id]["complexity"], seed, episodes)
        reuse_dataset = bool(request.get("reuse_dataset", False))
        cached_path = self._cached_dataset(protocol) if reuse_dataset else None
        collection_seconds = episodes * 2.1
        estimated_seconds = sum(float(item["estimated_wall_seconds"]) for item in estimates) - collection_seconds * (len(methods) - 1)
        if cached_path is not None:
            estimated_seconds -= collection_seconds
        reasons = list(dict.fromkeys(reason for item in estimates for reason in item["gate"]["reasons"]))
        return {
            "framework": self.ARENA_FRAMEWORK,
            "task_id": task_id,
            "methods": methods,
            "method_labels": [self.METHODS[method]["label"] for method in methods],
            "requested_profile": estimates[0]["requested_profile"],
            "resolved_profile": estimates[0]["resolved_profile"],
            "episodes": episodes,
            "epochs": estimates[0]["epochs"],
            "evaluation_episodes": estimates[0]["evaluation_episodes"],
            "evaluation_seeds": list(range(seed + episodes + 1000, seed + episodes + 1000 + estimates[0]["evaluation_episodes"])),
            "viewer": bool(request.get("viewer", True)),
            "reuse_dataset": reuse_dataset,
            "dataset_cache_hit": cached_path is not None,
            "cached_dataset_path": None if cached_path is None else str(cached_path),
            "dataset_protocol": protocol,
            "dataset_fingerprint": self._fingerprint(protocol),
            "collection_runs_saved": len(methods) - 1 + int(cached_path is not None),
            "sequential_peak_ram_mb": max(float(item["estimated_peak_ram_mb"]) for item in estimates),
            "total_trainable_params": sum(int(item["trainable_params"]) for item in estimates),
            "estimated_wall_seconds": round(max(0.1, estimated_seconds), 1),
            "candidate_estimates": estimates,
            "comparison_invariants": [
                "one shared demonstration dataset",
                "identical held-out seed sequence",
                "sequential execution under one resource gate",
                "failed candidates remain in the evidence record",
            ],
            "gate": {"passed": not reasons, "reasons": reasons},
        }

    def snapshot(self) -> dict:
        with self.lock:
            state = asdict(self.state)
            state.update({
                "framework": self.FRAMEWORK,
                "hardware": self._hardware(),
                "profiles": self.PROFILES,
                "methods": {
                    method_id: {key: value for key, value in method.items() if key != "script"}
                    for method_id, method in self.METHODS.items()
                },
                "task_sources": list(CUSTOM_TASK_SOURCES),
                "task_targets": list(CUSTOM_TASK_TARGETS),
                "tasks": self.task_catalogue(),
                "performance_portfolio": self.performance_portfolio(state.get("task_id")),
                "arena_efficiency": self.arena_efficiency(state.get("task_id")),
                "latest_arena": self.latest_arena(state.get("task_id")),
                "events": self.events[-40:],
                "logs": self.logs[-160:],
            })
            return state

    def preview(self, task_id: str, seed: int) -> bytes:
        if task_id not in TASKS or not TASKS[task_id].get("custom"):
            raise ValueError("unknown adaptation task")
        spec = TASKS[task_id]
        env = WidowXTabletopEnv(seed=seed, image_size=(480, 640), camera="front_rgb", workspace_profile="core_v2")
        env.reset(task=task_id, complexity=spec["complexity"], seed=seed)
        renderer = mujoco.Renderer(env.model, height=480, width=640)
        try:
            renderer.update_scene(env.data, camera="front_rgb")
            return png_bytes(renderer.render())
        finally:
            renderer.close()

    def open_viewer(self, request: dict) -> dict:
        task_id = str(request.get("task_id", ""))
        seed = int(request.get("seed", 0))
        if task_id not in TASKS or not TASKS[task_id].get("custom"):
            raise ValueError("unknown adaptation task")
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_viewer.py"),
            "--task", task_id,
            "--complexity", TASKS[task_id]["complexity"],
            "--seed", str(seed),
        ]
        flags = subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
        process = subprocess.Popen(command, cwd=ROOT, creationflags=flags)
        self.native_processes = [item for item in self.native_processes if item.poll() is None]
        self.native_processes.append(process)
        return {"started": True, "pid": process.pid, "command": subprocess.list2cmdline(command)}

    def _collect_command(self, task_id: str, seed: int, episodes: int, run_tag: str, viewer: bool) -> list[str]:
        command = [
            sys.executable, "-u", str(ROOT / "scripts" / "collect_demos.py"),
            "--task", task_id,
            "--complexity", TASKS[task_id]["complexity"],
            "--workspace-profile", "core_v2",
            "--seed", str(seed),
            "--episodes", str(episodes),
            "--output", str(ROOT / "data" / "demos"),
            "--run-name", run_tag,
            "--min-success-rate", "0.5",
            "--speed", "3.0",
            "--arm-kp", "150", "--arm-force", "100",
            "--gripper-kp", "1200", "--gripper-force", "200", "--friction", "5.0",
            "--retries", "1",
        ]
        if viewer:
            command.extend(["--viewer", "--duration", "6"])
        return command

    def _train_command(self, method: str, profile: dict, epochs: int, dataset_path: Path, model_output: Path, task_id: str) -> list[str]:
        if method in {"local_lora", "local_adapter"}:
            return [
                sys.executable, "-u", str(self.METHODS[method]["script"]),
                "--base-model", str(OBJECT_ACTION_HEAD_PATH),
                "--mode", "lora" if method == "local_lora" else "adapter",
                "--run-dir", str(dataset_path),
                "--output", str(model_output),
                "--rank", str(profile["rank"]),
                "--epochs", str(epochs),
                "--batch-size", str(profile["batch_size"]),
                "--lr", "0.001",
            ]
        if method == "micro_head":
            return [
                sys.executable, "-u", str(self.METHODS[method]["script"]),
                "--run-dir", str(dataset_path),
                "--output", str(model_output),
                "--model-prefix", "micro_object_action_head",
                "--hidden-sizes", f"{profile['hidden']},{profile['hidden']}",
                "--epochs", str(epochs),
                "--batch-size", str(profile["batch_size"]),
                "--lr", "0.001",
            ]
        return [
            sys.executable, "-u", str(self.METHODS[method]["script"]),
            "--task", task_id,
            "--output", str(model_output),
            "--calibration", str(CALIBRATION_PATH),
        ]

    def _evaluation_command(self, method: str, model_path: Path, task_id: str, seed: int, episodes: int, viewer: bool) -> list[str]:
        if method == "registry_rgb_skill":
            command = [
                sys.executable, "-u", str(ROOT / "scripts" / "run_clip_semantic_rgb_feedback.py"),
                "--intent-source", "task_registry",
                "--calibration", str(CALIBRATION_PATH),
                "--task", task_id,
                "--complexity", TASKS[task_id]["complexity"],
                "--workspace-profile", "core_v2",
                "--seed", str(seed),
                "--episodes", str(episodes),
                "--feedback-attempts", "1",
                "--recovery-search", "table",
                "--speed", "3.0",
                "--arm-kp", "105", "--arm-force", "70",
                "--gripper-kp", "550", "--gripper-force", "75", "--friction", "0.8",
            ]
        else:
            runner = "run_object_action_head.py" if method == "micro_head" else "run_peft_action_head.py"
            command = [
                sys.executable, "-u", str(ROOT / "scripts" / runner),
                "--model", str(model_path),
                "--task", task_id,
                "--complexity", TASKS[task_id]["complexity"],
                "--seed", str(seed),
                "--episodes", str(episodes),
                "--steps", "2840",
                "--speed", "3.0",
                "--gripper-kp", "1200", "--gripper-force", "200", "--friction", "5.0",
                "--log-every", "0",
            ]
        command.extend(["--viewer", "--duration", "6"] if viewer else ["--no-viewer"])
        return command

    def start(self, request: dict) -> dict:
        estimate = self.estimate(request)
        if not estimate["gate"]["passed"]:
            raise ValueError("resource gate failed: " + "; ".join(estimate["gate"]["reasons"]))
        if self.thread and self.thread.is_alive():
            raise RuntimeError("an adaptation job is already running")
        if self.training.snapshot()["status"] in {"starting", "running"}:
            raise RuntimeError("the baseline trainer is already using local resources")

        task_id = estimate["task_id"]
        profile = self.PROFILES[estimate["resolved_profile"]]
        method = estimate["method"]
        seed = int(request.get("seed", 0))
        viewer = bool(request.get("viewer", True))
        run_tag = f"adapt_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        dataset_path = ROOT / "data" / "demos" / run_tag
        model_output = ADAPTATION_MODEL_ROOT / task_id / run_tag
        collect_command = self._collect_command(task_id, seed, estimate["episodes"], run_tag, viewer)
        train_command = self._train_command(method, profile, estimate["epochs"], dataset_path, model_output, task_id)
        command_text = subprocess.list2cmdline(collect_command) + "\n" + subprocess.list2cmdline(train_command)
        config = {
            "task_id": task_id,
            "method": method,
            "profile": estimate["resolved_profile"],
            "episodes": estimate["episodes"],
            "epochs": estimate["epochs"],
            "evaluation_episodes": estimate["evaluation_episodes"],
            "seed": seed,
            "viewer": viewer,
            "resource_estimate": estimate,
        }
        run_id = self.ledger.start("adaptation", config, command=command_text)
        self.stop_event.clear()
        with self.lock:
            self.state = AdaptationState(
                run_id=run_id,
                status="starting",
                stage="validate",
                task_id=task_id,
                method=method,
                requested_profile=estimate["requested_profile"],
                resolved_profile=estimate["resolved_profile"],
                episodes=estimate["episodes"],
                epochs=estimate["epochs"],
                evaluation_episodes=estimate["evaluation_episodes"],
                seed=seed,
                viewer=viewer,
                started_at=time.time(),
                dataset_path=str(dataset_path),
                estimated=estimate,
                trainable_params=estimate["trainable_params"],
            )
            self.events = []
            self.logs = [command_text]
            self._append_event("gate", f"resource gate passed: {estimate['resolved_profile']}")
        self.thread = threading.Thread(
            target=self._run,
            args=(collect_command, train_command, dataset_path),
            daemon=True,
            name="low-resource-adaptation",
        )
        self.thread.start()
        return self.snapshot()

    def stop(self) -> dict:
        self.stop_event.set()
        process = self.process
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        with self.lock:
            if self.state.status in {"starting", "running"}:
                self.state.status = "stopped"
                self.state.stage = "stopped"
                self._append_event("stop", "adaptation stopped by operator")
        return self.snapshot()

    def _append_event(self, kind: str, message: str) -> None:
        self.events.append({"time": utc_now(), "kind": kind, "message": message})
        if len(self.events) > 100:
            self.events = self.events[-100:]

    def _set_stage(self, stage: str, total_items: int) -> None:
        with self.lock:
            self.state.stage = stage
            self.state.stage_progress = 0.0
            self.state.current_item = 0
            self.state.total_items = total_items
            self._append_event("stage", stage)

    def _update_resources(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        rss = process_tree_rss_bytes(process.pid)
        if rss <= 0:
            return
        rss_mb = rss / (1024 ** 2)
        with self.lock:
            self.state.process_rss_mb = rss_mb
            self.state.peak_rss_mb = max(self.state.peak_rss_mb, rss_mb)
            self.state.candidate_peak_rss_mb = max(self.state.candidate_peak_rss_mb, rss_mb)

    def _run_process(self, command: list[str], stage: str) -> int:
        self.process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        monitor_done = threading.Event()

        def monitor_resources() -> None:
            while not monitor_done.wait(0.05):
                self._update_resources()

        monitor = threading.Thread(target=monitor_resources, daemon=True, name="adaptation-resource-monitor")
        monitor.start()
        assert self.process.stdout is not None
        try:
            for raw_line in self.process.stdout:
                line = raw_line.rstrip()
                if not line:
                    continue
                self._update_resources()
                with self.lock:
                    self.logs.append(line)
                    if len(self.logs) > 500:
                        self.logs = self.logs[-500:]
                    if self.state.started_at is not None:
                        self.state.elapsed = time.time() - self.state.started_at
                    if stage == "collect":
                        match = self.collection_pattern.search(line)
                        if match:
                            self.state.current_item = int(match.group("episode")) + 1
                            self.state.collection_successes += int(match.group("success").lower() == "true")
                            self.state.stage_progress = self.state.current_item / max(1, self.state.total_items)
                            self.state.progress = (0.03 + 0.17 * self.state.stage_progress) if self.state.mode == "arena" else (0.08 + 0.42 * self.state.stage_progress)
                    elif stage == "train":
                        match = self.metric_pattern.search(line)
                        if match:
                            epoch = int(match.group("epoch"))
                            self.state.current_item = epoch
                            self.state.stage_progress = epoch / max(1, self.state.total_items)
                            if self.state.mode == "arena":
                                span = 0.77 / max(1, len(self.state.candidate_methods))
                                self.state.progress = 0.20 + span * ((self.state.current_candidate_index - 1) + 0.45 * self.state.stage_progress)
                            else:
                                self.state.progress = 0.50 + 0.35 * self.state.stage_progress
                            self.state.train_loss = float(match.group("train"))
                            self.state.val_loss = float(match.group("val"))
                            self.events.append({
                                "time": utc_now(),
                                "kind": "metric",
                                "message": f"epoch {epoch}: train={self.state.train_loss:.6g}, val={self.state.val_loss:.6g}",
                                "epoch": epoch,
                                "train": self.state.train_loss,
                                "val": self.state.val_loss,
                            })
                        if line.startswith("model_path:"):
                            self.state.model_path = line.split(":", 1)[1].strip()
                            if self.state.method == "registry_rgb_skill":
                                self.state.current_item = self.state.total_items
                                self.state.stage_progress = 1.0
                                if self.state.mode == "arena":
                                    span = 0.77 / max(1, len(self.state.candidate_methods))
                                    self.state.progress = 0.20 + span * ((self.state.current_candidate_index - 1) + 0.45)
                                else:
                                    self.state.progress = 0.85
                        if line.startswith("trainable_params:"):
                            self.state.trainable_params = int(line.split(":", 1)[1].strip())
                    elif line.startswith("episode_summary:"):
                        payload = line.split(":", 1)[1].strip()
                        try:
                            summary = json.loads(payload)
                        except json.JSONDecodeError:
                            summary = ast.literal_eval(payload)
                        success = bool(summary.get("task_success", summary.get("success")))
                        target_error = float(summary.get("target_distance", float("nan")))
                        row = {
                            "seed": int(summary.get("seed", 0)),
                            "success": success,
                            "target_error": target_error,
                            "steps": int(summary.get("steps_taken", 0)),
                            "semantic_correct": summary.get("semantic_correct"),
                            "visual_selection_correct": summary.get("visual_selection_correct"),
                        }
                        self.state.evaluation_rows.append(row)
                        self.state.current_item = len(self.state.evaluation_rows)
                        self.state.stage_progress = self.state.current_item / max(1, self.state.total_items)
                        if self.state.mode == "arena":
                            span = 0.77 / max(1, len(self.state.candidate_methods))
                            self.state.progress = 0.20 + span * ((self.state.current_candidate_index - 1) + 0.45 + 0.55 * self.state.stage_progress)
                        else:
                            self.state.progress = 0.85 + 0.12 * self.state.stage_progress
                        self.state.evaluation_successes = sum(int(item["success"]) for item in self.state.evaluation_rows)
                        self.state.evaluation_success_rate = self.state.evaluation_successes / self.state.current_item
                        self.state.evaluation_success = self.state.evaluation_successes == self.state.current_item
                        finite_errors = [item["target_error"] for item in self.state.evaluation_rows if np.isfinite(item["target_error"])]
                        self.state.evaluation_target_error = target_error
                        self.state.evaluation_mean_target_error = float(np.mean(finite_errors)) if finite_errors else None
                        self.state.evaluation_steps += row["steps"]
                        self._append_event(
                            "evaluation",
                            f"holdout seed={row['seed']} success={success}, target_error={target_error * 1000:.1f} mm",
                        )
                if self.stop_event.is_set() and self.process.poll() is None:
                    self.process.terminate()
            return self.process.wait()
        finally:
            monitor_done.set()
            monitor.join(timeout=1)

    def _run(self, collect_command: list[str], train_command: list[str], dataset_path: Path) -> None:
        run_id = self.state.run_id
        try:
            with self.lock:
                self.state.status = "running"
                self.state.progress = 0.05
            self._set_stage("collect", self.state.episodes)
            if self._run_process(collect_command, "collect") != 0:
                if self.stop_event.is_set():
                    return
                raise RuntimeError("demonstration collection failed")
            summary = json.loads((dataset_path / "summary.json").read_text(encoding="utf-8"))
            with self.lock:
                self.state.collection_successes = int(summary["successes"])
                self.state.progress = 0.50
                self._append_event("dataset", f"{summary['successes']}/{summary['episodes']} successful demonstrations")
            if int(summary["successes"]) < 2:
                raise RuntimeError("fewer than two successful demonstrations; training was not started")
            if self.stop_event.is_set():
                return
            self._set_stage("train", self.state.epochs)
            if self._run_process(train_command, "train") != 0:
                if self.stop_event.is_set():
                    return
                raise RuntimeError("adapter training failed")
            model_path = Path(str(self.state.model_path or ""))
            if not model_path.is_file():
                raise RuntimeError("trainer completed without a model artifact")
            evaluation_seed = self.state.seed + self.state.episodes + 1000
            evaluation_command = self._evaluation_command(
                self.state.method,
                model_path,
                str(self.state.task_id),
                evaluation_seed,
                self.state.evaluation_episodes,
                self.state.viewer,
            )
            with self.lock:
                self.state.evaluation_seed = evaluation_seed
                self.logs.append(subprocess.list2cmdline(evaluation_command))
            self._set_stage("evaluate", self.state.evaluation_episodes)
            if self._run_process(evaluation_command, "evaluate") != 0:
                if self.stop_event.is_set():
                    return
                raise RuntimeError("holdout evaluation failed to execute")
            if len(self.state.evaluation_rows) != self.state.evaluation_episodes:
                raise RuntimeError("holdout evaluation produced no episode summary")
            with self.lock:
                self.state.evaluation_success = self.state.evaluation_successes == self.state.evaluation_episodes
                self.state.evaluation_target_error = self.state.evaluation_mean_target_error
                self._append_event(
                    "promotion",
                    f"holdout {self.state.evaluation_successes}/{self.state.evaluation_episodes}, mean_error={self.state.evaluation_mean_target_error * 1000:.1f} mm",
                )
                self.state.status = "completed"
                self.state.stage = "complete"
                self.state.progress = 1.0
                self.state.stage_progress = 1.0
                self._append_event("complete", "dataset and lightweight adapter are ready")
        except Exception as error:
            with self.lock:
                self.state.status = "failed"
                self.state.stage = "failed"
                self.state.error = str(error)
                self._append_event("error", str(error))
        finally:
            with self.lock:
                if self.stop_event.is_set() and self.state.status not in {"completed", "failed"}:
                    self.state.status = "stopped"
                    self.state.stage = "stopped"
                if self.state.started_at is not None:
                    self.state.elapsed = time.time() - self.state.started_at
            snapshot = self.snapshot()
            run_dir = ADAPTATION_ROOT / str(run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            summary_path = run_dir / "adaptation.json"
            summary_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
            assets = {"adaptation_summary": artifact_url(summary_path)}
            dataset_summary = dataset_path / "summary.json"
            if dataset_summary.is_file():
                assets["dataset_summary"] = str(dataset_summary.relative_to(ROOT))
            self.ledger.finish(
                run_id,
                snapshot["status"],
                metrics={
                    "task_id": snapshot["task_id"],
                    "method": snapshot["method"],
                    "profile": snapshot["resolved_profile"],
                    "demonstration_successes": snapshot["collection_successes"],
                    "episodes": snapshot["episodes"],
                    "epochs": snapshot["epochs"] if snapshot["status"] == "completed" else 0,
                    "train_loss": snapshot["train_loss"],
                    "val_loss": snapshot["val_loss"],
                    "trainable_params": snapshot["trainable_params"],
                    "evaluation_seed": snapshot["evaluation_seed"],
                    "evaluation_episodes": snapshot["evaluation_episodes"],
                    "evaluation_successes": snapshot["evaluation_successes"],
                    "evaluation_success_rate": snapshot["evaluation_success_rate"],
                    "evaluation_success": snapshot["evaluation_success"],
                    "evaluation_target_error": snapshot["evaluation_target_error"],
                    "evaluation_mean_target_error": snapshot["evaluation_mean_target_error"],
                    "evaluation_steps": snapshot["evaluation_steps"],
                    "evaluation_rows": snapshot["evaluation_rows"],
                    "peak_rss_mb": snapshot["peak_rss_mb"],
                    "elapsed": snapshot["elapsed"],
                },
                artifact=snapshot["model_path"],
                assets=assets,
            )


@dataclass
class BenchmarkState:
    status: str = "idle"
    benchmark_id: str | None = None
    tasks: list[str] | None = None
    policies: list[str] | None = None
    policy: str = "rgb_grounded"
    seed_start: int = 100
    seeds_per_task: int = 1
    speed: float = 3.0
    total_episodes: int = 0
    completed_episodes: int = 0
    successes: int = 0
    failures: int = 0
    success_rate: float = 0.0
    mean_target_error: float | None = None
    current_task: str | None = None
    current_seed: int | None = None
    current_policy: str | None = None
    progress: float = 0.0
    elapsed: float = 0.0
    results: list[dict] | None = None
    policy_metrics: list[dict] | None = None
    task_metrics: list[dict] | None = None
    paired_summary: dict | None = None
    error: str | None = None


class BenchmarkManager:
    def __init__(self, simulation: SimulationManager, ledger: ExperimentLedger) -> None:
        self.simulation = simulation
        self.ledger = ledger
        self.lock = threading.RLock()
        self.state = BenchmarkState(tasks=[], policies=[], results=[], policy_metrics=[], task_metrics=[], paired_summary={})
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def snapshot(self) -> dict:
        with self.lock:
            return asdict(self.state)

    def start(self, request: dict) -> dict:
        if self.thread and self.thread.is_alive():
            raise RuntimeError("a benchmark is already running")
        simulation_status = self.simulation.snapshot()["status"]
        if simulation_status in {"starting", "running", "paused"}:
            raise RuntimeError("stop the current simulation before starting a benchmark")
        tasks = list(TASKS) if "tasks" not in request else request.get("tasks")
        if not isinstance(tasks, list) or not tasks or any(task not in TASKS for task in tasks):
            raise ValueError("benchmark tasks are invalid")
        tasks = list(dict.fromkeys(str(task) for task in tasks))
        requested_policies = request.get("policies")
        if requested_policies is None:
            requested_policies = [request.get("policy", "rgb_grounded")]
        if not isinstance(requested_policies, list) or not requested_policies:
            raise ValueError("benchmark policies are invalid")
        policies = list(dict.fromkeys(str(policy) for policy in requested_policies))
        seed_start = int(request.get("seed_start", 100))
        seeds_per_task = int(request.get("seeds_per_task", 1))
        speed = float(request.get("speed", 3.0))
        protocol_id = str(request.get("protocol_id", "")).strip() or None
        if any(policy not in {"rgb_grounded", "structured_state"} for policy in policies):
            raise ValueError("unsupported benchmark policy")
        if not 1 <= seeds_per_task <= 5:
            raise ValueError("seeds_per_task must be between 1 and 5")
        if len(tasks) * seeds_per_task * len(policies) > 40:
            raise ValueError("a benchmark may contain at most 40 episodes")
        if not 0.25 <= speed <= 3.0:
            raise ValueError("speed must be between 0.25 and 3.0")
        config = {
            "tasks": tasks,
            "policies": policies,
            "policy": policies[0],
            "seed_start": seed_start,
            "seeds_per_task": seeds_per_task,
            "speed": speed,
            "paired": len(policies) == 2,
            "protocol_id": protocol_id,
        }
        payload = json.dumps(config, ensure_ascii=False, separators=(",", ":")).replace("'", "''")
        command = f"Invoke-RestMethod 'http://127.0.0.1:8050/api/benchmark/start' -Method Post -ContentType 'application/json' -Body '{payload}'"
        run_id = self.ledger.start("benchmark", config, command=command)
        with self.lock:
            self.state = BenchmarkState(
                status="starting", benchmark_id=run_id, tasks=tasks, policies=policies, policy=policies[0],
                seed_start=seed_start, seeds_per_task=seeds_per_task, speed=speed,
                total_episodes=len(tasks) * seeds_per_task * len(policies), results=[],
                policy_metrics=[], task_metrics=[], paired_summary={},
            )
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, args=(config, run_id), daemon=True, name="benchmark-batch")
        self.thread.start()
        return self.snapshot()

    def stop(self) -> dict:
        self.stop_event.set()
        self.simulation.stop(wait=True)
        with self.lock:
            if self.state.status in {"starting", "running"}:
                self.state.status = "stopped"
        return self.snapshot()

    @staticmethod
    def _aggregate(results: list[dict], policies: list[str]) -> tuple[list[dict], list[dict], dict]:
        policy_metrics = []
        task_metrics = []
        for policy in policies:
            policy_rows = [row for row in results if row["policy"] == policy]
            successes = sum(row["success"] is True for row in policy_rows)
            errors = [row["target_distance"] for row in policy_rows if isinstance(row["target_distance"], (int, float))]
            grounding_errors = [row["rgb_grounding_error"] for row in policy_rows if isinstance(row["rgb_grounding_error"], (int, float))]
            low, high = wilson_interval(successes, len(policy_rows))
            policy_metrics.append({
                "policy": policy,
                "episodes": len(policy_rows),
                "successes": successes,
                "success_rate": successes / len(policy_rows) if policy_rows else 0.0,
                "ci95_low": low,
                "ci95_high": high,
                "mean_target_error": sum(errors) / len(errors) if errors else None,
                "mean_grounding_error": sum(grounding_errors) / len(grounding_errors) if grounding_errors else None,
            })
            for task in dict.fromkeys(row["task"] for row in policy_rows):
                rows = [row for row in policy_rows if row["task"] == task]
                task_successes = sum(row["success"] is True for row in rows)
                task_low, task_high = wilson_interval(task_successes, len(rows))
                task_errors = [row["target_distance"] for row in rows if isinstance(row["target_distance"], (int, float))]
                task_metrics.append({
                    "task": task,
                    "policy": policy,
                    "episodes": len(rows),
                    "successes": task_successes,
                    "success_rate": task_successes / len(rows) if rows else 0.0,
                    "ci95_low": task_low,
                    "ci95_high": task_high,
                    "mean_target_error": sum(task_errors) / len(task_errors) if task_errors else None,
                })

        paired = {"pairs": 0, "both_success": 0, "rgb_only": 0, "state_only": 0, "both_fail": 0, "success_rate_delta": None}
        if {"rgb_grounded", "structured_state"}.issubset(policies):
            indexed = {(row["task"], row["seed"], row["policy"]): row for row in results}
            pair_keys = sorted({(row["task"], row["seed"]) for row in results})
            for task, seed in pair_keys:
                rgb = indexed.get((task, seed, "rgb_grounded"))
                state = indexed.get((task, seed, "structured_state"))
                if rgb is None or state is None:
                    continue
                paired["pairs"] += 1
                if rgb["success"] is True and state["success"] is True:
                    paired["both_success"] += 1
                elif rgb["success"] is True:
                    paired["rgb_only"] += 1
                elif state["success"] is True:
                    paired["state_only"] += 1
                else:
                    paired["both_fail"] += 1
            rgb_metric = next((row for row in policy_metrics if row["policy"] == "rgb_grounded"), None)
            state_metric = next((row for row in policy_metrics if row["policy"] == "structured_state"), None)
            if rgb_metric and state_metric:
                paired["success_rate_delta"] = rgb_metric["success_rate"] - state_metric["success_rate"]
        return policy_metrics, task_metrics, paired

    def _run(self, config: dict, run_id: str) -> None:
        started = time.monotonic()
        try:
            with self.lock:
                self.state.status = "running"
            for task_index, task in enumerate(config["tasks"]):
                for offset in range(config["seeds_per_task"]):
                    seed = config["seed_start"] + task_index * config["seeds_per_task"] + offset
                    for policy in config["policies"]:
                        if self.stop_event.is_set():
                            raise SimulationCancelled("benchmark stopped")
                        with self.lock:
                            self.state.current_task = task
                            self.state.current_seed = seed
                            self.state.current_policy = policy
                        self.simulation.start(
                            {
                                "task": task,
                                "policy": policy,
                                "complexity": TASKS[task]["complexity"],
                                "seed": seed,
                                "speed": config["speed"],
                            },
                            parent_id=run_id,
                        )
                        thread = self.simulation.thread
                        if thread:
                            while thread.is_alive():
                                if self.stop_event.wait(0.1):
                                    self.simulation.stop(wait=True)
                                    raise SimulationCancelled("benchmark stopped")
                        episode = self.simulation.snapshot()
                        result = {
                            "run_id": episode["run_id"],
                            "task": task,
                            "seed": seed,
                            "policy": policy,
                            "status": episode["status"],
                            "success": episode["success"],
                            "target_distance": episode["target_distance"],
                            "rgb_grounding_error": episode["rgb_grounding_error"],
                            "elapsed": episode["elapsed"],
                            "source_name": episode["source_name"],
                            "assets": {
                                "initial_top": episode["initial_top_url"],
                                "final_front": episode["final_front_url"],
                            },
                        }
                        with self.lock:
                            assert self.state.results is not None
                            self.state.results.append(result)
                            self.state.completed_episodes += 1
                            if result["success"] is True:
                                self.state.successes += 1
                            else:
                                self.state.failures += 1
                            errors = [item["target_distance"] for item in self.state.results if isinstance(item["target_distance"], (int, float))]
                            self.state.success_rate = self.state.successes / self.state.completed_episodes
                            self.state.mean_target_error = sum(errors) / len(errors) if errors else None
                            self.state.policy_metrics, self.state.task_metrics, self.state.paired_summary = self._aggregate(self.state.results, config["policies"])
                            self.state.progress = self.state.completed_episodes / self.state.total_episodes
                            self.state.elapsed = time.monotonic() - started
            with self.lock:
                self.state.status = "completed"
                self.state.current_task = None
                self.state.current_seed = None
                self.state.current_policy = None
                self.state.progress = 1.0
        except SimulationCancelled:
            with self.lock:
                self.state.status = "stopped"
                self.state.elapsed = time.monotonic() - started
        except Exception as error:
            with self.lock:
                self.state.status = "failed"
                self.state.error = str(error)
                self.state.elapsed = time.monotonic() - started
        finally:
            snapshot = self.snapshot()
            run_dir = RUN_ARTIFACT_ROOT / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            report_path = run_dir / "benchmark.json"
            report_path.write_text(
                json.dumps({"run_id": run_id, "kind": "benchmark", "config": config, "results": snapshot}, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            self.ledger.finish(
                run_id,
                snapshot["status"],
                metrics={
                    "episodes": snapshot["completed_episodes"],
                    "successes": snapshot["successes"],
                    "failures": snapshot["failures"],
                    "success_rate": snapshot["success_rate"],
                    "mean_target_error": snapshot["mean_target_error"],
                    "policy_metrics": snapshot["policy_metrics"],
                    "task_metrics": snapshot["task_metrics"],
                    "paired_summary": snapshot["paired_summary"],
                    "elapsed": snapshot["elapsed"],
                },
                artifact=str(report_path.relative_to(ROOT)),
            )


class StudyRegistry:
    FRAMEWORK = "TRACE-1.0"

    def __init__(self, path: Path, ledger: ExperimentLedger, benchmark: BenchmarkManager) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger = ledger
        self.benchmark = benchmark
        self.lock = threading.RLock()
        self.studies: dict[str, dict] = {}
        self.launches: dict[str, list[dict]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            study_id = event.get("study_id")
            if not study_id:
                continue
            if event.get("event") == "protocol_locked":
                self.studies[study_id] = {key: value for key, value in event.items() if key != "event"}
            elif event.get("event") == "benchmark_launched":
                self.launches.setdefault(study_id, []).append({key: value for key, value in event.items() if key != "event"})

    def _append(self, event: dict) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

    @staticmethod
    def _validated_protocol(request: dict) -> dict:
        title = str(request.get("title", "")).strip()
        hypothesis = str(request.get("hypothesis", "")).strip()
        tasks = request.get("tasks")
        if not title or len(title) > 100:
            raise ValueError("study title must contain 1 to 100 characters")
        if not hypothesis or len(hypothesis) > 500:
            raise ValueError("hypothesis must contain 1 to 500 characters")
        if not isinstance(tasks, list) or not tasks or any(task not in TASKS for task in tasks):
            raise ValueError("study tasks are invalid")
        tasks = list(dict.fromkeys(str(task) for task in tasks))
        policies = ["rgb_grounded", "structured_state"]
        seed_start = int(request.get("seed_start", 1000))
        seeds_per_task = int(request.get("seeds_per_task", 3))
        speed = float(request.get("speed", 3.0))
        if not 0 <= seed_start <= 99999:
            raise ValueError("seed_start must be between 0 and 99999")
        if not 1 <= seeds_per_task <= 5:
            raise ValueError("seeds_per_task must be between 1 and 5")
        if len(tasks) * seeds_per_task * len(policies) > 40:
            raise ValueError("a TRACE study may contain at most 40 episodes")
        if not 0.25 <= speed <= 3.0:
            raise ValueError("speed must be between 0.25 and 3.0")

        criteria_request = request.get("criteria") or {}
        criteria = {
            "min_success_rate": float(criteria_request.get("min_success_rate", 0.8)),
            "max_target_error_mm": float(criteria_request.get("max_target_error_mm", 20.0)),
            "max_grounding_error_mm": float(criteria_request.get("max_grounding_error_mm", 15.0)),
            "max_ci_width": float(criteria_request.get("max_ci_width", 0.5)),
        }
        if not 0.0 <= criteria["min_success_rate"] <= 1.0:
            raise ValueError("min_success_rate must be between 0 and 1")
        if not 1.0 <= criteria["max_target_error_mm"] <= 100.0:
            raise ValueError("max_target_error_mm must be between 1 and 100")
        if not 1.0 <= criteria["max_grounding_error_mm"] <= 100.0:
            raise ValueError("max_grounding_error_mm must be between 1 and 100")
        if not 0.05 <= criteria["max_ci_width"] <= 1.0:
            raise ValueError("max_ci_width must be between 0.05 and 1")
        return {
            "title": title,
            "hypothesis": hypothesis,
            "tasks": tasks,
            "policies": policies,
            "seed_start": seed_start,
            "seeds_per_task": seeds_per_task,
            "speed": speed,
            "expected_pairs": len(tasks) * seeds_per_task,
            "expected_episodes": len(tasks) * seeds_per_task * len(policies),
            "criteria": criteria,
        }

    def create(self, request: dict) -> dict:
        protocol = self._validated_protocol(request)
        canonical = json.dumps(protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        protocol_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        study_id = f"stu-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{protocol_hash[:6]}-{uuid.uuid4().hex[:4]}"
        record = {
            "study_id": study_id,
            "framework": self.FRAMEWORK,
            "status": "locked",
            "locked_at": utc_now(),
            "protocol_hash": protocol_hash,
            "protocol": protocol,
        }
        with self.lock:
            self.studies[study_id] = record
            self._append({"event": "protocol_locked", **record})
        return self.get(study_id)

    def _evaluation(self, study: dict, launches: list[dict]) -> dict:
        protocol = study["protocol"]
        latest_launch = launches[-1] if launches else None
        benchmark_record = self.ledger.get(latest_launch["benchmark_id"]) if latest_launch else None
        benchmark_status = benchmark_record.get("status") if benchmark_record else None
        metrics = benchmark_record.get("metrics", {}) if benchmark_record else {}
        children = self.ledger.children(benchmark_record["run_id"]) if benchmark_record else []
        policy_metrics = metrics.get("policy_metrics") or []
        paired = metrics.get("paired_summary") or {}
        complete = benchmark_status == "completed"

        def gate(gate_id: str, label: str, passed: bool | None, observed: object, threshold: object = None) -> dict:
            return {
                "id": gate_id,
                "label": label,
                "status": "pending" if passed is None else "pass" if passed else "fail",
                "observed": observed,
                "threshold": threshold,
            }

        expected_episodes = protocol["expected_episodes"]
        expected_pairs = protocol["expected_pairs"]
        all_assets = bool(children) and all({"initial_top", "final_front"} <= set((row.get("assets") or {}).keys()) for row in children)
        success_floor = complete and len(policy_metrics) == 2 and all(
            row.get("success_rate", 0.0) >= protocol["criteria"]["min_success_rate"] for row in policy_metrics
        )
        target_precision = complete and len(policy_metrics) == 2 and all(
            isinstance(row.get("mean_target_error"), (int, float))
            and row["mean_target_error"] * 1000 <= protocol["criteria"]["max_target_error_mm"]
            for row in policy_metrics
        )
        rgb_metric = next((row for row in policy_metrics if row.get("policy") == "rgb_grounded"), None)
        grounding_precision = complete and rgb_metric is not None and isinstance(rgb_metric.get("mean_grounding_error"), (int, float)) and (
            rgb_metric["mean_grounding_error"] * 1000 <= protocol["criteria"]["max_grounding_error_mm"]
        )
        ci_widths = [row.get("ci95_high", 0.0) - row.get("ci95_low", 0.0) for row in policy_metrics]
        statistical_precision = complete and len(ci_widths) == 2 and all(width <= protocol["criteria"]["max_ci_width"] for width in ci_widths)

        gates = [
            gate("protocol", "Protocol fingerprint locked", True, study["protocol_hash"][:12]),
            gate("execution", "Expected episodes completed", None if not benchmark_record or benchmark_status in {"running", "starting"} else complete and metrics.get("episodes") == expected_episodes, metrics.get("episodes", 0), expected_episodes),
            gate("pairing", "Same-seed pairs complete", None if not complete else paired.get("pairs") == expected_pairs, paired.get("pairs", 0), expected_pairs),
            gate("artifacts", "Visual evidence archived", None if not complete else len(children) == expected_episodes and all_assets, len(children), expected_episodes),
            gate("success", "Policy success floor", None if not complete else success_floor, [row.get("success_rate") for row in policy_metrics], protocol["criteria"]["min_success_rate"]),
            gate("target_error", "Target error ceiling", None if not complete else target_precision, [None if row.get("mean_target_error") is None else row["mean_target_error"] * 1000 for row in policy_metrics], protocol["criteria"]["max_target_error_mm"]),
            gate("grounding", "RGB grounding error ceiling", None if not complete else grounding_precision, None if rgb_metric is None or rgb_metric.get("mean_grounding_error") is None else rgb_metric["mean_grounding_error"] * 1000, protocol["criteria"]["max_grounding_error_mm"]),
            gate("uncertainty", "Confidence interval width", None if not complete else statistical_precision, ci_widths, protocol["criteria"]["max_ci_width"]),
        ]
        if not latest_launch:
            verdict = "locked"
        elif benchmark_status in {"running", "starting"}:
            verdict = "executing"
        elif benchmark_status != "completed":
            verdict = "execution_failed"
        elif all(item["status"] == "pass" for item in gates):
            verdict = "ready_to_report"
        else:
            verdict = "needs_more_evidence"
        stages = {
            "target": "complete",
            "register": "complete",
            "acquire": "complete" if complete else "active" if latest_launch else "pending",
            "check": "complete" if complete else "pending",
            "explain": "complete" if verdict == "ready_to_report" else "blocked" if complete else "pending",
        }
        return {
            "verdict": verdict,
            "stages": stages,
            "gates": gates,
            "latest_benchmark_id": None if benchmark_record is None else benchmark_record["run_id"],
            "benchmark_status": benchmark_status,
            "policy_metrics": policy_metrics,
            "paired_summary": paired,
            "child_episodes": len(children),
        }

    def get(self, study_id: str) -> dict:
        with self.lock:
            study = self.studies.get(study_id)
            launches = list(self.launches.get(study_id, []))
        if study is None:
            raise ValueError("study not found")
        result = json.loads(json.dumps(study))
        result["launches"] = launches
        result["evaluation"] = self._evaluation(result, launches)
        return result

    def list(self) -> list[dict]:
        with self.lock:
            ids = list(self.studies)
        rows = [self.get(study_id) for study_id in ids]
        rows.sort(key=lambda row: row.get("locked_at") or "", reverse=True)
        return rows

    def summary(self) -> dict:
        rows = self.list()
        verdicts = {key: 0 for key in ["locked", "executing", "ready_to_report", "needs_more_evidence", "execution_failed"]}
        for row in rows:
            verdict = row["evaluation"]["verdict"]
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
        return {"total": len(rows), "verdicts": verdicts, "framework": self.FRAMEWORK}

    def launch(self, study_id: str) -> dict:
        study = self.get(study_id)
        protocol = study["protocol"]
        benchmark = self.benchmark.start({
            "tasks": protocol["tasks"],
            "policies": protocol["policies"],
            "seed_start": protocol["seed_start"],
            "seeds_per_task": protocol["seeds_per_task"],
            "speed": protocol["speed"],
            "protocol_id": study_id,
        })
        launch = {
            "study_id": study_id,
            "benchmark_id": benchmark["benchmark_id"],
            "launched_at": utc_now(),
            "protocol_hash": study["protocol_hash"],
        }
        with self.lock:
            self.launches.setdefault(study_id, []).append(launch)
            self._append({"event": "benchmark_launched", **launch})
        return {"study": self.get(study_id), "benchmark": benchmark}

    def memo_markdown(self, study_id: str) -> str:
        study = self.get(study_id)
        protocol = study["protocol"]
        evaluation = study["evaluation"]
        lines = [
            f"# TRACE Decision Memo: {study['study_id']}",
            "",
            f"**Title / 标题:** {protocol['title']}",
            "",
            f"**Hypothesis / 假设:** {protocol['hypothesis']}",
            "",
            f"**Protocol fingerprint / 协议指纹:** `{study['protocol_hash']}`",
            "",
            f"**Verdict / 判定:** `{evaluation['verdict']}`",
            "",
            "## Protocol / 协议",
            "",
            "```json",
            json.dumps(protocol, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Evidence gates / 证据门控",
            "",
            "| Gate | Status | Observed | Threshold |",
            "|---|---|---|---|",
        ]
        for item in evaluation["gates"]:
            lines.append(f"| {item['label']} | {item['status']} | `{json.dumps(item['observed'], ensure_ascii=False)}` | `{json.dumps(item['threshold'], ensure_ascii=False)}` |")
        lines.extend([
            "",
            "## Interpretation / 解释",
            "",
            "`ready_to_report` means every pre-registered gate passed. `needs_more_evidence` means the run is valid but at least one declared threshold was not met; the protocol and negative result remain archived.",
            "",
            "`ready_to_report` 表示全部预注册门槛通过；`needs_more_evidence` 表示实验有效，但至少一个预设阈值未达到，协议与负结果仍被保留。",
            "",
            "## Scope boundary / 结论边界",
            "",
            "MuJoCo-only. RGB-grounded runtime planning does not use simulator object positions; state is used only for offline diagnostics and scoring.",
            "",
        ])
        return "\n".join(lines)


class ResearchPortfolio:
    FRAMEWORK = "TRACE-PORTFOLIO-1.0"
    SOURCES = {
        "final_audit": FINAL_AUDIT_PATH,
        "method_registry": EXPERIMENT_VERSIONS_PATH,
        "resource_audit": MODEL_RESOURCE_PATH,
        "video_audit": VIDEO_AUDIT_PATH,
        "trace_registry": STUDY_REGISTRY_PATH,
    }

    def __init__(self, studies: StudyRegistry) -> None:
        self.studies = studies

    @staticmethod
    def _fraction(value: str) -> tuple[int, int] | None:
        match = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", value or "")
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    @staticmethod
    def _sha256(path: Path) -> str | None:
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

    def _source_catalogue(self) -> list[dict]:
        rows = []
        for source_id, path in self.SOURCES.items():
            exists = path.is_file()
            rows.append({
                "id": source_id,
                "path": str(path.relative_to(ROOT)),
                "exists": exists,
                "bytes": path.stat().st_size if exists else 0,
                "sha256": self._sha256(path),
                "url": f"/api/portfolio/sources/{source_id}" if exists else None,
            })
        return rows

    @staticmethod
    def _method_lifecycle(row: dict) -> str:
        stage = row.get("stage", "")
        if stage == "scripted_oracle":
            return "reference"
        if stage in {"structured_control_baseline", "data_verification"}:
            return "control_or_data"
        if any(token in stage for token in ("action_head", "peft", "vlm")):
            return "lightweight_probe"
        return "learned_baseline"

    def _methods(self, resources: list[dict], final_audit: dict) -> list[dict]:
        methods = []
        for row in resources:
            train = self._fraction(row.get("train_range_success", ""))
            heldout = self._fraction(row.get("heldout_success", ""))
            methods.append({
                "id": row["version"],
                "name": row["method"],
                "stage": row["stage"],
                "lifecycle": self._method_lifecycle(row),
                "outcome": "pilot_positive" if heldout and heldout[0] > 0 else "negative",
                "trainable_params": int(row["trainable_params"]) if row.get("trainable_params", "").isdigit() else None,
                "train_time_seconds": float(row["train_time_seconds"]) if row.get("train_time_seconds") else None,
                "peak_vram_mb": float(row["peak_vram_mb"]) if row.get("peak_vram_mb") else None,
                "train_success": None if train is None else {"successes": train[0], "episodes": train[1]},
                "heldout_success": None if heldout is None else {"successes": heldout[0], "episodes": heldout[1]},
                "artifact": row.get("artifact"),
                "boundary": row.get("note", ""),
                "evidence_ids": ["method_registry", "resource_audit", "video_audit"],
            })

        pooled = final_audit["v4_replication"]["pooled_descriptive"]
        methods.append({
            "id": "frozen_clip_rgb_structured_controller",
            "name": "Frozen CLIP intent + RGB geometry + structured execution",
            "stage": "retained_system",
            "lifecycle": "retained",
            "outcome": "replicated",
            "trainable_params": 0,
            "train_time_seconds": 0.0,
            "peak_vram_mb": None,
            "train_success": None,
            "heldout_success": {"successes": pooled["successes"], "episodes": pooled["episodes"]},
            "artifact": str(FINAL_MODEL_PATH.relative_to(ROOT)),
            "boundary": final_audit["scope"],
            "evidence_ids": ["final_audit", "video_audit"],
        })
        rejected = final_audit["rejected_candidates"]["contact_monitor_early_regrasp"]
        methods.append({
            "id": "contact_monitor_early_regrasp",
            "name": "Early contact-monitor regrasp candidate",
            "stage": "closed_loop_intervention",
            "lifecycle": "rejected",
            "outcome": "regressed",
            "trainable_params": None,
            "train_time_seconds": None,
            "peak_vram_mb": None,
            "train_success": None,
            "heldout_success": {"successes": rejected["candidate_success"][0], "episodes": rejected["candidate_success"][1]},
            "artifact": "docs/contact_phase_monitor_heldout_v1_analysis.md",
            "boundary": rejected["decision"],
            "evidence_ids": ["final_audit"],
        })
        return methods

    def _claims(self, final_audit: dict, methods: list[dict]) -> list[dict]:
        pooled = final_audit["v4_replication"]["pooled_descriptive"]
        rejected = final_audit["rejected_candidates"]["contact_monitor_early_regrasp"]
        lora = next(row for row in methods if row["id"] == "lora_action_head_lite_v1")
        baseline_ids = ["linear_bc_v1", "knn_bc_v1", "mlp_bc_v1", "act_lite_chunk_bc_v1", "diffusion_policy_lite_v1"]
        baselines = [next(row for row in methods if row["id"] == method_id) for method_id in baseline_ids]
        claims = [
            {
                "id": "claim-retained-controller",
                "status": "reportable",
                "readiness": 100,
                "title": {"zh": "最终控制器在两个独立队列中可复现", "en": "The retained controller reproduces across two independent cohorts"},
                "allowed": {
                    "zh": "在四个 MuJoCo 桌面任务及既定接触域内，冻结 CLIP 意图、RGB 几何与结构化执行组成的控制器在两个独立 seed 队列中取得 278/288 次严格成功（96.5%，Wilson 95% CI 93.7%–98.1%）。",
                    "en": "Within four MuJoCo tabletop tasks and the declared contact domains, frozen CLIP intent, RGB geometry and structured execution achieved 278/288 strict successes across two disjoint seed cohorts (96.5%, Wilson 95% CI 93.7%–98.1%).",
                },
                "blocked": {"zh": "不得表述为端到端 VLA、OpenVLA LoRA 或真实机械臂迁移。", "en": "Do not describe this as end-to-end VLA, OpenVLA LoRA or physical-robot transfer."},
                "metrics": [
                    {"label": "Strict success", "value": f"{pooled['successes']}/{pooled['episodes']}"},
                    {"label": "Semantic selection", "value": f"{pooled['semantic_correct']}/{pooled['episodes']}"},
                    {"label": "First attempt", "value": f"{pooled['first_attempt_success']}/{pooled['episodes']}"},
                ],
                "gates": [
                    {"label": "Two disjoint cohorts", "status": "pass"},
                    {"label": "n ≥ 100", "status": "pass"},
                    {"label": "95% interval reported", "status": "pass"},
                    {"label": "MuJoCo scope declared", "status": "pass"},
                ],
                "evidence_ids": ["final_audit", "video_audit"],
                "next_action": {"zh": "作为论文主要结果；继续保持范围限定。", "en": "Use as the primary thesis result while preserving the scope boundary."},
            },
            {
                "id": "claim-baseline-limit",
                "status": "bounded",
                "readiness": 75,
                "title": {"zh": "常规模仿学习在小样本闭环中表现不稳定", "en": "Conventional imitation learning was unstable in the small held-out pilot"},
                "allowed": {
                    "zh": "在同一蓝块到蓝区任务的 5-episode 留出试验中，Linear BC、kNN、MLP、ACT-lite 与 Diffusion Policy-lite 的结果介于 0/5 与 1/5。",
                    "en": "In the five-episode held-out pilot for blue-cube-to-blue-target, Linear BC, kNN, MLP, ACT-lite and Diffusion Policy-lite scored between 0/5 and 1/5.",
                },
                "blocked": {"zh": "样本量只有 5，不得推广为这些算法在一般机器人任务上无效。", "en": "With n=5, do not generalise this into a claim that these algorithms fail in robotics broadly."},
                "metrics": [{"label": row["name"], "value": f"{row['heldout_success']['successes']}/{row['heldout_success']['episodes']}"} for row in baselines],
                "gates": [
                    {"label": "Same registered task", "status": "pass"},
                    {"label": "Closed-loop metric", "status": "pass"},
                    {"label": "Adequate statistical power", "status": "fail"},
                    {"label": "Scope caveat present", "status": "pass"},
                ],
                "evidence_ids": ["method_registry", "resource_audit", "video_audit"],
                "next_action": {"zh": "作为限定性的负结果，不与 288-episode 结果做因果排名。", "en": "Report as bounded negative evidence, not as a causal ranking against the 288-episode result."},
            },
            {
                "id": "claim-local-peft",
                "status": "bounded",
                "readiness": 75,
                "title": {"zh": "LoRA-style 代理训练很轻，但闭环收益有限", "en": "The LoRA-style proxy is lightweight but offers limited closed-loop benefit"},
                "allowed": {
                    "zh": f"本地 LoRA-style action-head 仅更新 {lora['trainable_params']:,} 个参数，训练 {lora['train_time_seconds']:.1f} 秒；其留出成功率为 {lora['heldout_success']['successes']}/{lora['heldout_success']['episodes']}。",
                    "en": f"The local LoRA-style action head updates {lora['trainable_params']:,} parameters in {lora['train_time_seconds']:.1f} seconds, with {lora['heldout_success']['successes']}/{lora['heldout_success']['episodes']} held-out success.",
                },
                "blocked": {"zh": "这是本地 PEFT 代理，不是预训练机器人 VLA 或 OpenVLA LoRA。", "en": "This is a local PEFT proxy, not a pretrained robot VLA or OpenVLA LoRA run."},
                "metrics": [
                    {"label": "Trainable", "value": f"{lora['trainable_params']:,}"},
                    {"label": "Train time", "value": f"{lora['train_time_seconds']:.1f}s"},
                    {"label": "Held-out", "value": f"{lora['heldout_success']['successes']}/{lora['heldout_success']['episodes']}"},
                ],
                "gates": [
                    {"label": "Resource log available", "status": "pass"},
                    {"label": "Closed-loop result available", "status": "pass"},
                    {"label": "Pretrained robot VLA used", "status": "fail"},
                    {"label": "Proxy boundary declared", "status": "pass"},
                ],
                "evidence_ids": ["resource_audit", "method_registry"],
                "next_action": {"zh": "用于说明轻量适配的资源—性能权衡。", "en": "Use to explain the resource–performance trade-off of lightweight adaptation."},
            },
            {
                "id": "claim-contact-monitor",
                "status": "negative",
                "readiness": 100,
                "title": {"zh": "离线分类提升不保证闭环控制提升", "en": "Offline classification gains do not guarantee closed-loop control gains"},
                "allowed": {
                    "zh": f"接触监测器离线平衡准确率为 {rejected['offline_balanced_accuracy'] * 100:.2f}%，但闭环从 {rejected['v4_success'][0]}/{rejected['v4_success'][1]} 回退到 {rejected['candidate_success'][0]}/{rejected['candidate_success'][1]}；配对中改善 1 次、回退 17 次（双侧精确 p={rejected['exact_two_sided_p']:.6f}）。",
                    "en": f"The contact monitor reached {rejected['offline_balanced_accuracy'] * 100:.2f}% offline balanced accuracy, yet closed-loop success regressed from {rejected['v4_success'][0]}/{rejected['v4_success'][1]} to {rejected['candidate_success'][0]}/{rejected['candidate_success'][1]}; one paired case improved and 17 regressed (two-sided exact p={rejected['exact_two_sided_p']:.6f}).",
                },
                "blocked": {"zh": "不得因离线准确率高而将该候选保留为默认控制器。", "en": "Do not retain the candidate merely because its offline accuracy is high."},
                "metrics": [
                    {"label": "Offline balanced accuracy", "value": f"{rejected['offline_balanced_accuracy'] * 100:.2f}%"},
                    {"label": "Default", "value": f"{rejected['v4_success'][0]}/{rejected['v4_success'][1]}"},
                    {"label": "Candidate", "value": f"{rejected['candidate_success'][0]}/{rejected['candidate_success'][1]}"},
                ],
                "gates": [
                    {"label": "Held-out closed loop", "status": "pass"},
                    {"label": "Paired outcomes", "status": "pass"},
                    {"label": "Exact test", "status": "pass"},
                    {"label": "Retention criterion", "status": "fail"},
                ],
                "evidence_ids": ["final_audit"],
                "next_action": {"zh": "保留为强负结果，不进入默认控制器。", "en": "Retain as strong negative evidence; exclude from the default controller."},
            },
            {
                "id": "claim-openvla-boundary",
                "status": "blocked",
                "readiness": 20,
                "title": {"zh": "OpenVLA LoRA 与真实机械臂迁移尚无完成证据", "en": "OpenVLA LoRA and physical-robot transfer remain unsupported"},
                "allowed": {"zh": "OpenVLA LoRA 和真实 WidowX 验证属于后续工作，不属于当前已完成成果。", "en": "OpenVLA LoRA and physical WidowX validation are future work, not completed outcomes."},
                "blocked": {"zh": "不得继承当前 96.5% 成功率，也不得写成已完成 VLA 微调。", "en": "They cannot inherit the 96.5% result and must not be described as completed VLA fine-tuning."},
                "metrics": [{"label": "Completed runs", "value": "0"}, {"label": "Transfer trials", "value": "0"}],
                "gates": [
                    {"label": "Training artifact", "status": "fail"},
                    {"label": "Closed-loop evaluation", "status": "fail"},
                    {"label": "Physical trials", "status": "fail"},
                    {"label": "Boundary declared", "status": "pass"},
                ],
                "evidence_ids": ["final_audit", "method_registry"],
                "next_action": {"zh": "仅列入 Future Work。", "en": "List only under Future Work."},
            },
        ]

        studies = self.studies.list()
        if studies:
            latest = studies[0]
            evaluation = latest["evaluation"]
            passed = sum(gate["status"] == "pass" for gate in evaluation["gates"])
            claims.append({
                "id": "claim-latest-trace-study",
                "status": "reportable" if evaluation["verdict"] == "ready_to_report" else "bounded",
                "readiness": round(100 * passed / len(evaluation["gates"])),
                "title": {"zh": "最新 TRACE 协议的写作就绪度", "en": "Writing readiness of the latest TRACE protocol"},
                "allowed": {"zh": f"协议 {latest['study_id']} 已通过 {passed}/{len(evaluation['gates'])} 个证据门控，当前判定为 {evaluation['verdict']}。", "en": f"Protocol {latest['study_id']} passes {passed}/{len(evaluation['gates'])} evidence gates and is currently {evaluation['verdict']}."},
                "blocked": {"zh": "证据门控未全部通过时，不得把动作成功等同于统计结论充分。", "en": "Until every gate passes, task success must not be equated with adequate statistical evidence."},
                "metrics": [{"label": "Evidence gates", "value": f"{passed}/{len(evaluation['gates'])}"}, {"label": "Verdict", "value": evaluation["verdict"]}],
                "gates": evaluation["gates"],
                "evidence_ids": ["trace_registry"],
                "study_id": latest["study_id"],
                "next_action": {"zh": "在研究治理页按原协议增加样本或下载决策备忘录。", "en": "Add evidence under the locked protocol or download its decision memo from Research Governance."},
            })
        return claims

    def snapshot(self) -> dict:
        final_audit = json.loads(FINAL_AUDIT_PATH.read_text(encoding="utf-8"))
        experiment_versions = json.loads(EXPERIMENT_VERSIONS_PATH.read_text(encoding="utf-8"))
        with MODEL_RESOURCE_PATH.open(encoding="utf-8-sig", newline="") as handle:
            resources = list(csv.DictReader(handle))
        with VIDEO_AUDIT_PATH.open(encoding="utf-8-sig", newline="") as handle:
            videos = list(csv.DictReader(handle))
        methods = self._methods(resources, final_audit)
        claims = self._claims(final_audit, methods)
        source_catalogue = self._source_catalogue()
        registered_ids = {row["version"] for row in experiment_versions["methods"]}
        resource_ids = {row["version"] for row in resources}
        passed_videos = sum(row.get("审计状态") == "通过" for row in videos)
        integrity = [
            {"id": "source_files", "label": "All source files present", "status": "pass" if all(row["exists"] for row in source_catalogue) else "fail"},
            {"id": "method_alignment", "label": "Method registry matches resource audit", "status": "pass" if registered_ids == resource_ids else "fail", "observed": f"{len(registered_ids & resource_ids)}/{len(registered_ids | resource_ids)}"},
            {"id": "final_rate", "label": "Pooled rate recomputes from counts", "status": "pass" if abs(final_audit["v4_replication"]["pooled_descriptive"]["success_rate"] - 278 / 288) < 1e-12 else "fail"},
            {"id": "video_audit", "label": "Video audit rows passed", "status": "pass" if passed_videos == len(videos) else "fail", "observed": f"{passed_videos}/{len(videos)}"},
            {"id": "claim_scope", "label": "OpenVLA and real-robot exclusions declared", "status": "pass" if {"OpenVLA LoRA fine-tuning", "real-robot transfer"} <= set(final_audit["recommended_system"]["not_claimed"]) else "fail"},
        ]
        digest_input = "|".join(f"{row['id']}:{row['sha256']}" for row in source_catalogue)
        lifecycle_counts: dict[str, int] = {}
        for row in methods:
            lifecycle_counts[row["lifecycle"]] = lifecycle_counts.get(row["lifecycle"], 0) + 1
        status_counts: dict[str, int] = {}
        for row in claims:
            status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        return {
            "schema": "trace-evidence-portfolio-v1",
            "framework": self.FRAMEWORK,
            "generated_at": utc_now(),
            "source_digest": hashlib.sha256(digest_input.encode("utf-8")).hexdigest(),
            "summary": {
                "claims": len(claims),
                "methods": len(methods),
                "source_files": len(source_catalogue),
                "verified_videos": passed_videos,
                "claim_statuses": status_counts,
                "lifecycles": lifecycle_counts,
                "integrity_passed": sum(row["status"] == "pass" for row in integrity),
                "integrity_total": len(integrity),
            },
            "claims": claims,
            "methods": methods,
            "sources": source_catalogue,
            "integrity": integrity,
        }

    def source(self, source_id: str) -> tuple[bytes, str, str]:
        path = self.SOURCES.get(source_id)
        if path is None or not path.is_file():
            raise ValueError("evidence source not found")
        return path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream", path.name

    def report_markdown(self) -> str:
        portfolio = self.snapshot()
        lines = [
            "# TRACE Evidence Portfolio",
            "",
            f"- Generated: `{portfolio['generated_at']}`",
            f"- Source digest: `{portfolio['source_digest']}`",
            f"- Integrity gates: `{portfolio['summary']['integrity_passed']}/{portfolio['summary']['integrity_total']}`",
            "",
            "## Claims / 论文主张",
            "",
        ]
        for claim in portfolio["claims"]:
            lines.extend([
                f"### {claim['title']['en']} / {claim['title']['zh']}",
                "",
                f"- Status: `{claim['status']}`; readiness: `{claim['readiness']}%`",
                f"- Permitted wording: {claim['allowed']['en']}",
                f"- 允许表述：{claim['allowed']['zh']}",
                f"- Prohibited overclaim: {claim['blocked']['en']}",
                f"- 禁止越界：{claim['blocked']['zh']}",
                f"- Evidence: `{', '.join(claim['evidence_ids'])}`",
                "",
            ])
        lines.extend(["## Source fingerprints / 来源指纹", "", "| Source | Path | SHA-256 |", "|---|---|---|"])
        for source in portfolio["sources"]:
            lines.append(f"| {source['id']} | `{source['path']}` | `{source['sha256']}` |")
        lines.extend(["", "## Interpretation boundary / 解释边界", "", "This portfolio separates replicated evidence, bounded pilots, negative evidence and unsupported future work. Metrics from different protocols are not pooled into a causal leaderboard.", "", "该组合将重复验证、限定性试验、负结果与未完成工作分开管理；不同协议的数据不会被合并为因果排行榜。", ""])
        return "\n".join(lines)


class EvidenceReleaseManager:
    FRAMEWORK = "TRACE-RELEASE-1.0"

    def __init__(
        self,
        registry_path: Path,
        portfolio: ResearchPortfolio,
        ledger: ExperimentLedger,
        studies: StudyRegistry,
        simulation: SimulationManager,
        training: TrainingManager,
        adaptation: AdaptationManager,
        benchmark: BenchmarkManager,
    ) -> None:
        self.registry_path = registry_path
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        RELEASE_ROOT.mkdir(parents=True, exist_ok=True)
        self.portfolio = portfolio
        self.ledger = ledger
        self.studies = studies
        self.simulation = simulation
        self.training = training
        self.adaptation = adaptation
        self.benchmark = benchmark
        self.lock = threading.RLock()

    @staticmethod
    def _hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @classmethod
    def _hash_file(cls, path: Path) -> str | None:
        return cls._hash_bytes(path.read_bytes()) if path.is_file() else None

    @staticmethod
    def _canonical(payload: dict) -> bytes:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")

    def _registry_rows(self) -> list[dict]:
        if not self.registry_path.is_file():
            return []
        rows = []
        for line in self.registry_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "release_created" and row.get("release_id"):
                rows.append(row)
        return rows

    def _job_states(self) -> dict[str, str]:
        return {
            "simulation": self.simulation.snapshot()["status"],
            "training": self.training.snapshot()["status"],
            "adaptation": self.adaptation.snapshot()["status"],
            "benchmark": self.benchmark.snapshot()["status"],
        }

    def preview(self) -> dict:
        portfolio = self.portfolio.snapshot()
        job_states = self._job_states()
        idle_states = {"idle", "completed", "stopped", "failed"}
        reportable = sum(claim["status"] == "reportable" for claim in portfolio["claims"])
        trace_total = self.studies.summary()["total"]
        ledger_exists = LEDGER_PATH.is_file() and LEDGER_PATH.stat().st_size > 0
        gates = [
            {
                "id": "portfolio_integrity",
                "label": "Portfolio source integrity",
                "status": "pass" if portfolio["summary"]["integrity_passed"] == portfolio["summary"]["integrity_total"] else "fail",
                "observed": f"{portfolio['summary']['integrity_passed']}/{portfolio['summary']['integrity_total']}",
                "threshold": f"{portfolio['summary']['integrity_total']}/{portfolio['summary']['integrity_total']}",
            },
            {
                "id": "reportable_claim",
                "label": "At least one reportable claim",
                "status": "pass" if reportable >= 1 else "fail",
                "observed": reportable,
                "threshold": 1,
            },
            {
                "id": "trace_capture",
                "label": "TRACE protocol registry captured",
                "status": "pass" if trace_total >= 1 else "fail",
                "observed": trace_total,
                "threshold": 1,
            },
            {
                "id": "ledger_capture",
                "label": "Append-only ledger available",
                "status": "pass" if ledger_exists else "fail",
                "observed": LEDGER_PATH.stat().st_size if ledger_exists else 0,
                "threshold": "> 0 bytes",
            },
            {
                "id": "quiescent_state",
                "label": "Simulation, training, adaptation and benchmark are idle",
                "status": "pass" if all(status in idle_states for status in job_states.values()) else "fail",
                "observed": job_states,
                "threshold": "no active job",
            },
        ]
        return {
            "framework": self.FRAMEWORK,
            "ready": all(gate["status"] == "pass" for gate in gates),
            "gates": gates,
            "portfolio_digest": portfolio["source_digest"],
            "portfolio_summary": portfolio["summary"],
            "ledger_sha256": self._hash_file(LEDGER_PATH),
            "ledger_bytes": LEDGER_PATH.stat().st_size if LEDGER_PATH.is_file() else 0,
            "job_states": job_states,
            "bundle_file_count": len(portfolio["sources"]) + 2,
        }

    def create(self, request: dict) -> dict:
        label = str(request.get("label", "")).strip()
        note = str(request.get("note", "")).strip()
        if not label or len(label) > 80:
            raise ValueError("release label must contain 1 to 80 characters")
        if len(note) > 500:
            raise ValueError("release note must contain at most 500 characters")

        with self.lock:
            preview = self.preview()
            if not preview["ready"]:
                failed = ", ".join(gate["id"] for gate in preview["gates"] if gate["status"] != "pass")
                raise RuntimeError(f"release gates failed: {failed}")
            portfolio = self.portfolio.snapshot()
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            release_id = f"rel-{timestamp}-{portfolio['source_digest'][:8]}-{uuid.uuid4().hex[:4]}"
            release_dir = RELEASE_ROOT / release_id
            evidence_dir = release_dir / "evidence"
            evidence_dir.mkdir(parents=True, exist_ok=False)

            copied_paths: list[Path] = []
            for source in portfolio["sources"]:
                source_path = ROOT / source["path"]
                destination = evidence_dir / source["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination)
                copied_paths.append(destination)
            ledger_destination = evidence_dir / str(LEDGER_PATH.relative_to(ROOT))
            ledger_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(LEDGER_PATH, ledger_destination)
            copied_paths.append(ledger_destination)

            portfolio_path = release_dir / "portfolio.json"
            portfolio_path.write_text(json.dumps(portfolio, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            copied_paths.append(portfolio_path)

            files = []
            for path in copied_paths:
                files.append({
                    "path": str(path.relative_to(release_dir)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": self._hash_file(path),
                })
            files.sort(key=lambda row: row["path"])
            frozen_ledger = next(row for row in files if row["path"] == f"evidence/{str(LEDGER_PATH.relative_to(ROOT)).replace(chr(92), '/')}")
            study_rows = self.studies.list()
            payload = {
                "schema": "trace-evidence-release-v1",
                "framework": self.FRAMEWORK,
                "release_id": release_id,
                "label": label,
                "note": note,
                "created_at": utc_now(),
                "platform_version": "3.7",
                "portfolio_digest": portfolio["source_digest"],
                "portfolio_summary": portfolio["summary"],
                "claims": [
                    {
                        "id": claim["id"],
                        "status": claim["status"],
                        "readiness": claim["readiness"],
                        "title": claim["title"],
                        "metrics": claim["metrics"],
                        "evidence_ids": claim["evidence_ids"],
                    }
                    for claim in portfolio["claims"]
                ],
                "trace_studies": [
                    {
                        "study_id": study["study_id"],
                        "protocol_hash": study["protocol_hash"],
                        "verdict": study["evaluation"]["verdict"],
                    }
                    for study in study_rows
                ],
                "ledger": {
                    "path": str(LEDGER_PATH.relative_to(ROOT)),
                    "sha256": frozen_ledger["sha256"],
                    "bytes": frozen_ledger["bytes"],
                    "analytics": self.ledger.analytics(),
                },
                "release_gates": preview["gates"],
                "files": files,
                "claim_boundary": "MuJoCo-only evidence release. The bundle does not prove OpenVLA LoRA fine-tuning or physical-robot transfer.",
            }
            manifest_hash = self._hash_bytes(self._canonical(payload))
            manifest = {**payload, "manifest_hash": manifest_hash}
            manifest_path = release_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            readme_path = release_dir / "README.md"
            readme_path.write_text(self._markdown(manifest), encoding="utf-8")
            registry_row = {
                "event": "release_created",
                "release_id": release_id,
                "created_at": payload["created_at"],
                "label": label,
                "manifest_hash": manifest_hash,
                "manifest_path": str(manifest_path.relative_to(ROOT)),
            }
            with self.registry_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(registry_row, ensure_ascii=False, allow_nan=False) + "\n")
        return self.get(release_id)

    def _markdown(self, manifest: dict) -> str:
        lines = [
            f"# Evidence Release: {manifest['label']}",
            "",
            f"- Release ID: `{manifest['release_id']}`",
            f"- Created: `{manifest['created_at']}`",
            f"- Manifest SHA-256: `{manifest['manifest_hash']}`",
            f"- Portfolio digest: `{manifest['portfolio_digest']}`",
            f"- Bundled files: `{len(manifest['files'])}`",
            "",
            manifest["note"] or "No release note.",
            "",
            "## Claims / 论文主张",
            "",
            "| Status | Readiness | Claim |",
            "|---|---:|---|",
        ]
        for claim in manifest["claims"]:
            lines.append(f"| {claim['status']} | {claim['readiness']}% | {claim['title']['en']} / {claim['title']['zh']} |")
        lines.extend(["", "## Release gates / 发布门控", "", "| Gate | Status | Observed |", "|---|---|---|"])
        for gate in manifest["release_gates"]:
            lines.append(f"| {gate['label']} | {gate['status']} | `{json.dumps(gate['observed'], ensure_ascii=False)}` |")
        lines.extend(["", "## Files / 文件", "", "| Path | Bytes | SHA-256 |", "|---|---:|---|"])
        for row in manifest["files"]:
            lines.append(f"| `{row['path']}` | {row['bytes']} | `{row['sha256']}` |")
        lines.extend(["", "## Claim boundary / 结论边界", "", manifest["claim_boundary"], ""])
        return "\n".join(lines)

    def _manifest(self, release_id: str) -> dict:
        if not re.fullmatch(r"rel-[A-Za-z0-9-]+", release_id):
            raise ValueError("release not found")
        path = RELEASE_ROOT / release_id / "manifest.json"
        if not path.is_file():
            raise ValueError("release not found")
        return json.loads(path.read_text(encoding="utf-8"))

    def get(self, release_id: str) -> dict:
        manifest = self._manifest(release_id)
        payload = {key: value for key, value in manifest.items() if key != "manifest_hash"}
        manifest_valid = self._hash_bytes(self._canonical(payload)) == manifest.get("manifest_hash")
        release_dir = RELEASE_ROOT / release_id
        file_checks = []
        for row in manifest.get("files", []):
            path = (release_dir / row["path"]).resolve()
            inside = release_dir.resolve() in {path, *path.parents}
            valid = inside and path.is_file() and path.stat().st_size == row["bytes"] and self._hash_file(path) == row["sha256"]
            file_checks.append({"path": row["path"], "valid": valid})
        manifest_files = {row["path"]: row for row in manifest.get("files", [])}
        try:
            frozen_portfolio = json.loads((release_dir / "portfolio.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            frozen_portfolio = {}
        source_snapshot_valid = bool(frozen_portfolio.get("sources")) and all(
            manifest_files.get(f"evidence/{source['path'].replace(chr(92), '/')}", {}).get("sha256") == source.get("sha256")
            for source in frozen_portfolio.get("sources", [])
        )
        ledger_file = manifest_files.get(f"evidence/{str(LEDGER_PATH.relative_to(ROOT)).replace(chr(92), '/')}", {})
        ledger_snapshot_valid = ledger_file.get("sha256") == (manifest.get("ledger") or {}).get("sha256") and ledger_file.get("bytes") == (manifest.get("ledger") or {}).get("bytes")
        files_valid = (
            len(file_checks) == len(manifest.get("files", []))
            and all(row["valid"] for row in file_checks)
            and source_snapshot_valid
            and ledger_snapshot_valid
        )
        current_portfolio = self.portfolio.snapshot()["source_digest"] == manifest.get("portfolio_digest")
        current_ledger = self._hash_file(LEDGER_PATH) == (manifest.get("ledger") or {}).get("sha256")
        if not manifest_valid or not files_valid:
            status = "corrupted"
        elif current_portfolio and current_ledger:
            status = "verified_current"
        else:
            status = "verified_snapshot"
        return {
            "release_id": release_id,
            "label": manifest["label"],
            "note": manifest["note"],
            "created_at": manifest["created_at"],
            "manifest_hash": manifest["manifest_hash"],
            "portfolio_digest": manifest["portfolio_digest"],
            "portfolio_summary": manifest["portfolio_summary"],
            "claims": manifest["claims"],
            "trace_studies": manifest["trace_studies"],
            "ledger": manifest["ledger"],
            "release_gates": manifest["release_gates"],
            "files": manifest["files"],
            "status": status,
            "verification": {
                "manifest_valid": manifest_valid,
                "files_valid": files_valid,
                "source_snapshot_valid": source_snapshot_valid,
                "ledger_snapshot_valid": ledger_snapshot_valid,
                "verified_files": sum(row["valid"] for row in file_checks),
                "total_files": len(file_checks),
                "current_portfolio": current_portfolio,
                "current_ledger": current_ledger,
            },
            "manifest_url": f"/api/releases/{release_id}/manifest.json",
            "readme_url": f"/api/releases/{release_id}/README.md",
        }

    def list(self) -> list[dict]:
        rows = []
        for row in reversed(self._registry_rows()):
            try:
                rows.append(self.get(row["release_id"]))
            except (ValueError, json.JSONDecodeError):
                rows.append({
                    "release_id": row["release_id"],
                    "label": row.get("label", "unknown"),
                    "created_at": row.get("created_at"),
                    "manifest_hash": row.get("manifest_hash"),
                    "status": "corrupted",
                    "verification": {"manifest_valid": False, "files_valid": False, "verified_files": 0, "total_files": 0, "current_portfolio": False, "current_ledger": False},
                })
        return rows

    def summary(self) -> dict:
        rows = self.list()
        statuses: dict[str, int] = {}
        for row in rows:
            statuses[row["status"]] = statuses.get(row["status"], 0) + 1
        return {"total": len(rows), "statuses": statuses, "framework": self.FRAMEWORK}

    def artifact(self, release_id: str, filename: str) -> tuple[bytes, str, str]:
        if filename not in {"manifest.json", "README.md"}:
            raise ValueError("release artifact not found")
        self._manifest(release_id)
        path = RELEASE_ROOT / release_id / filename
        if not path.is_file():
            raise ValueError("release artifact not found")
        return path.read_bytes(), mimetypes.guess_type(filename)[0] or "application/octet-stream", filename


class PlatformApplication:
    def __init__(self) -> None:
        self.started_at = time.time()
        self.ledger = ExperimentLedger(LEDGER_PATH)
        self.simulation = SimulationManager(self.ledger)
        self.training = TrainingManager(self.ledger)
        self.adaptation = AdaptationManager(self.ledger, self.training)
        self.benchmark = BenchmarkManager(self.simulation, self.ledger)
        self.studies = StudyRegistry(STUDY_REGISTRY_PATH, self.ledger, self.benchmark)
        self.portfolio = ResearchPortfolio(self.studies)
        self.releases = EvidenceReleaseManager(
            RELEASE_REGISTRY_PATH,
            self.portfolio,
            self.ledger,
            self.studies,
            self.simulation,
            self.training,
            self.adaptation,
            self.benchmark,
        )

    def config(self) -> dict:
        return {
            "tasks": TASKS,
            "datasets": {key: {"path": str(path.relative_to(ROOT)), "available": path.exists()} for key, path in DATASETS.items()},
            "trainers": list(TRAINERS),
            "legacy_path": "/docs/integrated_research_showcase.html",
            "research_summary_path": "/docs/mujoco_research_summary_zh_en.html",
            "ledger_export_path": "/api/runs/export.csv",
            "platform_version": "3.7",
            "governance_framework": StudyRegistry.FRAMEWORK,
            "portfolio_framework": ResearchPortfolio.FRAMEWORK,
            "release_framework": EvidenceReleaseManager.FRAMEWORK,
            "adaptation_framework": AdaptationManager.FRAMEWORK,
            "arena_framework": AdaptationManager.ARENA_FRAMEWORK,
        }

    def run_report(self, run_id: str) -> dict:
        record = self.ledger.get(run_id)
        if record is None:
            raise ValueError("run not found")
        if record.get("kind") == "optimization":
            claim_boundary = (
                "MuJoCo-only paired candidate evidence. Completed candidates share one fingerprinted demonstration "
                "protocol and one held-out seed sequence. The LoRA-style residual is a local PEFT proxy, not an "
                "OpenVLA fine-tune; simulator state remains restricted to offline diagnostics and scoring in RGB runs."
            )
        else:
            claim_boundary = (
                "MuJoCo-only evidence. In rgb_grounded runs, MuJoCo state is excluded from runtime target "
                "selection and planning; it is used only for offline diagnostics and success scoring."
            )
        return {
            "schema": "widowx-research-report-v1",
            "generated_at": utc_now(),
            "platform_version": "3.7",
            "run": record,
            "children": self.ledger.children(run_id),
            "claim_boundary": claim_boundary,
        }

    def run_report_markdown(self, run_id: str) -> str:
        report = self.run_report(run_id)
        row = report["run"]
        metrics = row.get("metrics", {})
        lines = [
            f"# Experiment Report: {run_id}",
            "",
            f"- Type: `{row.get('kind')}`",
            f"- Status: `{row.get('status')}`",
            f"- Started: `{row.get('started_at')}`",
            f"- Finished: `{row.get('finished_at')}`",
            f"- Parent: `{row.get('parent_id') or 'none'}`",
            "",
            "## Configuration",
            "",
            "```json",
            json.dumps(row.get("config", {}), ensure_ascii=False, indent=2),
            "```",
            "",
            "## Metrics",
            "",
            "```json",
            json.dumps(metrics, ensure_ascii=False, indent=2),
            "```",
        ]
        assets = row.get("assets") or {}
        if assets:
            lines.extend(["", "## Visual Evidence", ""])
            lines.extend(f"- {name}: `{url}`" for name, url in assets.items())
        children = report["children"]
        if row.get("kind") == "optimization":
            paired = metrics.get("paired_summary", {})
            lines.extend([
                "", "## Paired Protocol", "",
                f"- Dataset fingerprint: `{metrics.get('dataset_fingerprint', '--')}`",
                f"- Dataset cache hit: `{metrics.get('dataset_cache_hit', False)}`",
                f"- Collection runs saved: `{metrics.get('collection_runs_saved', 0)}`",
                f"- Matched held-out seeds: `{paired.get('matched_seed_sets', False)}`",
                f"- Champion: `{paired.get('champion_method') or '--'}`",
                "", "| Method A | Method B | Both pass | A only | B only | Both fail | Exact p |",
                "|---|---|---:|---:|---:|---:|---:|",
            ])
            for comparison in paired.get("comparisons", []):
                lines.append(
                    f"| {comparison.get('method_a')} | {comparison.get('method_b')} | "
                    f"{comparison.get('both_success')} | {comparison.get('a_only')} | "
                    f"{comparison.get('b_only')} | {comparison.get('both_fail')} | "
                    f"{comparison.get('mcnemar_exact_p')} |"
                )
        if children and row.get("kind") == "optimization":
            lines.extend([
                "", "## Paired Candidates", "",
                "| Run | Method | Status | Seeds | Success | Error (mm) | Params | Peak RAM (MB) |",
                "|---|---|---|---|---:|---:|---:|---:|",
            ])
            for child in children:
                config = child.get("config", {})
                child_metrics = child.get("metrics", {})
                error = child_metrics.get("mean_target_error")
                error_mm = "--" if not isinstance(error, (int, float)) else f"{error * 1000:.1f}"
                seeds = ", ".join(str(seed) for seed in child_metrics.get("evaluation_seeds", []))
                success_rate = child_metrics.get("success_rate")
                success = "--" if not isinstance(success_rate, (int, float)) else f"{success_rate * 100:.1f}%"
                lines.append(
                    f"| {child.get('run_id')} | {config.get('method')} | {child.get('status')} | {seeds} | "
                    f"{success} | {error_mm} | {child_metrics.get('trainable_params', '--')} | "
                    f"{child_metrics.get('peak_rss_mb', '--')} |"
                )
        elif children:
            lines.extend(["", "## Child Episodes", "", "| Run | Task | Policy | Seed | Success | Error (mm) |", "|---|---|---|---:|---:|---:|"])
            for child in children:
                config = child.get("config", {})
                child_metrics = child.get("metrics", {})
                error = child_metrics.get("target_distance")
                error_mm = "--" if not isinstance(error, (int, float)) else f"{error * 1000:.1f}"
                lines.append(
                    f"| {child.get('run_id')} | {config.get('task')} | {config.get('policy')} | "
                    f"{config.get('seed')} | {child_metrics.get('success')} | {error_mm} |"
                )
        lines.extend([
            "",
            "## Reproduction",
            "",
            "```powershell",
            row.get("command") or "# No command recorded",
            "```",
            "",
            "## Claim Boundary",
            "",
            report["claim_boundary"],
            "",
        ])
        return "\n".join(lines)

    def health(self) -> dict:
        usage = shutil.disk_usage(ROOT)
        datasets = {key: path.exists() for key, path in DATASETS.items()}
        dataset_episodes = {key: len(list((path / "episodes").glob("*.npz"))) if path.exists() else 0 for key, path in DATASETS.items()}
        return {
            "status": "healthy" if all(datasets.values()) and CALIBRATION_PATH.exists() and FINAL_MODEL_PATH.exists() else "degraded",
            "platform_version": "3.7",
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "operating_system": platform.platform(),
            "pid": os.getpid(),
            "uptime": time.time() - self.started_at,
            "assets": {
                "calibration": CALIBRATION_PATH.exists(),
                "final_model": FINAL_MODEL_PATH.exists(),
                "original_showcase": (ROOT / "docs" / "integrated_research_showcase.html").exists(),
            },
            "datasets": datasets,
            "dataset_episodes": dataset_episodes,
            "disk_free_gb": usage.free / (1024 ** 3),
            "ledger_path": str(LEDGER_PATH.relative_to(ROOT)),
            "study_registry_path": str(STUDY_REGISTRY_PATH.relative_to(ROOT)),
            "trace_studies": self.studies.summary()["total"],
            "adaptation": {
                "framework": AdaptationManager.FRAMEWORK,
                "arena_framework": AdaptationManager.ARENA_FRAMEWORK,
                "status": self.adaptation.snapshot()["status"],
                "custom_tasks": len(self.adaptation.task_catalogue()),
                "base_action_head": OBJECT_ACTION_HEAD_PATH.exists(),
                "hardware": self.adaptation.snapshot()["hardware"],
            },
            "portfolio": self.portfolio.snapshot()["summary"],
            "evidence_releases": self.releases.summary(),
        }


APP = PlatformApplication()


class PlatformHTTPServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address) -> None:
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "WidowXResearchPlatform/3.7"

    def log_message(self, format: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}", flush=True)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/api/config":
            return self.send_json(APP.config())
        if path == "/api/status":
            return self.send_json({
                "simulation": APP.simulation.snapshot(),
                "training": APP.training.snapshot(),
                "adaptation": APP.adaptation.snapshot(),
                "benchmark": APP.benchmark.snapshot(),
                "analytics": APP.ledger.analytics(),
            })
        if path == "/api/health":
            return self.send_json(APP.health())
        if path == "/api/adaptation":
            return self.send_json(APP.adaptation.snapshot())
        if path == "/api/adaptation/portfolio":
            query = parse_qs(parsed.query)
            return self.send_json(APP.adaptation.performance_portfolio(query.get("task_id", [None])[0]))
        adaptation_preview_match = re.fullmatch(r"/api/adaptation/tasks/([^/]+)/preview\.png", path)
        if adaptation_preview_match:
            query = parse_qs(parsed.query)
            try:
                seed = int(query.get("seed", ["0"])[0])
                return self.send_bytes(
                    APP.adaptation.preview(adaptation_preview_match.group(1), seed),
                    "image/png",
                    {"Cache-Control": "no-store"},
                )
            except (ValueError, KeyError):
                return self.send_error(HTTPStatus.NOT_FOUND)
        if path == "/api/studies":
            return self.send_json({"studies": APP.studies.list(), "summary": APP.studies.summary()})
        if path == "/api/portfolio":
            return self.send_json(APP.portfolio.snapshot())
        if path == "/api/releases":
            return self.send_json({"preview": APP.releases.preview(), "releases": APP.releases.list(), "summary": APP.releases.summary()})
        release_artifact_match = re.fullmatch(r"/api/releases/([^/]+)/(manifest\.json|README\.md)", path)
        if release_artifact_match:
            try:
                data, content_type, filename = APP.releases.artifact(*release_artifact_match.groups())
                return self.send_bytes(data, content_type, {"Content-Disposition": f"attachment; filename={filename}", "Cache-Control": "no-store"})
            except ValueError:
                return self.send_error(HTTPStatus.NOT_FOUND)
        release_match = re.fullmatch(r"/api/releases/([^/]+)", path)
        if release_match:
            try:
                return self.send_json(APP.releases.get(release_match.group(1)))
            except ValueError:
                return self.send_error(HTTPStatus.NOT_FOUND)
        if path == "/api/portfolio/report.md":
            markdown = APP.portfolio.report_markdown().encode("utf-8")
            return self.send_bytes(markdown, "text/markdown; charset=utf-8", {"Content-Disposition": "attachment; filename=trace-evidence-portfolio.md", "Cache-Control": "no-store"})
        portfolio_source_match = re.fullmatch(r"/api/portfolio/sources/([^/]+)", path)
        if portfolio_source_match:
            try:
                data, content_type, filename = APP.portfolio.source(portfolio_source_match.group(1))
                return self.send_bytes(data, content_type, {"Content-Disposition": f"inline; filename={filename}", "Cache-Control": "no-store"})
            except ValueError:
                return self.send_error(HTTPStatus.NOT_FOUND)
        study_memo_match = re.fullmatch(r"/api/studies/([^/]+)/memo\.md", path)
        if study_memo_match:
            study_id = study_memo_match.group(1)
            try:
                markdown = APP.studies.memo_markdown(study_id).encode("utf-8")
                return self.send_bytes(
                    markdown,
                    "text/markdown; charset=utf-8",
                    {"Content-Disposition": f"attachment; filename={study_id}-decision-memo.md", "Cache-Control": "no-store"},
                )
            except ValueError:
                return self.send_error(HTTPStatus.NOT_FOUND)
        study_match = re.fullmatch(r"/api/studies/([^/]+)", path)
        if study_match:
            try:
                return self.send_json(APP.studies.get(study_match.group(1)))
            except ValueError:
                return self.send_error(HTTPStatus.NOT_FOUND)
        if path == "/api/runs":
            query = parse_qs(parsed.query)
            limit = int(query.get("limit", ["200"])[0])
            kind = query.get("kind", [None])[0]
            status = query.get("status", [None])[0]
            return self.send_json({"runs": APP.ledger.list(limit=limit, kind=kind, status=status), "analytics": APP.ledger.analytics()})
        if path == "/api/runs/export.csv":
            return self.send_bytes(
                APP.ledger.csv_bytes(),
                "text/csv; charset=utf-8",
                {"Content-Disposition": "attachment; filename=widowx_experiment_ledger.csv", "Cache-Control": "no-store"},
            )
        report_match = re.fullmatch(r"/api/runs/([^/]+)/report\.(json|md)", path)
        if report_match:
            run_id, extension = report_match.groups()
            try:
                if extension == "json":
                    return self.send_json(APP.run_report(run_id))
                markdown = APP.run_report_markdown(run_id).encode("utf-8")
                return self.send_bytes(
                    markdown,
                    "text/markdown; charset=utf-8",
                    {"Content-Disposition": f"attachment; filename={run_id}.md", "Cache-Control": "no-store"},
                )
            except ValueError:
                return self.send_error(HTTPStatus.NOT_FOUND)
        if path == "/api/sim/frame.png":
            frame, sequence = APP.simulation.frame_snapshot()
            return self.send_bytes(frame, "image/png", {"X-Frame-Sequence": str(sequence), "Cache-Control": "no-store"})
        if path == "/api/events":
            return self.send_events()
        if path in {"/", "/index.html"}:
            return self.send_file(STATIC_ROOT / "index.html")
        if path.startswith("/assets/"):
            return self.send_file(STATIC_ROOT / path.removeprefix("/assets/"), allowed_root=STATIC_ROOT)
        if path.startswith("/platform_artifacts/"):
            return self.send_file(PLATFORM_OUTPUT / path.removeprefix("/platform_artifacts/"), allowed_root=PLATFORM_OUTPUT)
        if path.startswith("/docs/"):
            return self.send_file((ROOT / path.lstrip("/")).resolve(), allowed_root=ROOT / "docs")
        if path.startswith("/showcase_assets/"):
            return self.send_file((ROOT / path.lstrip("/")).resolve(), allowed_root=ROOT / "showcase_assets")
        if path.startswith("/presentation_videos/"):
            return self.send_file((ROOT / path.lstrip("/")).resolve(), allowed_root=ROOT / "presentation_videos")
        if path.startswith("/videos/"):
            return self.send_file((ROOT / path.lstrip("/")).resolve(), allowed_root=ROOT / "videos")
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            body = self.read_json()
            if parsed.path == "/api/sim/start":
                return self.send_json(APP.simulation.start(body))
            if parsed.path == "/api/sim/control":
                action = str(body.get("action", ""))
                operations = {
                    "pause": APP.simulation.pause,
                    "resume": APP.simulation.resume,
                    "stop": APP.simulation.stop,
                    "reset": APP.simulation.reset,
                }
                if action not in operations:
                    raise ValueError("unsupported simulation action")
                return self.send_json(operations[action]())
            if parsed.path == "/api/sim/command":
                parsed_command = parse_instruction(str(body.get("command", "")))
                if parsed_command is None:
                    raise ValueError("command not recognised")
                kind, value = parsed_command
                if kind == "task":
                    request = {
                        "task": value,
                        "policy": body.get("policy", "rgb_grounded"),
                        "complexity": TASKS[value]["complexity"],
                        "seed": body.get("seed", 0),
                        "speed": body.get("speed", 1.0),
                    }
                    return self.send_json(APP.simulation.start(request))
                return self.send_json(getattr(APP.simulation, value)())
            if parsed.path == "/api/sim/native-viewer":
                return self.send_json(APP.simulation.open_native_viewer(body))
            if parsed.path == "/api/train/start":
                return self.send_json(APP.training.start(body))
            if parsed.path == "/api/train/stop":
                return self.send_json(APP.training.stop())
            if parsed.path == "/api/adaptation/tasks":
                return self.send_json(APP.adaptation.create_task(body), status=HTTPStatus.CREATED)
            if parsed.path == "/api/adaptation/estimate":
                return self.send_json(APP.adaptation.estimate(body))
            if parsed.path == "/api/adaptation/arena/estimate":
                return self.send_json(APP.adaptation.estimate_arena(body))
            if parsed.path == "/api/adaptation/start":
                return self.send_json(APP.adaptation.start(body))
            if parsed.path == "/api/adaptation/arena/start":
                return self.send_json(APP.adaptation.start_arena(body))
            if parsed.path == "/api/adaptation/stop":
                return self.send_json(APP.adaptation.stop())
            if parsed.path == "/api/adaptation/native-viewer":
                return self.send_json(APP.adaptation.open_viewer(body))
            if parsed.path == "/api/benchmark/start":
                return self.send_json(APP.benchmark.start(body))
            if parsed.path == "/api/benchmark/stop":
                return self.send_json(APP.benchmark.stop())
            if parsed.path == "/api/studies":
                return self.send_json(APP.studies.create(body), status=HTTPStatus.CREATED)
            if parsed.path == "/api/releases":
                return self.send_json(APP.releases.create(body), status=HTTPStatus.CREATED)
            study_launch_match = re.fullmatch(r"/api/studies/([^/]+)/launch", parsed.path)
            if study_launch_match:
                return self.send_json(APP.studies.launch(study_launch_match.group(1)))
            self.send_error(HTTPStatus.NOT_FOUND)
        except (ValueError, RuntimeError) as error:
            self.send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as error:  # pragma: no cover - defensive HTTP boundary
            self.send_json({"error": str(error)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_bytes(self, data: bytes, content_type: str, headers: dict[str, str] | None = None) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, file_path: Path, allowed_root: Path | None = None) -> None:
        file_path = file_path.resolve()
        if allowed_root is not None and allowed_root.resolve() not in {file_path, *file_path.parents}:
            return self.send_error(HTTPStatus.FORBIDDEN)
        if not file_path.is_file():
            return self.send_error(HTTPStatus.NOT_FOUND)
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        self.send_bytes(file_path.read_bytes(), content_type, {"Cache-Control": "no-cache"})

    def send_events(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            while True:
                payload = json.dumps({
                    "simulation": APP.simulation.snapshot(),
                    "training": APP.training.snapshot(),
                    "adaptation": APP.adaptation.snapshot(),
                    "benchmark": APP.benchmark.snapshot(),
                    "analytics": APP.ledger.analytics(),
                }, ensure_ascii=False, allow_nan=False)
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
                time.sleep(0.5)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the live WidowX MuJoCo research platform.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--open-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = PlatformHTTPServer((args.host, args.port), RequestHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Research platform: {url}", flush=True)
    print("Press Ctrl+C to stop the platform.", flush=True)
    if args.open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        APP.benchmark.stop()
        APP.simulation.stop(wait=True)
        APP.training.stop()
        server.server_close()


if __name__ == "__main__":
    main()
