from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from widowx_env import TASKS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile a registered task into a zero-gradient RGB skill adapter.")
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=ROOT / "runtime_assets" / "top_rgb_core_v2_calibration_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    task = TASKS[args.task]
    if not task.target_geom:
        raise ValueError("registry RGB skill adapters require a target region")
    if not args.calibration.is_file():
        raise FileNotFoundError(f"RGB calibration not found: {args.calibration}")

    args.output.mkdir(parents=True, exist_ok=True)
    artifact = args.output / f"registry_rgb_skill_adapter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "schema": "registry-rgb-skill-adapter-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_id": args.task,
        "instruction": task.instruction,
        "source_object": task.target_object,
        "target_region": task.target_geom,
        "intent_source": "task_registry",
        "position_source": "top_rgb + offline plane calibration",
        "executor": "structured_pick_place_with_one_rgb_retry",
        "calibration": str(args.calibration.resolve()),
        "trainable_params": 0,
        "note": "Compiled hierarchical skill adapter; not a learned VLA or OpenVLA fine-tune.",
    }
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"model_path: {artifact}", flush=True)
    print("trainable_params: 0", flush=True)
    print("frozen_params: 0", flush=True)
    print(f"compile_time_seconds: {time.monotonic() - started:.4f}", flush=True)


if __name__ == "__main__":
    main()
