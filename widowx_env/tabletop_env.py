from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = PROJECT_ROOT / "assets" / "mujoco" / "tabletop_wx250s_scene.xml"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    instruction: str
    kind: str
    target_object: str | None
    target_geom: str | None = None
    relation: str | None = None


TASKS: dict[str, TaskSpec] = {
    "pick_red_cube": TaskSpec(
        name="pick_red_cube",
        instruction="pick the red cube",
        kind="pick",
        target_object="red_cube",
    ),
    "place_red_cube_red_pad": TaskSpec(
        name="place_red_cube_red_pad",
        instruction="place the red cube on the red pad",
        kind="place",
        target_object="red_cube",
        target_geom="target_red_pad",
    ),
    "place_blue_cube_blue_pad": TaskSpec(
        name="place_blue_cube_blue_pad",
        instruction="place the blue cube on the blue pad",
        kind="place",
        target_object="blue_cube",
        target_geom="target_blue_pad",
    ),
    "place_blue_cube_red_pad": TaskSpec(
        name="place_blue_cube_red_pad",
        instruction="place the blue cube on the red pad",
        kind="place",
        target_object="blue_cube",
        target_geom="target_red_pad",
    ),
    "push_green_ball_blue_pad": TaskSpec(
        name="push_green_ball_blue_pad",
        instruction="push the green ball into the blue region",
        kind="push",
        target_object="green_ball",
        target_geom="target_blue_pad",
    ),
    "pick_red_cylinder": TaskSpec(
        name="pick_red_cylinder",
        instruction="pick the red cylinder",
        kind="pick",
        target_object="red_cylinder",
    ),
    "move_leftmost_to_bowl": TaskSpec(
        name="move_leftmost_to_bowl",
        instruction="move the leftmost object to the bowl",
        kind="place",
        target_object=None,
        target_geom="target_bowl",
        relation="leftmost",
    ),
    "move_leftmost_cube_to_bowl": TaskSpec(
        name="move_leftmost_cube_to_bowl",
        instruction="move the leftmost cube to the bowl",
        kind="place",
        target_object=None,
        target_geom="target_bowl",
        relation="leftmost_cube",
    ),
}


OBJECTS = (
    "red_cube",
    "blue_cube",
    "green_cube",
    "yellow_cube",
    "red_cylinder",
    "blue_cylinder",
    "green_ball",
)

CUSTOM_TASK_REGISTRY = PROJECT_ROOT / "outputs" / "platform_research" / "adaptation_tasks.json"
CUSTOM_TASK_SOURCES = tuple(name for name in OBJECTS if name.endswith("_cube"))
CUSTOM_TASK_TARGETS = ("target_red_pad", "target_blue_pad", "target_bowl")


def register_custom_task(record: dict[str, Any], persist: bool = True) -> TaskSpec:
    """Register a validated tabletop task that existing expert/data scripts can consume."""
    task_id = str(record.get("task_id", "")).strip()
    instruction = str(record.get("instruction", "")).strip()
    source = str(record.get("source", "")).strip()
    target = str(record.get("target", "")).strip()
    if not re.fullmatch(r"place_[a-z0-9_]{3,64}", task_id):
        raise ValueError("custom task id must use the place_<source>_<target> form")
    if not 3 <= len(instruction) <= 160:
        raise ValueError("custom task instruction must contain 3 to 160 characters")
    if source not in CUSTOM_TASK_SOURCES:
        raise ValueError("custom task source must be a tabletop cube")
    if target not in CUSTOM_TASK_TARGETS:
        raise ValueError("custom task target is not available in this MuJoCo scene")

    spec = TaskSpec(
        name=task_id,
        instruction=instruction,
        kind="place",
        target_object=source,
        target_geom=target,
    )
    existing = TASKS.get(task_id)
    if existing is not None and existing != spec:
        raise ValueError(f"task id already maps to a different task: {task_id}")
    TASKS[task_id] = spec

    if persist:
        CUSTOM_TASK_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": "widowx-custom-tasks-v1", "tasks": []}
        if CUSTOM_TASK_REGISTRY.is_file():
            payload = json.loads(CUSTOM_TASK_REGISTRY.read_text(encoding="utf-8"))
        rows = [item for item in payload.get("tasks", []) if item.get("task_id") != task_id]
        rows.append({
            "task_id": task_id,
            "instruction": instruction,
            "instruction_zh": str(record.get("instruction_zh", "")).strip(),
            "source": source,
            "target": target,
            "complexity": str(record.get("complexity", "medium")),
            "created_at": str(record.get("created_at", "")),
        })
        rows.sort(key=lambda item: item["task_id"])
        payload = {"schema": "widowx-custom-tasks-v1", "tasks": rows}
        temporary = CUSTOM_TASK_REGISTRY.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(CUSTOM_TASK_REGISTRY)
    return spec


def load_custom_tasks() -> list[dict[str, Any]]:
    if not CUSTOM_TASK_REGISTRY.is_file():
        return []
    payload = json.loads(CUSTOM_TASK_REGISTRY.read_text(encoding="utf-8"))
    rows = payload.get("tasks", [])
    for row in rows:
        register_custom_task(row, persist=False)
    return rows


load_custom_tasks()

COMPLEXITY_COUNTS = {
    "easy": 1,
    "medium": 3,
    "hard": len(OBJECTS),
    "language": 5,
}

WORKSPACE_PROFILES = {
    "legacy": (0.20, 0.46, -0.18, 0.18, 0.085),
    "core_v2": (0.23, 0.45, -0.10, 0.10, 0.085),
}
MIN_LEFTMOST_CUBE_X_GAP = 0.03


class WidowXTabletopEnv:
    """Small MuJoCo desktop manipulation environment for early VLA experiments."""

    def __init__(
        self,
        scene_path: str | Path = DEFAULT_SCENE,
        seed: int | None = None,
        image_size: tuple[int, int] = (128, 128),
        camera: str = "top_rgb",
        workspace_profile: str = "legacy",
    ) -> None:
        if workspace_profile not in WORKSPACE_PROFILES:
            raise KeyError(f"unknown workspace profile: {workspace_profile}")
        self.scene_path = Path(scene_path)
        self.load_path = self._prepared_load_path(self.scene_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.load_path))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.image_size = image_size
        self.camera = camera
        self.workspace_profile = workspace_profile

        self.object_body_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in OBJECTS
        }
        self.object_qpos_addr = {
            name: int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")])
            for name in OBJECTS
        }
        self.object_qvel_addr = {
            name: int(self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")])
            for name in OBJECTS
        }
        self.target_geom_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("target_red_pad", "target_blue_pad", "target_bowl")
        }
        self.tcp_geom_ids = tuple(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("left/left_g0", "left/left_g1", "right/right_g0", "right/right_g1")
        )
        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wx250s/gripper_link")

        self.home_qpos = np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 0.015, -0.015], dtype=float)
        self.home_ctrl = np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 0.015], dtype=float)
        self.robot_nq = len(self.home_qpos)
        self.robot_nv = 7
        self.active_objects = list(OBJECTS)
        self.task = TASKS["pick_red_cube"]
        self.episode_target_object = "red_cube"

    @property
    def action_size(self) -> int:
        return int(self.model.nu)

    def reset(
        self,
        task: str = "pick_red_cube",
        complexity: str = "medium",
        seed: int | None = None,
    ) -> dict[str, Any]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if task not in TASKS:
            raise KeyError(f"unknown task: {task}")
        if complexity not in COMPLEXITY_COUNTS:
            raise KeyError(f"unknown complexity: {complexity}")

        self.task = TASKS[task]
        self.data = mujoco.MjData(self.model)
        self.data.qpos[: self.robot_nq] = self.home_qpos
        self.data.qvel[: self.robot_nv] = 0.0
        self.data.ctrl[:] = self.home_ctrl

        count = COMPLEXITY_COUNTS[complexity]
        self.active_objects = self._choose_active_objects(count)
        self._place_objects()
        mujoco.mj_forward(self.model, self.data)
        return self.observation(render=False)

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], dict[str, float | bool]]:
        action = np.asarray(action, dtype=float)
        if action.shape != (self.action_size,):
            raise ValueError(f"action must have shape {(self.action_size,)}, got {action.shape}")
        ctrl_min = self.model.actuator_ctrlrange[:, 0]
        ctrl_max = self.model.actuator_ctrlrange[:, 1]
        self.data.ctrl[:] = np.clip(action, ctrl_min, ctrl_max)
        mujoco.mj_step(self.model, self.data)
        return self.observation(render=False), self.metrics()

    def observation(self, render: bool = False) -> dict[str, Any]:
        obs: dict[str, Any] = {
            "instruction": self.task.instruction,
            "task": self.task.name,
            "target_object": self.episode_target_object,
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "objects": {name: self.object_position(name).copy() for name in OBJECTS},
            "active_objects": tuple(self.active_objects),
        }
        if render:
            obs["rgb"] = self.render_rgb()
        return obs

    def metrics(self) -> dict[str, float | bool]:
        obj_pos = self.object_position(self.episode_target_object)
        ee_pos = self.data.xpos[self.ee_body_id]
        target_distance = np.nan
        if self.task.target_geom:
            target_pos = self.target_position(self.task.target_geom)
            target_distance = float(np.linalg.norm(obj_pos[:2] - target_pos[:2]))

        ee_object_distance = float(np.linalg.norm(ee_pos - obj_pos))
        lifted = bool(obj_pos[2] > 0.085)
        near_ee = bool(ee_object_distance < 0.09)
        placed = bool(np.isfinite(target_distance) and target_distance < 0.065 and obj_pos[2] < 0.08)

        if self.task.kind == "pick":
            success = lifted and near_ee
        elif self.task.kind in {"place", "push"}:
            success = placed
        else:
            success = False

        table_violation = bool(obj_pos[0] < -0.25 or obj_pos[0] > 0.85 or abs(obj_pos[1]) > 0.38)
        return {
            "success": success,
            "target_distance": target_distance,
            "ee_object_distance": ee_object_distance,
            "object_z": float(obj_pos[2]),
            "grasp_success": lifted and near_ee,
            "contact_count": float(self.data.ncon),
            "out_of_table": table_violation,
            "time": float(self.data.time),
        }

    def render_rgb(self) -> np.ndarray:
        height, width = self.image_size
        with mujoco.Renderer(self.model, height=height, width=width) as renderer:
            renderer.update_scene(self.data, camera=self.camera)
            return renderer.render()

    def object_position(self, name: str) -> np.ndarray:
        return self.data.xpos[self.object_body_ids[name]]

    def target_position(self, geom_name: str) -> np.ndarray:
        return self.data.geom_xpos[self.target_geom_ids[geom_name]]

    def tcp_position(self) -> np.ndarray:
        return np.mean(self.data.geom_xpos[list(self.tcp_geom_ids)], axis=0)

    def tcp_jacobian(self) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        total = np.zeros((3, self.model.nv))
        for geom_id in self.tcp_geom_ids:
            mujoco.mj_jacGeom(self.model, self.data, jacp, jacr, geom_id)
            total += jacp
        return total / len(self.tcp_geom_ids)

    def set_arm_actuator_strength(self, kp: float = 100.0, force_limit: float = 80.0) -> None:
        self.model.actuator_gainprm[:6, 0] = kp
        self.model.actuator_biasprm[:6, 1] = -kp
        self.model.actuator_forcerange[:6, 0] = -force_limit
        self.model.actuator_forcerange[:6, 1] = force_limit

    def set_gripper_actuator_strength(self, kp: float = 800.0, force_limit: float = 140.0) -> None:
        self.model.actuator_gainprm[6, 0] = kp
        self.model.actuator_biasprm[6, 1] = -kp
        self.model.actuator_forcerange[6, 0] = -force_limit
        self.model.actuator_forcerange[6, 1] = force_limit

    def set_grasp_contact_friction(self, sliding: float = 3.0, torsional: float = 0.02, rolling: float = 0.002) -> None:
        friction = np.array([sliding, torsional, rolling], dtype=float)
        for name in OBJECTS:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom")
            self.model.geom_friction[geom_id] = friction
        for geom_id in self.tcp_geom_ids:
            self.model.geom_friction[geom_id] = friction

    def _choose_active_objects(self, count: int) -> list[str]:
        if self.task.relation == "leftmost_cube":
            spatial_objects = ["red_cube", "blue_cube", "green_cube", "yellow_cube", "green_ball"]
            if count > len(spatial_objects):
                spatial_objects.extend(["red_cylinder", "blue_cylinder"])
            return spatial_objects[:count]
        target = self._resolve_target_object_for_sampling()
        distractors = [name for name in OBJECTS if name != target]
        self.rng.shuffle(distractors)
        active = [target, *distractors[: max(0, count - 1)]]
        return active

    def _resolve_target_object_for_sampling(self) -> str:
        if self.task.target_object is not None:
            self.episode_target_object = self.task.target_object
            return self.episode_target_object
        self.episode_target_object = "red_cube"
        return self.episode_target_object

    def _place_objects(self) -> None:
        active_positions = self._sample_relation_positions() if self.task.relation == "leftmost_cube" else self._sample_workspace_positions(len(self.active_objects))
        parked_positions = self._parking_positions()
        for name in OBJECTS:
            pos = active_positions.pop(0) if name in self.active_objects else parked_positions.pop(0)
            self._set_free_object_pose(name, pos)

        if self.task.relation in {"leftmost", "leftmost_cube"}:
            # The free-joint poses above are stored in qpos; update xpos before comparing them.
            mujoco.mj_forward(self.model, self.data)
            candidates = self.active_objects
            if self.task.relation == "leftmost_cube":
                candidates = [name for name in candidates if name.endswith("_cube")]
            self.episode_target_object = min(candidates, key=lambda item: self.object_position(item)[0])

    def _sample_relation_positions(self) -> list[np.ndarray]:
        """Make the leftmost-cube label resolvable at the RGB calibration precision."""
        for _ in range(200):
            positions = self._sample_workspace_positions(len(self.active_objects))
            by_name = dict(zip(self.active_objects, positions))
            cube_x = sorted(float(by_name[name][0]) for name in self.active_objects if name.endswith("_cube"))
            if len(cube_x) < 2 or cube_x[1] - cube_x[0] >= MIN_LEFTMOST_CUBE_X_GAP:
                return positions
        raise RuntimeError("could not sample a visually separable leftmost cube")

    def _sample_workspace_positions(self, count: int) -> list[np.ndarray]:
        x_min, x_max, y_min, y_max, minimum_distance = WORKSPACE_PROFILES[self.workspace_profile]
        if self.workspace_profile == "core_v2" and count > COMPLEXITY_COUNTS["language"]:
            minimum_distance = 0.06
        positions: list[np.ndarray] = []
        attempts = 0
        max_attempts = 5000 if self.workspace_profile == "core_v2" else 500
        while len(positions) < count and attempts < max_attempts:
            attempts += 1
            candidate = np.array([
                self.rng.uniform(x_min, x_max),
                self.rng.uniform(y_min, y_max),
                0.026,
            ])
            if all(np.linalg.norm(candidate[:2] - pos[:2]) > minimum_distance for pos in positions):
                positions.append(candidate)
        if len(positions) != count:
            raise RuntimeError("could not sample non-overlapping object positions")
        return positions

    def _parking_positions(self) -> list[np.ndarray]:
        return [
            np.array([0.68, -0.28, 0.026]),
            np.array([0.76, -0.20, 0.026]),
            np.array([0.76, -0.10, 0.026]),
            np.array([0.76, 0.00, 0.026]),
            np.array([0.76, 0.10, 0.026]),
            np.array([0.76, 0.20, 0.026]),
            np.array([0.68, 0.28, 0.026]),
        ]

    def _set_free_object_pose(self, name: str, pos: np.ndarray) -> None:
        qadr = self.object_qpos_addr[name]
        vadr = self.object_qvel_addr[name]
        self.data.qpos[qadr : qadr + 7] = [pos[0], pos[1], pos[2], 1.0, 0.0, 0.0, 0.0]
        self.data.qvel[vadr : vadr + 6] = 0.0

    @staticmethod
    def _prepared_load_path(scene_path: Path) -> Path:
        cache_root = Path(tempfile.gettempdir()) / f"vla_robot_grasping_mujoco_{os.getpid()}"
        src_asset_dir = PROJECT_ROOT / "external" / "wx250s_assets"
        dst_asset_dir = cache_root / "external" / "mujoco_menagerie" / "trossen_wx250s"
        shutil.copytree(src_asset_dir, dst_asset_dir, dirs_exist_ok=True)

        cached_scene = dst_asset_dir / scene_path.name
        shutil.copy2(scene_path, cached_scene)
        return cached_scene


def _main() -> None:
    env = WidowXTabletopEnv(seed=0)
    obs = env.reset(task="pick_red_cube", complexity="medium", seed=0)
    image = env.render_rgb()
    print("WidowXTabletopEnv self-check OK")
    print(f"python scene: {env.load_path}")
    print(f"instruction: {obs['instruction']}")
    print(f"active_objects: {', '.join(obs['active_objects'])}")
    print(f"tcp_position: {np.round(env.tcp_position(), 4).tolist()}")
    print(f"render_shape: {image.shape}, render_nonblank: {bool(image.max() > image.min())}")


if __name__ == "__main__":
    _main()
