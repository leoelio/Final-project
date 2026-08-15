from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


COLOR_NAMES = ("red", "blue", "green", "yellow")
CUBE_MIN_AREA = 160
CUBE_MAX_AREA = 650
CUBE_MIN_FILL_RATIO = 0.70
RECOVERY_CUBE_MIN_FILL_RATIO = 0.50
STATIC_COLOR_TARGETS = {
    "red": np.array([0.50, -0.16], dtype=float),
    "blue": np.array([0.50, 0.16], dtype=float),
}


@dataclass(frozen=True)
class Detection:
    color: str
    center_uv: np.ndarray
    area: int
    fill_ratio: float
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class PlaneCalibration:
    matrix: np.ndarray
    image_size: int
    camera: str
    workspace_profile: str
    rms_error_m: float

    def pixel_to_world(self, center_uv: np.ndarray, z: float = 0.026) -> np.ndarray:
        uv1 = np.array([float(center_uv[0]), float(center_uv[1]), 1.0], dtype=float)
        xy = self.matrix @ uv1
        return np.array([xy[0], xy[1], z], dtype=float)

    def to_dict(self) -> dict:
        return {
            "matrix": self.matrix.tolist(),
            "image_size": self.image_size,
            "camera": self.camera,
            "workspace_profile": self.workspace_profile,
            "rms_error_m": self.rms_error_m,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlaneCalibration":
        return cls(
            matrix=np.asarray(data["matrix"], dtype=float),
            image_size=int(data["image_size"]),
            camera=str(data["camera"]),
            workspace_profile=str(data["workspace_profile"]),
            rms_error_m=float(data["rms_error_m"]),
        )


def save_calibration(path: Path, calibration: PlaneCalibration, anchors: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"calibration": calibration.to_dict(), "anchors": anchors}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_calibration(path: Path) -> PlaneCalibration:
    return PlaneCalibration.from_dict(json.loads(path.read_text(encoding="utf-8"))["calibration"])


def fit_plane_calibration(
    pixels_uv: np.ndarray,
    world_xy: np.ndarray,
    image_size: int,
    camera: str,
    workspace_profile: str,
) -> PlaneCalibration:
    design = np.column_stack((pixels_uv, np.ones(len(pixels_uv), dtype=float)))
    coefficients, _, _, _ = np.linalg.lstsq(design, world_xy, rcond=None)
    matrix = coefficients.T
    predicted = design @ coefficients
    rms_error_m = float(np.sqrt(np.mean(np.sum((predicted - world_xy) ** 2, axis=1))))
    return PlaneCalibration(
        matrix=matrix,
        image_size=image_size,
        camera=camera,
        workspace_profile=workspace_profile,
        rms_error_m=rms_error_m,
    )


def color_mask(image: np.ndarray, color: str) -> np.ndarray:
    rgb = image.astype(np.int16)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if color == "red":
        return (red > 100) & (red > green * 3 // 2) & (red > blue * 3 // 2)
    if color == "blue":
        return (blue > 90) & (blue > red * 3 // 2) & (blue > green * 6 // 5)
    if color == "green":
        return (green > 90) & (green > red * 5 // 4) & (green > blue * 6 // 5)
    if color == "yellow":
        return (red > 120) & (green > 100) & (blue < 120) & (red > blue * 3 // 2)
    raise KeyError(f"unknown color: {color}")


def connected_components(mask: np.ndarray, min_area: int = 8) -> list[Detection]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    detections: list[Detection] = []
    for start_v, start_u in zip(*np.nonzero(mask)):
        if visited[start_v, start_u]:
            continue
        stack = [(int(start_v), int(start_u))]
        visited[start_v, start_u] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            v, u = stack.pop()
            pixels.append((v, u))
            for next_v, next_u in ((v - 1, u), (v + 1, u), (v, u - 1), (v, u + 1)):
                if 0 <= next_v < height and 0 <= next_u < width and mask[next_v, next_u] and not visited[next_v, next_u]:
                    visited[next_v, next_u] = True
                    stack.append((next_v, next_u))
        if len(pixels) < min_area:
            continue
        array = np.asarray(pixels, dtype=float)
        min_v, min_u = np.min(array, axis=0).astype(int)
        max_v, max_u = np.max(array, axis=0).astype(int)
        bbox_area = max(1, (max_v - min_v + 1) * (max_u - min_u + 1))
        detections.append(
            Detection(
                color="",
                center_uv=np.array([float(array[:, 1].mean()), float(array[:, 0].mean())]),
                area=len(pixels),
                fill_ratio=float(len(pixels) / bbox_area),
                bbox=(min_u, min_v, max_u, max_v),
            )
        )
    return detections


def detect_colored_regions(image: np.ndarray, color: str, min_area: int = 8) -> list[Detection]:
    return [
        Detection(color=color, center_uv=item.center_uv, area=item.area, fill_ratio=item.fill_ratio, bbox=item.bbox)
        for item in connected_components(color_mask(image, color), min_area=min_area)
    ]


def source_workspace_mask(shape: tuple[int, int], calibration: PlaneCalibration) -> np.ndarray:
    """Keep only pixels that project to the randomized source-object workspace."""
    height, width = shape
    u, v = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    x = calibration.matrix[0, 0] * u + calibration.matrix[0, 1] * v + calibration.matrix[0, 2]
    y = calibration.matrix[1, 0] * u + calibration.matrix[1, 1] * v + calibration.matrix[1, 2]
    return (0.21 <= x) & (x <= 0.47) & (-0.13 <= y) & (y <= 0.13)


def static_target_exclusion_mask(shape: tuple[int, int], calibration: PlaneCalibration, color: str) -> np.ndarray:
    """Exclude a fixed same-color pad so it cannot merge with a source cube."""
    target_xy = STATIC_COLOR_TARGETS.get(color)
    if target_xy is None:
        return np.zeros(shape, dtype=bool)
    height, width = shape
    u, v = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    x = calibration.matrix[0, 0] * u + calibration.matrix[0, 1] * v + calibration.matrix[0, 2]
    y = calibration.matrix[1, 0] * u + calibration.matrix[1, 1] * v + calibration.matrix[1, 2]
    return (x - target_xy[0]) ** 2 + (y - target_xy[1]) ** 2 <= 0.065**2


def cube_candidates(image: np.ndarray, calibration: PlaneCalibration) -> dict[str, Detection]:
    """Return the visible cube of each color from the randomized workspace, not the static pads."""
    candidates: dict[str, Detection] = {}
    workspace = source_workspace_mask(image.shape[:2], calibration)
    for color in COLOR_NAMES:
        regions = [
            Detection(color=color, center_uv=item.center_uv, area=item.area, fill_ratio=item.fill_ratio, bbox=item.bbox)
            for item in connected_components(color_mask(image, color) & workspace & ~static_target_exclusion_mask(image.shape[:2], calibration, color))
        ]
        scored: list[tuple[float, Detection]] = []
        for region in regions:
            world = calibration.pixel_to_world(region.center_uv)
            in_workspace = 0.21 <= world[0] <= 0.47 and -0.13 <= world[1] <= 0.13
            square_like = region.fill_ratio >= CUBE_MIN_FILL_RATIO
            small_object = CUBE_MIN_AREA <= region.area <= CUBE_MAX_AREA
            if in_workspace and square_like and small_object:
                scored.append((float(region.area), region))
        if scored:
            candidates[color] = max(scored, key=lambda item: item[0])[1]
    return candidates


def locate_object(
    image: np.ndarray,
    calibration: PlaneCalibration,
    object_name: str,
) -> tuple[np.ndarray, Detection]:
    candidates = cube_candidates(image, calibration)
    color = object_name.split("_", 1)[0]
    if color not in candidates:
        raise LookupError(f"no visible {color} cube candidate")
    detection = candidates[color]
    return calibration.pixel_to_world(detection.center_uv), detection


def locate_leftmost_cube(image: np.ndarray, calibration: PlaneCalibration) -> tuple[str, np.ndarray, Detection]:
    candidates = cube_candidates(image, calibration)
    if not candidates:
        raise LookupError("no visible cube candidates")
    color, detection = min(
        candidates.items(),
        key=lambda item: float(calibration.pixel_to_world(item[1].center_uv)[0]),
    )
    return f"{color}_cube", calibration.pixel_to_world(detection.center_uv), detection


def locate_color_near(
    image: np.ndarray,
    calibration: PlaneCalibration,
    color: str,
    reference_xy: np.ndarray,
    radius: float,
    min_area: int = CUBE_MIN_AREA,
) -> tuple[np.ndarray, Detection]:
    """Locate a cube-sized colored object near a known target, independent of the source workspace."""
    reference_xy = np.asarray(reference_xy, dtype=float)[:2]
    candidates: list[tuple[float, Detection]] = []
    for region in detect_colored_regions(image, color):
        if not (min_area <= region.area <= CUBE_MAX_AREA and region.fill_ratio >= CUBE_MIN_FILL_RATIO):
            continue
        world = calibration.pixel_to_world(region.center_uv)
        distance = float(np.linalg.norm(world[:2] - reference_xy))
        if distance <= radius:
            candidates.append((distance, region))
    if not candidates:
        raise LookupError(f"no compact {color} object within {radius:.3f} m of the reference")
    _, detection = min(candidates, key=lambda item: item[0])
    return calibration.pixel_to_world(detection.center_uv), detection


def relocate_known_object(
    image: np.ndarray,
    calibration: PlaneCalibration,
    object_name: str,
    previous_xy: np.ndarray,
    search_scope: str = "source",
) -> tuple[np.ndarray, Detection]:
    """Recover a known colored object from the source workspace or a bounded tabletop fallback."""
    if search_scope not in {"source", "table"}:
        raise ValueError(f"unknown recovery search scope: {search_scope}")
    color = object_name.split("_", 1)[0]
    previous_xy = np.asarray(previous_xy, dtype=float)[:2]
    candidates: list[tuple[float, Detection]] = []
    mask = color_mask(image, color) & ~static_target_exclusion_mask(image.shape[:2], calibration, color)
    for region in connected_components(mask):
        if not (CUBE_MIN_AREA <= region.area <= CUBE_MAX_AREA and region.fill_ratio >= RECOVERY_CUBE_MIN_FILL_RATIO):
            continue
        world = calibration.pixel_to_world(region.center_uv)
        in_scope = (
            0.21 <= world[0] <= 0.47 and -0.13 <= world[1] <= 0.13
            if search_scope == "source"
            else 0.18 <= world[0] <= 0.62 and -0.28 <= world[1] <= 0.28
        )
        if in_scope:
            candidates.append((float(np.linalg.norm(world[:2] - previous_xy)), region))
    if not candidates:
        raise LookupError(f"no recoverable {color} object in the {search_scope} workspace")
    _, detection = min(candidates, key=lambda item: item[0])
    return calibration.pixel_to_world(detection.center_uv), detection


def draw_detection_overlay(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    output = image.copy()
    for detection in detections:
        u, v = np.rint(detection.center_uv).astype(int)
        color = np.array([255, 255, 255], dtype=np.uint8)
        output[max(0, v - 3) : min(output.shape[0], v + 4), max(0, u - 1) : min(output.shape[1], u + 2)] = color
        output[max(0, v - 1) : min(output.shape[0], v + 2), max(0, u - 3) : min(output.shape[1], u + 4)] = color
    return output


def write_ppm(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError("PPM output expects RGB image")
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + image.astype(np.uint8).tobytes())
