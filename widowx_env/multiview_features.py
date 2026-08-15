from __future__ import annotations

import numpy as np

from .vision_grounding import COLOR_NAMES, PlaneCalibration, detect_colored_regions


TASK_NAMES = (
    "place_blue_cube_blue_pad",
    "place_blue_cube_red_pad",
    "place_red_cube_red_pad",
    "move_leftmost_cube_to_bowl",
)
TARGET_NAMES = ("target_blue_pad", "target_red_pad", "target_bowl")


def _pool_rgb(image: np.ndarray, grid: int) -> np.ndarray:
    """Average-pool RGB without introducing another learned visual encoder."""
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError(f"expected RGB image, got {image.shape}")
    rows = np.array_split(np.arange(height), grid)
    cols = np.array_split(np.arange(width), grid)
    return np.asarray([[image[np.ix_(row, col)].mean(axis=(0, 1)) / 255.0 for col in cols] for row in rows], dtype=np.float32)


def _region_features(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    values: list[float] = []
    for color in COLOR_NAMES:
        regions = detect_colored_regions(image, color)
        mask_area = float(sum(region.area for region in regions)) / float(height * width)
        if regions:
            largest = max(regions, key=lambda item: item.area)
            u0, v0, u1, v1 = largest.bbox
            values.extend(
                [
                    mask_area,
                    float(largest.area) / float(height * width),
                    float(largest.center_uv[0]) / float(width),
                    float(largest.center_uv[1]) / float(height),
                    float(largest.fill_ratio),
                    float(u1 - u0 + 1) / float(width),
                    float(v1 - v0 + 1) / float(height),
                ]
            )
        else:
            values.extend([mask_area, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0])
    return np.asarray(values, dtype=np.float32)


def _world_to_pixel(calibration: PlaneCalibration, xy: np.ndarray) -> np.ndarray:
    matrix = np.asarray(calibration.matrix, dtype=float)
    return np.linalg.solve(matrix[:, :2], np.asarray(xy, dtype=float) - matrix[:, 2])


def _target_patch(top_rgb: np.ndarray, calibration: PlaneCalibration, target_xy: np.ndarray, grid: int = 4) -> np.ndarray:
    center_u, center_v = np.rint(_world_to_pixel(calibration, target_xy)).astype(int)
    radius = max(12, top_rgb.shape[0] // 9)
    padded = np.pad(top_rgb, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    center_u += radius
    center_v += radius
    patch = padded[center_v - radius : center_v + radius, center_u - radius : center_u + radius]
    return _pool_rgb(patch, grid).reshape(-1)


def extract_multiview_features(
    top_rgb: np.ndarray,
    front_rgb: np.ndarray,
    task: str,
    source_name: str,
    target_name: str,
    target_xy: np.ndarray,
    calibration: PlaneCalibration,
    pool_grid: int = 8,
) -> np.ndarray:
    """Features available to the runtime gate: two RGB views plus known task configuration."""
    if task not in TASK_NAMES:
        raise KeyError(f"unsupported task: {task}")
    if target_name not in TARGET_NAMES:
        raise KeyError(f"unsupported target: {target_name}")
    source_color = source_name.split("_", 1)[0]
    source_one_hot = np.asarray([float(source_color == color) for color in COLOR_NAMES], dtype=np.float32)
    task_one_hot = np.asarray([float(task == name) for name in TASK_NAMES], dtype=np.float32)
    target_one_hot = np.asarray([float(target_name == name) for name in TARGET_NAMES], dtype=np.float32)
    same_color = float(
        (source_color == "blue" and target_name == "target_blue_pad")
        or (source_color == "red" and target_name == "target_red_pad")
    )
    return np.concatenate(
        [
            _pool_rgb(top_rgb, pool_grid).reshape(-1),
            _pool_rgb(front_rgb, pool_grid).reshape(-1),
            _region_features(top_rgb),
            _region_features(front_rgb),
            _target_patch(top_rgb, calibration, target_xy),
            task_one_hot,
            source_one_hot,
            target_one_hot,
            np.asarray([same_color], dtype=np.float32),
        ]
    ).astype(np.float32)


def feature_spec(pool_grid: int = 8) -> dict[str, int]:
    return {
        "top_rgb_pooled": pool_grid * pool_grid * 3,
        "front_rgb_pooled": pool_grid * pool_grid * 3,
        "top_regions": len(COLOR_NAMES) * 7,
        "front_regions": len(COLOR_NAMES) * 7,
        "top_target_patch": 4 * 4 * 3,
        "task_one_hot": len(TASK_NAMES),
        "source_color_one_hot": len(COLOR_NAMES),
        "target_one_hot": len(TARGET_NAMES),
        "same_color_flag": 1,
    }


def feature_indices(spec: dict[str, int], view: str) -> np.ndarray:
    """Return a fixed ablation slice without changing labels or task metadata."""
    order = ("top_rgb_pooled", "front_rgb_pooled", "top_regions", "front_regions", "top_target_patch", "task_one_hot", "source_color_one_hot", "target_one_hot", "same_color_flag")
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name in order:
        offsets[name] = (cursor, cursor + int(spec[name]))
        cursor += int(spec[name])
    if view == "top_front":
        return np.arange(cursor, dtype=np.int32)
    if view != "top":
        raise KeyError(f"unknown view ablation: {view}")
    selected = ("top_rgb_pooled", "top_regions", "top_target_patch", "task_one_hot", "source_color_one_hot", "target_one_hot", "same_color_flag")
    return np.concatenate([np.arange(*offsets[name], dtype=np.int32) for name in selected])
