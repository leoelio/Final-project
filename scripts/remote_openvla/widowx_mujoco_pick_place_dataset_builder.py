"""TFDS builder for the versioned WidowX MuJoCo RLDS source.

Run this file on the remote OpenVLA machine after setting
WIDOWX_MUJOCO_RLDS_SOURCE_DIR to the unpacked source directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator, Tuple

import numpy as np
import tensorflow_datasets as tfds


class WidowXMujocoPickPlace(tfds.core.GeneratorBasedBuilder):
    """Builds a TFDS/RLDS train split from versioned successful MuJoCo episodes."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial Core V2 successful-demo source conversion."}

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(shape=(128, 128, 3), dtype=np.uint8),
                                    "joint_state": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                                    "gripper_state": tfds.features.Scalar(dtype=np.float32),
                                }
                            ),
                            "action": tfds.features.Tensor(shape=(8,), dtype=np.float32),
                            "discount": tfds.features.Scalar(dtype=np.float32),
                            "reward": tfds.features.Scalar(dtype=np.float32),
                            "is_first": tfds.features.Scalar(dtype=np.bool_),
                            "is_last": tfds.features.Scalar(dtype=np.bool_),
                            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                            "language_instruction": tfds.features.Text(),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "source_episode_path": tfds.features.Text(),
                            "source_task": tfds.features.Text(),
                            "source_episode_index": tfds.features.Scalar(dtype=np.int32),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        del dl_manager
        return {"train": self._generate_examples()}

    def _source_dir(self) -> Path:
        value = os.environ.get("WIDOWX_MUJOCO_RLDS_SOURCE_DIR")
        if not value:
            raise ValueError("set WIDOWX_MUJOCO_RLDS_SOURCE_DIR before invoking tfds build")
        source_dir = Path(value).resolve()
        manifest = source_dir / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        return source_dir

    def _generate_examples(self) -> Iterator[Tuple[str, Any]]:
        source_dir = self._source_dir()
        episode_paths = sorted((source_dir / "episodes").glob("episode_*.npz"))
        if not episode_paths:
            raise FileNotFoundError(f"no source episodes found under {source_dir}")
        for episode_path in episode_paths:
            with np.load(episode_path) as data:
                image_paths = [source_dir / str(item) for item in data["image_paths"]]
                state = data["state"].astype(np.float32)
                action = data["action"].astype(np.float32)
                count = len(image_paths)
                if state.shape != (count, 8) or action.shape != (count, 8):
                    raise ValueError(f"invalid source episode shape: {episode_path}")
                instruction = str(data["language_instruction"].item())
                steps = []
                for index, image_path in enumerate(image_paths):
                    if not image_path.exists():
                        raise FileNotFoundError(image_path)
                    steps.append(
                        {
                            "observation": {
                                "image": str(image_path),
                                "joint_state": state[index, :7],
                                "gripper_state": state[index, 7],
                            },
                            "action": action[index],
                            "discount": float(data["discount"][index]),
                            "reward": float(data["reward"][index]),
                            "is_first": bool(data["is_first"][index]),
                            "is_last": bool(data["is_last"][index]),
                            "is_terminal": bool(data["is_terminal"][index]),
                            "language_instruction": instruction,
                        }
                    )
                yield episode_path.stem, {
                    "steps": steps,
                    "episode_metadata": {
                        "source_episode_path": str(episode_path),
                        "source_task": str(data["task"].item()),
                        "source_episode_index": int(data["source_episode_index"].item()),
                    },
                }
