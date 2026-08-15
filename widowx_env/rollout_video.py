from __future__ import annotations

from pathlib import Path
import subprocess

import mujoco
import numpy as np


class Mp4FrameRecorder:
    """Write sparse MuJoCo RGB frames to a single MP4 without storing raw video in memory."""

    def __init__(self, env, output: Path, width: int, height: int, fps: int, frame_stride: int, camera: str) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        self.frame_stride = max(1, int(frame_stride))
        self.step_count = 0
        self.frame_count = 0
        self.renderer = mujoco.Renderer(env.model, height=height, width=width)
        self.env = env
        self.camera = camera
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def capture(self) -> None:
        self.renderer.update_scene(self.env.data, camera=self.camera)
        if self.process.stdin is None:
            raise RuntimeError("video writer is closed")
        self.process.stdin.write(np.ascontiguousarray(self.renderer.render()).tobytes())
        self.frame_count += 1

    def sync(self) -> None:
        self.step_count += 1
        if self.step_count % self.frame_stride == 0:
            self.capture()

    def close(self) -> None:
        self.capture()
        self.renderer.close()
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.wait():
            raise RuntimeError("ffmpeg video export failed")
