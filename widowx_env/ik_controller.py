from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .tabletop_env import WidowXTabletopEnv


@dataclass(frozen=True)
class IKResult:
    qpos: np.ndarray
    target: np.ndarray
    tcp_position: np.ndarray
    error_norm: float
    iterations: int
    converged: bool


class DampedLeastSquaresIK:
    """Position-only IK for the WidowX arm joints."""

    def __init__(
        self,
        env: WidowXTabletopEnv,
        damping: float = 0.04,
        max_step: float = 0.08,
        tolerance: float = 0.03,
        max_iterations: int = 200,
    ) -> None:
        self.env = env
        self.damping = damping
        self.max_step = max_step
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.arm_dofs = np.arange(6)
        self.arm_qpos = np.arange(6)

    def solve(self, target: np.ndarray) -> IKResult:
        target = np.asarray(target, dtype=float)
        model = self.env.model
        data = self.env.data
        q_min = model.jnt_range[:6, 0]
        q_max = model.jnt_range[:6, 1]

        converged = False
        error_norm = float("inf")

        for iteration in range(1, self.max_iterations + 1):
            mujoco.mj_forward(model, data)
            tcp_position = self.env.tcp_position()
            error = target - tcp_position
            error_norm = float(np.linalg.norm(error))
            if error_norm < self.tolerance:
                converged = True
                break

            jacp = self.env.tcp_jacobian()
            jac = jacp[:, self.arm_dofs]
            lhs = jac @ jac.T + (self.damping**2) * np.eye(3)
            dq = jac.T @ np.linalg.solve(lhs, error)
            step_norm = float(np.linalg.norm(dq))
            if step_norm > self.max_step:
                dq *= self.max_step / step_norm

            data.qpos[self.arm_qpos] = np.clip(data.qpos[self.arm_qpos] + dq, q_min, q_max)

        mujoco.mj_forward(model, data)
        tcp_position = self.env.tcp_position()
        error_norm = float(np.linalg.norm(target - tcp_position))
        return IKResult(
            qpos=data.qpos[self.arm_qpos].copy(),
            target=target.copy(),
            tcp_position=tcp_position,
            error_norm=error_norm,
            iterations=iteration,
            converged=converged or error_norm < self.tolerance,
        )


def target_above_object(env: WidowXTabletopEnv, object_name: str, z_offset: float = 0.09) -> np.ndarray:
    object_position = env.object_position(object_name)
    return object_position + np.array([0.0, 0.0, z_offset], dtype=float)
