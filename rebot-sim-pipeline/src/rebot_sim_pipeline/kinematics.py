"""Numerical inverse kinematics against any forward-kinematics callback.

The solver never needs a robot model: it calls ``fk(q)`` for the end-effector
pose and differentiates task-space residuals by finite differences. In Isaac
Sim, ``fk`` can simply set the articulation's joints and read a link pose,
which updates without a physics step.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


Pose = tuple[np.ndarray, np.ndarray]  # position (3,), rotation matrix (3, 3)
ForwardKinematics = Callable[[np.ndarray], Pose]
Residual = Callable[[np.ndarray, np.ndarray], np.ndarray]


def quat_to_matrix(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    """Rotation matrix of a (w, x, y, z) quaternion; need not be normalized."""
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    w, x, y, z = q / np.linalg.norm(q)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def point_residual(point_local: np.ndarray, target: np.ndarray) -> Residual:
    """Error of an end-effector-fixed point from a world target."""
    return lambda position, rotation: position + rotation @ point_local - target


def solve_ik(
    fk: ForwardKinematics,
    residual: Residual,
    seeds: Sequence[np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    prefer_near: np.ndarray | None = None,
    iterations: int = 150,
    tolerance: float = 5e-4,
    damping: float = 1e-3,
    max_step: float = 0.2,
    fd_epsilon: float = 1e-4,
) -> tuple[np.ndarray, float]:
    """Damped least squares from each seed; returns (best q, residual norm).

    Joint limits are enforced by clipping. Among near-equal fits the solution
    closest to ``prefer_near`` wins, which avoids needless configuration flips.
    """
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    best_q = np.clip(np.asarray(seeds[0], dtype=np.float64), lower, upper)
    best_score = float("inf")
    for seed in seeds:
        q = np.clip(np.asarray(seed, dtype=np.float64), lower, upper)
        error = residual(*fk(q))
        for _ in range(iterations):
            if np.linalg.norm(error) < tolerance:
                break
            jacobian = np.empty((error.size, q.size))
            for joint in range(q.size):
                probe = q.copy()
                probe[joint] += fd_epsilon
                jacobian[:, joint] = (residual(*fk(probe)) - error) / fd_epsilon
            step = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(error.size), -error
            )
            q = np.clip(q + np.clip(step, -max_step, max_step), lower, upper)
            error = residual(*fk(q))
        score = float(np.linalg.norm(error))
        if prefer_near is not None:
            score += 1e-3 * float(np.linalg.norm(q - prefer_near))
        if score < best_score:
            best_q, best_score = q, score
    return best_q, float(np.linalg.norm(residual(*fk(best_q))))
