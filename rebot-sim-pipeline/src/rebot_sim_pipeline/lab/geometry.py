"""Small rotation helpers shared by the scene and the expert driver (numpy only).

Isaac Lab 3 orders quaternions (x, y, z, w); the YAML files and USD use (w, x, y, z).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Unit (w, x, y, z) quaternion of a rotation matrix."""
    m = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(m)
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def wxyz_to_xyzw(q: Sequence[float]) -> tuple[float, float, float, float]:
    w, x, y, z = (float(v) for v in q)
    return (x, y, z, w)


def yaw_quat_wxyz(yaw_rad: float) -> np.ndarray:
    return np.asarray([np.cos(yaw_rad / 2.0), 0.0, 0.0, np.sin(yaw_rad / 2.0)])


def look_at_opengl(eye: Sequence[float], target: Sequence[float], up: Sequence[float]) -> np.ndarray:
    """Rotation of a camera at ``eye`` looking at ``target``, OpenGL/USD convention.

    The camera looks along its local -Z with +Y up, so the columns are
    (right, up, -forward) expressed in the parent frame.
    """
    forward = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, dtype=np.float64))
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    return np.column_stack((right, true_up, -forward))


def quat_xyzw_to_matrix(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
