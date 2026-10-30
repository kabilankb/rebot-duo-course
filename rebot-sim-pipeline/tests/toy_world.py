"""A toy 6-joint arm and a fake world that carries objects, for tests."""

from __future__ import annotations

import numpy as np

BASE = np.asarray([0.0, 0.455, 0.12])
LINKS = (0.22, 0.20, 0.03)
LOWER = np.radians([-160.0, -180.0, -180.0, -100.0, -90.0, -180.0])
UPPER = np.radians([160.0, 0.0, 0.0, 100.0, 90.0, 180.0])
FINGERTIP = np.asarray([0.0, 0.0, 0.166])
GRASP_OFFSET = np.asarray([0.0, 0.0, 0.091])


def _rx(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def toy_fk(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Upright at q=0; joints 2-4 pitch toward world -Y (the zero heading)."""
    rotation = _rz(q[0])
    position = BASE.copy()
    for joint, length in zip(q[1:4], LINKS):
        rotation = rotation @ _rx(-joint)
        position = position + rotation @ np.asarray([0.0, 0.0, length])
    rotation = rotation @ _ry(q[4]) @ _rz(q[5])
    return position, rotation


def yaw_rotation(yaw: float) -> np.ndarray:
    return _rz(yaw)


class FakeWorld:
    """Attaches the nearest object when the gripper closes near it, carries it
    rigidly with the wrist, and drops it straight down when the gripper opens."""

    def __init__(self, objects: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        self.objects = {name: (p.copy(), r.copy()) for name, (p, r) in objects.items()}
        self.attached: str | None = None
        self.relative: tuple[np.ndarray, np.ndarray] | None = None
        self.released_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def object_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        position, rotation = self.objects[name]
        return position.copy(), rotation.copy()

    def is_attached(self) -> bool:
        return self.attached is not None

    def step(self, q: np.ndarray, gripper: float) -> None:
        position, rotation = toy_fk(q)
        if self.attached is None and gripper <= 0.014:
            tip = position + rotation @ FINGERTIP
            name = min(self.objects, key=lambda n: np.linalg.norm(self.objects[n][0] - tip))
            if np.linalg.norm(self.objects[name][0] - tip) <= 0.06:
                object_position, object_rotation = self.objects[name]
                self.attached = name
                self.relative = (
                    rotation.T @ (object_position - position),
                    rotation.T @ object_rotation,
                )
        elif self.attached is not None and gripper >= 0.026:
            self.released_poses[self.attached] = self.object_pose(self.attached)
            self.attached = None
        if self.attached is not None:
            relative_position, relative_rotation = self.relative
            self.objects[self.attached] = (
                position + rotation @ relative_position,
                rotation @ relative_rotation,
            )
