"""Forward kinematics of a serial chain described by USD physics-joint frames.

A USD joint places its child body so that ``body0 * localFrame0`` and
``body1 * localFrame1`` coincide after the joint motion. Walking the chain gives

    T_child = T_parent @ F(localPos0, localRot0) @ M(q) @ inv(F(localPos1, localRot1))

where ``M(q)`` is a rotation about (revolute) or translation along (prismatic)
the joint axis. The joint frames are plain numbers (see the robot YAML), so this
runs without USD or Isaac Sim and is fast enough for finite-difference IK.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .kinematics import Pose, quat_to_matrix

_AXES = {"X": 0, "Y": 1, "Z": 2}


def transform(position: Sequence[float], rotation: np.ndarray) -> np.ndarray:
    """4x4 homogeneous transform."""
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = position
    return matrix


def _axis_rotation(axis: int, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    i, j = [k for k in range(3) if k != axis]
    rotation = np.eye(3)
    rotation[i, i], rotation[i, j], rotation[j, i], rotation[j, j] = c, -s, s, c
    return rotation


@dataclass(frozen=True)
class ChainJoint:
    """One joint: parent-side frame, child-side frame, axis and type."""

    name: str
    kind: str  # "revolute" | "prismatic"
    axis: int
    frame0: np.ndarray  # 4x4, in the parent body
    frame1_inv: np.ndarray  # 4x4, inverse of the frame in the child body

    @classmethod
    def from_spec(cls, spec: Mapping) -> "ChainJoint":
        """Build from ``{name, type, axis, local_pos0, local_rot0_wxyz, local_pos1, local_rot1_wxyz}``."""
        frame0 = transform(spec["local_pos0"], quat_to_matrix(spec["local_rot0_wxyz"]))
        frame1 = transform(spec["local_pos1"], quat_to_matrix(spec["local_rot1_wxyz"]))
        return cls(
            name=str(spec["name"]),
            kind=str(spec["type"]),
            axis=_AXES[str(spec["axis"]).upper()],
            frame0=frame0,
            frame1_inv=np.linalg.inv(frame1),
        )

    def motion(self, q: float) -> np.ndarray:
        if self.kind == "revolute":
            return transform(np.zeros(3), _axis_rotation(self.axis, q))
        offset = np.zeros(3)
        offset[self.axis] = q
        return transform(offset, np.eye(3))


class SerialChain:
    """Base-to-tip forward kinematics for a fixed-base arm placed in the world."""

    def __init__(self, joints: Sequence[Mapping], base_pose: np.ndarray | None = None) -> None:
        self.joints = [ChainJoint.from_spec(spec) for spec in joints]
        self.base_pose = np.eye(4) if base_pose is None else np.asarray(base_pose, dtype=np.float64)

    def link_transforms(self, q: Sequence[float]) -> list[np.ndarray]:
        """World transform of the base and of each joint's child body."""
        if len(q) != len(self.joints):
            raise ValueError(f"expected {len(self.joints)} joint values, got {len(q)}")
        current = self.base_pose
        frames = [current]
        for joint, value in zip(self.joints, q):
            current = current @ joint.frame0 @ joint.motion(float(value)) @ joint.frame1_inv
            frames.append(current)
        return frames

    def fk(self, q: Sequence[float]) -> Pose:
        """World (position, rotation) of the last body in the chain."""
        tip = self.link_transforms(q)[-1]
        return tip[:3, 3].copy(), tip[:3, :3].copy()


def base_pose_from_xy_yaw(position: Sequence[float], yaw_rad: float) -> np.ndarray:
    """World transform of an arm base standing on the table, turned by ``yaw_rad`` about Z."""
    return transform(position, _axis_rotation(2, yaw_rad))
