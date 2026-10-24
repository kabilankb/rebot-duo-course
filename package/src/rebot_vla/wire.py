"""Simulator units <-> the bimanual reBot wire format (see ``embodiment``).

Per arm the conversions are ``rebot_sim_pipeline.conventions``, the same code
that wrote the datasets, so the client and the recordings cannot disagree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from rebot_sim_pipeline import conventions

from .embodiment import CAMERA_SOURCES

ARMS = ("left_arm", "right_arm")


def state_from_sim(q: Mapping[str, Sequence[float]], opening: Mapping[str, float]) -> np.ndarray:
    """14 wire state values from measured joints (rad) and mean finger openings (m) per arm."""
    return np.concatenate([conventions.arm_state(q[arm], opening[arm]) for arm in ARMS]).astype(np.float32)


def command_from_action(row: Sequence[float]) -> np.ndarray:
    """One 14-value wire action row -> the env's action ``[left q(6) rad, gripper m, right ...]``."""
    row = np.asarray(row, dtype=np.float64).reshape(-1)
    if row.shape != (14,):
        raise ValueError(f"action row must have 14 values, got {row.shape}")
    command = []
    for index in range(len(ARMS)):
        q, gripper = conventions.action_to_command(row[7 * index: 7 * index + 7])
        command += [*q, gripper]
    return np.asarray(command, dtype=np.float32)


def cameras_from_sim(images: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Wire camera fields (``top_cam``...) from simulator camera images (``top``...), uint8 HxWx3."""
    return {wire: np.ascontiguousarray(np.asarray(images[source])[..., :3], dtype=np.uint8)
            for wire, source in CAMERA_SOURCES.items()}
