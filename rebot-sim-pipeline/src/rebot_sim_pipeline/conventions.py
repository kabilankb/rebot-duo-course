"""reBot B601 dataset conventions: simulator joint values <-> recorded values.

The single-arm reBot datasets (``rebot-arm-dli-isaacsim``) record what the real
leader/follower pair reports, so a policy trained in simulation drives the real
arm unchanged. The bimanual pipeline uses the same per-arm convention:

* **action** (leader convention): arm joint ``i`` in degrees divided by
  ``LEADER_SCALE[i]``; gripper ``45 * open fraction`` (0 closed, 45 open).
* **observation.state** (follower convention): arm joints are ``-degrees`` (the
  follower directions times the leader scale give -1 on every joint); gripper
  ``270 * open fraction``.

So state and action differ in sign on joints 3, 4 and 5. ``GRIPPER_OPEN_M`` is the
per-finger opening of a fully open gripper.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

LEADER_SCALE = np.asarray([-1.0, -1.0, 1.0, 1.0, 1.0, -1.0])
FOLLOWER_SIGN = -np.ones(6)
GRIPPER_OPEN_M = 0.05
GRIPPER_ACTION_FULL = 45.0
GRIPPER_STATE_FULL = 270.0
JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)


def _open_fraction(opening_m: float) -> float:
    return float(np.clip(opening_m / GRIPPER_OPEN_M, 0.0, 1.0))


def arm_action(q_rad: Sequence[float], gripper_cmd_m: float) -> np.ndarray:
    """7 recorded action values from 6 commanded joint targets and the gripper command."""
    arm = np.degrees(np.asarray(q_rad, dtype=np.float64)) / LEADER_SCALE
    return np.append(arm, GRIPPER_ACTION_FULL * _open_fraction(gripper_cmd_m)).astype(np.float32)


def arm_state(q_rad: Sequence[float], finger_mean_m: float) -> np.ndarray:
    """7 recorded state values from 6 measured joints and the mean finger opening."""
    arm = np.degrees(np.asarray(q_rad, dtype=np.float64)) * FOLLOWER_SIGN
    return np.append(arm, GRIPPER_STATE_FULL * _open_fraction(finger_mean_m)).astype(np.float32)


def action_to_command(action: Sequence[float]) -> tuple[np.ndarray, float]:
    """Inverse of :func:`arm_action`: (6 joint targets in rad, gripper command in m)."""
    action = np.asarray(action, dtype=np.float64)
    q = np.radians(action[:6] * LEADER_SCALE)
    gripper = float(np.clip(action[6] / GRIPPER_ACTION_FULL, 0.0, 1.0)) * GRIPPER_OPEN_M
    return q, gripper


def feature_names(arm_prefixes: Sequence[str]) -> list[str]:
    """LeRobot names, e.g. ``left_shoulder_pan.pos`` ... ``right_gripper.pos``."""
    return [f"{prefix}_{joint}.pos" for prefix in arm_prefixes for joint in JOINT_NAMES]
