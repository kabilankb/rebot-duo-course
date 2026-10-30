"""Sampling object layouts that a fixed-base arm can actually reach."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def footprint_points(
    xy: np.ndarray, yaw: float, half_length: float, samples: int = 9
) -> np.ndarray:
    """Points along an object's long (local X) axis, used for spacing checks."""
    direction = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    offsets = np.linspace(-half_length, half_length, samples)
    return np.asarray(xy, dtype=np.float64)[None, :] + offsets[:, None] * direction[None, :]


def sample_workspace_layout(
    rng: np.random.Generator,
    names: Sequence[str],
    config: Mapping[str, Any],
) -> dict[str, tuple[np.ndarray, float]] | None:
    """Rejection-sample non-overlapping planar poses ``{name: (xy, yaw)}``.

    Each object centre lies in an annulus around the robot base, in front of
    the robot; its whole footprint stays inside the workspace bounds, away from
    the robot base and any keep-out circles, and apart from the other objects.
    Returns ``None`` if no layout was found within ``max_attempts``.

    Config keys: ``robot_base_xy_m``, ``center_radius_m`` [min, max],
    ``bounds_x_m``, ``bounds_y_m``, ``robot_base_clearance_m``,
    ``min_separation_m``, ``footprint_half_length_m`` {name: m}, ``yaw_deg``
    [min, max]; optional ``robot_heading_xy`` (default [0, -1]),
    ``keep_out_circles`` [{center_xy_m, radius_m}] and ``max_attempts``.
    """
    base = np.asarray(config["robot_base_xy_m"], dtype=np.float64)
    heading = np.asarray(config.get("robot_heading_xy", (0.0, -1.0)), dtype=np.float64)
    heading /= np.linalg.norm(heading)
    left = np.asarray([-heading[1], heading[0]])
    radius_min, radius_max = (float(v) for v in config["center_radius_m"])
    x_min, x_max = (float(v) for v in config["bounds_x_m"])
    y_min, y_max = (float(v) for v in config["bounds_y_m"])
    yaw_min, yaw_max = (float(v) for v in config["yaw_deg"])
    base_clearance = float(config["robot_base_clearance_m"])
    separation = float(config["min_separation_m"])
    half_lengths = config["footprint_half_length_m"]
    keep_out = [
        (np.asarray(item["center_xy_m"], dtype=np.float64), float(item["radius_m"]))
        for item in config.get("keep_out_circles", [])
    ]
    for _ in range(int(config.get("max_attempts", 500))):
        layout: dict[str, tuple[np.ndarray, float]] = {}
        placed: list[np.ndarray] = []
        for name in names:
            # Uniform over the annulus area, not over the radius.
            radius = math.sqrt(rng.uniform(radius_min**2, radius_max**2))
            angle = rng.uniform(-math.pi / 2.0, math.pi / 2.0)
            xy = base + radius * (math.cos(angle) * heading + math.sin(angle) * left)
            yaw = math.radians(float(rng.uniform(yaw_min, yaw_max)))
            points = footprint_points(xy, yaw, float(half_lengths[name]))
            if not (
                np.all((points[:, 0] >= x_min) & (points[:, 0] <= x_max))
                and np.all((points[:, 1] >= y_min) & (points[:, 1] <= y_max))
                and np.min(np.linalg.norm(points - base, axis=1)) >= base_clearance
                and all(
                    np.min(np.linalg.norm(points - center, axis=1)) >= keep_radius
                    for center, keep_radius in keep_out
                )
                and all(
                    np.min(np.linalg.norm(points[:, None] - other[None], axis=2))
                    >= separation
                    for other in placed
                )
            ):
                break
            layout[name] = (xy, yaw)
            placed.append(points)
        else:
            return layout
    return None
