"""Two-arm layout sampling and work assignment for the bimanual stationery task.

Two reBot arms stand side by side and share one pencil cup. Each object is
assigned to one arm and must lie in that arm's reach and on its side of the
table; the arms then work one at a time, so they never meet over the cup.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .workspace import footprint_points


@dataclass(frozen=True)
class ArmBase:
    """Where an arm stands and which way it faces at joint-1 zero."""

    name: str
    base_xy: np.ndarray
    heading_xy: np.ndarray


def arm_bases(config: Mapping[str, Any]) -> dict[str, ArmBase]:
    """``{arm: ArmBase}`` from ``{arm: {base_xy_m, heading_xy}}``."""
    bases = {}
    for name, item in config.items():
        heading = np.asarray(item.get("heading_xy", (0.0, -1.0)), dtype=np.float64)
        bases[name] = ArmBase(
            name, np.asarray(item["base_xy_m"], dtype=np.float64), heading / np.linalg.norm(heading)
        )
    return bases


def _nearest_arm(xy: np.ndarray, bases: Mapping[str, ArmBase]) -> str:
    return min(bases, key=lambda name: float(np.linalg.norm(xy - bases[name].base_xy)))


def sample_bimanual_layout(
    rng: np.random.Generator,
    names: Sequence[str],
    assignment: Mapping[str, str],
    bases: Mapping[str, ArmBase],
    config: Mapping[str, Any],
) -> tuple[dict[str, tuple[np.ndarray, float]], dict[str, str]] | None:
    """Sample ``({name: (xy, yaw)}, {name: arm})``, or ``None`` if no layout was found.

    ``assignment[name]`` is an arm name, or ``"any"`` to draw the arm at random.
    Each object centre lies in an annulus in front of its arm and is nearer its
    own arm than any other; its footprint stays inside the bounds, clear of every
    arm base and keep-out circle, and apart from the other objects.

    Config keys as in :func:`workspace.sample_workspace_layout`, without
    ``robot_base_xy_m``: ``center_radius_m``, ``bounds_x_m``, ``bounds_y_m``,
    ``robot_base_clearance_m``, ``min_separation_m``, ``footprint_half_length_m``,
    ``yaw_deg``, optional ``keep_out_circles`` and ``max_attempts``.
    """
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
    arm_names = sorted(bases)
    for _ in range(int(config.get("max_attempts", 500))):
        layout: dict[str, tuple[np.ndarray, float]] = {}
        arms: dict[str, str] = {}
        placed: list[np.ndarray] = []
        for name in names:
            arm_name = assignment.get(name, "any")
            if arm_name == "any":
                arm_name = arm_names[int(rng.integers(len(arm_names)))]
            arm = bases[arm_name]
            left = np.asarray([-arm.heading_xy[1], arm.heading_xy[0]])
            # Uniform over the annulus area, not over the radius.
            radius = math.sqrt(rng.uniform(radius_min**2, radius_max**2))
            angle = rng.uniform(-math.pi / 2.0, math.pi / 2.0)
            xy = arm.base_xy + radius * (math.cos(angle) * arm.heading_xy + math.sin(angle) * left)
            yaw = math.radians(float(rng.uniform(yaw_min, yaw_max)))
            points = footprint_points(xy, yaw, float(half_lengths[name]))
            if not (
                _nearest_arm(xy, bases) == arm_name
                and np.all((points[:, 0] >= x_min) & (points[:, 0] <= x_max))
                and np.all((points[:, 1] >= y_min) & (points[:, 1] <= y_max))
                and all(
                    np.min(np.linalg.norm(points - other.base_xy, axis=1)) >= base_clearance
                    for other in bases.values()
                )
                and all(
                    np.min(np.linalg.norm(points - center, axis=1)) >= keep_radius
                    for center, keep_radius in keep_out
                )
                and all(
                    np.min(np.linalg.norm(points[:, None] - other[None], axis=2)) >= separation
                    for other in placed
                )
            ):
                break
            layout[name] = (xy, yaw)
            arms[name] = arm_name
            placed.append(points)
        else:
            return layout, arms
    return None


def work_order(order: Sequence[str], arms: Mapping[str, str]) -> list[tuple[str, list[str]]]:
    """Group the global object order into consecutive per-arm turns.

    ``["pen_red", "pen_blue", "eraser"]`` with arms ``{pen_red: left, pen_blue:
    right, eraser: right}`` gives ``[("left", ["pen_red"]), ("right", ["pen_blue",
    "eraser"])]``. Pens should come first so they stand on the flat cup floor.
    """
    turns: list[tuple[str, list[str]]] = []
    for name in order:
        arm = arms[name]
        if turns and turns[-1][0] == arm:
            turns[-1][1].append(name)
        else:
            turns.append((arm, [name]))
    return turns
