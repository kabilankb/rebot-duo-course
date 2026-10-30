"""Scripted pick-and-place expert: stationery into a pencil cup.

The expert only produces per-step ``(arm joint targets, gripper target)``
commands, the same thing a teleoperation leader sends. Grasping, carrying,
release and success detection belong to the host simulator; the expert reads
the world through three callbacks:

* ``fk(q) -> (position, rotation)``: wrist (end-effector link) world pose,
* ``object_pose(name) -> (position, rotation)``: an object's world pose,
* ``is_attached() -> bool``: whether the gripper currently holds an object.

Every motion is planned when it starts, from the poses measured at that
moment, so the expert works on randomized layouts. Pens (long objects along
their local X axis) are held near one end and stood upright in the cup; other
objects are dropped in upright.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .kinematics import ForwardKinematics, Pose, Residual, point_residual, solve_ik


WORLD_UP = np.asarray([0.0, 0.0, 1.0])
Command = tuple[np.ndarray, float]


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


@dataclass(frozen=True)
class SceneGeometry:
    """World geometry the expert plans against (metres)."""

    robot_base_xy: np.ndarray
    cup_center_xy: np.ndarray
    cup_top_z: float
    cup_floor_z: float
    # Objects listed here are treated as pens: {name: half length along local X}.
    pen_half_lengths: Mapping[str, float]
    # Direction the arm points at joint-1 zero; positive joint 1 turns it CCW.
    robot_zero_heading_xy: tuple[float, float] = (0.0, -1.0)


@dataclass(frozen=True)
class GripperGeometry:
    """End-effector-link frame quantities."""

    grasp_offset: np.ndarray  # point between the finger roots
    fingertip: np.ndarray  # centre of the fingertips
    finger_axis: np.ndarray  # finger closing direction
    approach_axis: np.ndarray = field(
        default_factory=lambda: np.asarray([0.0, 0.0, 1.0])
    )  # points from the wrist out between the fingers


@dataclass(frozen=True)
class ExpertTuning:
    home_deg: tuple[float, ...] = (0.0, -45.0, -45.0, 0.0, 0.0, 0.0)
    # Pens are held this far from their centre and stood with that end up, so
    # the fingers stay above the cup rim when they open.
    pen_grip_from_center_m: float = 0.035
    # Pens stand on the cup floor on either side of the centre so they neither
    # land on each other nor on an object dropped in later between them.
    pen_slot_offset_m: float = 0.014
    pregrasp_height_m: float = 0.07
    lift_height_m: float = 0.13


class ScriptedPickPlace:
    """Generates one ``(arm_target, gripper_target)`` command per physics step."""

    def __init__(
        self,
        *,
        fk: ForwardKinematics,
        object_pose: Callable[[str], Pose],
        is_attached: Callable[[], bool],
        arm_lower: np.ndarray,
        arm_upper: np.ndarray,
        gripper: GripperGeometry,
        geometry: SceneGeometry,
        order: Sequence[str],
        open_m: float,
        closed_m: float,
        dt: float,
        start_q: np.ndarray,
        tuning: ExpertTuning = ExpertTuning(),
        log: Callable[[str], None] = print,
    ) -> None:
        self.fk = fk
        self.object_pose = object_pose
        self.is_attached = is_attached
        self.lower = np.asarray(arm_lower, dtype=np.float64)
        self.upper = np.asarray(arm_upper, dtype=np.float64)
        self.grasp_offset = np.asarray(gripper.grasp_offset, dtype=np.float64)
        self.fingertip = np.asarray(gripper.fingertip, dtype=np.float64)
        self.finger_axis = np.asarray(gripper.finger_axis, dtype=np.float64)
        self.finger_axis = self.finger_axis / np.linalg.norm(self.finger_axis)
        self.approach_axis = np.asarray(gripper.approach_axis, dtype=np.float64)
        self.geometry = geometry
        self.tuning = tuning
        self.order = list(order)
        self.open_m = float(open_m)
        self.closed_m = float(closed_m)
        self.dt = float(dt)
        self.log = log
        self.q = np.asarray(start_q, dtype=np.float64).copy()
        self.gripper = self.open_m
        self.grasped: list[str] = []
        self._pen_grip_end: dict[str, float] = {}
        self._cup_slots = self._assign_cup_slots()
        self._program = self._run()

    # ------------------------------------------------------------------ API
    def next_command(self) -> Command | None:
        """Advance one physics step; ``None`` once the program has finished."""
        try:
            self.q, self.gripper = next(self._program)
        except StopIteration:
            return None
        return self.q.copy(), self.gripper

    # ------------------------------------------------------------ planning
    def _is_pen(self, name: str) -> bool:
        return name in self.geometry.pen_half_lengths

    def _assign_cup_slots(self) -> dict[str, np.ndarray]:
        cup = np.asarray(self.geometry.cup_center_xy, dtype=np.float64)
        toward_base = np.asarray(self.geometry.robot_base_xy) - cup
        across = np.asarray([-toward_base[1], toward_base[0]])
        across /= np.linalg.norm(across)
        slots = {name: cup.copy() for name in self.order}
        pens = [name for name in self.order if self._is_pen(name)]
        for index, name in enumerate(pens):
            side = 1.0 if index % 2 == 0 else -1.0
            slots[name] = cup + side * self.tuning.pen_slot_offset_m * across
        return slots

    def _ik(
        self, residual: Residual, seeds: list[np.ndarray], *, iterations: int = 150
    ) -> tuple[np.ndarray, float]:
        return solve_ik(
            self.fk,
            residual,
            seeds,
            self.lower,
            self.upper,
            prefer_near=self.q,
            iterations=iterations,
        )

    def _seeds(self, target_xy: np.ndarray) -> list[np.ndarray]:
        heading = np.asarray(self.geometry.robot_zero_heading_xy, dtype=np.float64)
        to_target = np.asarray(target_xy) - np.asarray(self.geometry.robot_base_xy)
        yaw = float(
            np.arctan2(
                heading[0] * to_target[1] - heading[1] * to_target[0],
                heading @ to_target,
            )
        )
        seeds = [self.q.copy()]
        for shoulder in (-30.0, -60.0, -90.0):
            for elbow in (-30.0, -60.0, -90.0):
                seed = np.zeros_like(self.q)
                seed[:3] = np.radians([0.0, shoulder, elbow])
                seed[0] = yaw
                seeds.append(seed)
        return seeds

    def _grasp_residual(
        self, target: np.ndarray, object_axis: np.ndarray, *, down_weight: float
    ) -> Residual:
        """Fingertips on target, gripper pointing down, fingers across the object."""
        point = point_residual(self.fingertip, target)
        horizontal_axis = object_axis - WORLD_UP * float(object_axis @ WORLD_UP)
        horizontal_axis = horizontal_axis / max(float(np.linalg.norm(horizontal_axis)), 1e-9)

        def residual(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
            forward = rotation @ self.approach_axis
            fingers = rotation @ self.finger_axis
            return np.concatenate(
                (
                    point(position, rotation),
                    down_weight * (forward + WORLD_UP),
                    [0.3 * float(fingers @ horizontal_axis)],
                )
            )

        return residual

    @staticmethod
    def _hold_orientation_residual(
        point_local: np.ndarray, target: np.ndarray, rotation_ref: np.ndarray, weight: float
    ) -> Residual:
        point = point_residual(point_local, target)
        return lambda position, rotation: np.concatenate(
            (point(position, rotation), weight * (rotation - rotation_ref).ravel())
        )

    # --------------------------------------------------------------- motions
    def _steps(self, duration: float) -> int:
        return max(int(round(duration / self.dt)), 1)

    def _hold(self, duration: float, gripper: float | None = None) -> Iterator[Command]:
        start = self.gripper
        target = self.gripper if gripper is None else gripper
        steps = self._steps(duration)
        for step in range(steps):
            alpha = _smoothstep((step + 1) / steps)
            yield self.q.copy(), start + alpha * (target - start)

    def _hold_until(self, condition: Callable[[], bool], timeout: float) -> Iterator[Command]:
        for _ in range(self._steps(timeout)):
            if condition():
                return
            yield self.q.copy(), self.gripper

    def _joint_move(self, q_target: np.ndarray, duration: float) -> Iterator[Command]:
        q_start = self.q.copy()
        steps = self._steps(duration)
        for step in range(steps):
            alpha = _smoothstep((step + 1) / steps)
            yield q_start + alpha * (q_target - q_start), self.gripper

    def _cartesian_move(
        self,
        point_local: np.ndarray,
        target: np.ndarray,
        duration: float,
        label: str,
        *,
        orientation_weight: float = 0.2,
    ) -> Iterator[Command]:
        """Straight line of a wrist-fixed point, keeping the current orientation.

        A low ``orientation_weight`` lets the wrist turn where holding the
        orientation exactly would leave the workspace.
        """
        position, rotation = self.fk(self.q)
        start = position + rotation @ point_local
        samples = max(int(duration * 20), 2)
        path = [self.q.copy()]
        worst = 0.0
        for index in range(1, samples + 1):
            waypoint = start + (index / samples) * (target - start)
            self.q = path[-1]
            q, error = self._ik(
                self._hold_orientation_residual(
                    point_local, waypoint, rotation, orientation_weight
                ),
                [path[-1]],
                iterations=40,
            )
            path.append(q)
            worst = max(worst, error)
        self.q = path[0]
        self.log(f"[script] {label}: {samples} IK waypoints, worst residual {worst:.4f}")
        path_array = np.asarray(path)
        steps = self._steps(duration)
        for step in range(steps):
            s = _smoothstep((step + 1) / steps) * (len(path) - 1)
            low = min(int(s), len(path) - 2)
            frac = s - low
            yield (1 - frac) * path_array[low] + frac * path_array[low + 1], self.gripper

    # --------------------------------------------------------------- program
    def _run(self) -> Iterator[Command]:
        home = np.radians(self.tuning.home_deg)
        yield from self._hold(0.5, self.open_m)
        yield from self._joint_move(home, 1.5)
        for name in self.order:
            yield from self._pick(name)
            if name in self.grasped:
                yield from self._place(name)
            yield from self._joint_move(home, 1.5)
        yield from self._hold(2.0)

    def _pick(self, name: str) -> Iterator[Command]:
        position, rotation = self.object_pose(name)
        object_axis = rotation[:, 0]
        grip_xy = position[:2]
        if self._is_pen(name):
            # Hold the end nearer the robot: shorter reach, and it goes up in the cup.
            toward_base = np.asarray(self.geometry.robot_base_xy) - position[:2]
            end = 1.0 if float(object_axis[:2] @ toward_base) >= 0.0 else -1.0
            self._pen_grip_end[name] = end
            grip_xy = grip_xy + end * self.tuning.pen_grip_from_center_m * object_axis[:2]
        # Fingertips just above the work surface, so the pads close on the object.
        grasp = np.asarray([grip_xy[0], grip_xy[1], max(position[2] - 0.001, 0.003)])
        above = grasp + np.asarray([0.0, 0.0, self.tuning.pregrasp_height_m])

        q_above, error = self._ik(
            self._grasp_residual(above, object_axis, down_weight=0.3),
            self._seeds(above[:2]),
        )
        self.log(
            f"[script] {name}: pregrasp above ({above.round(3).tolist()}), residual {error:.4f}"
        )
        yield from self._joint_move(q_above, 2.0)
        yield from self._cartesian_move(self.fingertip, grasp, 1.2, f"{name} descend")
        yield from self._hold(0.9, self.closed_m)
        yield from self._hold(0.3)
        if not self.is_attached():
            self.log(f"[script-warn] {name} was not grasped; opening and moving on")
            yield from self._hold(0.5, self.open_m)
            yield from self._cartesian_move(self.fingertip, above, 1.0, f"{name} retreat")
            return
        self.grasped.append(name)
        lift = grasp + np.asarray([0.0, 0.0, self.tuning.lift_height_m])
        yield from self._cartesian_move(self.fingertip, lift, 1.2, f"{name} lift")

    def _place(self, name: str) -> Iterator[Command]:
        geometry = self.geometry
        link_position, link_rotation = self.fk(self.q)
        object_position, object_rotation = self.object_pose(name)
        # The carried object keeps this wrist-relative transform until release.
        center_local = link_rotation.T @ (object_position - link_position)
        axis_local = link_rotation.T @ object_rotation[:, 0]
        cup_xy = self._cup_slots[name]
        is_pen = self._is_pen(name)

        if is_pen:
            half = float(geometry.pen_half_lengths[name])
            above_z = geometry.cup_top_z + half + 0.035
            lower_z = geometry.cup_floor_z + half + 0.004
        else:
            above_z = geometry.cup_top_z + 0.05
            lower_z = geometry.cup_top_z + 0.012
        above = np.asarray([cup_xy[0], cup_xy[1], above_z])

        point = point_residual(center_local, above)
        # Stand a pen upright with the gripped end on top.
        pen_up = self._pen_grip_end.get(name, 1.0) * axis_local

        def place_residual(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
            if is_pen:
                orient = rotation @ pen_up - WORLD_UP
            else:
                orient = 0.3 * (rotation @ self.approach_axis + WORLD_UP)
            return np.concatenate((point(position, rotation), orient))

        q_above, error = self._ik(place_residual, self._seeds(cup_xy))
        self.log(f"[script] {name}: above cup ({above.round(3).tolist()}), residual {error:.4f}")
        yield from self._joint_move(q_above, 2.2)
        yield from self._cartesian_move(
            center_local, np.asarray([cup_xy[0], cup_xy[1], lower_z]), 1.2, f"{name} lower"
        )
        yield from self._hold(0.6, self.open_m)
        yield from self._hold_until(lambda: not self.is_attached(), 1.5)
        if self.is_attached():
            self.log(f"[script-warn] {name} has not released yet; backing out anyway")
        yield from self._hold(0.3)
        # Back out along the gripper axis so the fingers slide off the object.
        position, rotation = self.fk(self.q)
        grasp_point = position + rotation @ self.grasp_offset
        retreat = (
            grasp_point
            - 0.06 * (rotation @ self.approach_axis)
            + np.asarray([0.0, 0.0, 0.03])
        )
        yield from self._cartesian_move(
            self.grasp_offset, retreat, 0.9, f"{name} retreat", orientation_weight=0.02
        )
