"""Bimanual scripted demonstrator: the single-arm expert, one arm at a time.

Both arms first move to the expert's home pose. Then, for each per-arm turn
from :func:`bimanual.work_order` (pens first), a :class:`ScriptedPickPlace`
plans that arm's picks and placements against the live scene, while the other
arm holds its last command. Each turn ends at home, so the arms never meet over
the shared cup. Pens go to fixed slots either side of the cup centre (task
YAML ``cup.pen_slots_xy``).
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import torch

from ..bimanual import work_order
from ..chain import SerialChain, base_pose_from_xy_yaw
from ..expert import ExpertTuning, GripperGeometry, SceneGeometry, ScriptedPickPlace

ARMS = ("left_arm", "right_arm")


class TurnExpert(ScriptedPickPlace):
    """ScriptedPickPlace with given cup slots and a configurable final hold."""

    def __init__(self, *, cup_slots: Mapping[str, np.ndarray], final_hold_s: float, **kwargs) -> None:
        self._slot_override = {name: np.asarray(xy, dtype=np.float64) for name, xy in cup_slots.items()}
        self._final_hold_s = float(final_hold_s)
        super().__init__(**kwargs)

    def _assign_cup_slots(self) -> dict[str, np.ndarray]:
        slots = super()._assign_cup_slots()
        slots.update({name: xy for name, xy in self._slot_override.items() if name in slots})
        return slots

    def _run(self):
        home = np.radians(self.tuning.home_deg)
        yield from self._hold(0.5, self.open_m)
        yield from self._joint_move(home, 1.5)
        for name in self.order:
            yield from self._pick(name)
            if name in self.grasped:
                yield from self._place(name)
            yield from self._joint_move(home, 1.5)
        if self._final_hold_s > 0.0:
            yield from self._hold(self._final_hold_s)


class BimanualExpertDriver:
    """Yields one 14-D action (torch, shape (1, 14)) per env step for env 0."""

    def __init__(self, env, task: Mapping[str, Any], robot: Mapping[str, Any], log=print) -> None:
        self.env = env
        self.task = task
        self.robot = robot
        self.log = log
        self.dt = float(env.step_dt)
        expert = task["expert"]
        self.tuning = ExpertTuning(
            home_deg=tuple(float(v) for v in expert["home_deg"]),
            pen_grip_from_center_m=float(expert["pen_grip_from_center_m"]),
            pregrasp_height_m=float(expert["pregrasp_height_m"]),
            lift_height_m=float(expert["lift_height_m"]),
        )
        gripper = robot["gripper"]
        self.gripper = GripperGeometry(
            grasp_offset=np.asarray(gripper["grasp_offset"], dtype=np.float64),
            fingertip=np.asarray(gripper["fingertip"], dtype=np.float64),
            finger_axis=np.asarray(gripper["finger_axis"], dtype=np.float64),
            approach_axis=np.asarray(gripper["approach_axis"], dtype=np.float64),
        )
        self.open_m = float(gripper["open_m"])
        self.closed_m = float(gripper["closed_m"])
        self.chains = {}
        for arm in ARMS:
            placement = robot["arms"][arm]
            self.chains[arm] = SerialChain(
                robot["chain"], base_pose_from_xy_yaw(placement["position"], math.radians(placement["yaw_deg"])))
        # Commands start from the measured pose with the gripper open.
        self.command = {arm: (env.arm_q(arm), self.open_m) for arm in ARMS}
        layout_arms = {name: item["arm"] for name, item in env.layouts[0]["objects"].items()}
        self.turns = work_order(task["order"], layout_arms)
        self._program = self._run()

    def next_action(self) -> torch.Tensor | None:
        try:
            next(self._program)
        except StopIteration:
            return None
        return self.current_action()

    def current_action(self) -> torch.Tensor:
        """The last command without advancing (the start pose, before the first step)."""
        values = []
        for arm in ARMS:
            q, gripper = self.command[arm]
            values.extend(np.asarray(q, dtype=np.float32).tolist())
            values.append(float(gripper))
        return torch.tensor(values, dtype=torch.float32, device=self.env.device)[None]

    # ---------------------------------------------------------------- program
    def _expert(self, arm: str, names: list[str], last_turn: bool) -> TurnExpert:
        task = self.task
        cup = task["cup"]
        pens = {name: float(spec["pen_half_length_m"]) for name, spec in task["objects"].items()
                if "pen_half_length_m" in spec}
        placement = self.robot["arms"][arm]
        yaw = math.radians(float(placement["yaw_deg"]))
        return TurnExpert(
            fk=self.chains[arm].fk,
            object_pose=self.env.object_pose,
            is_attached=lambda: self.env.is_attached(arm),
            arm_lower=np.asarray(self.robot["arm_lower"], dtype=np.float64),
            arm_upper=np.asarray(self.robot["arm_upper"], dtype=np.float64),
            gripper=self.gripper,
            geometry=SceneGeometry(
                robot_base_xy=np.asarray(placement["position"][:2], dtype=np.float64),
                cup_center_xy=np.asarray(cup["center_xy"], dtype=np.float64),
                cup_top_z=float(self.robot["table_z"]) + float(cup["top_z"]),
                cup_floor_z=float(self.robot["table_z"]) + float(cup["floor_z"]),
                pen_half_lengths=pens,
                robot_zero_heading_xy=(math.cos(yaw), math.sin(yaw)),
            ),
            order=names,
            open_m=self.open_m,
            closed_m=self.closed_m,
            dt=self.dt,
            start_q=self.command[arm][0],
            tuning=self.tuning,
            log=lambda message: self.log(f"[{arm}] {message}"),
            cup_slots={name: np.asarray(cup["center_xy"]) + np.asarray(offset)
                       for name, offset in cup.get("pen_slots_xy", {}).items()},
            final_hold_s=float(task["expert"]["final_hold_s"]) if last_turn else 0.0,
        )

    def _run(self) -> Iterator[None]:
        # Both arms to home together.
        home = np.radians(self.tuning.home_deg)
        start = {arm: np.asarray(self.command[arm][0]) for arm in ARMS}
        steps = max(int(round(1.5 / self.dt)), 1)
        for step in range(steps):
            alpha = (step + 1) / steps
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            for arm in ARMS:
                self.command[arm] = (start[arm] + alpha * (home - start[arm]), self.open_m)
            yield
        for index, (arm, names) in enumerate(self.turns):
            self.log(f"[script] turn {index + 1}/{len(self.turns)}: {arm} -> {', '.join(names)}")
            expert = self._expert(arm, names, last_turn=index == len(self.turns) - 1)
            while (command := expert.next_command()) is not None:
                self.command[arm] = command
                yield
