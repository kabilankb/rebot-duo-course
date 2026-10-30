"""Grasp assist and the gripper action term that drives it every physics step.

The single-arm reBot app carries an object rigidly once the gripper closes on
it, because a 4 mm pen between convex-hull fingers is not a reliable physical
grasp. The same rules apply here, per arm:

* the distance from an arm to each object is the smallest distance from probe
  points on the gripper's approach axis (around ``grasp_offset``) to the object
  (pens: capsules along local X; others: boxes);
* **attach** when the command is at most ``attach_command_m``, the measured
  opening at most ``attach_opening_m`` and an object (not held by the other arm)
  within ``max_distance_m``; the object then keeps its pose relative to the
  end-effector link and its velocity is zeroed every physics step;
* **release** when both the command and the opening are at least ``release_m``.

While attached the fingers are held just inside their measured opening, so they
press lightly on the object instead of fighting the carried pose.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    quat_apply_inverse,
    subtract_frame_transforms,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class GraspAssist:
    """Attach/release bookkeeping for every arm and object, vectorized over envs."""

    def __init__(
        self,
        env: "ManagerBasedEnv",
        arms: Sequence[str],
        objects: Mapping[str, Mapping[str, Any]],
        robot: Mapping[str, Any],
    ) -> None:
        self.env = env
        self.device = env.device
        self.arms = list(arms)
        self.objects = list(objects)
        self.cfg = robot["grasp_assist"]
        self.enabled = bool(self.cfg["enabled"])
        n = env.num_envs
        self.held = {arm: torch.full((n,), -1, dtype=torch.long, device=self.device) for arm in self.arms}
        self.rel_pos = {arm: torch.zeros(n, 3, device=self.device) for arm in self.arms}
        self.rel_quat = {arm: torch.zeros(n, 4, device=self.device) for arm in self.arms}
        self.ever_grasped = torch.zeros(n, len(self.objects), dtype=torch.bool, device=self.device)

        ee = robot["end_effector_link"]
        self.ee_index = {arm: env.scene[arm].find_bodies(ee)[0][0] for arm in self.arms}
        self.finger_ids = {arm: env.scene[arm].find_joints(robot["gripper_joints"], preserve_order=True)[0]
                           for arm in self.arms}
        offset = torch.tensor(robot["gripper"]["grasp_offset"], dtype=torch.float32, device=self.device)
        approach = torch.tensor(robot["gripper"]["approach_axis"], dtype=torch.float32, device=self.device)
        span = float(self.cfg["probe_half_span_m"])
        steps = torch.linspace(-span, span, int(self.cfg["probe_samples"]), device=self.device)
        self.probes = offset[None, :] + steps[:, None] * approach[None, :]  # (P, 3) in the EE frame

        # Per-object shape for the distance test: capsule (pen) or box.
        self.capsule = torch.zeros(len(self.objects), dtype=torch.bool, device=self.device)
        self.half = torch.zeros(len(self.objects), 3, device=self.device)
        self.radius = torch.zeros(len(self.objects), device=self.device)
        for index, name in enumerate(self.objects):
            spec = objects[name]
            if "pen_half_length_m" in spec:
                self.capsule[index] = True
                self.half[index, 0] = float(spec["pen_half_length_m"])
                self.radius[index] = float(spec["radius_m"])
            else:
                self.half[index] = torch.tensor(spec["box_half_extents_m"], device=self.device)

    # ----------------------------------------------------------------- queries
    def ee_pose(self, arm: str) -> tuple[torch.Tensor, torch.Tensor]:
        pose = self.env.scene[arm].data.body_link_pose_w.torch[:, self.ee_index[arm]]
        return pose[:, :3], pose[:, 3:7]

    def opening(self, arm: str) -> torch.Tensor:
        """Mean measured finger opening (m), shape (N,)."""
        return self.env.scene[arm].data.joint_pos.torch[:, self.finger_ids[arm]].mean(dim=1)

    def is_attached(self, arm: str) -> torch.Tensor:
        return self.held[arm] >= 0

    def holder_of(self, object_index: int) -> torch.Tensor:
        """Bool (N,): some arm holds this object."""
        held = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
        for arm in self.arms:
            held |= self.held[arm] == object_index
        return held

    def distances(self, arm: str) -> torch.Tensor:
        """(N, num_objects) distance from the arm's grasp probes to each object surface."""
        ee_pos, ee_quat = self.ee_pose(arm)
        n, p = ee_pos.shape[0], self.probes.shape[0]
        probes_w = ee_pos[:, None] + quat_apply(
            ee_quat[:, None].expand(n, p, 4).reshape(-1, 4), self.probes[None].expand(n, p, 3).reshape(-1, 3)
        ).view(n, p, 3)
        result = torch.empty(n, len(self.objects), device=self.device)
        for index, name in enumerate(self.objects):
            data = self.env.scene[name].data
            obj_pos, obj_quat = data.root_pos_w.torch, data.root_quat_w.torch
            local = quat_apply_inverse(
                obj_quat[:, None].expand(n, p, 4).reshape(-1, 4), (probes_w - obj_pos[:, None]).reshape(-1, 3)
            ).view(n, p, 3)
            if bool(self.capsule[index]):
                axis_x = local[..., 0].clamp(-self.half[index, 0], self.half[index, 0])
                nearest = torch.stack((axis_x, torch.zeros_like(axis_x), torch.zeros_like(axis_x)), dim=-1)
                dist = torch.linalg.norm(local - nearest, dim=-1) - self.radius[index]
            else:
                q = local.abs() - self.half[index]
                dist = torch.linalg.norm(q.clamp(min=0.0), dim=-1) + q.max(dim=-1).values.clamp(max=0.0)
            result[:, index] = dist.min(dim=1).values
        return result

    # ------------------------------------------------------------------ update
    def update(self, arm: str, command: torch.Tensor) -> torch.Tensor:
        """Advance attach/release for ``arm``; return the finger target (N,) to apply."""
        target = command.clone()
        if not self.enabled:
            return target
        opening = self.opening(arm)
        attached = self.is_attached(arm)
        release_m = float(self.cfg["release_m"])

        released = attached & (command >= release_m) & (opening >= release_m)
        if released.any():
            self._zero_velocity(self.held[arm], released)
            self.held[arm][released] = -1
            attached = self.is_attached(arm)

        candidates = ~attached & (command <= float(self.cfg["attach_command_m"])) & (
            opening <= float(self.cfg["attach_opening_m"]))
        if candidates.any():
            dist = self.distances(arm)
            for index in range(len(self.objects)):
                dist[self.holder_of(index), index] = float("inf")
            best_dist, best = dist.min(dim=1)
            attach = candidates & (best_dist <= float(self.cfg["max_distance_m"]))
            if attach.any():
                ee_pos, ee_quat = self.ee_pose(arm)
                for index, name in enumerate(self.objects):
                    rows = attach & (best == index)
                    if not rows.any():
                        continue
                    data = self.env.scene[name].data
                    rel_pos, rel_quat = subtract_frame_transforms(
                        ee_pos[rows], ee_quat[rows], data.root_pos_w.torch[rows], data.root_quat_w.torch[rows])
                    self.rel_pos[arm][rows] = rel_pos
                    self.rel_quat[arm][rows] = rel_quat
                    self.held[arm][rows] = index
                    self.ever_grasped[rows, index] = True
                attached = self.is_attached(arm)

        if attached.any():
            self._carry(arm, attached)
            target[attached] = torch.maximum(command[attached], opening[attached] - 0.002)
        return target

    def _carry(self, arm: str, rows: torch.Tensor) -> None:
        ee_pos, ee_quat = self.ee_pose(arm)
        pos, quat = combine_frame_transforms(ee_pos, ee_quat, self.rel_pos[arm], self.rel_quat[arm])
        for index, name in enumerate(self.objects):
            mask = rows & (self.held[arm] == index)
            env_ids = mask.nonzero(as_tuple=False).squeeze(-1)
            if env_ids.numel() == 0:
                continue
            obj = self.env.scene[name]
            obj.write_root_pose_to_sim_index(root_pose=torch.cat((pos[env_ids], quat[env_ids]), dim=-1),
                                             env_ids=env_ids)
            obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros(env_ids.numel(), 6, device=self.device),
                                                 env_ids=env_ids)

    def _zero_velocity(self, held: torch.Tensor, rows: torch.Tensor) -> None:
        for index, name in enumerate(self.objects):
            env_ids = (rows & (held == index)).nonzero(as_tuple=False).squeeze(-1)
            if env_ids.numel():
                self.env.scene[name].write_root_velocity_to_sim_index(
                    root_velocity=torch.zeros(env_ids.numel(), 6, device=self.device), env_ids=env_ids)

    def reset(self, env_ids: torch.Tensor) -> None:
        for arm in self.arms:
            self.held[arm][env_ids] = -1
        self.ever_grasped[env_ids] = False


class GripperAction(ActionTerm):
    """One value per arm: the per-finger opening (m), mirrored to both finger joints.

    Each physics step it runs the env's grasp assist (``env.grasp_assist``) for its
    arm before setting the finger targets.
    """

    cfg: "GripperActionCfg"

    def __init__(self, cfg: "GripperActionCfg", env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)
        self._joint_ids, _ = self._asset.find_joints(list(cfg.joint_names), preserve_order=True)
        self._raw = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw[:] = actions
        self._processed[:] = actions.clamp(self.cfg.closed_m, self.cfg.open_m)

    def apply_actions(self) -> None:
        command = self._processed[:, 0]
        assist = getattr(self._env, "grasp_assist", None)
        target = assist.update(self.cfg.asset_name, command) if assist is not None else command
        self._asset.set_joint_position_target_index(
            target=target[:, None].expand(-1, len(self._joint_ids)).contiguous(), joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw[env_ids] = self.cfg.open_m


@configclass
class GripperActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = GripperAction
    joint_names: tuple[str, ...] = ("joint_left", "joint_right")
    open_m: float = 0.05
    closed_m: float = 0.0
