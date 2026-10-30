"""Bimanual reBot stationery env: layout reset, grasp assist and the success latch.

The manager terms referenced by ``env_cfg.py`` (``reset_layout``,
``gripper_opening``, ``success_termination``, ``success_reward``) live here,
next to the env class whose state they use.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv

from ..bimanual import arm_bases, sample_bimanual_layout
from .geometry import quat_xyzw_to_matrix
from .grasp_assist import GraspAssist

ARMS = ("left_arm", "right_arm")


class StationeryBimanualEnv(ManagerBasedRLEnv):
    """Two reBot arms, two pens, an eraser and a pencil cup."""

    def __init__(self, cfg, task: Mapping[str, Any], robot: Mapping[str, Any], *, seed: int = 0,
                 render_mode: str | None = None, **kwargs) -> None:
        # Read by the reset event, which runs inside super().__init__ and env.reset().
        self.task = task
        self.robot_spec = robot
        self.object_names = list(task["objects"])
        self.base_seed = int(seed)
        self.reset_count = 0
        self.terminate_on_success = task["mode"] == "evaluation"
        self.bases = arm_bases({
            arm: {"base_xy_m": robot["arms"][arm]["position"][:2],
                  "heading_xy": (math.cos(math.radians(robot["arms"][arm]["yaw_deg"])),
                                 math.sin(math.radians(robot["arms"][arm]["yaw_deg"])))}
            for arm in ARMS
        })
        n = cfg.scene.num_envs
        self.layouts: list[dict[str, Any]] = [{} for _ in range(n)]
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self.grasp_assist = GraspAssist(self, ARMS, task["objects"], robot)
        self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.hold_time = torch.zeros(self.num_envs, device=self.device)
        cup = task["cup"]
        self.cup_xy = torch.tensor(cup["center_xy"], dtype=torch.float32, device=self.device)
        self.is_pen = [("pen_half_length_m" in task["objects"][name]) for name in self.object_names]

    # ------------------------------------------------------------- accessors
    def arm_q(self, arm: str, env_id: int = 0) -> np.ndarray:
        asset = self.scene[arm]
        ids = asset.find_joints(self.robot_spec["arm_joints"], preserve_order=True)[0]
        return asset.data.joint_pos.torch[env_id, ids].cpu().numpy().astype(np.float64)

    def object_pose(self, name: str, env_id: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """(position relative to the env origin, rotation matrix)."""
        data = self.scene[name].data
        position = (data.root_pos_w.torch[env_id] - self.scene.env_origins[env_id]).cpu().numpy()
        return position.astype(np.float64), quat_xyzw_to_matrix(data.root_quat_w.torch[env_id].cpu().numpy())

    def is_attached(self, arm: str, env_id: int = 0) -> bool:
        return bool(self.grasp_assist.is_attached(arm)[env_id])

    # --------------------------------------------------------------- success
    def update_success(self) -> torch.Tensor:
        """Every object in the cup (pens upright), all still, both grippers open; held ``hold_s``."""
        rules = self.task["success"]
        origins = self.scene.env_origins
        table_z = float(self.robot_spec["table_z"])
        in_cup = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        still = torch.ones_like(in_cup)
        z_min, z_max = (float(v) for v in rules["z_range_m"])
        for name, pen in zip(self.object_names, self.is_pen):
            data = self.scene[name].data
            pos = data.root_pos_w.torch - origins
            ok = torch.linalg.norm(pos[:, :2] - self.cup_xy, dim=1) <= float(rules["cup_radius_m"])
            ok &= (pos[:, 2] - table_z >= z_min) & (pos[:, 2] - table_z <= z_max)
            if pen:
                qx, qy, qz, qw = data.root_quat_w.torch.unbind(-1)
                axis_z = 2.0 * (qx * qz - qw * qy)  # world z of the local X axis
                ok &= axis_z.abs() >= float(rules["pen_min_vertical"])
            in_cup &= ok
            still &= torch.linalg.norm(data.root_lin_vel_w.torch, dim=1) <= float(rules["max_speed_mps"])
        released = torch.ones_like(in_cup)
        release_m = float(self.robot_spec["grasp_assist"]["release_m"])
        for arm in ARMS:
            released &= ~self.grasp_assist.is_attached(arm) & (self.grasp_assist.opening(arm) >= release_m)
        good = in_cup & still & released
        self.hold_time = torch.where(good, self.hold_time + self.step_dt, torch.zeros_like(self.hold_time))
        self.success |= self.hold_time >= float(rules["hold_s"])
        return self.success

    def episode_info(self, env_id: int = 0) -> dict[str, Any]:
        """Layout, seed and per-object outcome, for the episode sidecar."""
        objects = {}
        for index, name in enumerate(self.object_names):
            position, rotation = self.object_pose(name, env_id)
            objects[name] = {
                "final_position": position.round(4).tolist(),
                "final_axis_z": round(float(rotation[2, 0]), 3),
                "grasped": bool(self.grasp_assist.ever_grasped[env_id, index]),
            }
        return {"layout": self.layouts[env_id], "objects": objects, "success": bool(self.success[env_id])}


# ---------------------------------------------------------------- manager terms
def gripper_opening(env: StationeryBimanualEnv, arm: str) -> torch.Tensor:
    """Mean finger opening (m), shape (N, 1)."""
    asset = env.scene[arm]
    ids = asset.find_joints(list(env.robot_spec["gripper_joints"]), preserve_order=True)[0]
    return asset.data.joint_pos.torch[:, ids].mean(dim=1, keepdim=True)


def success_termination(env: StationeryBimanualEnv) -> torch.Tensor:
    success = env.update_success()
    if env.terminate_on_success:
        return success
    return torch.zeros_like(success)


def success_reward(env: StationeryBimanualEnv) -> torch.Tensor:
    return env.success.float()


def reset_layout(env: StationeryBimanualEnv, env_ids: torch.Tensor) -> None:
    """Sample a fresh reachable layout per env and drop the objects onto the table."""
    task = env.task
    names = env.object_names
    assignment = {name: task["objects"][name].get("arm", "any") for name in names}
    spawn_z = float(env.robot_spec["table_z"]) + float(task["object_physics"]["spawn_z_m"])
    poses = {name: torch.zeros(len(env_ids), 7, device=env.device) for name in names}
    for row, env_id in enumerate(env_ids.tolist()):
        seed = env.base_seed + env.reset_count
        env.reset_count += 1
        sampled = sample_bimanual_layout(np.random.default_rng(seed), names, assignment, env.bases, task["layout"])
        if sampled is None:
            raise RuntimeError(f"no reachable layout found (seed {seed}); relax configs/tasks layout settings")
        layout, arms = sampled
        env.layouts[env_id] = {
            "seed": seed,
            "objects": {name: {"xy": np.round(xy, 4).tolist(), "yaw_deg": round(math.degrees(yaw), 2),
                               "arm": arms[name]} for name, (xy, yaw) in layout.items()},
        }
        origin = env.scene.env_origins[env_id]
        for name, (xy, yaw) in layout.items():
            poses[name][row, :3] = origin + torch.tensor((xy[0], xy[1], spawn_z), device=env.device)
            poses[name][row, 3:] = torch.tensor((0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)),
                                                device=env.device)
    for name in names:
        obj = env.scene[name]
        obj.write_root_pose_to_sim_index(root_pose=poses[name], env_ids=env_ids)
        obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(env_ids), 6, device=env.device),
                                             env_ids=env_ids)
    if hasattr(env, "grasp_assist"):
        env.grasp_assist.reset(env_ids)
        env.success[env_ids] = False
        env.hold_time[env_ids] = 0.0
