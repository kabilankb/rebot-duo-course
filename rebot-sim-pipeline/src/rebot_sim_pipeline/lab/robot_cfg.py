"""Isaac Lab articulation config for one reBot B601 arm (built from the robot YAML)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .geometry import wxyz_to_xyzw, yaw_quat_wxyz


def arm_cfg(robot: Mapping[str, Any], arm: str, usd_path: str | Path) -> ArticulationCfg:
    """ArticulationCfg for ``arm`` (a key of ``robot["arms"]``), placed per the YAML.

    ``usd_path`` is the coloured wrapper from :func:`assets.prepare_robot_asset`.
    """
    placement = robot["arms"][arm]
    actuators = {
        group: ImplicitActuatorCfg(
            joint_names_expr=[spec["joints"]],
            stiffness=float(spec["stiffness"]),
            damping=float(spec["damping"]),
            effort_limit_sim=float(spec["effort_limit"]),
            velocity_limit_sim=float(spec["velocity_limit"]),
        )
        for group, spec in robot["actuators"].items()
    }
    # Start fully open at the zero pose; the expert moves to its home pose itself.
    joint_pos = {name: 0.0 for name in robot["arm_joints"]}
    joint_pos.update({name: float(robot["gripper"]["open_m"]) for name in robot["gripper_joints"]})
    return ArticulationCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{placement['prim_name']}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=int(robot["solver_position_iterations"]),
                solver_velocity_iteration_count=int(robot["solver_velocity_iterations"]),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(float(v) for v in placement["position"]),
            rot=wxyz_to_xyzw(yaw_quat_wxyz(math.radians(float(placement["yaw_deg"])))),
            joint_pos=joint_pos,
        ),
        actuators=actuators,
    )
