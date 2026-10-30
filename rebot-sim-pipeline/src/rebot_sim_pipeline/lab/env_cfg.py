"""Isaac Lab scene + manager config for the bimanual reBot stationery task.

:func:`build_env_cfg` turns the resolved task YAML, the robot YAML and the
generated object USDs into a :class:`StationeryEnvCfg`. Action layout (14):
``[left joint1..6 (rad), left gripper (m), right joint1..6, right gripper]``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.camera import TiledCameraCfg
from isaaclab.utils import configclass

from . import env as task_env
from .geometry import look_at_opengl, matrix_to_quat_wxyz, wxyz_to_xyzw
from .grasp_assist import GripperActionCfg
from .robot_cfg import arm_cfg

ARMS = ("left_arm", "right_arm")


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Floor and light as YAMLab's scene; the builder adds the workstation, arms, cup,
    objects and cameras."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=800.0, color=(0.75, 0.75, 0.75))
    )


@configclass
class ActionsCfg:
    left_arm = mdp.JointPositionActionCfg(asset_name="left_arm", joint_names=["joint[1-6]"], scale=1.0,
                                          use_default_offset=False)
    left_gripper = GripperActionCfg(asset_name="left_arm")
    right_arm = mdp.JointPositionActionCfg(asset_name="right_arm", joint_names=["joint[1-6]"], scale=1.0,
                                           use_default_offset=False)
    right_gripper = GripperActionCfg(asset_name="right_arm")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        left_joint_pos = ObsTerm(func=mdp.joint_pos,
                                 params={"asset_cfg": SceneEntityCfg("left_arm", joint_names=["joint[1-6]"])})
        left_gripper = ObsTerm(func=task_env.gripper_opening, params={"arm": "left_arm"})
        right_joint_pos = ObsTerm(func=mdp.joint_pos,
                                  params={"asset_cfg": SceneEntityCfg("right_arm", joint_names=["joint[1-6]"])})
        right_gripper = ObsTerm(func=task_env.gripper_opening, params={"arm": "right_arm"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})
    reset_layout = EventTerm(func=task_env.reset_layout, mode="reset")


@configclass
class RewardsCfg:
    success = RewTerm(func=task_env.success_reward, weight=1.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Always evaluated (it updates the success latch); ends the episode only when
    # env.terminate_on_success is set (evaluation).
    task_success = DoneTerm(func=task_env.success_termination)


@configclass
class StationeryEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 120.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (-0.35, 0.0, 1.45)
        self.viewer.lookat = (0.6, 0.0, 0.75)


def _camera_cfg(name: str, spec: Mapping[str, Any], robot: Mapping[str, Any], width: int, height: int):
    if "quaternion_opengl" in spec:
        position, quat_wxyz = spec["position"], spec["quaternion_opengl"]
    else:
        position = spec["eye"]
        quat_wxyz = matrix_to_quat_wxyz(look_at_opengl(spec["eye"], spec["target"], spec["up"]))
    if "arm" in spec:
        prim = robot["arms"][spec["arm"]]["prim_name"]
        prim_path = f"{{ENV_REGEX_NS}}/{prim}/{robot['end_effector_link']}/{name}_camera"
    else:
        prim_path = f"{{ENV_REGEX_NS}}/{name}_camera"
    return TiledCameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        width=width,
        height=height,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=float(spec["focal_length"]),
            horizontal_aperture=float(spec["horizontal_aperture"]),
            vertical_aperture=float(spec["vertical_aperture"]),
            clipping_range=(0.01, 100.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=tuple(float(v) for v in position),
            rot=wxyz_to_xyzw(quat_wxyz),
            convention="opengl",
        ),
    )


def build_env_cfg(
    task: Mapping[str, Any],
    robot: Mapping[str, Any],
    assets: Mapping[str, Any],
    *,
    robot_usd: str,
    workstation_usd: str,
    enable_cameras: bool,
) -> StationeryEnvCfg:
    """Full env config for the resolved task (see lab/config.py).

    ``assets`` comes from :func:`assets.prepare_stationery_assets`, ``robot_usd`` from
    :func:`assets.prepare_robot_asset` and ``workstation_usd`` from :func:`assets.find_workstation`.
    """
    cfg = StationeryEnvCfg()
    sim = task["sim"]
    cfg.sim.dt = float(sim["dt"])
    cfg.decimation = int(sim["decimation"])
    cfg.sim.render_interval = int(sim["render_interval"])
    cfg.sim.device = str(sim["device"])
    cfg.episode_length_s = float(sim["episode_length_s"])
    cfg.scene.num_envs = int(sim["num_envs"])

    for arm in ARMS:
        setattr(cfg.scene, arm, arm_cfg(robot, arm, robot_usd))
        gripper = getattr(cfg.actions, arm.replace("_arm", "_gripper"))
        gripper.joint_names = tuple(robot["gripper_joints"])
        gripper.open_m = float(robot["gripper"]["open_m"])
        gripper.closed_m = float(robot["gripper"]["closed_m"])

    # YAMLab's workstation (table, walls, camera gate) as an immovable collider.
    cfg.scene.workstation = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workstation",
        spawn=sim_utils.UsdFileCfg(
            usd_path=workstation_usd,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    table_z = float(robot["table_z"])
    cup = task["cup"]
    cup_bottom = float(assets["bbox"]["pencil_cup"]["min"][2])
    cfg.scene.pencil_cup = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PencilCup",
        spawn=sim_utils.UsdFileCfg(usd_path=assets["cup"]),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(float(cup["center_xy"][0]), float(cup["center_xy"][1]),
                                                     table_z - cup_bottom)),
    )

    spawn_z = table_z + float(task["object_physics"]["spawn_z_m"])
    for index, name in enumerate(task["objects"]):
        setattr(cfg.scene, name, RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=sim_utils.UsdFileCfg(usd_path=assets["objects"][name]),
            # Placeholder; reset_layout samples the real pose every episode.
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8, 0.3 * (index - 1), spawn_z)),
        ))

    if enable_cameras:
        width, height = int(task["rendering"]["width"]), int(task["rendering"]["height"])
        for name, spec in robot["cameras"].items():
            setattr(cfg.scene, f"{name}_camera", _camera_cfg(name, spec, robot, width, height))
            setattr(cfg.observations.policy, f"{name}_rgb", ObsTerm(
                func=mdp.image,
                params={"sensor_cfg": SceneEntityCfg(f"{name}_camera"), "data_type": "rgb", "normalize": False},
            ))
    return cfg
