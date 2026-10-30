"""Collect bimanual scripted demonstrations in Isaac Lab and stage them.

Run with the Isaac Lab Python (it starts Isaac Sim)::

    python -m rebot_sim_pipeline.lab.generate --episodes 20 --headless
    python -m rebot_sim_pipeline.lab.generate --episodes 1 --no-recording --viz kit   # watch a dry run

Each attempt is a freshly sampled layout (seed = --seed + attempt). Successful
episodes are written to ``--staging`` (see rebot_sim_pipeline.staging); failed
ones are discarded and retried, up to 2N+5 attempts. ``write_dataset`` (LeRobot
venv) turns the staged episodes into a LeRobot v3 dataset. Exit code 0 only if
all N episodes were collected.
"""

from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Bimanual reBot scripted demonstrations (Isaac Lab).")
parser.add_argument("--task", default="stationery_sorting", help="Task name in lab/configs/tasks or a YAML path.")
parser.add_argument("--robot", default="rebot_b601_bimanual", help="Robot name in lab/configs/robot or a YAML path.")
parser.add_argument("--episodes", type=int, default=1, help="Successful episodes to collect.")
parser.add_argument("--seed", type=int, default=0, help="Layout seed of the first attempt.")
parser.add_argument("--staging", default="datasets/.rebot_bimanual_staging", help="Staged-episode directory.")
parser.add_argument("--keep-failed", action="store_true", help="Stage failed attempts too (N counts attempts).")
parser.add_argument("--no-recording", action="store_true", help="Dry run: no cameras, nothing staged.")
parser.add_argument("--max-attempts", type=int, default=None, help="Default 2N+5.")
parser.add_argument("--rebot-project", default=None, help="Single-arm reBot checkout with the assets.")
parser.add_argument("--realtime", action="store_true", help="Pace the simulation to wall-clock time (watching).")
parser.add_argument("--hold-open", action="store_true", help="Keep the window open after the last episode.")
parser.add_argument("--camera-views", action="store_true",
                    help="GUI: open a viewport window per camera (top, left/right wrist); renders the cameras "
                         "even with --no-recording.")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device=None)
args = parser.parse_args()
args.enable_cameras = not args.no_recording or args.camera_views
if args.device is None:  # the YAML per-mode device, unless --device was given
    from rebot_sim_pipeline.lab.config import load_task as _load_task

    args.device = str(_load_task(args.task, mode="generate")["sim"]["device"])
app = AppLauncher(args).app

import numpy as np  # noqa: E402

from rebot_sim_pipeline import conventions  # noqa: E402
from rebot_sim_pipeline.lab.assets import (  # noqa: E402
    find_workstation,
    prepare_robot_asset,
    prepare_stationery_assets,
)
from rebot_sim_pipeline.lab.config import find_rebot_project, load_robot, load_task  # noqa: E402
from rebot_sim_pipeline.lab.env import ARMS, StationeryBimanualEnv  # noqa: E402
from rebot_sim_pipeline.lab.env_cfg import build_env_cfg  # noqa: E402
from rebot_sim_pipeline.lab.expert_driver import BimanualExpertDriver  # noqa: E402
from rebot_sim_pipeline.staging import EpisodeStager  # noqa: E402


def _record_row(obs: dict, action) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Images + (action, state) in the reBot convention and in simulator units, for env 0."""
    policy = obs["policy"]
    command = action[0].cpu().numpy().astype(np.float64)
    rec_action, rec_state, raw_state = [], [], []
    for index, arm in enumerate(ARMS):
        prefix = arm.replace("_arm", "")
        q = policy[f"{prefix}_joint_pos"][0].cpu().numpy().astype(np.float64)
        opening = float(policy[f"{prefix}_gripper"][0, 0])
        cmd = command[7 * index: 7 * index + 7]
        rec_action.append(conventions.arm_action(cmd[:6], cmd[6]))
        rec_state.append(conventions.arm_state(q, opening))
        raw_state.append(np.append(q, opening))
    images = {key[: -len("_rgb")]: value[0].cpu().numpy() for key, value in policy.items() if key.endswith("_rgb")}
    return (images, np.concatenate(rec_action), np.concatenate(rec_state), command.astype(np.float32),
            np.concatenate(raw_state).astype(np.float32))


class CameraViews:
    """One 640x480 viewport window per scene camera (env 0), as the single-arm reBot app opens them.

    Each window is set to its camera prim with ``set_active_camera`` (as Isaac
    Lab's Kit visualizer does); :meth:`apply` re-sets any window that fell back
    to the default perspective camera while it was still initializing.
    """

    def __init__(self, env, robot: dict) -> None:
        from omni.kit.viewport.utility import create_viewport_window

        width, height = 640, 480
        self.views = []
        for index, name in enumerate(robot["cameras"]):
            # The scene has already expanded {ENV_REGEX_NS} to the regex /World/envs/env_.* here.
            path = env.scene[f"{name}_camera"].cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
            path = path.replace("/World/envs/env_.*", "/World/envs/env_0")
            window = create_viewport_window(name=f"{name} camera", width=width, height=height,
                                            position_x=30 + index * (width + 20), position_y=90)
            self.views.append((name, path, window))
        self.apply(log=True)

    def apply(self, log: bool = False) -> None:
        import omni.usd

        for name, path, window in self.views:
            api = window.viewport_api
            if str(api.get_active_camera()) != path:
                api.set_active_camera(path)
            if log:
                stage = omni.usd.get_context(api.usd_context_name).get_stage()
                prim = stage.GetPrimAtPath(path) if stage else None
                ok = bool(prim and prim.IsValid())
                print(f"[script] camera view '{name}': {api.get_active_camera()}"
                      + ("" if ok else f" (camera prim {path} not found)"), flush=True)


def main() -> int:
    task = load_task(args.task, mode="generate", overrides={"sim": {"device": args.device}} if args.device else None)
    robot = load_robot(args.robot)
    project = find_rebot_project(args.rebot_project)
    assets = prepare_stationery_assets(task, project)
    cfg = build_env_cfg(task, robot, assets, robot_usd=prepare_robot_asset(robot, project),
                        workstation_usd=find_workstation(robot), enable_cameras=args.enable_cameras)
    env = StationeryBimanualEnv(cfg, task, robot, seed=args.seed)
    views = CameraViews(env, robot) if args.camera_views else None

    cameras = list(robot["cameras"])
    stager = None if args.no_recording else EpisodeStager(args.staging, cameras, task["recording"]["jpeg_quality"])
    fps = round(1.0 / env.step_dt)
    settle_steps = int(round(float(task["expert"]["settle_s"]) / env.step_dt))
    max_attempts = args.max_attempts or 2 * args.episodes + 5
    kept = 0
    attempt = 0
    while kept < args.episodes and attempt < max_attempts:
        attempt += 1
        started = time.time()
        obs, _ = env.reset()
        if views is not None:
            for _ in range(3):  # let the new windows finish initializing
                app.update()
            views.apply(log=attempt == 1)
        print(f"[RESET] attempt {attempt} seed={env.layouts[0]['seed']} " + ", ".join(
            f"{n}=({o['xy'][0]:.3f},{o['xy'][1]:.3f})->{o['arm']}" for n, o in env.layouts[0]["objects"].items()))
        # Let the objects settle, arms holding still and open (not recorded).
        driver = BimanualExpertDriver(env, task, robot)
        hold = driver.current_action()
        for _ in range(settle_steps):
            obs, *_ = env.step(hold)
        if stager is not None:
            stager.start(attempt)
        wall_start = time.time()
        steps = 0
        while (action := driver.next_action()) is not None:
            if stager is not None:
                images, rec_action, rec_state, raw_action, raw_state = _record_row(obs, action)
                stager.add(images, action=rec_action, state=rec_state, raw_action=raw_action, raw_state=raw_state)
            obs, *_ = env.step(action)
            steps += 1
            if views is not None and steps % 30 == 0:
                views.apply()
            if args.realtime:
                time.sleep(max(0.0, wall_start + steps * env.step_dt - time.time()))
        info = env.episode_info()
        success = info["success"]
        keep = success or args.keep_failed
        if keep:
            kept += 1
        print(f"[script] Attempt {attempt} ({kept}/{args.episodes} successful): frames={stager.frames if stager else 0}"
              f" time={time.time() - started:.0f}s success={success}")
        for name, item in info["objects"].items():
            print(f"[script]   {name}: final={item['final_position']} axis_z={item['final_axis_z']} "
                  f"grasped={item['grasped']}")
        if stager is None:
            continue
        if keep:
            path = stager.finish({
                "task": task["task_description"],
                "task_name": task["task_name"],
                "robot_type": "seeed_b601_rs_bimanual_follower",
                "fps": fps,
                "resolution": [int(task["rendering"]["width"]), int(task["rendering"]["height"])],
                "feature_names": conventions.feature_names([arm.replace("_arm", "") for arm in ARMS]),
                "camera_keys": {name: spec["lerobot_key"] for name, spec in robot["cameras"].items()},
                "attempt": attempt,
                **info,
            })
            print(f"[dataset] Episode staged: {path}")
        else:
            stager.discard()
    print(f"[script] Done: {kept}/{args.episodes} episodes in {attempt} attempts")
    if args.hold_open:
        print("[script] Holding the window open; close it to exit.")
        while app.is_running():
            env.sim.render()
    env.close()
    return 0 if kept >= args.episodes else 1


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
