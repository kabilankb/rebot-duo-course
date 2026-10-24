"""Drive the bimanual reBot Isaac Lab scene from a vla-edge inference server.

Run with the Isaac Lab Python (it starts Isaac Sim), with ``rebot-sim-pipeline``
importable::

    python -m rebot_vla.sim_rollout --server 127.0.0.1:8202                  # headless
    python -m rebot_vla.sim_rollout --server 127.0.0.1:8202 --viz kit        # watch it

Per episode: reset to a layout (seed = ``--seed`` + episode; with a replay
server the default is the recorded episode's layout seed), let the objects
settle with the arms still, then loop: observation -> ``POST /act`` -> execute
``--execute`` rows of the chunk, until the task succeeds or ``--max-steps``.
The sim waits for each chunk (synchronous inference). Exit code 0 only if
every episode succeeded.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Bimanual reBot sim rollout against a vla-edge server.")
parser.add_argument("--server", default="127.0.0.1:8202", help="Inference server host:port.")
parser.add_argument("--instruction", default=None, help="Default: the task description from the task YAML.")
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--seed", type=int, default=None,
                    help="Layout seed of the first episode. Default: the replay server's recorded seed, else 0.")
parser.add_argument("--execute", type=int, default=None, help="Rows executed per chunk (default: the whole chunk).")
parser.add_argument("--max-steps", type=int, default=None,
                    help="Control steps per episode (default: the task's episode length).")
parser.add_argument("--task", default="stationery_sorting", help="Task name or YAML path (rebot-sim-pipeline).")
parser.add_argument("--robot", default="rebot_b601_bimanual", help="Robot name or YAML path (rebot-sim-pipeline).")
parser.add_argument("--rebot-project", default=None, help="Single-arm reBot checkout with the assets.")
parser.add_argument("--video", default=None, help="Write the three camera views side by side to this .mp4.")
parser.add_argument("--results", default=None, help="Write per-episode results to this JSON file.")
parser.add_argument("--realtime", action="store_true", help="Pace control steps to wall-clock time (watching).")
parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds.")
parser.add_argument("--episode-timeout", type=float, default=None,
                    help="Wall-clock limit per episode in seconds; a slower episode ends as a failure (timed_out).")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device=None)
args = parser.parse_args()
args.enable_cameras = True
if args.device is None:  # the YAML evaluation-mode device, unless --device was given
    from rebot_sim_pipeline.lab.config import load_task as _load_task

    args.device = str(_load_task(args.task, mode="evaluation")["sim"]["device"])
app = AppLauncher(args).app

import numpy as np
import requests
import torch
from rebot_sim_pipeline.lab.assets import (
    find_workstation,
    prepare_robot_asset,
    prepare_stationery_assets,
)
from rebot_sim_pipeline.lab.config import find_rebot_project, load_robot, load_task
from rebot_sim_pipeline.lab.env import ARMS, StationeryBimanualEnv
from rebot_sim_pipeline.lab.env_cfg import build_env_cfg
from rebot_sim_pipeline.lab.expert_driver import BimanualExpertDriver
from vla_edge.protocol.client import ActClient

from rebot_vla.embodiment import EMBODIMENT
from rebot_vla.wire import cameras_from_sim, command_from_action, state_from_sim


def observation(obs: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Wire state and camera fields for env 0."""
    policy = obs["policy"]
    q, opening = {}, {}
    for arm in ARMS:
        prefix = arm.replace("_arm", "")
        q[arm] = policy[f"{prefix}_joint_pos"][0].cpu().numpy().astype(np.float64)
        opening[arm] = float(policy[f"{prefix}_gripper"][0, 0])
    images = {key[: -len("_rgb")]: value[0].cpu().numpy() for key, value in policy.items() if key.endswith("_rgb")}
    return state_from_sim(q, opening), cameras_from_sim(images)


def act(client: ActClient, *call_args, **call_kwargs):
    """``client.act`` with one retry when the connection drops before any response.

    vla-edge's server (uvicorn) closes idle keep-alive connections after 5 s, about
    as long as the sim takes to execute one chunk, so a reused connection can be
    closed under the request. The server never saw that request, so resending it is safe.
    """
    try:
        return client.act(*call_args, **call_kwargs)
    except requests.ConnectionError:
        return client.act(*call_args, **call_kwargs)


def main() -> int:
    client = ActClient(args.server, timeout_s=args.timeout)
    contract = client.bind(embodiment=EMBODIMENT.name, cameras=EMBODIMENT.camera_names,
                           state_dim=EMBODIMENT.state_dim, action_dim=EMBODIMENT.action_dim)
    runtime = client.health().get("runtime", {})
    print(f"[rollout] server {args.server}: policy={contract.policy} family={contract.model_family} "
          f"horizon={contract.action_horizon} runtime={runtime}", flush=True)
    seed0 = args.seed if args.seed is not None else int(runtime.get("layout_seed") or 0)
    execute = min(args.execute or contract.action_horizon, contract.action_horizon)

    task = load_task(args.task, mode="evaluation", overrides={"sim": {"device": args.device}})
    robot = load_robot(args.robot)
    instruction = args.instruction or task["task_description"]
    project = find_rebot_project(args.rebot_project)
    assets = prepare_stationery_assets(task, project)
    cfg = build_env_cfg(task, robot, assets, robot_usd=prepare_robot_asset(robot, project),
                        workstation_usd=find_workstation(robot), enable_cameras=True)
    env = StationeryBimanualEnv(cfg, task, robot, seed=seed0)
    # Keep the final scene for the report; the loop below stops on success itself.
    env.terminate_on_success = False
    settle_steps = round(float(task["expert"]["settle_s"]) / env.step_dt)
    # One step short of the time-out, which would reset the env inside step().
    max_steps = args.max_steps or int(env.max_episode_length) - settle_steps - 1

    writer = None
    if args.video:
        import imageio

        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args.video, fps=round(1.0 / env.step_dt), macro_block_size=1)

    results = []
    for episode in range(args.episodes):
        seed = seed0 + episode
        env.base_seed = seed - env.reset_count  # reset_layout uses base_seed + reset_count
        obs, _ = env.reset()
        print(f"[RESET] episode {episode} seed={env.layouts[0]['seed']} " + ", ".join(
            f"{n}=({o['xy'][0]:.3f},{o['xy'][1]:.3f})->{o['arm']}" for n, o in env.layouts[0]["objects"].items()),
            flush=True)
        hold = BimanualExpertDriver(env, task, robot).current_action()
        for _ in range(settle_steps):
            obs, *_ = env.step(hold)

        chunk, row, steps, requests, latency = None, 0, 0, 0, []
        started = wall_start = time.time()
        timed_out = False
        while steps < max_steps and not bool(env.success[0]):
            if args.episode_timeout is not None and time.time() - started > args.episode_timeout:
                timed_out = True
                break
            state, cameras = observation(obs)
            if writer is not None:
                writer.append_data(np.concatenate([cameras[name] for name in EMBODIMENT.camera_names], axis=1))
            if chunk is None or row >= execute:
                chunk, dt_ms = act(client, cameras, instruction, state, seed=seed)
                row, requests = 0, requests + 1
                latency.append(dt_ms)
            action = torch.as_tensor(command_from_action(chunk[row]), device=env.device)[None]
            obs, *_ = env.step(action)
            row, steps = row + 1, steps + 1
            if args.realtime:
                time.sleep(max(0.0, wall_start + steps * env.step_dt - time.time()))
        info = env.episode_info()
        result = {"episode": episode, "seed": seed, "success": info["success"], "steps": steps,
                  "requests": requests, "server_ms_mean": float(np.mean(latency)) if latency else None,
                  "wall_s": round(time.time() - started, 1), "timed_out": timed_out, "objects": info["objects"]}
        results.append(result)
        if args.results:  # after every episode, so a stalled run still leaves its results
            Path(args.results).parent.mkdir(parents=True, exist_ok=True)
            Path(args.results).write_text(json.dumps(results, indent=2), encoding="utf-8")
        server_ms = "n/a" if result["server_ms_mean"] is None else f"{result['server_ms_mean']:.1f}"
        print(f"[rollout] episode {episode}: success={result['success']} steps={steps} requests={requests} "
              f"server_ms_mean={server_ms} wall={result['wall_s']}s" + (" TIMED OUT" if timed_out else ""), flush=True)
        for name, item in info["objects"].items():
            print(f"[rollout]   {name}: final={item['final_position']} axis_z={item['final_axis_z']} "
                  f"grasped={item['grasped']}", flush=True)

    if writer is not None:
        writer.close()
    successes = sum(r["success"] for r in results)
    print(f"[rollout] Done: {successes}/{len(results)} successful", flush=True)
    client.close()
    env.close()
    return 0 if successes == len(results) else 1


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
