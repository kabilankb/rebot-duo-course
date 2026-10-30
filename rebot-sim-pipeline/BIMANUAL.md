# Bimanual reBot pipeline (Isaac Lab)

Two reBot B601 arms share one pencil cup on YAMLab's workstation. They put two
pens and an eraser into the cup, pens standing upright. Demonstrations come from
the same scripted expert as the single-arm pipeline ([`WORKFLOW.md`](WORKFLOW.md)),
run one arm at a time. They are recorded as a LeRobot v3 dataset in the reBot
real-robot conventions.

```bash
rebot-bimanual-generate -n 20                      # collect 20 successful episodes
rebot-bimanual-generate -n 1 --gui --no-recording  # watch one dry run, with camera windows
```

> **Status (2026-10-20):**
> - On the YAMLab station, 3/3 headless dry runs succeeded (seeds 5–7). The GUI runs with the
>   three camera windows on their cameras.
> - Recording on the YAMLab station (2026-10-24): 2/2 episodes (seeds 5–6, 1,332 frames each)
>   written as LeRobot v3. Episode 0 replayed through a vla-edge server into this scene
>   (`~/rebot-bimanual-vla`) succeeds with near-zero state drift.
> - See [Known issues](#7-known-issues).

| Part | From |
|---|---|
| Workstation, arm mounts, top camera, world frame, layered YAML configs | YAMLab (`~/yamlab6`) |
| Robot USD, colors, wrist camera mount, grasp assist, success rules, stationery assets, dataset conventions | single-arm reBot (`~/rebot-arm-dli-isaacsim`) |
| Scripted expert, IK, layout sampling | this package (`expert.py`, `kinematics.py`, `workspace.py`) |

---

## 1. The scene

```
            camera gate (top camera, 1.70 m)
   ┌──────────────────────────────────────────┐   YAMLab workstation
   │                                          │   table top z = 0.7517
   │   LeftArm  (0.2525, +0.305) ── faces +X ─▶│
   │                 pencil cup (0.45, 0)     │
   │   RightArm (0.2525, -0.305) ── faces +X ─▶│
   └──────────────────────────────────────────┘
```

- **Arms:** reBot B601 at YAMLab's YAM mounts, facing +X. The robot's left is +Y.
  - The PhysX asset ships plain grey. A generated wrapper binds the Seeed materials
    (green, black, aluminium, motor) to 23 visual parts, as the single-arm scene builder does.
  - It also adds the UVC32 wrist camera mount, board and lens on `link6`.
- **Cameras** (640×480):

  | Camera | Mount | Pose and lens |
  |---|---|---|
  | `top` | YAMLab's calibrated top camera on the gate | focal 12.84 mm, i.e. fx ≈ 392 px |
  | `left_wrist`, `right_wrist` | under each `link6` | the single-arm reBot FrontCamera transform exactly: position (0.065394, 0, 0.096031), same rotation; focal 4.2 mm |

- **Objects:**
  - `pen_red` (assigned to the left arm) and `pen_blue` (right arm).
  - `eraser`: each episode it goes to whichever arm it lands nearer to.
  - Every episode samples new poses. Each object lies 0.18–0.35 m in front of its arm, on its
    arm's side, inside the table, at least 5 cm from the others and clear of the cup.
- **Cup:** static and concave, at (0.45, 0). That is 0.363 m from each arm base, the same reach
  as in the single-arm scene.

## 2. One episode

1. **Reset:** sample a layout (seed = `--seed` + attempt), drop the objects, settle 0.5 s.
   Settling is not recorded.
2. **Home:** both arms move to the expert home pose (0, −45, −45, 0, 0, 0)°.
3. **Turns:** pens first, then the eraser, grouped per arm. For example:
   - left arm → `pen_red`;
   - right arm → `pen_blue`, `eraser`.

   Each turn is a `ScriptedPickPlace` for that arm while the other arm holds still. Every turn
   ends at home, so the arms never meet over the cup.
4. **Pens:** held 3.5 cm from their centre and stood upright in slots ±1.4 cm either side of
   the cup centre along Y, with the left arm's pen on +Y.
5. **End:** a 2 s hold at the end of the last turn lets the success check settle.
6. **Outcome:** a successful episode is staged; a failed one is discarded and retried, up to
   2N+5 attempts.

**Grasp assist** (as in the single-arm app): a 4 mm pen between convex-hull fingers is not a
reliable physical grasp, so a closed gripper carries the object rigidly.

| Event | Condition |
|---|---|
| attach | command ≤ 0.014 m, opening ≤ 0.028 m, and the object within 4 cm of the grasp probes |
| release | command and opening both ≥ 0.026 m |

It runs every physics step inside the gripper action term.

**Success** (the single-arm checker), held for 0.35 s:

- Every object's origin is within 3.8 cm of the cup centre and 1–16 cm above the table.
- Each pen axis is within about 57° of vertical.
- Every object moves slower than 0.25 m/s.
- Both grippers are open and holding nothing.

## 3. Prerequisites

| What | Where (this machine) |
|---|---|
| Isaac Lab 3 / Isaac Sim 6 Python | `~/miniconda3/envs/env_yamlab6/bin/python` (or set `$REBOT_ISAACLAB_PYTHON`) |
| YAMLab (workstation USD) | the `yamlab` package installed in that env, or set `$YAMLAB_HOME` |
| Single-arm reBot project (robot, stationery and camera assets) | `~/rebot-arm-dli-isaacsim` (or set `$REBOT_HOME`, or pass `--rebot-project`) |
| LeRobot venv (dataset writing) | `~/rebot_lerobot/.venv/bin/python`, LeRobot 0.6.2 (or set `$REBOT_LEROBOT_PYTHON`) |

The first run generates the object, cup and robot wrapper USDs into `assets/generated/`, which
is git-ignored. They are rebuilt automatically when their config changes.

## 4. Run it

```bash
cd ~/rebot-sim-pipeline
pip install --no-deps -e .     # or prefix commands with PYTHONPATH=src
```

| Command | What it does |
|---|---|
| `rebot-bimanual-generate -n 20` | collect 20 successful episodes into `datasets/rebot_bimanual_stationery` |
| `rebot-bimanual-generate -n 20 --seed 100` | add 20 more on new layouts (reusing a seed repeats its layouts) |
| `rebot-bimanual-generate -n 1 --gui --no-recording` | watch a dry run in real time, with a window per camera; the window stays open until closed |
| `rebot-bimanual-generate -n 1 --gui` | the same, but recorded |
| `rebot-bimanual-generate --print-command` | only show the simulator command |

Other options: `--dataset-root`, `--repo-id`, `--keep-failed`, `--task` (a name or a YAML
path). Arguments after `--` go to the simulator script unchanged, e.g. `-- --device cuda:0`.
Exit code 0 only if all N episodes were collected and written.

The two stages can also run separately:

```bash
# 1. simulate and stage (Isaac Lab Python)
python -m rebot_sim_pipeline.lab.generate --episodes 20 --seed 0 --staging datasets/.staging
# 2. write the LeRobot v3 dataset (LeRobot venv)
~/rebot_lerobot/.venv/bin/python -m rebot_sim_pipeline.write_dataset \
    --staging datasets/.staging --dataset-root datasets/rebot_bimanual_stationery \
    --repo-id local/rebot_bimanual_stationery
```

Progress lines:

```
[RESET] attempt 1 seed=5 pen_red=(0.421,0.438)->left_arm, pen_blue=(0.511,-0.261)->right_arm, ...
[script] turn 1/2: left_arm -> pen_red
[script] Attempt 1 (1/20 successful): frames=1392 time=87s success=True
[dataset] Episode staged: .../episode_20261020_200137_0001
[bimanual] dataset .../rebot_bimanual_stationery: 20 episode(s) added, 20 total
```

## 5. Dataset

| Feature | Shape | Contents |
|---|---|---|
| `observation.images.top` | 480×640×3 video | YAMLab top camera |
| `observation.images.left_wrist`, `right_wrist` | 480×640×3 video | reBot wrist cameras |
| `observation.state` | 14 | `left_*.pos` (7) then `right_*.pos` (7), follower convention |
| `action` | 14 | same names, leader convention |
| task | text | "Put the two pens and the eraser into the pencil cup" |

- **Format:** 30 fps, `robot_type = seeed_b601_rs_bimanual_follower`.
- **Per arm:** the values are the single-arm reBot conventions (`conventions.py`), so a policy
  drives the real leader/follower pair unchanged:
  - **action:** joint degrees ÷ leader scale [−1, −1, 1, 1, 1, −1]; gripper 0 (closed) to 45 (open).
  - **state:** −degrees on every joint; gripper 0 to 270.
  - So state and action differ in sign on joints 3, 4 and 5.
- **Raw values:** the staged episodes also keep simulator units (rad, m) in
  `raw_actions.npy` / `raw_states.npy`.
- **Sidecars:** `episode_metadata/episode_NNNNNN.json` records the layout, seed, arm
  assignment, per-object final pose and whether it was grasped, and success.

## 6. Configuration

YAMLab-style layers, lowest to highest precedence:
`configs/defaults.yaml` < its `modes.<mode>` < `configs/tasks/<task>.yaml` < its `modes.<mode>`
< CLI. Nested dicts merge; scalars and lists replace. Modes are `generate` (CPU, 1 env), `view`
and `evaluation` (GPU).

| File (under `src/rebot_sim_pipeline/lab/configs/`) | Holds |
|---|---|
| `robot/rebot_b601_bimanual.yaml` | robot facts: arm mounts, kinematic chain, joint limits, gripper geometry, drives, grasp-assist thresholds, cameras, wrist camera model, Seeed colors, workstation |
| `defaults.yaml` | physics (120 Hz, decimation 4 → 30 Hz), rendering 640×480, JPEG quality, per-mode device |
| `tasks/stationery_sorting.yaml` | objects and their arms, order, cup, layout sampling, success rules, expert tuning |

In the task file, XY values are world coordinates and every height is measured from the table
top (`table_z`). A new task is a new file in `configs/tasks/` passed with `--task`.

## 7. Known issues

- **GUI startup can stall.** Twice the GUI sat in loading for minutes; once another Isaac Sim
  process was running at the same time. Killing it and relaunching with nothing else running
  worked. The cause is not known. Headless runs were not affected.
- **No MimicGen or teleop yet.** Demonstrations are scripted only.
- **No policy evaluation script yet.** The `evaluation` mode exists (GPU, ends an episode on
  success), but no rollout script uses it.
- **No domain randomization.** Lighting, materials and cameras are fixed; the single-arm
  project's randomization is not ported.
- **The eraser is not checked for tilt.** As in the single-arm checker, it may stand on end in
  the cup.

## 8. Files

| File | Role |
|---|---|
| `chain.py` | forward kinematics from USD joint frames (matches the simulator to 5e-7 m) |
| `bimanual.py` | two-arm layout sampling, per-arm turn order |
| `conventions.py` | reBot leader/follower recording conventions |
| `staging.py` | staged-episode format shared by simulator and writer |
| `write_dataset.py` | staged episodes → LeRobot v3 (LeRobot venv) |
| `bimanual_cli.py` | `rebot-bimanual-generate` |
| `lab/config.py` | layered YAML loader (no Isaac Lab import) |
| `lab/assets.py` | generated object, cup and robot wrapper USDs; workstation lookup |
| `lab/robot_cfg.py` | reBot `ArticulationCfg` |
| `lab/env_cfg.py` | scene, 14-D actions, observations, events, terminations |
| `lab/env.py` | env: layout reset, success latch, accessors for the expert |
| `lab/grasp_assist.py` | grasp assist and the gripper action term |
| `lab/expert_driver.py` | runs `ScriptedPickPlace` per arm turn |
| `lab/generate.py` | the collection loop; camera windows for GUI runs |
| `tests/test_bimanual.py` | chain vs USD, conventions, layout constraints, config layers, wrist camera pose |

Tests (27, no simulator needed): `PYTHONPATH=src:tests python3 -m unittest discover -s tests -v`
