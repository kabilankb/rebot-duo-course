# rebot-sim-pipeline

Scripted-expert demonstrations for the reBot arm stationery-sorting task: the
arm picks two pens and an eraser from randomized positions and puts them in a
pencil cup, pens standing upright.

The core package is pure numpy. It never imports Isaac Sim; the host simulator is
reached only through callbacks, so everything here is testable on its own. It
is used by [`rebot-arm-dli-isaacsim`](../rebot-arm-dli-isaacsim)
(`./run.sh --scripted`), which supplies the callbacks, grasping, success
detection and LeRobot recording.

End-to-end guide (collect, verify, publish to Hugging Face): [`WORKFLOW.md`](WORKFLOW.md).

**Bimanual (two arms, Isaac Lab):** the `lab/` subpackage runs the same task
with two reBot arms sharing one cup, as an Isaac Lab environment configured by
layered YAML in the style of YAMLab. One command collects demonstrations into a
LeRobot v3 dataset: `rebot-bimanual-generate -n 20`. Guide: [`BIMANUAL.md`](BIMANUAL.md).

## Contents

| Module | What it does |
|---|---|
| `expert.py` | `ScriptedPickPlace`: yields one `(arm joint targets, gripper target)` per physics step, like a teleoperation leader |
| `kinematics.py` | `solve_ik`: damped-least-squares IK against any `fk(q)` callback, finite-difference Jacobian, joint limits by clipping; `quat_to_matrix` |
| `workspace.py` | `sample_workspace_layout`: non-overlapping object poses inside the arm's reach, clear of the robot base and keep-out areas (the cup) |
| `chain.py` | `SerialChain`: forward kinematics from USD physics-joint frames (the reBot chain is in the robot YAML), for IK without the simulator |
| `bimanual.py` | `sample_bimanual_layout` (each object in its arm's reach and on its side), `work_order` (per-arm turns) |
| `conventions.py` | reBot leader/follower recording convention: sim joints and gripper <-> recorded degrees and 0-45 / 0-270 gripper |
| `staging.py` | staged episodes (JPEG frames + npy + metadata) shared by the simulator and the LeRobot writer |
| `write_dataset.py` | staged episodes -> LeRobot v3 dataset (run in the LeRobot venv) |
| `bimanual_cli.py` | `rebot-bimanual-generate`: collect, write and report in one command |
| `lab/` | Isaac Lab env for two arms: YAML configs, scene, grasp assist, success check, expert driver, `generate` |

## Install

Into the Python that runs the simulator (Isaac Sim's own interpreter):

```bash
~/isaacsim/_build/linux-x86_64/release/python.sh -m pip install --no-deps -e ~/rebot-sim-pipeline
```

`--no-deps` keeps pip from touching Isaac Sim's numpy. `-e` means edits here
take effect without reinstalling.

## Using the expert

```python
from rebot_sim_pipeline import GripperGeometry, SceneGeometry, ScriptedPickPlace

expert = ScriptedPickPlace(
    fk=fk,                      # q -> (wrist position, wrist rotation matrix)
    object_pose=object_pose,    # name -> (position, rotation matrix)
    is_attached=is_attached,    # () -> bool, the gripper currently holds something
    arm_lower=lower, arm_upper=upper,
    gripper=GripperGeometry(grasp_offset=..., fingertip=..., finger_axis=...),
    geometry=SceneGeometry(
        robot_base_xy=..., cup_center_xy=..., cup_top_z=..., cup_floor_z=...,
        pen_half_lengths={"pen_red": 0.068, "pen_blue": 0.078},
    ),
    order=["pen_blue", "pen_red", "eraser"],
    open_m=0.05, closed_m=0.0, dt=1 / 120, start_q=q0,
)
while (command := expert.next_command()) is not None:
    arm_targets, gripper_target = command
    ...  # apply, step physics
```

`fk` may move the simulated arm (in Isaac Sim, set the joints and read the link
pose, which needs no physics step); restore the commanded pose before stepping.

Motions are planned when they start, from the poses measured then:

1. above the object, gripper pointing down, fingers across its long axis;
2. straight down until the fingertips are just above the surface; close;
3. lift; carry above the cup; lower; open; back out along the gripper axis.

Pens are held 3.5 cm from their centre (the end nearer the robot) and stood
upright with that end on top, in two slots either side of the cup centre, so
the fingers open above the rim. Other objects are dropped in upright. Put pens
before other objects in `order` so they stand on the flat cup floor.

`ExpertTuning` holds the home pose, pen grip and slot offsets, and the
pregrasp and lift heights.

## Generating a dataset (auto mode)

`rebot-sim-generate` collects demonstrations unattended: it runs the reBot
project headless in `--scripted` mode, one freshly randomized scene per episode,
keeps the successful ones, retries the failures (up to 2N+5 attempts), and then
reads the dataset back to report how many episodes were added. It exits 0 only if
all requested episodes were written.

```bash
pip install --no-deps -e ~/rebot-sim-pipeline      # any Python; the CLI is stdlib-only
rebot-sim-generate -n 50                           # -> <project>/datasets/rebot_stationery_scripted_dr
rebot-sim-generate -n 50 --seed 2                  # more episodes, new scenes, same dataset
rebot-sim-generate -n 1 --gui --no-recording       # watch one dry run
rebot-sim-generate -n 20 --dataset-root /data/rebot --repo-id me/rebot_scripted
rebot-sim-generate --print-command                 # show the run.sh call only
```

Without installing: `PYTHONPATH=~/rebot-sim-pipeline/src python3 -m rebot_sim_pipeline ...`.
It finds the project through `--project`, `$REBOT_HOME`, a sibling checkout or
`~/rebot-arm-dli-isaacsim`, and Isaac Sim through `--isaacsim-root`,
`$ISAACSIM_ROOT`, `/isaac-sim` or `~/isaacsim` (a source build's
`_build/linux-x86_64/release` is found too). Arguments after `--` go to `run.sh`
unchanged (e.g. `-- --strict-physics`). Relative `--dataset-root` paths are taken
from the project. Reusing a seed on an existing dataset repeats its scenes.

## Tests

```bash
cd ~/rebot-sim-pipeline
PYTHONPATH=src:tests python3 -m unittest discover -s tests -v
```

The end-to-end test runs the whole program against a toy 6-joint arm and a
fake world that carries objects, and checks every object is released upright
over the cup.
