# Scripted demonstration workflow

How the reBot stationery-sorting demonstrations are generated in Isaac Sim,
recorded as a LeRobot dataset, and published to Hugging Face.

- **Task:** the reBot B601 arm picks two pens and an eraser from the table and
  puts them in a pencil cup, pens standing upright.
- **Demonstrator:** a scripted expert instead of a human on the leader arm. It
  reads the exact object poses from the simulator (privileged information) and
  plans the motion with numerical IK.
- **Output:** a LeRobot v3 dataset with two cameras and 7-D joint actions,
  ready for imitation learning (ACT, GR00T, ...).

Published result: [`kabilanKB/rebot_pick_place`](https://huggingface.co/datasets/kabilanKB/rebot_pick_place),
20 successful episodes (24,560 frames at 30 FPS), collected in 22 attempts.

---

## 1. How it fits together

```
 rebot-sim-pipeline (this package, pure numpy)      rebot-arm-dli-isaacsim (Isaac Sim app)
 ┌──────────────────────────────────────────┐      ┌─────────────────────────────────────────┐
 │ workspace.sample_workspace_layout        │◀─────│ DomainRandomizer.reset()   (each reset) │
 │   reachable, non-overlapping object poses│      │   + lighting, mass, wrist-camera jitter │
 │                                          │      │                                         │
 │ expert.ScriptedPickPlace                 │◀────▶│ leader_teleop.py --scripted             │
 │   next_command() -> (6 joints, gripper)  │ fk,  │   same path as leader-arm teleop:       │
 │ kinematics.solve_ik (uses fk callback)   │ poses│   drive arm, grasp assist, success check│
 └──────────────────────────────────────────┘      │   EpisodeRecorder (side + wrist, 30 FPS)│
                                                   └────────────────────┬────────────────────┘
                                                                        ▼
                                          dataset_writer.py (LeRobot venv) → LeRobot v3 dataset
                                                                        ▼
                                                    push_to_hub → huggingface.co/datasets/...
```

The expert produces the same thing a leader arm does: one `(arm joint
targets, gripper target)` command per physics step. Grasping, carrying,
release, success detection and recording are the project's normal
teleoperation code, so scripted and human episodes have the same format and can
be mixed for training.

### Where the object poses come from

| Step | Question | How |
|---|---|---|
| Object pose | Where is the pen? | `get_world_pose()` from the simulator: exact, no camera |
| Grasp target | Where should the gripper go? | Geometry: 3.5 cm from the pen centre (the end nearer the robot), fingertips just above the table, pointing down, fingers across the pen |
| IK | Which joint angles reach it? | `solve_ik`: damped least squares; each iteration calls FK, which Isaac Sim computes by setting the joints and reading the wrist pose (no physics step, so the scene is not disturbed) |

Only the cameras and actions go into the dataset. A policy trained on it has to
find the objects from pixels, which is what makes it usable on the real arm.

### One episode

1. **Reset:** objects are placed at random, reachable, non-overlapping poses,
   fall onto the table and settle. Lighting and the wrist camera are
   randomized. Recording starts.
2. **Each pen:** move above it, then descend in a straight line. Close; grasp
   assist attaches the pen. Lift, carry it above its cup slot, and stand it
   upright with the gripped end on top. Lower until its bottom is just above
   the cup floor, open, and back out along the gripper axis.
3. **Eraser:** picked the same way and dropped into the cup between the pens.
4. **Home:** return, then hold 2 s so the success check can settle (every
   object in the cup, stable, gripper open).
5. **End:** a successful episode is written to the dataset in the background.
   A failed one is discarded and the scene re-randomized.

The pens go in first, standing on the flat cup floor either side of the centre,
and the eraser last. With the eraser first, a pen landing on it could topple out
of the cup.

---

## 2. Prerequisites

| What | Where (this machine) |
|---|---|
| Isaac Sim (source build) | `~/isaacsim/_build/linux-x86_64/release` (set `ISAACSIM_ROOT` to this) |
| reBot Isaac Sim project | `~/rebot-arm-dli-isaacsim` |
| This package | `~/rebot-sim-pipeline` (`run.sh` adds `src/` to `PYTHONPATH`; override with `REBOT_SIM_PIPELINE`) |
| LeRobot venv (dataset writing, Hub upload) | `~/rebot_lerobot/.venv` |
| Hugging Face login | `huggingface-cli login` in the LeRobot venv |

Isaac Sim's Python has no `pip`, so this package is not installed into it.
`run.sh` puts it on `PYTHONPATH` instead.

Object placement is configured in
`rebot-arm-dli-isaacsim/config/domain_randomization.json` → `workspace_sampling`:

| Key | Value | Meaning |
|---|---|---|
| `center_radius_m` | 0.18–0.35 | object centres this far from the robot base, where IK reliably reaches with the gripper pointing down |
| `robot_base_xy_m` | [0.0, 0.455] | robot base; it faces world −Y |
| `bounds_x_m`, `bounds_y_m` | ±0.34, −0.09–0.53 | whole footprint inside the enclosure |
| `keep_out_circles` | cup at (−0.16, 0.12), r = 0.085 | nothing spawns in or against the cup |
| `min_separation_m` | 0.05 | gap between object footprints |
| `yaw_deg` | −180–180 | any orientation |

Set `enabled: false` to go back to the original small jitter around fixed
positions (that mode does not need this package).

---

## 3. Run it

All commands are run from `~/rebot-arm-dli-isaacsim` with:

```bash
export ISAACSIM_ROOT=~/isaacsim/_build/linux-x86_64/release
```

### 3.1 Check the package

```bash
cd ~/rebot-sim-pipeline
PYTHONPATH=src:tests python3 -m unittest discover -s tests -v
```

The end-to-end test drives the whole expert program against a toy 6-joint arm
and a fake world, and checks every object is released upright over the cup.

### 3.2 Dry run (no recording)

```bash
cd ~/rebot-arm-dli-isaacsim
./run.sh --headless --scripted --no-recording --seed 700
```

Expect `[script] Attempt 1 (1/1 successful) ... success=True` and a final pose
line per object. Drop `--headless` to watch in the GUI.

### 3.3 Collect

```bash
./run.sh --scripted --episodes 20 --dataset-root datasets/rebot_stationery_scripted_dr
```

| Flag | Effect |
|---|---|
| `--episodes N` | collect N **successful** episodes; failures are discarded and retried, up to 2N+5 attempts |
| `--keep-failed` | save failed attempts too (`--episodes` then counts attempts) |
| `--headless` | no GUI; faster |
| `--seed N` | start another randomization sequence (scene *k* uses seed N+k) |
| `--no-dr` | fixed scene, no randomization |
| `--no-recording` | dry run |

The app exits by itself when done (exit code 0 only if all N succeeded). Allow
about 1–1.5 min per attempt in the GUI.

To add more episodes to the same dataset, use a **new** `--seed`. Reusing a
seed repeats the same scenes.

Progress lines to watch in the log:

```
[RESET] seed=602 objects: pen_red=(0.224,0.372), ...
[GRASP] Stress-free attachment created for pen_blue ...
[SORT] In pencil cup: eraser, pen_blue, pen_red (3/3)
[SUCCESS] Both pens and the eraser are stable in the pencil cup ...
[script] Attempt 2 (2/20 successful): ... success=True
[dataset] Episode saved; total completed: 2
```

`rebot-sim-generate` (see `README.md`) wraps this in a single unattended
command.

### 3.4 Check the dataset

```bash
cd ~/rebot_lerobot && source .venv/bin/activate
lerobot-dataset-viz --repo-id local/rebot_stationery_front_side \
  --root ~/rebot-arm-dli-isaacsim/datasets/rebot_stationery_scripted_dr --episode-index 0
```

| Feature | Shape | Contents |
|---|---|---|
| `observation.images.side` | 480×640×3, video | fixed side camera |
| `observation.images.front` | 480×640×3, video | wrist camera |
| `observation.state` | 7 | follower joints + gripper |
| `action` | 7 | commanded joints + gripper (leader convention) |
| task | text | "Put the two pens and the eraser into the pencil cup" |

`episode_metadata/episode_NNNNNN.json` records each episode's layout, seed,
lighting and success.

### 3.5 Publish to Hugging Face

Run with the LeRobot venv's Python. The script refuses to upload unless the
dataset has the expected number of episodes and every one succeeded:

```python
import json, sys
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path.home() / "rebot-arm-dli-isaacsim/datasets/rebot_stationery_scripted_dr"
REPO_ID = "kabilanKB/rebot_pick_place"
EXPECTED = 20

dataset = LeRobotDataset(REPO_ID, root=ROOT)
sidecars = sorted((ROOT / "episode_metadata").glob("episode_*.json"))
ok = all(json.loads(p.read_text())["success"] for p in sidecars)
print(f"episodes={dataset.num_episodes} frames={len(dataset)} sidecars={len(sidecars)} all_success={ok}")
if dataset.num_episodes != EXPECTED or len(sidecars) != EXPECTED or not ok:
    sys.exit("check failed; not pushing")
dataset.push_to_hub(
    private=False,
    tags=["LeRobot", "reBot", "isaac-sim", "simulation", "scripted-expert", "pick-and-place"],
)
```

This uploads `data/`, `meta/`, `videos/` and `episode_metadata/`, writes a
dataset card (license `apache-2.0` by default), and tags the revision `v3.0`.
Pass `private=True` for a private repository.

---

## 4. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Isaac Sim Python was not found: ~/isaacsim/python.sh` | Source build; set `ISAACSIM_ROOT=~/isaacsim/_build/linux-x86_64/release` |
| `--scripted needs the rebot-sim-pipeline package` | Package not at `~/rebot-sim-pipeline`; set `REBOT_SIM_PIPELINE` |
| Long `libtorchcodec` / `libavutil.so` tracebacks on every save | Harmless: LeRobot falls back to PyAV for decoding |
| `Cannot call 'add_frame()' on a read-only dataset` | Old `dataset_writer.py` opening an existing dataset; the current one uses `LeRobotDataset.resume()`. Failed episodes stay in `datasets/.<name>_staging/` and can be converted with `scripts/dataset_writer.py --staging <dir> --dataset-root <dataset> --repo-id <id>` |
| Second instance fails to start | Only one app can bind UDP 5005; wait for the other run to exit |
| Log stuck right after "Starting kit application" | Isaac Sim startup hang; stop the process and relaunch |
| `[script-warn] <object> was not grasped` | Occasional grasp miss; the attempt is discarded and retried |
| A pen counted out of the cup although it fell in | It is leaning past the checker's limits (centre > 3.8 cm from the cup centre, or tilted more than ~57° from vertical); the attempt is retried |

---

## 5. Files

| File | Role |
|---|---|
| `rebot-sim-pipeline/src/rebot_sim_pipeline/expert.py` | scripted pick-and-place program |
| `rebot-sim-pipeline/src/rebot_sim_pipeline/kinematics.py` | numerical IK, quaternion helpers |
| `rebot-sim-pipeline/src/rebot_sim_pipeline/workspace.py` | reachable, non-overlapping layout sampling |
| `rebot-arm-dli-isaacsim/scripts/leader_teleop.py` | `--scripted`, `--episodes`, `--keep-failed`; FK/pose callbacks for the expert |
| `rebot-arm-dli-isaacsim/scripts/domain_randomization.py` | calls the layout sampler when `workspace_sampling.enabled` |
| `rebot-arm-dli-isaacsim/scripts/dataset_writer.py` | appends episodes with `LeRobotDataset.resume()` |
| `rebot-arm-dli-isaacsim/config/domain_randomization.json` | `workspace_sampling` settings |
| `rebot-arm-dli-isaacsim/run.sh` | adds this package to `PYTHONPATH` |
