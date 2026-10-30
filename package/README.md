# reBot Duo: Sim-to-Policy

*Simulate, collect, train and serve vision-language-action policies for two reBot B601 arms.*

This repository (`rebot-bimanual-vla`) runs bimanual reBot B601 policies through
[vla-edge](https://github.com/Agents2AgentsAI/vla-edge), the serving stack that
runs MolmoAct2, ABC-VLA and π0.5 on two YAM arms. This repository adds:

- the bimanual reBot embodiment;
- a replay test policy;
- a SmolVLA runtime that serves a LeRobot SmolVLA checkpoint trained on reBot data;
- a client that drives the Isaac Lab bimanual reBot scene from
  rebot-sim-pipeline through vla-edge's `POST /act`, on the same machine or
  against a server on a Jetson AGX Thor (hardware in the loop).

The step-by-step course is in the separate
[rebot-duo-course](https://github.com/kabilankb/rebot-duo-course) repository.

```
 Isaac Lab scene (rebot-sim-pipeline)            vla-edge server (rebot-vla-serve)
 ┌───────────────────────────────┐   POST /act   ┌─────────────────────────────────┐
 │ 2× reBot B601, YAMLab station │ ────────────▶ │ embodiment  bimanual-rebot      │
 │ top / left / right cameras    │  3 images,    │ policy      replay-bimanual-... │
 │ sim_rollout.py                │  14-D state   │             (or your model)     │
 │   executes chunk rows at 30Hz │ ◀──────────── │ action chunk (30, 14)           │
 └───────────────────────────────┘               └─────────────────────────────────┘
```

The vla-edge YAM example (`examples/bimanual-yam`) is the model for this split:

| | Bimanual YAM (vla-edge) | Bimanual reBot (here) |
|---|---|---|
| Embodiment | `bimanual-yam` | `bimanual-rebot` |
| Cameras (training order) | `top_cam`, `left_cam`, `right_cam` | the same names: the dataset's `top`, `left_wrist`, `right_wrist`; 480×640 RGB |
| State / action | 14 = 2 × (6 joints rad + gripper 0..1) | 14 = 2 × (6 joints + gripper) in the **reBot LeRobot convention** (below) |
| Robot side | `run_task.sh`: CAN arms, RealSense | `sim_rollout.py`: Isaac Lab scene; no real-robot client yet |
| Server | `vla-edge-serve` | `rebot-vla-serve` = `vla-edge-serve` + reBot registration |

## Wire format

These are the `observation.state` and `action` features of the datasets that
`rebot-bimanual-generate` writes. A policy trained on them is served without
conversion, and the same values drive the real leader/follower pair.

| Field | Values per arm (left then right) |
|---|---|
| `state` (14) | follower convention: `-degrees` on every joint; gripper 0 (closed) to 270 (open), **measured** opening |
| action rows (14) | leader convention: degrees ÷ `[-1, -1, 1, 1, 1, -1]`; gripper 0 (closed) to 45 (open), commanded |

The conversions to simulator units are `rebot_sim_pipeline.conventions`, the
same code that recorded the data (`src/rebot_vla/wire.py`). The grippers are
not 0..1, so the embodiment declares no vla-edge gripper channels.

## Install (this machine)

`rebot-sim-pipeline` and this package ship in the
[rebot-duo-course](https://github.com/kabilankb/rebot-duo-course) repository
(`rebot-sim-pipeline/` and `package/`). Clone it, then link them to the paths used below:
`ln -s ~/rebot-duo-course/rebot-sim-pipeline ~/rebot-sim-pipeline && ln -s ~/rebot-duo-course/package ~/rebot-bimanual-vla`.
The recorded dataset is not included: generate it with `rebot-bimanual-generate` (see `rebot-sim-pipeline/BIMANUAL.md`).

| Environment | Used for | Install |
|---|---|---|
| `~/vla-edge/.venv` (Python 3.14) | the server and the unit tests | `python3 -m venv ~/vla-edge/.venv && ~/vla-edge/.venv/bin/pip install --build-constraint $HOME/vla-edge/build-constraints.txt -e "$HOME/vla-edge[dev]" pyarrow` then `pip install --no-deps -e ~/rebot-sim-pipeline -e .` |
| `env_yamlab6` (Isaac Lab 3) | the sim client | `pip install json-numpy && pip install --no-deps -e ~/vla-edge`; this repo and rebot-sim-pipeline go on `PYTHONPATH` (the script does it) |

On Jetson Thor (JetPack 7), see [Hardware in the loop on Jetson Thor](#hardware-in-the-loop-on-jetson-thor).
It is not needed for this repository's tests.

## Test

```bash
# Unit tests: registration, wire conversions, replay server over HTTP. No GPU, no simulator.
~/vla-edge/.venv/bin/pytest -q

# End to end: replay episode 0 of a recorded dataset through the server into Isaac Lab.
scripts/sim_replay_test.sh ~/rebot-sim-pipeline/datasets/rebot_bimanual_stationery 0
```

The end-to-end script:

1. Starts `rebot-vla-serve` on port 8212 with the replay policy.
2. Checks `GET /act`.
3. Runs `sim_rollout` headless on CPU (the device the data was recorded on), at the
   episode's recorded layout seed, which the server reports.
4. Writes `outputs/replay_ep<N>/` containing `server.log`, `results.json`, and
   `rollout.mp4` with the three cameras side by side.

The server logs how far the robot's state is from the recorded state at every
chunk. That is the check that conventions and camera/state wiring are right.
Exit code 0 only if the task succeeds.

Options after `--` go to `sim_rollout`, e.g. `-- --viz kit --realtime` to watch.

**Verified (2026-10-24):**

| Check | Result |
|---|---|
| vla-edge's own suite on x86 + RTX PRO 5000 | 238 passed, 2 skipped |
| This repository's unit tests | 10 passed |
| End-to-end replay of episode 0 (layout seed 5, 1,332 frames) | **success** |

End-to-end details:

- 1,200 control steps and 40 `POST /act` requests, 1.3 ms per request on the server.
- **State drift** from the recording, checked at every chunk:
  - mean 0.006 over the 14 channels;
  - worst 3.3 out of 270 on the right gripper during a grasp, about 0.6 mm.
- All three objects ended in the cup.

## Serve your own policy

`rebot-vla-serve` passes every `vla-edge-serve` option through. Defaults:

| Option | Default |
|---|---|
| `--policy` | `replay-bimanual-rebot` |
| `--backend` | `replay` |
| `--episode` | 0 (a replay-only option added by `rebot-vla-serve`) |

To serve a trained model:

1. Add a `PolicySpec` with `embodiment="bimanual-rebot"`.
2. Register its model family the way `embodiment.register()` does for the replay policy.
3. Start `rebot-vla-serve --policy <name> --backend <backend>`.

A policy has to be trained on reBot data, e.g. a dataset from
`rebot-bimanual-generate`. The released YAM checkpoints expect YAM kinematics
and radians, so they do not transfer.

### SmolVLA

`smolvla-bimanual-rebot` serves a `lerobot-train` SmolVLA checkpoint (chunk 50).
It needs `lerobot[smolvla]` in the server's environment. The checkpoint's
saved processors rename the cameras, tokenize and normalize.

```bash
rebot-vla-serve --policy smolvla-bimanual-rebot --backend torch \
  --checkpoint outputs/smolvla_rebot_bimanual/checkpoints/last/pretrained_model --host localhost --port <port>
```

To evaluate a checkpoint on unseen layouts in one command, run
`scripts/sim_policy_eval.sh`. It starts the server, runs `sim_rollout`, and
writes everything to `outputs/eval_<step>_seed<S>x<N>/`:

```bash
scripts/sim_policy_eval.sh outputs/smolvla_rebot_bimanual/checkpoints/last/pretrained_model 20 1000
```

The sim client works with any server that reports `embodiment: bimanual-rebot`:

```bash
python -m rebot_vla.sim_rollout --server HOST:PORT --episodes 10 --seed 1000 --results eval.json
```

| Option | Default |
|---|---|
| `--execute` | the whole chunk; set the rows run per chunk |
| `--instruction` | the task description |
| `--device` | the task's evaluation-mode device, `cuda:0` |
| `--video` | off |
| `--viz kit` | off; opens the GUI |
| `--episode-timeout` | off; wall-clock seconds before an episode is abandoned (`timed_out` in the results) |

`results.json` is rewritten after every episode, so a stopped run keeps the
episodes it finished. If the connection drops before any response arrives, the
client retries the request once: uvicorn closes idle connections after 5 s,
which is about how long the sim takes to execute one chunk.

## Hardware in the loop on Jetson Thor

The simulator stays on the workstation and the policy is served from the Thor
on the local network. Every chunk the arms execute is computed on the Thor.

On the Thor (JetPack 7.0, Python 3.12):

```bash
python3 -m venv ~/rebot-hil/.venv && source ~/rebot-hil/.venv/bin/activate
pip install torch==2.10.0 torchvision --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple
pip install nvpl-blas nvpl-lapack "nvidia-cudss-cu13<1"   # libraries the Jetson wheel links
python ~/rebot-hil/vla-edge/scripts/jetson_thor_postinstall.py
pip freeze | grep -E "^(torch|torchvision|nvpl|nvidia-cudss)" > ~/rebot-hil/pins.txt
pip install -c ~/rebot-hil/pins.txt -e ~/rebot-hil/lerobot[smolvla] -e ~/rebot-hil/vla-edge fastapi uvicorn json-numpy
pip install --no-deps -e ~/rebot-hil/rebot-bimanual-vla
python -c "import torch; print(torch.cuda.is_available())"   # must print True

rebot-vla-serve --policy smolvla-bimanual-rebot --backend torch \
  --checkpoint ~/rebot-hil/smolvla_020000 --host <thor-lan-address> --port <port>
```

On the workstation, point `sim_rollout` at it with `--server <thor-host>:<port>`.
The server has no authentication: bind it to a private network you trust.

Results for the step-20,000 SmolVLA checkpoint, CPU physics, layout seeds from 1000:

| Server | Episodes | Success | Inference per chunk |
|---|---|---|---|
| Workstation, RTX PRO 5000 Blackwell | 5 | 0 / 5 | ≈ 160 ms |
| Jetson AGX Thor, MAXN | 20 | 1 / 20 (seed 1015) | ≈ 278 ms |

In both runs the policy grasps and carries objects but rarely releases them.

Over Wi-Fi, one request (three raw 480×640 images, 3.7 MB) takes about
0.6 s to upload, longer than inference. Use a wired link for latency numbers.

## Files

| File | Role |
|---|---|
| `src/rebot_vla/embodiment.py` | `bimanual-rebot` embodiment, replay and SmolVLA `PolicySpec`s, `register()` |
| `src/rebot_vla/replay.py` | replay policy runtime: reads a LeRobot v3 dataset with pyarrow |
| `src/rebot_vla/smolvla.py` | SmolVLA policy runtime for a LeRobot checkpoint |
| `src/rebot_vla/serve.py` | `rebot-vla-serve` |
| `src/rebot_vla/wire.py` | simulator units ↔ wire values, camera renaming |
| `src/rebot_vla/sim_rollout.py` | Isaac Lab client: reset, settle, chunked control, success, video |
| `scripts/sim_replay_test.sh` | end-to-end server + sim test |
| `scripts/sim_policy_eval.sh` | evaluate a trained checkpoint on unseen layouts |
| `tests/test_rebot_vla.py` | unit tests |
