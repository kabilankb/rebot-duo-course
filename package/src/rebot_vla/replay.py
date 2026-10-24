"""A vla-edge policy runtime that replays one episode of a reBot LeRobot dataset.

It ignores the images and returns the recorded action rows in order, one
chunk per request, so a client that executes every row of every chunk
reproduces the recording. A new ``seed`` (the client sends one per episode)
restarts the episode. Each request logs how far the robot's state is from the
recorded state at that point, which is the test signal: a wire-convention or
camera/state mix-up shows up as drift long before the task fails.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("rebot_vla.replay")

DATASET_ENV = "REBOT_REPLAY_DATASET"
EPISODE_ENV = "REBOT_REPLAY_EPISODE"


def load_episode(root: str | Path, episode: int) -> tuple[np.ndarray, np.ndarray]:
    """``(actions, states)``, each ``(frames, 14)`` float32, from a LeRobot v3 dataset."""
    import pyarrow.parquet as pq

    root = Path(root).expanduser()
    files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no LeRobot v3 data files under {root / 'data'}")
    columns = ["episode_index", "frame_index", "action", "observation.state"]
    rows = []
    for path in files:
        table = pq.read_table(path, columns=columns).to_pydict()
        rows += [
            (frame, action, state)
            for index, frame, action, state in zip(
                table["episode_index"], table["frame_index"], table["action"], table["observation.state"]
            )
            if index == episode
        ]
    if not rows:
        raise ValueError(f"episode {episode} not found in {root}")
    rows.sort(key=lambda row: row[0])
    actions = np.asarray([row[1] for row in rows], dtype=np.float32)
    states = np.asarray([row[2] for row in rows], dtype=np.float32)
    return actions, states


def episode_sidecar(root: str | Path, episode: int) -> dict[str, Any]:
    """The pipeline's ``episode_metadata`` sidecar (layout seed etc.), or ``{}``."""
    path = Path(root).expanduser() / "episode_metadata" / f"episode_{episode:06d}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


class ReplayPolicy:
    """Whole-policy runtime (``vla_edge.backends.base.PolicyRuntime``)."""

    name = "replay"

    def __init__(self, root: str | Path, episode: int, action_horizon: int) -> None:
        self.root = Path(root).expanduser().resolve()
        self.episode = int(episode)
        self.action_horizon = int(action_horizon)
        self.actions, self.states = load_episode(self.root, self.episode)
        self.sidecar = episode_sidecar(self.root, self.episode)
        self._seed: int | None = None
        self._cursor = 0
        log.info("replaying %s episode %d: %d frames, layout seed %s",
                 self.root, self.episode, len(self.actions), self.layout_seed)

    @property
    def layout_seed(self) -> int | None:
        seed = self.sidecar.get("layout", {}).get("seed")
        return None if seed is None else int(seed)

    def generate_actions(self, *, images: list[Any], instruction: str, state: Any, num_steps: int,
                         enable_cuda_graph: bool = False, seed: int | None = None) -> np.ndarray:
        if seed is not None and seed != self._seed:
            self._seed, self._cursor = seed, 0
        frames = len(self.actions)
        recorded = self.states[min(self._cursor, frames - 1)]
        drift = np.abs(np.asarray(state, dtype=np.float32).reshape(-1) - recorded)
        log.info("frame %d/%d: state drift max %.2f (index %d), mean %.2f",
                 self._cursor, frames, float(drift.max()), int(drift.argmax()), float(drift.mean()))
        index = np.minimum(np.arange(self._cursor, self._cursor + self.action_horizon), frames - 1)
        self._cursor += self.action_horizon
        return self.actions[index]

    def warmup(self) -> None:
        pass

    def serving_metadata(self) -> dict[str, Any]:
        return {"dataset": str(self.root), "episode": self.episode, "frames": len(self.actions),
                "layout_seed": self.layout_seed}


def load_backend(*, backend: str, policy: Any, embodiment: Any, checkpoint: str | None = None,
                 **_: Any) -> ReplayPolicy:
    """vla-edge family loader. ``--checkpoint`` (or ``$REBOT_REPLAY_DATASET``) is the dataset root."""
    if backend != "replay":
        raise ValueError(f"policy {policy.name!r} runs only with --backend replay, got {backend!r}")
    root = checkpoint or os.environ.get(DATASET_ENV)
    if not root:
        raise ValueError(f"replay needs the dataset root: --checkpoint DIR or ${DATASET_ENV}")
    return ReplayPolicy(root, int(os.environ.get(EPISODE_ENV, "0")), policy.action_horizon)
