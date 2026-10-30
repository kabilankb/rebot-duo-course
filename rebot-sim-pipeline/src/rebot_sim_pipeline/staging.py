"""Staged episodes: what the simulator writes and the LeRobot writer reads.

Same layout as the single-arm reBot recorder, generalized to any cameras and
vector sizes, so the simulator never imports LeRobot::

    <staging>/episode_<timestamp>_<attempt>/
        images/<camera>/frame_000000.jpg
        actions.npy        (frames, D) float32, recorded convention
        states.npy         (frames, D) float32, recorded convention
        raw_actions.npy    (frames, D) float32, simulator units (rad, m)
        raw_states.npy     (frames, D) float32, simulator units
        metadata.json      task, fps, resolution, cameras, feature names, success, layout ...

An episode directory is complete once ``metadata.json`` exists (written last).
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class EpisodeStager:
    def __init__(self, root: Path, cameras: Sequence[str], jpeg_quality: int = 92) -> None:
        self.root = Path(root)
        self.cameras = list(cameras)
        self.jpeg_quality = int(jpeg_quality)
        self._dir: Path | None = None
        self._rows: dict[str, list[np.ndarray]] = {}
        self.frames = 0

    @property
    def active(self) -> bool:
        return self._dir is not None

    def start(self, attempt: int) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._dir = self.root / f"episode_{time.strftime('%Y%m%d_%H%M%S')}_{attempt:04d}"
        for camera in self.cameras:
            (self._dir / "images" / camera).mkdir(parents=True, exist_ok=True)
        self._rows = {key: [] for key in ("actions", "states", "raw_actions", "raw_states")}
        self.frames = 0

    def add(self, images: Mapping[str, np.ndarray], *, action: np.ndarray, state: np.ndarray,
            raw_action: np.ndarray, raw_state: np.ndarray) -> None:
        assert self._dir is not None, "start() first"
        for camera in self.cameras:
            Image.fromarray(np.ascontiguousarray(images[camera])).save(
                self._dir / "images" / camera / f"frame_{self.frames:06d}.jpg", quality=self.jpeg_quality)
        for key, value in (("actions", action), ("states", state), ("raw_actions", raw_action),
                           ("raw_states", raw_state)):
            self._rows[key].append(np.asarray(value, dtype=np.float32))
        self.frames += 1

    def finish(self, metadata: Mapping[str, Any]) -> Path:
        """Write the arrays and metadata; return the episode directory."""
        assert self._dir is not None, "start() first"
        for key, rows in self._rows.items():
            np.save(self._dir / f"{key}.npy", np.stack(rows) if rows else np.zeros((0, 0), np.float32))
        meta = dict(metadata)
        meta.update(frames=self.frames, cameras=self.cameras)
        (self._dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        path, self._dir = self._dir, None
        return path

    def discard(self) -> None:
        if self._dir is not None:
            shutil.rmtree(self._dir, ignore_errors=True)
        self._dir = None


def staged_episodes(root: Path) -> list[Path]:
    """Complete staged episodes under ``root``, oldest first."""
    return sorted(path.parent for path in Path(root).glob("episode_*/metadata.json"))
