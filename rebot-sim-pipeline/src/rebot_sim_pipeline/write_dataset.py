"""Append staged bimanual episodes to a LeRobot v3 dataset.

Run with the LeRobot venv's Python (not Isaac Sim's)::

    python -m rebot_sim_pipeline.write_dataset --staging <dir> --dataset-root <dir> --repo-id local/rebot_bimanual

Each complete staged episode (rebot_sim_pipeline.staging) becomes one dataset
episode; its metadata is kept as ``episode_metadata/episode_NNNNNN.json`` and
the staged directory is deleted. Features follow the episode metadata: 14-D
``observation.state`` / ``action`` (``left_*.pos`` then ``right_*.pos``) and one
video per camera (``observation.images.<lerobot key>``).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from rebot_sim_pipeline.staging import staged_episodes

try:  # LeRobot >= 0.5 replaced the vcodec argument with an encoder config
    from lerobot.configs.video import RGBEncoderConfig
except ImportError:  # pragma: no cover - older LeRobot releases
    RGBEncoderConfig = None


def _video_kwargs() -> dict:
    if RGBEncoderConfig is None:
        return {"vcodec": "h264"}
    return {"rgb_encoder": RGBEncoderConfig(vcodec="h264")}


def features(meta: dict) -> dict:
    width, height = (int(v) for v in meta["resolution"])
    names = list(meta["feature_names"])
    result = {
        "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
        "action": {"dtype": "float32", "shape": (len(names),), "names": names},
    }
    for camera in meta["cameras"]:
        result[f"observation.images.{meta['camera_keys'][camera]}"] = {
            "dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channels"]}
    return result


def open_dataset(root: Path, repo_id: str, meta: dict) -> LeRobotDataset:
    if (root / "meta" / "info.json").is_file():
        # Newer LeRobot opens existing datasets read-only; resume() appends.
        dataset = getattr(LeRobotDataset, "resume", LeRobotDataset)(
            repo_id=repo_id, root=root, batch_encoding_size=1, **_video_kwargs())
        if int(dataset.fps) != int(meta["fps"]):
            raise ValueError(f"dataset fps {dataset.fps} != episode fps {meta['fps']}")
        if dict(dataset.features).keys() < features(meta).keys():
            raise ValueError(f"dataset at {root} has different features; use a new --dataset-root")
        return dataset
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"not a LeRobot dataset, refusing to write into it: {root}")
    return LeRobotDataset.create(
        repo_id=repo_id, fps=int(meta["fps"]), features=features(meta), root=root,
        robot_type=meta["robot_type"], use_videos=True, batch_encoding_size=1, **_video_kwargs())


def write_episode(dataset: LeRobotDataset, episode: Path, root: Path) -> int:
    meta = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
    actions = np.load(episode / "actions.npy")
    states = np.load(episode / "states.npy")
    frames = int(meta["frames"])
    if actions.shape[0] != frames or states.shape[0] != frames:
        raise ValueError(f"{episode}: {frames} frames but actions {actions.shape}, states {states.shape}")
    index = int(dataset.meta.total_episodes)
    for frame in range(frames):
        row = {"action": actions[frame], "observation.state": states[frame], "task": meta["task"]}
        for camera in meta["cameras"]:
            with Image.open(episode / "images" / camera / f"frame_{frame:06d}.jpg") as image:
                row[f"observation.images.{meta['camera_keys'][camera]}"] = np.asarray(image.convert("RGB"))
        dataset.add_frame(row)
    dataset.save_episode(parallel_encoding=False)
    sidecars = root / "episode_metadata"
    sidecars.mkdir(parents=True, exist_ok=True)
    meta["episode_index"] = index
    (sidecars / f"episode_{index:06d}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/rebot_bimanual_stationery")
    parser.add_argument("--keep-staging", action="store_true", help="Do not delete written staged episodes.")
    args = parser.parse_args()

    episodes = staged_episodes(args.staging)
    if not episodes:
        print(f"[dataset-writer] no staged episodes in {args.staging}")
        return 0
    root = args.dataset_root.resolve()
    first = json.loads((episodes[0] / "metadata.json").read_text(encoding="utf-8"))
    dataset = open_dataset(root, args.repo_id, first)
    try:
        for episode in episodes:
            index = write_episode(dataset, episode, root)
            print(f"[dataset-writer] episode={index} from {episode.name}")
            if not args.keep_staging:
                shutil.rmtree(episode)
    finally:
        dataset.finalize()
    print(f"[dataset-writer] total episodes={dataset.meta.total_episodes} frames={dataset.meta.total_frames} "
          f"root={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
