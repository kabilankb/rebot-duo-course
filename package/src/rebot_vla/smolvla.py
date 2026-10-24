"""A vla-edge policy runtime for a LeRobot SmolVLA checkpoint trained on reBot data.

The checkpoint is a ``lerobot-train`` ``pretrained_model`` directory. Its
saved pre/post-processors do the dataset-specific work (camera renaming,
tokenization, normalization), so this runtime only turns the wire inputs back
into the dataset's observation keys: ``observation.images.<top|left_wrist|
right_wrist>`` and ``observation.state``, both in the reBot LeRobot
convention that the wire already uses. Needs ``lerobot[smolvla]`` installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .embodiment import CAMERA_SOURCES

log = logging.getLogger("rebot_vla.smolvla")


class SmolVLAPolicy:
    """Whole-policy runtime (``vla_edge.backends.base.PolicyRuntime``)."""

    name = "smolvla"

    def __init__(self, checkpoint: str | Path, action_horizon: int, device: str = "cuda") -> None:
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as LeRobotSmolVLA

        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.device = torch.device(device)
        self.policy = LeRobotSmolVLA.from_pretrained(str(self.checkpoint)).to(self.device).eval()
        overrides = {"device_processor": {"device": str(self.device)}}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config, pretrained_path=str(self.checkpoint), preprocessor_overrides=overrides)
        chunk = int(self.policy.config.n_action_steps)
        if chunk != action_horizon:
            raise ValueError(f"checkpoint predicts {chunk}-step chunks, policy spec says {action_horizon}")
        self.action_horizon = chunk
        self._seed: int | None = None
        self._generator = torch.Generator(device=self.device)
        log.info("SmolVLA %s on %s: chunk %d, %d denoising steps",
                 self.checkpoint, self.device, chunk, self.policy.config.num_steps)

    def generate_actions(self, *, images: list[Any], instruction: str, state: Any, num_steps: int,
                         enable_cuda_graph: bool = False, seed: int | None = None) -> np.ndarray:
        import torch

        if seed is not None and seed != self._seed:
            self._seed = seed
            self._generator.manual_seed(seed)
            self.policy.reset()
        self.policy.config.num_steps = int(num_steps)
        observation = {"observation.state": torch.as_tensor(np.asarray(state, dtype=np.float32).reshape(1, -1))}
        for source, image in zip(CAMERA_SOURCES.values(), images):
            pixels = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.uint8))
            observation[f"observation.images.{source}"] = pixels.permute(2, 0, 1)[None].float() / 255
        observation["task"] = [instruction]
        config = self.policy.config
        noise = torch.randn((1, config.chunk_size, config.max_action_dim), generator=self._generator,
                            device=self.device)
        with torch.inference_mode():
            batch = self.preprocessor(observation)
            actions = self.postprocessor(self.policy.predict_action_chunk(batch, noise=noise))
        return actions[0, : self.action_horizon].float().cpu().numpy()

    def warmup(self) -> None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        from PIL import Image

        self.generate_actions(images=[Image.fromarray(blank)] * len(CAMERA_SOURCES), instruction="warmup",
                              state=np.zeros(14, dtype=np.float32), num_steps=self.policy.config.num_steps)
        self.policy.reset()

    def serving_metadata(self) -> dict[str, Any]:
        return {"checkpoint": str(self.checkpoint), "device": str(self.device),
                "chunk_size": int(self.policy.config.chunk_size)}


def load_backend(*, backend: str, policy: Any, embodiment: Any, checkpoint: str | None = None,
                 **_: Any) -> SmolVLAPolicy:
    """vla-edge family loader. ``--checkpoint`` is the ``pretrained_model`` directory."""
    if backend not in ("torch", "lerobot"):
        raise ValueError(f"policy {policy.name!r} runs only with --backend torch, got {backend!r}")
    if not checkpoint:
        raise ValueError("SmolVLA needs --checkpoint .../checkpoints/<step>/pretrained_model")
    return SmolVLAPolicy(checkpoint, policy.action_horizon)
