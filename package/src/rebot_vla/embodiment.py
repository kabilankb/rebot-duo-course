"""The bimanual reBot contract, registered with vla-edge.

The wire format is the reBot LeRobot dataset format of ``rebot-sim-pipeline``
(``rebot_sim_pipeline.conventions``), so a policy trained on those datasets is
served without any conversion:

* ``state`` (14): ``left_*.pos`` then ``right_*.pos``, follower convention:
  ``-degrees`` per arm joint, gripper 0 (closed) to 270 (open), measured.
* action rows (14): the same names, leader convention: degrees divided by
  ``[-1, -1, 1, 1, 1, -1]``, gripper 0 (closed) to 45 (open), commanded.
* cameras, in training order: ``top_cam``, ``left_cam``, ``right_cam``
  (the dataset's ``top``, ``left_wrist``, ``right_wrist``), 480x640 RGB.

The grippers are not normalized to 0..1, so no vla-edge gripper channels are
declared; a model-family host that needs them must convert itself.
"""

from __future__ import annotations

from vla_edge.backends.families import FAMILIES, register_model_family
from vla_edge.config import EMBODIMENTS, POLICIES, Embodiment, PolicySpec

EMBODIMENT_NAME = "bimanual-rebot"
REPLAY_POLICY_NAME = "replay-bimanual-rebot"
REPLAY_FAMILY = "rebot-replay"
SMOLVLA_POLICY_NAME = "smolvla-bimanual-rebot"
SMOLVLA_FAMILY = "rebot-smolvla"

#: Wire camera name -> camera name in the simulator and the LeRobot dataset.
CAMERA_SOURCES = {"top_cam": "top", "left_cam": "left_wrist", "right_cam": "right_wrist"}

EMBODIMENT = Embodiment(
    name=EMBODIMENT_NAME,
    state_dim=14,
    action_dim=14,
    camera_names=tuple(CAMERA_SOURCES),
    description="Bimanual reBot B601 (Seeed), 3 cameras, absolute joint control in the reBot LeRobot convention.",
)

REPLAY_POLICY = PolicySpec(
    name=REPLAY_POLICY_NAME,
    model_family=REPLAY_FAMILY,
    embodiment=EMBODIMENT_NAME,
    repo_id="local/rebot_bimanual_stationery",
    norm_tag="none",
    action_horizon=30,
    default_num_steps=1,
    gripper_state="measured",
    description="Open-loop replay of one recorded episode; tests the serving and robot paths, not a model.",
)

SMOLVLA_POLICY = PolicySpec(
    name=SMOLVLA_POLICY_NAME,
    model_family=SMOLVLA_FAMILY,
    embodiment=EMBODIMENT_NAME,
    repo_id="kabilanKB/rebot_bimanual_stationery",
    norm_tag="none",
    action_horizon=50,
    default_num_steps=10,
    gripper_state="measured",
    description="LeRobot SmolVLA fine-tuned on reBot sim demonstrations; --checkpoint is its pretrained_model dir.",
)


def register() -> None:
    """Add the embodiment, the reBot policies and their families to vla-edge (idempotent)."""
    EMBODIMENTS.setdefault(EMBODIMENT.name, EMBODIMENT)
    POLICIES.setdefault(REPLAY_POLICY.name, REPLAY_POLICY)
    POLICIES.setdefault(SMOLVLA_POLICY.name, SMOLVLA_POLICY)
    if REPLAY_FAMILY not in FAMILIES.available():
        register_model_family(REPLAY_FAMILY, "rebot_vla.replay:load_backend")
    if SMOLVLA_FAMILY not in FAMILIES.available():
        register_model_family(SMOLVLA_FAMILY, "rebot_vla.smolvla:load_backend")
