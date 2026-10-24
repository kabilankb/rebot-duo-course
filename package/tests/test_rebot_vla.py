"""Embodiment registration, wire conversions and the replay server (no simulator, no GPU)."""

from __future__ import annotations

import json

import json_numpy
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient
from rebot_sim_pipeline import conventions
from vla_edge.config import get_policy_embodiment, resolve_policy
from vla_edge.pipeline import Pipeline
from vla_edge.serving.server import build_app

from rebot_vla import embodiment, replay, wire
from rebot_vla.embodiment import EMBODIMENT, REPLAY_POLICY, register

FRAMES = 70  # more than two chunks of 30


@pytest.fixture(scope="module", autouse=True)
def registered():
    register()
    register()  # idempotent


@pytest.fixture()
def dataset(tmp_path):
    """A two-episode LeRobot v3 data file plus the pipeline's sidecars."""
    rows = {"episode_index": [], "frame_index": [], "action": [], "observation.state": []}
    for episode in range(2):
        for frame in range(FRAMES):
            rows["episode_index"].append(episode)
            rows["frame_index"].append(frame)
            rows["action"].append([episode * 1000.0 + frame + k / 100 for k in range(14)])
            rows["observation.state"].append([float(frame)] * 14)
    path = tmp_path / "data" / "chunk-000" / "file-000.parquet"
    path.parent.mkdir(parents=True)
    # Store frames out of order: the loader must sort them.
    table = pa.table(rows).take(pa.array(np.random.default_rng(0).permutation(2 * FRAMES)))
    pq.write_table(table, path)
    sidecars = tmp_path / "episode_metadata"
    sidecars.mkdir()
    for episode in range(2):
        (sidecars / f"episode_{episode:06d}.json").write_text(json.dumps({"layout": {"seed": 40 + episode}}))
    return tmp_path


def test_policy_resolves_to_rebot_embodiment():
    policy = resolve_policy(policy="replay-bimanual-rebot")
    assert policy is REPLAY_POLICY
    assert get_policy_embodiment(policy) == EMBODIMENT
    assert EMBODIMENT.camera_names == ("top_cam", "left_cam", "right_cam")
    assert (EMBODIMENT.state_dim, EMBODIMENT.action_dim) == (14, 14)
    assert len(conventions.feature_names(["left", "right"])) == EMBODIMENT.state_dim


def test_smolvla_policy_resolves_to_rebot_embodiment():
    policy = resolve_policy(policy=embodiment.SMOLVLA_POLICY_NAME)
    assert policy is embodiment.SMOLVLA_POLICY
    assert get_policy_embodiment(policy) == EMBODIMENT
    assert policy.action_horizon == 50


def test_action_row_round_trip():
    left_q = np.radians([10.0, -30.0, -60.0, 20.0, -15.0, 90.0])
    right_q = np.radians([-5.0, -45.0, -45.0, 0.0, 30.0, -20.0])
    row = np.concatenate([conventions.arm_action(left_q, 0.05), conventions.arm_action(right_q, 0.01)])
    command = wire.command_from_action(row)
    np.testing.assert_allclose(command[:6], left_q, atol=1e-5)
    np.testing.assert_allclose(command[7:13], right_q, atol=1e-5)
    assert command[6] == pytest.approx(0.05) and command[13] == pytest.approx(0.01)
    with pytest.raises(ValueError):
        wire.command_from_action(row[:7])


def test_state_matches_dataset_convention():
    q = {"left_arm": np.radians([10, 20, 30, 40, 50, 60]), "right_arm": np.zeros(6)}
    state = wire.state_from_sim(q, {"left_arm": 0.025, "right_arm": 0.05})
    np.testing.assert_allclose(state[:7], [-10, -20, -30, -40, -50, -60, 135], atol=1e-4)
    np.testing.assert_allclose(state[7:], [0, 0, 0, 0, 0, 0, 270], atol=1e-4)


def test_cameras_are_renamed_and_rgb_uint8():
    images = {name: np.full((4, 6, 4), index, dtype=np.uint8)
              for index, name in enumerate(["top", "left_wrist", "right_wrist"])}
    cameras = wire.cameras_from_sim(images)
    assert list(cameras) == list(EMBODIMENT.camera_names)
    assert cameras["left_cam"].shape == (4, 6, 3) and cameras["left_cam"].dtype == np.uint8
    assert int(cameras["right_cam"][0, 0, 0]) == 2


def test_load_episode_sorts_frames(dataset):
    actions, states = replay.load_episode(dataset, 1)
    assert actions.shape == (FRAMES, 14) and states.shape == (FRAMES, 14)
    np.testing.assert_allclose(actions[:, 0], 1000.0 + np.arange(FRAMES))
    with pytest.raises(ValueError):
        replay.load_episode(dataset, 5)


def test_loader_requires_replay_backend(dataset):
    with pytest.raises(ValueError, match="--backend replay"):
        Pipeline.load(REPLAY_POLICY, backend="torch", checkpoint=str(dataset))


def _request(seed: int | None) -> str:
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    payload = {name: image for name in EMBODIMENT.camera_names}
    payload.update(instruction="put the pens in the cup", state=np.zeros(14, dtype=np.float32))
    if seed is not None:
        payload["seed"] = seed
    return json_numpy.dumps(payload)


def test_server_replays_episode_in_chunks(dataset, monkeypatch):
    monkeypatch.setenv(replay.EPISODE_ENV, "1")
    pipeline = Pipeline.load(REPLAY_POLICY, backend="replay", checkpoint=str(dataset))
    client = TestClient(build_app(pipeline, "replay"))

    health = client.get("/act").json()
    assert health["embodiment"] == embodiment.EMBODIMENT_NAME
    assert health["cameras"] == ["top_cam", "left_cam", "right_cam"]
    assert health["action_horizon"] == 30
    assert health["runtime"]["layout_seed"] == 41 and health["runtime"]["frames"] == FRAMES

    def chunk(seed: int) -> np.ndarray:
        response = client.post("/act", content=_request(seed))
        assert response.status_code == 200, response.text
        return np.asarray(json_numpy.loads(response.text)["actions"])

    first, second, third = chunk(7), chunk(7), chunk(7)
    assert first.shape == (30, 14)
    np.testing.assert_allclose(first[:, 0], 1000.0 + np.arange(30))
    np.testing.assert_allclose(second[:, 0], 1000.0 + np.arange(30, 60))
    # Past the end the last recorded row repeats.
    np.testing.assert_allclose(third[:, 0], 1000.0 + np.minimum(np.arange(60, 90), FRAMES - 1))
    # A new episode seed restarts the replay.
    np.testing.assert_allclose(chunk(8)[:, 0], 1000.0 + np.arange(30))


def test_server_replay_advances_without_seed(dataset):
    pipeline = Pipeline.load(REPLAY_POLICY, backend="replay", checkpoint=str(dataset))
    client = TestClient(build_app(pipeline, "replay"))

    def chunk() -> np.ndarray:
        response = client.post("/act", content=_request(None))
        assert response.status_code == 200, response.text
        return np.asarray(json_numpy.loads(response.text)["actions"])

    np.testing.assert_allclose(chunk()[:, 0], np.arange(30))
    np.testing.assert_allclose(chunk()[:, 0], np.arange(30, 60))


def test_server_rejects_wrong_cameras(dataset):
    pipeline = Pipeline.load(REPLAY_POLICY, backend="replay", checkpoint=str(dataset))
    client = TestClient(build_app(pipeline, "replay"))
    payload = json_numpy.loads(_request(1))
    payload["wrist_cam"] = payload.pop("left_cam")
    response = client.post("/act", content=json_numpy.dumps(payload))
    assert response.status_code == 400
