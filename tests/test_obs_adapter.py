"""
test_obs_adapter.py  –  7 tests for the observation adapter (Day 7a).

Run:  pytest -q tests/test_obs_adapter.py
Expected: 7 passed
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.policy.obs_adapter import (
    STATE_DIM,
    TASK_PROMPTS,
    extract_image,
    extract_state,
    feature_key,
    image_key,
    image_to_tensor,
    observation_to_batch,
)

from src.fleet.simulator import TASKS


def _fake_obs(h=64, w=64, camera="agentview"):
    return {
        image_key(camera): np.random.randint(0, 256, (h, w, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.zeros(4),
        "robot0_gripper_qpos": np.zeros(2),
    }


def test_state_shape_and_dtype():
    obs = _fake_obs()
    state = extract_state(obs)
    assert state.shape == (STATE_DIM,)
    assert state.dtype == np.float32


def test_missing_key_raises():
    obs = _fake_obs()
    del obs["robot0_eef_pos"]
    with pytest.raises(KeyError, match="robot0_eef_pos"):
        extract_state(obs)


def test_image_flip():
    obs = _fake_obs()
    raw = obs[image_key("agentview")]
    flipped = extract_image(obs, "agentview")
    np.testing.assert_array_equal(flipped[0], raw[-1])


def test_tensor_chw_and_range():
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    t = image_to_tensor(img)
    assert t.shape == (3, 64, 64)
    assert t.dtype == torch.float32
    assert t.min() >= 0.0 and t.max() <= 1.0


def test_batch_keys():
    obs = _fake_obs(camera="agentview")
    batch = observation_to_batch(obs, cameras=("agentview",))
    assert feature_key("agentview") in batch
    assert "observation.state" in batch


def test_unknown_camera_raises():
    obs = _fake_obs(camera="agentview")
    with pytest.raises(KeyError, match="no_such_cam"):
        extract_image(obs, "no_such_cam")


def test_every_task_has_prompt():
    for task_name in TASKS:
        assert task_name in TASK_PROMPTS
