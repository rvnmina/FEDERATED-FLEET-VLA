
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import torch

STATE_KEYS: Tuple[str, ...] = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)
STATE_DIM: int = 9

TASK_PROMPTS: Dict[str, str] = {
    "lift": "Pick up the red cube from the table.",
    "stack": "Stack the red cube on top of the green cube.",
    "door": "Open the door by pushing the handle.",
    "nut_assembly": "Place the nut onto the peg.",
}


def image_key(camera: str) -> str:
    return f"{camera}_image"


def feature_key(camera: str) -> str:
    return f"observation.images.{camera}"


def extract_state(obs: dict) -> np.ndarray:
    parts = []
    for key in STATE_KEYS:
        if key not in obs:
            raise KeyError(
                f"observation missing '{key}' — available: {sorted(obs.keys())}"
            )
        parts.append(obs[key])
    return np.concatenate(parts).astype(np.float32)


def extract_image(obs: dict, camera: str) -> np.ndarray:
    key = image_key(camera)
    if key not in obs:
        raise KeyError(
            f"no image for camera '{camera}' — expected key '{key}' in observation"
        )
    raw = np.asarray(obs[key])
    flipped = raw[::-1]
    return np.ascontiguousarray(flipped)


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(image).to(torch.float32)
    t = t / 255.0
    t = t.permute(2, 0, 1)
    return t


def observation_to_batch(
    obs: dict,
    cameras: Tuple[str, ...],
) -> Dict[str, torch.Tensor]:
    batch: Dict[str, torch.Tensor] = {}
    for cam in cameras:
        img = extract_image(obs, cam)
        tensor = image_to_tensor(img)
        batch[feature_key(cam)] = tensor.unsqueeze(0)
    state = extract_state(obs)
    batch["observation.state"] = torch.from_numpy(state).unsqueeze(0)
    return batch


def build_features(
    cameras: Tuple[str, ...],
    image_size: int,
    action_dim: int,
) -> Tuple[dict, dict]:
    input_features = {}
    for cam in cameras:
        input_features[feature_key(cam)] = {
            "type": "VISUAL",
            "shape": (3, image_size, image_size),
        }
    input_features["observation.state"] = {
        "type": "STATE",
        "shape": (STATE_DIM,),
    }
    output_features = {
        "action": {
            "type": "ACTION",
            "shape": (action_dim,),
        }
    }
    return input_features, output_features
