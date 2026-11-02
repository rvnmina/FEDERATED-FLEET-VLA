"""SmolVLA policy wrapper: load, inspect, save, and run inference.

Thin layer over LeRobot's SmolVLAPolicy that handles:
- loading / saving checkpoints
- size accounting for federated payloads (M3)
- retargeting the checkpoint to a new robot (configure_for_env)
- running inference (select_action)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.utils.instrument import payload_bytes, trainable_keys

BASE_CHECKPOINT = "lerobot/smolvla_base"


@dataclass
class PolicyConfig:
    """Everything that decides which policy we get."""

    checkpoint: str = BASE_CHECKPOINT
    device: str = "cpu"
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True


class SmolVLAWrapper:
    """Owns one SmolVLA policy and reports its size and speed."""

    def __init__(self, cfg: PolicyConfig | None = None):
        self.cfg = cfg or PolicyConfig()
        self.policy = None
        self.cameras: tuple = ()  # set by configure_for_env

    # ── lifecycle ──────────────────────────────────────────────────

    def load(self) -> "SmolVLAWrapper":
        """Load the checkpoint. Downloads it on first use."""
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        policy = SmolVLAPolicy.from_pretrained(self.cfg.checkpoint)
        policy.config.freeze_vision_encoder = self.cfg.freeze_vision_encoder
        policy.config.train_expert_only = self.cfg.train_expert_only
        expert = policy.model.vlm_with_expert
        expert.freeze_vision_encoder = self.cfg.freeze_vision_encoder
        expert.train_expert_only = self.cfg.train_expert_only
        expert.set_requires_grad()

        self.policy = policy.to(self.cfg.device)
        return self

    def save(self, directory: str | Path) -> Path:
        """Write config + weights to directory."""
        directory = Path(directory)
        self._require_loaded().save_pretrained(directory)
        return directory

    # ── configure for a specific robot / environment ───────────────

    def configure_for_env(
        self,
        cameras: tuple,
        image_size: int,
        action_dim: int,
        n_action_steps: int | None = None,
    ) -> None:
        """
        Retarget the loaded checkpoint to our robosuite robot.

        SmolVLA was pretrained on ALOHA robots with different cameras
        and action dimensions.  This method overwrites the feature
        descriptors so the model accepts OUR observation/action shapes.
        """
        from src.policy.obs_adapter import build_features

        policy = self._require_loaded()
        self.cameras = cameras

        # Build new feature descriptors for our robot
        inp, out = build_features(cameras, image_size, action_dim)
        policy.config.input_features = inp
        policy.config.output_features = out

        # ALOHA-specific indexing — crashes on non-ALOHA robots
        policy.config.adapt_to_pi_aloha = False

        # n_action_steps = how many actions replayed per forward pass
        if n_action_steps is not None:
            policy.config.n_action_steps = n_action_steps

        # Reset clears the internal action queue (a deque)
        policy.reset()

    # ── inference ──────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear the action queue between episodes."""
        self._require_loaded().reset()

    @torch.no_grad()
    def select_action(self, obs: dict, task: str) -> np.ndarray:
        """
        Pick the next action given a robosuite observation.

        Steps:
        1. Convert robosuite obs → SmolVLA batch (via obs_adapter)
        2. Tokenize the task prompt
        3. Call the policy's select_action
        4. Return a numpy array the simulator can use
        """
        from src.policy.obs_adapter import observation_to_batch

        policy = self._require_loaded()

        # Build image + state tensors
        batch = observation_to_batch(obs, self.cameras)

        # Add language tokens
        tokens, mask = self._tokenize(task)
        batch["observation.language.tokens"] = tokens
        batch["observation.language.attention_mask"] = mask

        # Move everything to the right device
        batch = {k: v.to(self.cfg.device) for k, v in batch.items()}

        # Run the model — returns (1, action_dim) tensor
        action = policy.select_action(batch)

        # Convert to numpy for robosuite
        return action.squeeze(0).cpu().numpy()

    def _tokenize(self, task: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize a task prompt the way SmolVLA expects.

        The tokenizer lives deep inside the model. We add a newline
        (matching LeRobot's own processor) and pad to the model's
        expected length (48 tokens).
        """
        policy = self._require_loaded()
        tokenizer = policy.model.vlm_with_expert.processor.tokenizer
        max_len = policy.config.tokenizer_max_length  # 48

        encoded = tokenizer(
            task + "\n",            # LeRobot appends a newline
            padding="max_length",   # pad to max_len
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"]

    # ── accounting ─────────────────────────────────────────────────

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._require_loaded().state_dict()

    def trainable_keys(self) -> set[str]:
        return trainable_keys(self._require_loaded())

    def size_report(self) -> dict[str, float]:
        """Parameter counts and transmit sizes."""
        policy = self._require_loaded()
        state = policy.state_dict()
        keys = trainable_keys(policy)
        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params = sum(p.numel()
                               for p in policy.parameters() if p.requires_grad)
        full = payload_bytes(state)
        trainable = payload_bytes(state, only=keys)
        return {
            "total_params_m": total_params / 1e6,
            "trainable_params_m": trainable_params / 1e6,
            "full_payload_mb": full / 1e6,
            "trainable_payload_mb": trainable / 1e6,
            "trainable_fraction": (trainable / full) if full else 0.0,
        }

    # ── internals ──────────────────────────────────────────────────

    def _require_loaded(self):
        if self.policy is None:
            raise RuntimeError("call load() first")
        return self.policy
