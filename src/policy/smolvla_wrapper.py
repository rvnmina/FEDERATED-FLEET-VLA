"""SmolVLA policy wrapper: load, inspect, save.

Thin layer over LeRobot's ``SmolVLAPolicy`` that answers the three questions
this project asks of a policy:

- how big is the update a federated client would send (M3's currency),
- how long does one forward pass take on this machine,
- can the weights be saved and reloaded unchanged (FedAvg needs this).

The base checkpoint is downloaded from the Hugging Face Hub the first time
and cached under ``~/.cache/huggingface``; later loads work offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from src.utils.instrument import payload_bytes, trainable_keys

BASE_CHECKPOINT = "lerobot/smolvla_base"


@dataclass
class PolicyConfig:
    """Everything that decides which policy we get."""

    checkpoint: str = BASE_CHECKPOINT
    device: str = "cpu"  # Apple Silicon: "mps" is an option, but CPU is the baseline
    # SmolVLA's own defaults: fine-tune the action expert, leave the VLM frozen.
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True


class SmolVLAWrapper:
    """Owns one SmolVLA policy and reports its size and speed."""

    def __init__(self, cfg: PolicyConfig | None = None):
        self.cfg = cfg or PolicyConfig()
        self.policy = None

    # ------------------------------------------------------------ lifecycle

    def load(self) -> "SmolVLAWrapper":
        """Load the checkpoint. Downloads it on first use."""
        # Imported here so the rest of the module (and the tests) work even
        # when the heavy VLM dependencies are not installed.
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        policy = SmolVLAPolicy.from_pretrained(self.cfg.checkpoint)
        policy.config.freeze_vision_encoder = self.cfg.freeze_vision_encoder
        policy.config.train_expert_only = self.cfg.train_expert_only
        # The flags are applied when the submodule is built, so re-apply them
        # after loading in case we changed them.
        expert = policy.model.vlm_with_expert
        expert.freeze_vision_encoder = self.cfg.freeze_vision_encoder
        expert.train_expert_only = self.cfg.train_expert_only
        expert.set_requires_grad()

        self.policy = policy.to(self.cfg.device)
        return self

    def save(self, directory: str | Path) -> Path:
        """Write config + weights to ``directory``; reloadable with ``load``."""
        directory = Path(directory)
        self._require_loaded().save_pretrained(directory)
        return directory

    # ------------------------------------------------------------ accounting

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._require_loaded().state_dict()

    def trainable_keys(self) -> set[str]:
        return trainable_keys(self._require_loaded())

    def size_report(self) -> dict[str, float]:
        """Parameter counts and transmit sizes.

        ``trainable_bytes`` is what one federated client would actually send
        each round, which is the number M3's payload-vs-accuracy curves are
        plotted against.
        """
        policy = self._require_loaded()
        state = policy.state_dict()
        keys = trainable_keys(policy)
        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        full = payload_bytes(state)
        trainable = payload_bytes(state, only=keys)
        return {
            "total_params_m": total_params / 1e6,
            "trainable_params_m": trainable_params / 1e6,
            "full_payload_mb": full / 1e6,
            "trainable_payload_mb": trainable / 1e6,
            "trainable_fraction": (trainable / full) if full else 0.0,
        }

    # ------------------------------------------------------------ internals

    def _require_loaded(self):
        if self.policy is None:
            raise RuntimeError("call load() first")
        return self.policy
