"""Fixed-seed helper so every run is reproducible from its config."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch.

    ``deterministic=True`` also asks PyTorch for deterministic kernels
    (warn-only, so an op without a deterministic variant does not crash a run).
    Note: ``PYTHONHASHSEED`` only takes effect for child processes.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
