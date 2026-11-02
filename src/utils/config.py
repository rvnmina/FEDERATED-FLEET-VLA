"""Load a YAML run config into plain, typed Python objects."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from src.fleet.simulator import SimConfig


@dataclass
class RunConfig:
    """One baseline run. Mirrors configs/*.yaml."""

    task: str = "lift"
    robot: str = "Panda"
    seed: int = 0
    horizon: int = 200
    control_freq: int = 20
    n_episodes: int = 10
    policy: str = "random"
    run_dir: str = "results/run"
    # ── Day 7b: camera & policy fields ──
    cameras: tuple = ()           # empty = no camera obs (random baseline)
    camera_size: int = 128        # square image resolution
    checkpoint: str = ""          # HF hub id or local path; empty = random
    device: str = "cpu"           # "cpu", "mps", or "cuda"
    task_prompt: str = ""         # language instruction for VLA
    n_action_steps: int = 10      # how many actions per forward pass

    @property
    def use_camera_obs(self) -> bool:
        """True when at least one camera is configured."""
        return len(self.cameras) > 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        # YAML gives a list for cameras; dataclass wants a tuple
        if "cameras" in data:
            data["cameras"] = tuple(data["cameras"])
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def sim_config(self) -> SimConfig:
        """The subset of settings the simulator needs."""
        return SimConfig(
            task=self.task,
            robot=self.robot,
            seed=self.seed,
            horizon=self.horizon,
            control_freq=self.control_freq,
            use_camera_obs=self.use_camera_obs,
            camera_names=self.cameras or ("agentview",),
            camera_size=self.camera_size,
        )
