"""Load a YAML run config into plain, typed Python objects."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from src.fleet.simulator import SimConfig


@dataclass
class RunConfig:
    """One baseline run. Mirrors ``configs/single_robot.yaml``."""

    task: str = "lift"
    robot: str = "Panda"
    seed: int = 0
    horizon: int = 200
    control_freq: int = 20
    n_episodes: int = 10
    policy: str = "random"
    run_dir: str = "results/run"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
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
        )
