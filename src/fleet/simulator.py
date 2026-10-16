"""Single-robot robosuite environment with reproducible, seeded episodes.

One ``RobotSim`` is one robot. In M2/M3 a fleet is just N of these.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import robosuite as suite

# Our short task names -> robosuite environment names.
TASKS = {
    "lift": "Lift",
    "stack": "Stack",
    "door": "Door",
    "nut_assembly": "NutAssemblySquare",
}


@dataclass
class SimConfig:
    task: str = "lift"
    robot: str = "Panda"
    seed: int = 0
    horizon: int = 500            # max steps per episode
    control_freq: int = 20        # policy steps per simulated second
    use_camera_obs: bool = False  # True when SmolVLA needs images (Day 6+)
    camera_names: tuple[str, ...] = ("agentview", "robot0_eye_in_hand")
    camera_size: int = 128


class RobotSim:
    """Thin wrapper over ``robosuite.make`` with per-episode seeding."""

    def __init__(self, cfg: SimConfig):
        if cfg.task not in TASKS:
            raise ValueError(f"unknown task {cfg.task!r}; choose from {sorted(TASKS)}")
        self.cfg = cfg
        self.env = suite.make(
            TASKS[cfg.task],
            robots=cfg.robot,
            has_renderer=False,
            has_offscreen_renderer=cfg.use_camera_obs,
            use_camera_obs=cfg.use_camera_obs,
            camera_names=list(cfg.camera_names),
            camera_heights=cfg.camera_size,
            camera_widths=cfg.camera_size,
            horizon=cfg.horizon,
            control_freq=cfg.control_freq,
            reward_shaping=True,
            seed=cfg.seed,
        )

    @property
    def action_dim(self) -> int:
        return self.env.action_dim

    def episode_seed(self, episode: int) -> int:
        """Seed for one episode; distinct per (run seed, episode)."""
        return self.cfg.seed * 1_000_000 + episode

    def reset(self, episode: int) -> dict:
        # robosuite's placement sampler holds a reference to env.rng,
        # so reseed that same Generator in place instead of replacing it.
        fresh = np.random.default_rng(self.episode_seed(episode))
        self.env.rng.bit_generator.state = fresh.bit_generator.state
        return self.env.reset()

    def step(self, action: np.ndarray):
        return self.env.step(action)

    def is_success(self) -> bool:
        return bool(self.env._check_success())

    def close(self) -> None:
        self.env.close()
