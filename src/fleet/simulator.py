"""MuJoCo + robosuite wrapper for the simulated robot fleet.

Wraps ``robosuite.SingleArmEnv`` with:
- Deterministic reset (seeded)
- Per-robot task/object-distribution assignment (see :mod:`.heterogeneity`)
- Optional gripper / arm morphology override for the M2 stretch condition

Usage
-----
>>> sim = FleetSimulator(config="configs/fleet_8.yaml", seed=42)
>>> for step in sim.rollout(policy, n_steps=1000):
...     ...  # collect trajectory
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FleetSimulator:
    """One entry point per robot; N instances form a fleet."""

    config_path: str
    seed: int = 42

    def rollout(self, policy, n_steps: int):
        """Run policy for ``n_steps`` steps in the simulator; yield transitions."""
        raise NotImplementedError("M1 — implement single-robot rollout first")
