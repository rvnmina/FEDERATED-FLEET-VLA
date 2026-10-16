"""M1 driver: run N episodes on one task and report the success rate.

The policy here is a placeholder (random or zero actions) so the whole
measurement path -- seeding, rollout, success check, logging -- is working
and instrumented before SmolVLA arrives on Day 6. Its success rate is the
floor that a trained policy has to beat.

Example
-------
    python -m scripts.run_baseline --config configs/single_robot.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.fleet.simulator import RobotSim
from src.utils.config import RunConfig
from src.utils.instrument import RunLogger
from src.utils.seed import seed_everything


def act(policy: str, rng: np.random.Generator, action_dim: int) -> np.ndarray:
    if policy == "random":
        return rng.uniform(-1.0, 1.0, size=action_dim)
    if policy == "zero":
        return np.zeros(action_dim)
    raise ValueError(f"unknown policy {policy!r}; choose from ['random', 'zero']")


def run_episode(sim: RobotSim, episode: int, policy: str, log: RunLogger) -> bool:
    """One episode. Returns whether the task was solved."""
    # Separate stream from the environment's, so actions and scene layout
    # do not shift each other when either changes.
    rng = np.random.default_rng([sim.cfg.seed, episode])
    sim.reset(episode=episode)

    success = False
    steps = 0
    for step in range(sim.cfg.horizon):
        with log.step(phase="rollout", episode=episode, step=step):
            _, _, done, _ = sim.step(act(policy, rng, sim.action_dim))
        steps = step + 1
        if sim.is_success():
            success = True
            break
        if done:
            break

    log.log(phase="episode", episode=episode, success=success, steps=steps)
    return success


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--n-episodes", type=int, help="override config")
    parser.add_argument("--seed", type=int, help="override config")
    args = parser.parse_args()

    cfg = RunConfig.from_yaml(args.config)
    if args.n_episodes is not None:
        cfg.n_episodes = args.n_episodes
    if args.seed is not None:
        cfg.seed = args.seed

    seed_everything(cfg.seed)
    sim = RobotSim(cfg.sim_config())

    with RunLogger(cfg.run_dir, seed=cfg.seed, extra=vars(cfg)) as log:
        successes = [
            run_episode(sim, ep, cfg.policy, log) for ep in range(cfg.n_episodes)
        ]
        rate = float(np.mean(successes))
        log.log(phase="summary", success_rate=rate, n_episodes=cfg.n_episodes)

    sim.close()

    summary = {
        "task": cfg.task,
        "policy": cfg.policy,
        "seed": cfg.seed,
        "n_episodes": cfg.n_episodes,
        "success_rate": rate,
    }
    Path(cfg.run_dir, "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"metrics: {cfg.run_dir}/metrics.jsonl")


if __name__ == "__main__":
    main()
