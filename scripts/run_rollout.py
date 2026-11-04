"""
run_rollout.py  –  Run SmolVLA on a robosuite task and log results.

This is the Day 7 deliverable: first end-to-end rollout on one
manipulation task, logging success rate and per-step wall time.

Usage:
    python -m scripts.run_rollout --config configs/smolvla_lift.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.fleet.simulator import RobotSim
from src.policy.smolvla_wrapper import SmolVLAWrapper, PolicyConfig
from src.utils.config import RunConfig
from src.utils.instrument import write_metrics


def build_policy(cfg: RunConfig) -> SmolVLAWrapper:
    """Load SmolVLA and configure it for the robosuite environment."""
    wrapper = SmolVLAWrapper(PolicyConfig(
        checkpoint=cfg.checkpoint,
        device=cfg.device,
    ))
    wrapper.load()
    wrapper.configure_for_env(
        cameras=cfg.cameras,
        image_size=cfg.camera_size,
        action_dim=7,            # Panda: 6 joints + 1 gripper
        n_action_steps=cfg.n_action_steps,
    )
    return wrapper


def run_episode(
    sim: RobotSim,
    policy: SmolVLAWrapper,
    task_prompt: str,
    horizon: int,
) -> dict:
    """Run one episode and return metrics."""
    obs = sim.reset()
    policy.reset()

    step_times = []
    total_reward = 0.0

    for step in range(horizon):
        t0 = time.perf_counter()
        action = policy.select_action(obs, task_prompt)
        elapsed = time.perf_counter() - t0
        step_times.append(elapsed)

        # Clip actions to [-1, 1] — robosuite's action space
        action = np.clip(action, -1.0, 1.0)

        obs, reward, done, info = sim.step(action)
        total_reward += reward

        if done:
            break

    return {
        "steps": step + 1,
        "total_reward": total_reward,
        "success": float(info.get("is_success", False)),
        "mean_step_s": float(np.mean(step_times)),
        "median_step_s": float(np.median(step_times)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SmolVLA rollout")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = RunConfig.from_yaml(args.config)
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading policy from {cfg.checkpoint} ...")
    policy = build_policy(cfg)
    print("Policy loaded.")

    sim_cfg = cfg.sim_config()
    sim = RobotSim(sim_cfg)

    results = []
    for ep in range(cfg.n_episodes):
        print(f"Episode {ep + 1}/{cfg.n_episodes} ...", end=" ", flush=True)
        sim_cfg_ep = sim_cfg._replace(seed=cfg.seed + ep)
        sim = RobotSim(sim_cfg_ep)
        metrics = run_episode(sim, policy, cfg.task_prompt, cfg.horizon)
        metrics["episode"] = ep
        results.append(metrics)
        print(f"reward={metrics['total_reward']:.1f}  "
              f"success={metrics['success']:.0f}  "
              f"median_step={metrics['median_step_s']:.3f}s")
        sim.close()

    # Summary
    successes = [r["success"] for r in results]
    step_times = [r["median_step_s"] for r in results]
    print(f"\n{'='*40}")
    print(f"Success rate: {np.mean(successes):.0%}  "
          f"({int(sum(successes))}/{len(successes)})")
    print(f"Median step time: {np.median(step_times):.3f}s")

    # Save results
    out_path = run_dir / "rollout_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()