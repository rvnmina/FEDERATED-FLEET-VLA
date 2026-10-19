"""Day 6: load SmolVLA, report its size, and time the load.

First run downloads ~450M of weights from the Hugging Face Hub and caches
them under ~/.cache/huggingface; later runs are offline.

    python -m scripts.inspect_policy
    python -m scripts.inspect_policy --save checkpoints/smolvla_base_copy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.policy.smolvla_wrapper import PolicyConfig, SmolVLAWrapper
from src.utils.instrument import RunLogger
from src.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu (baseline) or mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", help="also save a copy of the policy here")
    parser.add_argument("--run-dir", default="results/policy_inspect")
    args = parser.parse_args()

    seed_everything(args.seed)

    with RunLogger(args.run_dir, seed=args.seed, extra={"device": args.device}) as log:
        with log.step(phase="load_policy"):
            wrapper = SmolVLAWrapper(PolicyConfig(device=args.device)).load()

        report = wrapper.size_report()
        log.log(phase="policy_size", **report)

        if args.save:
            with log.step(phase="save_policy"):
                wrapper.save(args.save)
            report["saved_to"] = args.save

    print(json.dumps(report, indent=2))
    print(f"metrics: {Path(args.run_dir) / 'metrics.jsonl'}")


if __name__ == "__main__":
    main()
