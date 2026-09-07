"""M3 driver — FedAvg training over a heterogeneous fleet, no stressors."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[fedavg] rounds={args.n_rounds}, local_epochs={args.local_epochs}")
    raise NotImplementedError("Implement in M3")


if __name__ == "__main__":
    main()
