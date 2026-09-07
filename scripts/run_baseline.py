"""M1 / M2 driver — single-robot or centralised-fleet training baseline.

Examples
--------
Single robot (M1):
    python scripts/run_baseline.py --config configs/single_robot.yaml

Centralised heterogeneous fleet (M2):
    python scripts/run_baseline.py --config configs/fleet_8.yaml --centralised
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--centralised", action="store_true",
                        help="Pool all robots' data and train one policy")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[baseline] loading {args.config} (seed={args.seed}, "
          f"centralised={args.centralised})")
    raise NotImplementedError("Implement in M1 / M2")


if __name__ == "__main__":
    main()
