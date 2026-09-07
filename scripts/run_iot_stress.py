"""M3 driver — FedAvg with one of the four IoT stressors turned on.

Stressors
---------
- ``bandwidth`` — top-k gradient sparsification, cap in bytes/round
- ``stale``     — async delayed updates, delay distribution configurable
- ``dropout``   — probability p of client silence in a round
- ``energy``    — finite local compute budget per round
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stressor", choices=["bandwidth", "stale", "dropout", "energy"],
                        required=True)
    parser.add_argument("--cap-bytes", type=float, default=None,
                        help="Bandwidth stressor: bytes/round cap")
    parser.add_argument("--delay-mean", type=float, default=None,
                        help="Stale stressor: mean delay rounds")
    parser.add_argument("--dropout-p", type=float, default=None,
                        help="Dropout stressor: per-round dropout probability")
    parser.add_argument("--energy-flops", type=float, default=None,
                        help="Energy stressor: max FLOPs/round")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[iot-stress] stressor={args.stressor}")
    raise NotImplementedError("Implement in M3")


if __name__ == "__main__":
    main()
