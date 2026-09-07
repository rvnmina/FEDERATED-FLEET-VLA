"""FedAvg from scratch — no framework.

Deliberately hand-rolled: the whole point of this project is to instrument
FedAvg with four IoT stressors that a framework abstraction would hide.
Kept short (< 300 lines total) so every line is auditable.

Reference
---------
McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017).
Communication-Efficient Learning of Deep Networks from Decentralized Data.
AISTATS. https://arxiv.org/abs/1602.05629
"""

from __future__ import annotations

from typing import Iterable


def fedavg_round(client_updates: Iterable[dict], client_weights: list[float]) -> dict:
    """One FedAvg aggregation round.

    Parameters
    ----------
    client_updates
        Iterable of per-client parameter deltas (state_dict format).
    client_weights
        Number of local samples per client, used as the aggregation weight.

    Returns
    -------
    dict
        Averaged parameter delta ready to apply to the global model.
    """
    raise NotImplementedError("M3 — implement FedAvg aggregation")


def fedavg_train(
    global_model,
    clients,
    n_rounds: int,
    local_epochs: int,
    stressor=None,
):
    """Full FedAvg training loop with optional IoT stressor injection."""
    raise NotImplementedError("M3 — implement full FedAvg training loop")
