"""Run instrumentation: wall-clock per step, process memory, payload size.

Payload size is the currency of M3 (bandwidth-capped FedAvg), so it is
measured from day one, exactly as a client would send it: the bytes of the
tensors in a ``state_dict`` (or of a delta between two of them), at the
dtype they are stored in.

Every record is one JSON line in ``<run_dir>/metrics.jsonl``; the run's
seed, library versions and git commit go to ``<run_dir>/run_info.json``.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Iterator, Mapping

import psutil
import torch

TRACKED_PACKAGES = ("torch", "lerobot", "robosuite", "mujoco", "numpy", "transformers")


# ---------------------------------------------------------------- payload size

def payload_bytes(
    state: Mapping[str, torch.Tensor],
    only: set[str] | None = None,
) -> int:
    """Bytes needed to transmit ``state`` uncompressed.

    ``only`` restricts the count to the given keys, e.g. the trainable
    parameters, since SmolVLA fine-tuning freezes the vision encoder by default
    and a client only needs to send what changed.
    """
    return sum(
        t.numel() * t.element_size()
        for k, t in state.items()
        if only is None or k in only
    )


def trainable_keys(model: torch.nn.Module) -> set[str]:
    """Names of parameters with ``requires_grad=True`` (state_dict keys)."""
    return {name for name, p in model.named_parameters() if p.requires_grad}


def state_delta(
    new: Mapping[str, torch.Tensor],
    old: Mapping[str, torch.Tensor],
    only: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    """``new - old`` for floating-point tensors (the FedAvg client update)."""
    return {
        k: new[k].detach() - old[k].detach()
        for k in new
        if (only is None or k in only) and new[k].is_floating_point()
    }


# ---------------------------------------------------------------- memory

def process_memory_bytes() -> int:
    """Current resident set size of this process."""
    return psutil.Process().memory_info().rss


# ---------------------------------------------------------------- run metadata

def run_info(seed: int, extra: Mapping | None = None) -> dict:
    """Everything needed to say which code and stack produced a number."""
    versions = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "seed": seed,
        "git_commit": commit,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "versions": versions,
        **(extra or {}),
    }


# ---------------------------------------------------------------- logger

class RunLogger:
    """Append-only JSONL metrics log for one run.

    >>> log = RunLogger("results/lift_seed0", seed=0)
    >>> with log.step(phase="train", step=1):
    ...     loss = train_step()
    >>> log.log(phase="payload", bytes=payload_bytes(model.state_dict()))
    """

    def __init__(self, run_dir: str | Path, seed: int, extra: Mapping | None = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "run_info.json").write_text(
            json.dumps(run_info(seed, extra), indent=2) + "\n"
        )
        self._f = (self.run_dir / "metrics.jsonl").open("a")
        self._t0 = time.perf_counter()

    def log(self, **fields) -> dict:
        record = {"t": round(time.perf_counter() - self._t0, 6), **fields}
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()
        return record

    @contextmanager
    def step(self, **fields) -> Iterator[dict]:
        """Time the enclosed block; log wall-clock seconds and RSS after it.

        Fields added to the yielded dict (e.g. ``rec["loss"] = ...``) are
        logged too.
        """
        rec: dict = dict(fields)
        start = time.perf_counter()
        yield rec
        rec["wall_s"] = time.perf_counter() - start
        rec["rss_bytes"] = process_memory_bytes()
        self.log(**rec)

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
