import json

import torch

from src.utils.run_logger import (
    RunLogger,
    payload_bytes,
    state_delta,
    trainable_keys,
)
from src.utils.seed import seed_everything


def _model() -> torch.nn.Module:
    m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    m[0].requires_grad_(False)  # frozen "encoder", like SmolVLA's default
    return m


def test_payload_bytes_counts_every_tensor():
    m = _model()
    # (4*3 + 3) + (3*2 + 2) = 23 float32 params
    assert payload_bytes(m.state_dict()) == 23 * 4
    assert payload_bytes(m.half().state_dict()) == 23 * 2


def test_payload_bytes_trainable_only():
    m = _model()
    assert trainable_keys(m) == {"1.weight", "1.bias"}
    assert payload_bytes(m.state_dict(), only=trainable_keys(m)) == 8 * 4


def test_state_delta_is_new_minus_old():
    m = _model()
    old = {k: v.clone() for k, v in m.state_dict().items()}
    with torch.no_grad():
        m[1].weight.add_(1.0)
    delta = state_delta(m.state_dict(), old, only=trainable_keys(m))
    assert set(delta) == {"1.weight", "1.bias"}
    assert torch.all(delta["1.weight"] == 1.0)
    assert torch.all(delta["1.bias"] == 0.0)


def test_seed_everything_is_reproducible():
    seed_everything(0)
    a = torch.randn(5)
    seed_everything(0)
    assert torch.equal(a, torch.randn(5))


def test_run_logger_writes_jsonl_and_run_info(tmp_path):
    with RunLogger(tmp_path, seed=7, extra={"task": "Lift"}) as log:
        with log.step(phase="train", step=1) as rec:
            rec["loss"] = 0.5
        log.log(phase="payload", bytes=123)

    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["seed"] == 7 and info["task"] == "Lift"
    assert "torch" in info["versions"]

    lines = [json.loads(x) for x in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert lines[0]["phase"] == "train" and lines[0]["loss"] == 0.5
    assert lines[0]["wall_s"] >= 0 and lines[0]["rss_bytes"] > 0
    assert lines[1] == {"t": lines[1]["t"], "phase": "payload", "bytes": 123}
