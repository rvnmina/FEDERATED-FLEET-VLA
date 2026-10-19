"""Tests for the SmolVLA wrapper.

The parts that need the 450M checkpoint are skipped unless it is already
cached locally, so the suite stays fast and works offline.
"""

import pytest
import torch

from src.policy.smolvla_wrapper import BASE_CHECKPOINT, PolicyConfig, SmolVLAWrapper


def checkpoint_is_cached() -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(BASE_CHECKPOINT, local_files_only=True)
        return True
    except Exception:
        return False


needs_checkpoint = pytest.mark.skipif(
    not checkpoint_is_cached(),
    reason=f"{BASE_CHECKPOINT} not cached; run scripts/inspect_policy.py once",
)


def test_defaults_match_the_project_plan():
    cfg = PolicyConfig()
    assert cfg.checkpoint == BASE_CHECKPOINT
    assert cfg.device == "cpu"  # the whole project is CPU-only
    assert cfg.freeze_vision_encoder and cfg.train_expert_only


def test_methods_refuse_before_load():
    w = SmolVLAWrapper()
    with pytest.raises(RuntimeError, match="load"):
        w.size_report()
    with pytest.raises(RuntimeError, match="load"):
        w.save("/tmp/nope")


@needs_checkpoint
def test_load_reports_a_sane_size():
    w = SmolVLAWrapper().load()
    r = w.size_report()
    assert 400 < r["total_params_m"] < 600       # SmolVLA is ~450M
    assert 0 < r["trainable_params_m"] < r["total_params_m"]
    assert 0 < r["trainable_fraction"] < 1


@needs_checkpoint
def test_save_then_reload_keeps_the_weights(tmp_path):
    w = SmolVLAWrapper().load()
    before = w.state_dict()

    w.save(tmp_path)
    reloaded = SmolVLAWrapper(PolicyConfig(checkpoint=str(tmp_path))).load()
    after = reloaded.state_dict()

    assert set(before) == set(after)
    for k in list(before)[:50]:
        assert torch.equal(before[k], after[k]), k
