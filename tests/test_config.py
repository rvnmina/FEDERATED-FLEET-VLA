import pytest
import yaml

from src.utils.config import RunConfig


def test_from_yaml_reads_every_field(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"task": "door", "seed": 7, "n_episodes": 3}))

    cfg = RunConfig.from_yaml(p)

    assert cfg.task == "door" and cfg.seed == 7 and cfg.n_episodes == 3
    assert cfg.robot == "Panda"  # default kept


def test_unknown_key_is_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"tsak": "door"}))  # typo

    with pytest.raises(ValueError, match="tsak"):
        RunConfig.from_yaml(p)


def test_sim_config_carries_the_sim_settings():
    cfg = RunConfig(task="stack", seed=4, horizon=50)
    sim = cfg.sim_config()
    assert (sim.task, sim.seed, sim.horizon) == ("stack", 4, 50)


def test_shipped_config_loads():
    cfg = RunConfig.from_yaml("configs/single_robot.yaml")
    assert cfg.task == "lift" and cfg.policy == "random"
