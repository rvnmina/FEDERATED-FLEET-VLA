import numpy as np
import pytest

pytest.importorskip("robosuite")

from src.fleet.simulator import TASKS, RobotSim, SimConfig  # noqa: E402

OBJECT_KEY = {
    "lift": "cube_pos",
    "stack": "cubeA_pos",
    "door": "door_pos",
    "nut_assembly": "SquareNut_pos",
}


@pytest.mark.parametrize("task", sorted(TASKS))
def test_reset_is_reproducible(task):
    sim = RobotSim(SimConfig(task=task, seed=0))
    key = OBJECT_KEY[task]

    a = sim.reset(episode=3)
    b = sim.reset(episode=4)
    c = sim.reset(episode=3)

    assert np.allclose(a[key], c[key])
    assert np.allclose(a["robot0_joint_pos"], c["robot0_joint_pos"])
    assert not np.allclose(a[key], b[key])
    sim.close()


def test_step_runs():
    sim = RobotSim(SimConfig(task="lift", seed=0))
    sim.reset(episode=0)
    obs, reward, done, info = sim.step(np.zeros(sim.action_dim))
    assert "robot0_eef_pos" in obs
    assert sim.is_success() is False
    sim.close()
