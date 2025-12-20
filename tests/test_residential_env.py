import numpy as np

from heatpump_rl_env import ResidentialHeatPumpEnv


def test_residential_env_reset():
    env = ResidentialHeatPumpEnv()
    obs, info = env.reset()
    assert obs.shape == (13,)
    assert isinstance(info, dict)


def test_residential_env_step():
    env = ResidentialHeatPumpEnv()
    obs, info = env.reset()

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == (13,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "T_room" in info
    assert "cumulative_energy_kwh" in info


def test_residential_env_episode():
    env = ResidentialHeatPumpEnv(config={"max_episode_steps": 100})
    obs, info = env.reset()

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if truncated:
            break

    assert truncated
    assert env.episode_step == 100