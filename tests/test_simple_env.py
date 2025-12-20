from heatpump_rl_env import SimpleThermalEnv


def test_simple_env_runs_one_step():
    env = SimpleThermalEnv()
    obs, info = env.reset()
    assert obs.shape == (1,)

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == (1,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "T_room" in info