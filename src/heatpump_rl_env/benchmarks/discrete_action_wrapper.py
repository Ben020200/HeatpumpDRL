from __future__ import annotations

from typing import Sequence

import numpy as np
from gymnasium import Env, spaces


class DiscreteActionWrapper(Env):
    """
    Wraps a continuous-action env into a discrete one by mapping discrete
    indices to fixed continuous values.

    Example:
        levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        Discrete(5) -> Box([level], dtype=float32)
    """

    def __init__(self, env: Env, levels: Sequence[float]):
        super().__init__()
        self.env = env
        self.levels = np.array(levels, dtype=np.float32)

        # Original obs/action spaces
        self.observation_space = env.observation_space
        self.action_space = spaces.Discrete(len(self.levels))

        # Gymnasium metadata passthrough
        self.metadata = getattr(env, "metadata", {})

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        # Map discrete index -> continuous scalar
        level = float(self.levels[int(action)])
        cont_action = np.array([level], dtype=np.float32)
        return self.env.step(cont_action)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()