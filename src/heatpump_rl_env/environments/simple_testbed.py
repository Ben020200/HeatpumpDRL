from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import spaces

from heatpump_rl_env.core.base_environment import ThermalEnvironment
from heatpump_rl_env.thermal_engine.rc_network import RC1R1C


class SimpleThermalEnv(ThermalEnvironment):
    """Minimal 1R1C environment for quick testing."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        config.setdefault("max_episode_steps", 24)  # 1 day

        super().__init__(config)

        # 1D observation: room temperature
        self.observation_space = spaces.Box(
            low=np.array([-30.0], dtype=np.float32),
            high=np.array([50.0], dtype=np.float32),
            dtype=np.float32,
        )

        # 1D action: normalized heating power in [0, 1]
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Simple 1R1C building model
        self.model = RC1R1C(C=5e6, UA=100.0, T_init=20.0)

        # Environment parameters
        self.ambient_temp = 5.0  # fixed outdoor temp
        self.max_power_w = 5000.0
        self.internal_gains_w = 200.0

    def _get_observation(self) -> np.ndarray:
        return np.array([self.model.get_state()[0]], dtype=np.float32)

    def _compute_reward(self) -> float:
        T = float(self.model.get_state()[0])
        # Simple comfort around 21°C, quadratic penalty
        return -((T - 21.0) ** 2) / 4.0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.episode_step = 0
        self.model.set_state(np.array([20.0], dtype=float))
        obs = self._get_observation()
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: Any):
        self.episode_step += 1

        # Clip action to [0, 1]
        a = float(np.clip(action[0], 0.0, 1.0))
        Q_hvac = a * self.max_power_w

        inputs = np.array(
            [Q_hvac, self.ambient_temp, self.internal_gains_w],
            dtype=float,
        )
        self.model.step(inputs, dt=3600.0)

        obs = self._get_observation()
        reward = float(self._compute_reward())
        terminated = False
        truncated = self.episode_step >= self.max_episode_steps
        info: Dict[str, Any] = {"T_room": float(obs[0]), "Q_hvac": Q_hvac}
        return obs, reward, terminated, truncated, info