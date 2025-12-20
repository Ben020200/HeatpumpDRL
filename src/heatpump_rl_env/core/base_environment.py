from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ThermalEnvironment(gym.Env, ABC):
    """Abstract base class for thermal RL environments (Gymnasium API)."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.episode_step: int = 0
        self.max_episode_steps: int = int(config.get("max_episode_steps", 8760))

        # Subclasses must set these
        self.observation_space: spaces.Space
        self.action_space: spaces.Space

    @abstractmethod
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        ...

    @abstractmethod
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the environment by one step."""
        ...

    @abstractmethod
    def _get_observation(self) -> np.ndarray:
        """Build observation vector from internal state."""
        ...

    @abstractmethod
    def _compute_reward(self) -> float:
        """Compute reward at current time step."""
        ...

    def render(self):
        """Optional render; no-op for now."""
        return None

    def close(self):
        """Optional cleanup hook."""
        return None

    def set_weather(self, weather_df: "Any") -> None:
        """Override weather profile (subclasses may implement)."""
        return None

    def get_state_dict(self) -> Dict[str, Any]:
        """Return basic state info."""
        return {
            "episode_step": self.episode_step,
            "max_episode_steps": self.max_episode_steps,
        }