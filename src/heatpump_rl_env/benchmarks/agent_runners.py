from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    from stable_baselines3 import DQN, PPO, SAC
    from stable_baselines3.common.callbacks import EvalCallback
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

from heatpump_rl_env.benchmarks.discrete_action_wrapper import DiscreteActionWrapper


class AgentRunner:
    """Base class for RL agent runners."""

    def __init__(self, env, agent_name: str):
        if not HAS_SB3:
            raise ImportError("stable-baselines3 not installed. Install with: pip install stable-baselines3")
        self.env = env
        self.agent_name = agent_name
        self.model = None

    def train(self, total_timesteps: int, verbose: int = 1):
        """Train the agent."""
        raise NotImplementedError

    def evaluate(self, n_episodes: int = 10) -> dict:
        """Evaluate agent over n episodes."""
        episode_rewards = []
        episode_lengths = []
        episode_energies = []

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0.0
            episode_length = 0

            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1

                if terminated or truncated:
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    episode_energies.append(info.get("cumulative_energy_kwh", 0.0))
                    break

        return {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "mean_energy_kwh": float(np.mean(episode_energies)),
        }


class DQNRunner(AgentRunner):
    """DQN agent (uses a discrete wrapper over the continuous env)."""

    def __init__(self, env, learning_rate: float = 1e-4, buffer_size: int = 10000):
        # Wrap env into discrete version with 5 power levels
        discrete_env = DiscreteActionWrapper(env, levels=[0.0, 0.25, 0.5, 0.75, 1.0])
        super().__init__(discrete_env, "DQN")
        self.model = DQN(
            "MlpPolicy",
            discrete_env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            verbose=0,
        )

    def train(self, total_timesteps: int, verbose: int = 1):
        """Train DQN."""
        self.model.learn(total_timesteps=total_timesteps, log_interval=10, progress_bar=verbose > 0)


class SACRunner(AgentRunner):
    """SAC agent (Soft Actor-Critic, ideal for continuous control)."""

    def __init__(self, env, learning_rate: float = 3e-4, batch_size: int = 256):
        super().__init__(env, "SAC")
        self.model = SAC(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            verbose=0,
        )

    def train(self, total_timesteps: int, verbose: int = 1):
        """Train SAC."""
        self.model.learn(total_timesteps=total_timesteps, log_interval=10, progress_bar=verbose > 0)


class PPORunner(AgentRunner):
    """PPO agent (Proximal Policy Optimization, good balance for continuous control)."""

    def __init__(self, env, learning_rate: float = 3e-4, batch_size: int = 64):
        super().__init__(env, "PPO")
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            verbose=0,
        )

    def train(self, total_timesteps: int, verbose: int = 1):
        """Train PPO."""
        self.model.learn(total_timesteps=total_timesteps, log_interval=10, progress_bar=verbose > 0)