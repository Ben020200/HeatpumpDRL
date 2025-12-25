from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import spaces

from heatpump_rl_env.core.base_environment import ThermalEnvironment
from heatpump_rl_env.core.hvac_components import DHWController, HeatPump
from heatpump_rl_env.rewards.standard_rewards import multiobjective_reward
from heatpump_rl_env.thermal_engine.rc_network import RC1R1C
from heatpump_rl_env.utils.weather_data import WeatherProfile


class ResidentialHeatPumpEnv(ThermalEnvironment):
    """Residential heat pump environment (13-D observation, continuous action)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}

        config.setdefault("max_episode_steps", 8760)
        config.setdefault("building_area_m2", 150)
        config.setdefault("u_value_w_m2k", 0.3)
        config.setdefault("thermal_mass_j_k", 5000000)
        config.setdefault("hp_capacity_w", 10000)
        config.setdefault("hp_cop_rated", 3.5)
        config.setdefault("comfort_setpoint_c", 21.0)
        config.setdefault("comfort_bandwidth_c", 2.0)
        config.setdefault("reward_w_energy", 0.7)
        config.setdefault("reward_w_comfort", 0.2)
        config.setdefault("reward_w_stability", 0.1)

        super().__init__(config)

        # Building parameters
        self.area = float(config["building_area_m2"])
        self.u_value = float(config["u_value_w_m2k"])
        self.ua = self.u_value * self.area
        self.thermal_mass = float(config.get("thermal_mass_j_k", 5e6))

        # HVAC
        self.hp = HeatPump(
            capacity_w=float(config["hp_capacity_w"]),
            cop_rated=float(config["hp_cop_rated"]),
        )
        self.dhw_controller = DHWController()

        # Thermal model (RC1R1C for simplicity; can extend to RC2R2C)
        self.model = RC1R1C(C=self.thermal_mass, UA=self.ua, T_init=20.0)

        # Weather
        self.weather = WeatherProfile()

        # Comfort
        self.setpoint = float(config["comfort_setpoint_c"])
        self.bandwidth = float(config["comfort_bandwidth_c"])

        # Reward weights
        self.w_energy = float(config["reward_w_energy"])
        self.w_comfort = float(config["reward_w_comfort"])
        self.w_stability = float(config["reward_w_stability"])

        # State tracking
        self.previous_power = 0.0
        self.internal_gains = float(config.get("internal_gains_w", 200.0))  # W (fixed for now)
        self.price = 0.25  # €/kWh (fixed for now; can be time-varying)

        # Occupancy schedule (simplified: 6-9, 17-23)
        self.occupancy = False

        # 13-D observation: [T_room, T_amb, hour, day, price, occupancy, (6 more padding)]
        self.observation_space = spaces.Box(
            low=np.array(
                [-30, -30, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1],
                dtype=np.float32,
            ),
            high=np.array(
                [50, 50, 23, 365, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # Continuous action: [0, 1] normalized heat pump power
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Cumulative tracking for logging
        self.cumulative_energy_kwh = 0.0
        self.cumulative_cost = 0.0
        self.comfort_violations = 0.0

    def _update_occupancy(self, hour: int, day: int) -> None:
        """Simple occupancy schedule."""
        hour_of_day = hour % 24
        self.occupancy = (6 <= hour_of_day <= 9) or (17 <= hour_of_day <= 23)

    def _get_observation(self) -> np.ndarray:
        """Build 13-D observation."""
        T_room = float(self.model.get_state()[0])
        weather = self.weather.get_hourly(self.episode_step)
        T_amb = weather["T_ambient"]

        hour = self.episode_step % 24
        day = self.episode_step // 24

        obs = np.array(
            [
                T_room,
                T_amb,
                float(hour),
                float(day % 365),
                self.price,
                float(self.occupancy),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        return obs

    def _compute_reward(self) -> float:
        """Multiobjective reward: energy + comfort + stability."""
        T_room = float(self.model.get_state()[0])
        power_w = self.previous_power

        reward = multiobjective_reward(
            T_room=T_room,
            power_w=power_w,
            occupancy=self.occupancy,
            price=self.price,
            previous_power_w=self.previous_power if self.episode_step > 0 else None,
            w_comfort=self.w_comfort,
            w_energy=self.w_energy,
            w_stability=self.w_stability,
            setpoint=self.setpoint,
            bandwidth=self.bandwidth,
        )
        return reward

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)

        self.episode_step = 0
        self.previous_power = 0.0
        self.model.set_state(np.array([15.0], dtype=float))
        self.cumulative_energy_kwh = 0.0
        self.cumulative_cost = 0.0
        self.comfort_violations = 0.0

        self._update_occupancy(0, 0)

        obs = self._get_observation()
        info: Dict[str, Any] = {}

        return obs, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step."""
        self.episode_step += 1

        # Parse action
        a = float(np.clip(action[0], 0.0, 1.0))
        Q_hp_elec = a * self.hp.capacity_w

        # Weather
        weather = self.weather.get_hourly(self.episode_step)
        T_amb = weather["T_ambient"]

                # Compute COP
        cop = self.hp.compute_cop(T_amb, 35.0)  # Sink = 35°C (DHW tank return)

        # Heat output
        Q_hp = self.hp.compute_heat_output(cop, Q_hp_elec)

        # Update occupancy FIRST (before using it)
        self._update_occupancy(self.episode_step, self.episode_step // 24)
        
        # Internal gains only when occupied
        Q_internal = 200.0 if self.occupancy else 0.0

        # Update thermal model
        inputs = np.array([Q_hp, T_amb, Q_internal], dtype=float)
        self.model.step(inputs, dt=3600.0)

        # Logging
        energy_kwh = Q_hp_elec * 3600.0 / (1000.0 * 1000.0)  # W * s / (kW * s) = kWh
        self.cumulative_energy_kwh += energy_kwh
        self.cumulative_cost += energy_kwh * self.price

        T_room = float(self.model.get_state()[0])
        if self.occupancy:
            if T_room < self.setpoint - self.bandwidth or T_room > self.setpoint + self.bandwidth:
                self.comfort_violations += 1

        # Compute reward
        self.previous_power = Q_hp_elec
        reward = float(self._compute_reward())

        # Termination
        obs = self._get_observation()
        terminated = False
        truncated = self.episode_step >= self.max_episode_steps

        info: Dict[str, Any] = {
            "T_room": T_room,
            "T_ambient": T_amb,
            "Q_hvac_w": Q_hp,
            "P_elec_w": Q_hp_elec,
            "COP": cop,
            "occupancy": self.occupancy,
            "cumulative_energy_kwh": self.cumulative_energy_kwh,
            "cumulative_cost": self.cumulative_cost,
            "comfort_violations": self.comfort_violations,
        }

        return obs, reward, terminated, truncated, info