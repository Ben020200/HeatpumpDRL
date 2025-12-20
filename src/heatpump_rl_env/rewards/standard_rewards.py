from __future__ import annotations

from typing import Optional

import numpy as np


def comfort_reward(
    T_room: float,
    occupancy: bool,
    setpoint: float = 21.0,
    bandwidth: float = 2.0,
) -> float:
    """Penalty for temperature outside comfort band (if occupied)."""
    if not occupancy:
        return 0.0

    t_min = setpoint - bandwidth
    t_max = setpoint + bandwidth

    below = max(0, t_min - T_room)
    above = max(0, T_room - t_max)

    return -(below + above)


def energy_reward(power_w: float, price: float) -> float:
    """Penalty for energy cost."""
    energy_kwh = power_w / 3600 / 1000
    cost = energy_kwh * price
    return -cost * 100  # Amplify for learning scale


def stability_reward(power_w: float, previous_power_w: Optional[float] = None) -> float:
    """Penalty for rapid power changes."""
    if previous_power_w is None:
        return 0.0

    ramp = abs(power_w - previous_power_w)
    return -ramp / 10000


def multiobjective_reward(
    T_room: float,
    power_w: float,
    occupancy: bool,
    price: float,
    previous_power_w: Optional[float] = None,
    w_comfort: float = 0.2,
    w_energy: float = 0.7,
    w_stability: float = 0.1,
    setpoint: float = 21.0,
    bandwidth: float = 2.0,
) -> float:
    """Weighted multiobjective reward."""
    w_sum = w_comfort + w_energy + w_stability
    if w_sum > 0:
        w_comfort /= w_sum
        w_energy /= w_sum
        w_stability /= w_sum

    r_comfort = comfort_reward(T_room, occupancy, setpoint, bandwidth)
    r_energy = energy_reward(power_w, price)
    r_stability = stability_reward(power_w, previous_power_w)

    return (w_comfort * r_comfort + w_energy * r_energy + w_stability * r_stability)