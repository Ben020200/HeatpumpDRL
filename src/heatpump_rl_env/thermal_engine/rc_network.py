from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from scipy.integrate import solve_ivp


class ThermalModel(ABC):
    """Abstract base class for thermal models."""

    def __init__(self, name: str, n_states: int, n_inputs: int):
        self.name = name
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.state = np.zeros(n_states, dtype=float)

    @abstractmethod
    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """dx/dt = f(x, u)."""
        ...

    def step(self, inputs: np.ndarray, dt: float = 3600.0) -> np.ndarray:
        """Integrate one step forward using solve_ivp (RK45)."""

        def f(t, x):
            return self.dynamics(t, x, inputs)

        sol = solve_ivp(
            f,
            t_span=(0.0, dt),
            y0=self.state,
            method="RK45",
            max_step=min(300.0, dt / 4.0),
        )
        self.state = sol.y[:, -1]
        return self.state.copy()

    def get_state(self) -> np.ndarray:
        return self.state.copy()

    def set_state(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)
        if x.shape != self.state.shape:
            raise ValueError("State shape mismatch.")
        self.state = x.copy()


class RC1R1C(ThermalModel):
    """1R1C model: single-zone, single capacitance.

    Dynamics:
        C * dT/dt = UA * (T_amb - T_room) + Q_hvac + Q_internal
    """

    def __init__(self, C: float, UA: float, T_init: float = 20.0):
        super().__init__(name="RC1R1C", n_states=1, n_inputs=3)
        self.C = float(C)
        self.UA = float(UA)
        self.state[:] = T_init

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        T_room = x[0]
        Q_hvac, T_amb, Q_internal = float(u[0]), float(u[1]), float(u[2])
        dTdt = (self.UA * (T_amb - T_room) + Q_hvac + Q_internal) / self.C
        return np.array([dTdt], dtype=float)


def create_thermal_model(model_type: str, **params) -> ThermalModel:
    """Factory to create thermal models (currently only RC1R1C)."""

    if model_type.upper() == "RC1R1C":
        return RC1R1C(
            C=params.get("C", 5e6),
            UA=params.get("UA", 100.0),
            T_init=params.get("T_init", 20.0),
        )
    raise ValueError(f"Unknown model_type: {model_type}")