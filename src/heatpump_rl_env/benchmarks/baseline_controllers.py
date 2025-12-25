from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaselineController(ABC):
    """Abstract baseline controller."""

    @abstractmethod
    def compute_action(self, observation: np.ndarray) -> float:
        """Compute control action (0-1 power level)."""
        ...


class OnOffController(BaselineController):
    """Simple on-off hysteresis control."""

    def __init__(self, setpoint: float = 21.0, deadband: float = 0.8): 
        self.setpoint = float(setpoint)
        self.deadband = float(deadband)
        self.state = False

    def compute_action(self, observation: np.ndarray) -> float:
        """Hysteresis control based on room temperature."""
        T_room = float(observation[0])

        if self.state:
            if T_room >= self.setpoint + self.deadband:
                self.state = False
        else:
            if T_room <= self.setpoint - self.deadband:
                self.state = True

        return 1.0 if self.state else 0.0


class PIDControllerBaseline(BaselineController):
    """PID feedback controller."""

    def __init__(self, setpoint: float = 21.0, kp: float = 0.1, ki: float = 0.01, kd: float = 0.05):
        self.setpoint = float(setpoint)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)

        self.integral = 0.0
        self.last_error = 0.0

    def compute_action(self, observation: np.ndarray) -> float:
        """PID control."""
        T_room = float(observation[0])
        error = self.setpoint - T_room

        self.integral += error
        self.integral = np.clip(self.integral, -100, 100)

        derivative = error - self.last_error
        self.last_error = error

        u = self.kp * error + self.ki * self.integral + self.kd * derivative
        return float(np.clip(u, 0.0, 1.0))


class RuleBasedController(BaselineController):
    """Physics-informed rule-based control."""

    def __init__(self, setpoint: float = 21.0):
        self.setpoint = float(setpoint)

    def compute_action(self, observation: np.ndarray) -> float:
        """Rule-based action based on temperature only (for now)."""
        T_room = float(observation[0])

        # Simple rule: proportional to temperature error
        power = max(0.0, min(1.0, 0.5 * (self.setpoint - T_room) / 5))
        return power