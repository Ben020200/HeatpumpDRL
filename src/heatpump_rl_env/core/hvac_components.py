from __future__ import annotations

from typing import Tuple

import numpy as np


class HeatPump:
    """Heat pump with temperature-dependent COP."""

    def __init__(
        self,
        capacity_w: float = 10000.0,
        cop_rated: float = 3.5,
        t_ref_source: float = 7.0,
        t_ref_sink: float = 35.0,
        cop_temp_factor: float = 0.03,
    ):
        self.capacity_w = float(capacity_w)
        self.cop_rated = float(cop_rated)
        self.t_ref_source = float(t_ref_source)
        self.t_ref_sink = float(t_ref_sink)
        self.cop_temp_factor = float(cop_temp_factor)

    def compute_cop(self, t_source: float, t_sink: float) -> float:
        """COP as function of source and sink temperatures."""
        dt = t_sink - t_source
        dt_ref = self.t_ref_sink - self.t_ref_source
        cop = self.cop_rated * (1.0 - self.cop_temp_factor * (dt - dt_ref))
        return max(1.1, min(cop, self.cop_rated * 1.2))

    def compute_heat_output(self, cop: float, power_elec: float) -> float:
        """Compute heat output given COP and electrical power."""
        return cop * power_elec

    def compute_electrical_power(self, heat_demand: float, cop: float) -> float:
        """Compute electrical power needed to meet heat demand."""
        if heat_demand <= 0:
            return 0.0
        return min(heat_demand / cop, self.capacity_w)

    def compute_max_heat(self, cop: float) -> float:
        """Maximum heat output at given COP."""
        return self.capacity_w * cop


class DHWController:
    """Domestic hot water on-off hysteresis control."""

    def __init__(
        self,
        setpoint_c: float = 50.0,
        deadband_c: float = 5.0,
        min_on_time: float = 300.0,
        min_off_time: float = 300.0,
    ):
        self.setpoint = float(setpoint_c)
        self.deadband = float(deadband_c)
        self.min_on_time = float(min_on_time)
        self.min_off_time = float(min_off_time)

        self.state = False
        self.state_time = 0.0

    def step(self, tank_temp: float, dt: float = 3600.0) -> bool:
        """Update controller state based on tank temperature."""
        self.state_time += dt

        if self.state:  # Currently on
            if (tank_temp >= self.setpoint + self.deadband) or (self.state_time > 7200):
                if self.state_time >= self.min_on_time:
                    self.state = False
                    self.state_time = 0.0
        else:  # Currently off
            if tank_temp <= self.setpoint - self.deadband:
                if self.state_time >= self.min_off_time:
                    self.state = True
                    self.state_time = 0.0

        return self.state


class PIDController:
    """Simple PID controller for comfort."""

    def __init__(self, setpoint: float = 21.0, kp: float = 0.1, ki: float = 0.01, kd: float = 0.05):
        self.setpoint = float(setpoint)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)

        self.integral = 0.0
        self.last_error = 0.0

    def step(self, measurement: float, dt: float = 3600.0) -> float:
        """Compute control signal."""
        error = self.setpoint - measurement
        self.integral += error * dt
        self.integral = np.clip(self.integral, -100, 100)

        derivative = (error - self.last_error) / dt if dt > 0 else 0
        self.last_error = error

        u = self.kp * error + self.ki * self.integral + self.kd * derivative
        return float(np.clip(u, 0.0, 1.0))

    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.last_error = 0.0