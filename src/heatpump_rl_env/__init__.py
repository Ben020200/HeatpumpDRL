"""
HeatPumpRL-Environment: Thermal RL environments for heat pump and HVAC control.
"""

__version__ = "0.1.0"

from .environments import ResidentialHeatPumpEnv, SimpleThermalEnv
from .benchmarks.baseline_controllers import OnOffController, PIDControllerBaseline, RuleBasedController
from .core.hvac_components import HeatPump, DHWController, PIDController
from .thermal_engine.rc_network import RC1R1C, create_thermal_model

__all__ = [
    "ResidentialHeatPumpEnv",
    "SimpleThermalEnv",
    "OnOffController",
    "PIDControllerBaseline",
    "RuleBasedController",
    "HeatPump",
    "DHWController",
    "PIDController",
    "RC1R1C",
    "create_thermal_model",
]