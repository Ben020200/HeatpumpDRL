from __future__ import annotations

from typing import Dict

import numpy as np


class WeatherProfile:
    """Synthetic annual weather profile (TMY-like)."""

    def __init__(self, year: int = 2023):
        self.year = year
        self.data = self._generate_weather()

    @staticmethod
    def _generate_weather() -> Dict[int, Dict[str, float]]:
        """Generate 8760 hours of synthetic weather."""
        hours = np.arange(8760)
        day_of_year = hours / 24.0
        hour_of_day = hours % 24

        # Temperature: annual + diurnal variation + noise
        T_annual = 12 + 13 * np.sin(2 * np.pi * day_of_year / 365 - np.pi / 2)
        T_diurnal = 8 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi / 2)
        T_noise = np.random.normal(0, 1.5, 8760)
        T_ambient = np.clip(T_annual + T_diurnal + T_noise, -20, 40)

        # Solar radiation: peak at solar noon
        solar_rad = np.maximum(0, 800 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi / 2))

        return {
            h: {
                "T_ambient": float(T_ambient[h]),
                "solar_radiation": float(solar_rad[h]),
            }
            for h in hours
        }

    def get_hourly(self, hour: int) -> Dict[str, float]:
        """Get weather for a specific hour of year."""
        hour = hour % 8760
        return self.data.get(hour, {"T_ambient": 15.0, "solar_radiation": 0.0})