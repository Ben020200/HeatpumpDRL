from __future__ import annotations

from typing import Dict

import numpy as np


class WeatherProfile:
    """Synthetic annual weather profile (TMY-like) for Central Europe."""

    def __init__(self, year: int = 2023, seed: int = None):
        self.year = year
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(42)  # Fixed seed for reproducibility
        self.data = self._generate_weather()

    def _generate_weather(self) -> Dict[int, Dict[str, float]]:
        """Generate 8760 hours of synthetic weather (NOT @staticmethod)."""
        hours = np.arange(8760)
        day_of_year = hours / 24.0
        hour_of_day = hours % 24

        # Temperature: Central European climate (Berlin-like)
        # Winter: -5°C, Summer: +25°C, Mean: 10°C
        T_annual = 10.0 + 15.0 * np.sin(2 * np.pi * day_of_year / 365 - np.pi / 2)
        
        # Diurnal: ±6°C daily variation (stronger in summer)
        seasonal_factor = 0.5 + 0.5 * np.sin(2 * np.pi * day_of_year / 365)
        T_diurnal = 6 * seasonal_factor * np.sin(2 * np.pi * hour_of_day / 24 - np.pi / 2)
        
        # Random noise: realistic weather variability
        T_noise = np.random.normal(0, 1.2, 8760)
        
        T_ambient = np.clip(T_annual + T_diurnal + T_noise, -15, 35)

        # Solar radiation: peak at solar noon (simplified)
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
        return self.data.get(hour, {"T_ambient": 10.0, "solar_radiation": 0.0})