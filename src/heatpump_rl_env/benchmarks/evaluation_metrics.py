from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


class EvaluationMetrics:
    """Compute and aggregate evaluation metrics."""

    def __init__(self):
        self.episodes: List[Dict[str, Any]] = []

    def record_episode(self, info: Dict[str, Any]) -> None:
        """Record metrics from an episode."""
        self.episodes.append(info.copy())

    def summary(self) -> Dict[str, float]:
        """Compute aggregate metrics across all episodes."""
        if not self.episodes:
            return {}

        energies = [e.get("cumulative_energy_kwh", 0.0) for e in self.episodes]
        costs = [e.get("cumulative_cost", 0.0) for e in self.episodes]
        comforts = [e.get("comfort_violations", 0) for e in self.episodes]

        return {
            "mean_energy_kwh": float(np.mean(energies)),
            "std_energy_kwh": float(np.std(energies)),
            "mean_cost_eur": float(np.mean(costs)),
            "mean_comfort_violations": float(np.mean(comforts)),
            "n_episodes": len(self.episodes),
        }

    def print_summary(self, agent_name: str = "Agent") -> None:
        """Print summary in a human-readable format."""
        summary = self.summary()
        print(f"\n{'='*60}")
        print(f"  {agent_name} Evaluation Summary")
        print(f"{'='*60}")
        print(f"  Mean annual energy:       {summary.get('mean_energy_kwh', 0):.1f} kWh")
        print(f"  Std energy:               {summary.get('std_energy_kwh', 0):.1f} kWh")
        print(f"  Mean annual cost:         €{summary.get('mean_cost_eur', 0):.2f}")
        print(f"  Mean comfort violations:  {summary.get('mean_comfort_violations', 0):.1f} hours")
        print(f"  Episodes evaluated:       {summary.get('n_episodes', 0)}")
        print(f"{'='*60}")