from heatpump_rl_env import ResidentialHeatPumpEnv, OnOffController
import numpy as np


def main():
    # Create residential environment
    env = ResidentialHeatPumpEnv()
    obs, info = env.reset()
    print("ResidentialHeatPumpEnv initialized.")
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    # Run one day (24 hours) with random actions
    print("\n=== Running 24-hour episode with random policy ===")
    total_reward = 0.0
    for step in range(24):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 6 == 0:
            print(
                f"Hour {step:2d}: T_room={info['T_room']:.1f}°C, "
                f"P_elec={info['P_elec_w']/1000:.2f}kW, "
                f"Occ={info['occupancy']}, reward={reward:.3f}"
            )

        if terminated or truncated:
            break

    print(f"\nTotal 24-hour energy: {info['cumulative_energy_kwh']:.2f} kWh")
    print(f"Total cost: €{info['cumulative_cost']:.2f}")
    print(f"Comfort violations: {info['comfort_violations']}")

    # Test baseline controller
    print("\n=== Running 24-hour episode with On-Off controller ===")
    obs, info = env.reset()
    controller = OnOffController(setpoint=21.0, deadband=2.0)
    total_reward = 0.0

    for step in range(24):
        action = [controller.compute_action(obs)]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 6 == 0:
            print(
                f"Hour {step:2d}: T_room={info['T_room']:.1f}°C, "
                f"P_elec={info['P_elec_w']/1000:.2f}kW, "
                f"reward={reward:.3f}"
            )

        if terminated or truncated:
            break

    print(f"\nTotal 24-hour energy: {info['cumulative_energy_kwh']:.2f} kWh")
    print(f"Total cost: €{info['cumulative_cost']:.2f}")
    print(f"Comfort violations: {info['comfort_violations']}")


if __name__ == "__main__":
    main()