from heatpump_rl_env import SimpleThermalEnv
import numpy as np


def main():
    env = SimpleThermalEnv()
    obs, info = env.reset()
    print("Initial observation:", obs)

    total_reward = 0.0
    for step in range(10):
        # Simple random policy
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"step={step}, T_room={info['T_room']:.2f}, reward={reward:.3f}")

        if terminated or truncated:
            break

    print("Total reward:", total_reward)


if __name__ == "__main__":
    main()