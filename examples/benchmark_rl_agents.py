from heatpump_rl_env import ResidentialHeatPumpEnv, OnOffController, PIDControllerBaseline, RuleBasedController
from heatpump_rl_env.benchmarks.agent_runners import DQNRunner, SACRunner, PPORunner
from heatpump_rl_env.benchmarks.evaluation_metrics import EvaluationMetrics
import numpy as np


def run_baseline_controller(env, controller, controller_name: str, n_episodes: int = 3):
    """Evaluate a baseline controller over n episodes."""
    metrics = EvaluationMetrics()
    print(f"\n{'='*60}")
    print(f"Evaluating {controller_name} (baseline)")
    print(f"{'='*60}")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        while True:
            action = [controller.compute_action(obs)]
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                metrics.record_episode(info)
                print(f"  Episode {ep+1}/{n_episodes}: "
                      f"Energy={info['cumulative_energy_kwh']:.1f} kWh, "
                      f"Cost=€{info['cumulative_cost']:.2f}, "
                      f"Comfort violations={int(info['comfort_violations'])}")
                break

    metrics.print_summary(controller_name)
    return metrics.summary()


def run_rl_agent(env, agent_runner, n_train_steps: int = 50000, n_eval_episodes: int = 3):
    """Train and evaluate an RL agent."""
    print(f"\n{'='*60}")
    print(f"Training {agent_runner.agent_name} agent...")
    print(f"{'='*60}")

    agent_runner.train(total_timesteps=n_train_steps, verbose=0)
    print(f"Training complete.")

    print(f"\nEvaluating {agent_runner.agent_name} over {n_eval_episodes} episodes...")
    eval_result = agent_runner.evaluate(n_episodes=n_eval_episodes)

    print(f"  Mean reward: {eval_result['mean_reward']:.3f}")
    print(f"  Mean energy: {eval_result['mean_energy_kwh']:.1f} kWh")

    return eval_result


def main():
    # Initialize environment
    env = ResidentialHeatPumpEnv(config={"max_episode_steps": 8760})

    # Dictionary to store all results
    results = {}

    # ===== BASELINE CONTROLLERS =====
    print("\n" + "="*60)
    print("PHASE 1: BASELINE CONTROLLERS")
    print("="*60)

    # On-Off controller
    on_off = OnOffController(setpoint=21.0, deadband=2.0)
    results["OnOffController"] = run_baseline_controller(env, on_off, "OnOffController", n_episodes=2)

    # PID controller
    pid = PIDControllerBaseline(setpoint=21.0, kp=0.1, ki=0.01, kd=0.05)
    results["PIDController"] = run_baseline_controller(env, pid, "PIDController", n_episodes=2)

    # Rule-based controller
    rule = RuleBasedController(setpoint=21.0)
    results["RuleBasedController"] = run_baseline_controller(env, rule, "RuleBasedController", n_episodes=2)

    # ===== RL AGENTS =====
    print("\n" + "="*60)
    print("PHASE 2: RL AGENTS")
    print("="*60)

    # DQN
    print("\n>>> Training DQN (discrete action approximation)...")
    dqn_runner = DQNRunner(env)
    dqn_result = run_rl_agent(env, dqn_runner, n_train_steps=5000, n_eval_episodes=2)
    results["DQN"] = dqn_result

    # SAC
    print("\n>>> Training SAC (Soft Actor-Critic - continuous)...")
    sac_runner = SACRunner(env)
    sac_result = run_rl_agent(env, sac_runner, n_train_steps=5000, n_eval_episodes=2)
    results["SAC"] = sac_result

    # PPO
    print("\n>>> Training PPO (Proximal Policy Optimization - continuous)...")
    ppo_runner = PPORunner(env)
    ppo_result = run_rl_agent(env, ppo_runner, n_train_steps=5000, n_eval_episodes=2)
    results["PPO"] = ppo_result

    # ===== SUMMARY COMPARISON =====
    print("\n" + "="*60)
    print("FINAL BENCHMARK SUMMARY")
    print("="*60)
    print(f"{'Agent':<25} {'Energy (kWh)':<15} {'Reward':<15}")
    print("-" * 55)

    for agent_name, result in results.items():
        energy = result.get("mean_energy_kwh", result.get("mean_energy_kwh", "N/A"))
        reward = result.get("mean_reward", "N/A")
        print(f"{agent_name:<25} {energy:<15.1f} {reward:<15.3f}")

    print("="*60)

    # Recommendation
    print("\nRECOMMENDATION:")
    print("  - SAC and PPO are best for continuous control (this problem)")
    print("  - PPO is more stable and easier to tune")
    print("  - SAC may achieve better asymptotic performance (sample-efficient)")
    print("  - For production use: choose PPO (balance of performance & stability)")
    print("="*60)


if __name__ == "__main__":
    main()