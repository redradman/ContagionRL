#!/usr/bin/env python
"""
Quick test script to demonstrate the benchmark functionality without needing a trained model.
This creates a temporary random model and runs a quick benchmark to show how the plots look.
"""

import os
import sys
import tempfile
import json
import shutil
import datetime
from stable_baselines3 import PPO

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import SIRSEnvironment
from config import env_config, ppo_config
from result_utils import (
    run_benchmark, 
    plot_cumulative_rewards,
    plot_survival_boxplot,
    save_benchmark_results,
    get_summary_stats
)
def main():
    """
    Create a temporary model and run a quick benchmark to demonstrate functionality.
    """
    print("Creating a temporary model and running a quick benchmark demonstration...")
    
    # Create a temporary directory for the model
    temp_dir = tempfile.mkdtemp(prefix="contagion_demo_")
    model_dir = os.path.join(temp_dir, "test_model")
    os.makedirs(model_dir, exist_ok=True)
    
    # Create a test environment
    test_env_config = env_config.copy()
    test_env_config["render_mode"] = None
    test_env_config["simulation_time"] = 50  # Shorter episodes for quick testing
    env = SIRSEnvironment(**test_env_config)
    
    # Create a simple (untrained) model
    model = PPO(
        ppo_config["policy_type"],
        env,
        **{k: v for k, v in ppo_config.items() if k not in ["policy_type", "total_timesteps", "n_envs"]}
    )
    
    # Save the model
    model_path = os.path.join(model_dir, "test_model.zip")
    model.save(model_path)
    
    # Save the config
    config = {
        "environment": test_env_config,
        "ppo": ppo_config,
    }
    
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f)
    
    # Create graphs directory
    graphs_dir = "results/graphs"
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Run a quick benchmark (with fewer runs for speed)
    print(f"Running benchmark with model at {model_path}")
    results = run_benchmark(
        model_path=model_path,
        n_runs=3,  # Just a few runs for speed
        include_random=True,
        random_seed=42
    )
    
    # Generate timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot the cumulative rewards
    reward_filename = f"quick_test_rewards_{timestamp}.png"
    plot_cumulative_rewards(
        results,
        title="Quick Test: Untrained Model vs Random Actions",
        filename=reward_filename,
        save_dir=graphs_dir,
        show_std=True
    )
    
    # Plot the survival boxplot
    boxplot_filename = f"quick_test_survival_{timestamp}.png"
    plot_survival_boxplot(
        results,
        title="Quick Test: Episode Duration Comparison",
        filename=boxplot_filename,
        save_dir=graphs_dir
    )
    
    # Save results data
    data_filename = f"quick_test_data_{timestamp}.json"
    save_benchmark_results(
        results,
        filename=data_filename,
        save_dir=graphs_dir
    )
    
    # Print summary statistics
    stats = get_summary_stats(results)
    
    print("\nQuick Test Summary:")
    print("  Untrained Model (randomly initialized):")
    print(f"    Mean Episode Length: {stats['trained']['mean_episode_length']:.2f} steps (±{stats['trained']['std_episode_length']:.2f})")
    print(f"    Mean Final Reward: {stats['trained']['mean_final_reward']:.2f} (±{stats['trained']['std_final_reward']:.2f})")
    
    if "random" in stats:
        print("  Random Actions:")
        print(f"    Mean Episode Length: {stats['random']['mean_episode_length']:.2f} steps (±{stats['random']['std_episode_length']:.2f})")
        print(f"    Mean Final Reward: {stats['random']['mean_final_reward']:.2f} (±{stats['random']['std_final_reward']:.2f})")
    
    print(f"\nTest reward plot saved to {graphs_dir}/{reward_filename}")
    print(f"Test survival boxplot saved to {graphs_dir}/{boxplot_filename}")
    print(f"Test data saved to {graphs_dir}/{data_filename}")
    
    # Cleanup temporary directory
    env.close()
    shutil.rmtree(temp_dir)
    print(f"Cleaned up temporary model directory: {temp_dir}")

if __name__ == "__main__":
    main() 