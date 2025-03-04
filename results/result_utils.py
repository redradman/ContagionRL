import os
import json
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import sys
from tqdm import tqdm
import seaborn as sns

# Add the parent directory to the path so we can import environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import SIRSEnvironment

def load_config(model_path: str) -> Dict[str, Any]:
    """Load the configuration file associated with the model."""
    config_path = os.path.join(os.path.dirname(model_path), "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def create_env_from_config(env_config: Dict[str, Any], seed: Optional[int] = None) -> SIRSEnvironment:
    """Create a SIRS environment from a configuration dictionary."""
    env = SIRSEnvironment(**env_config)
    env.reset(seed=seed)
    return env

def run_episode(env: SIRSEnvironment, model=None, seed: Optional[int] = None) -> Tuple[List[float], int]:
    """
    Run a single episode using either a trained model or random actions.
    
    Args:
        env: The environment to run the episode in
        model: The trained model (if None, random actions will be used)
        seed: Optional seed for reproducibility
        
    Returns:
        Tuple of (cumulative_rewards_over_time, episode_length)
    """
    obs = env.reset(seed=seed)[0]
    done = False
    cumulative_rewards = [0]  # Start with 0 reward
    
    while not done:
        if model is not None:
            # Use the trained model
            action, _ = model.predict(obs, deterministic=True)
        else:
            # Use random actions
            action = env.action_space.sample()
        
        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Update cumulative reward
        cumulative_rewards.append(cumulative_rewards[-1] + reward)
        
        done = terminated or truncated
    
    return cumulative_rewards, len(cumulative_rewards) - 1  # -1 because we start with 0

def run_benchmark(
    model_path: str, 
    n_runs: int, 
    include_random: bool = True,
    random_seed: Optional[int] = 42
) -> Dict[str, Any]:
    """
    Run a benchmark comparing a trained model to random actions.
    
    Args:
        model_path: Path to the trained model
        n_runs: Number of runs to evaluate
        include_random: Whether to include a random agent benchmark
        random_seed: Base seed for reproducibility
        
    Returns:
        Dictionary with benchmark results
    """
    # Load model config
    config = load_config(model_path)
    env_config = config["environment"]
    
    # Set render mode to None for faster execution
    env_config["render_mode"] = None
    
    # Create environment
    env = create_env_from_config(env_config, seed=random_seed)
    
    # Load the trained model
    model = PPO.load(model_path)
    
    # Run trained model episodes
    trained_rewards_over_time = []
    trained_episode_lengths = []
    
    print("Running episodes with trained model...")
    for i in tqdm(range(n_runs), desc="Trained Model Episodes", unit="episode"):
        # Use different seeds for each run
        seed = random_seed + i if random_seed is not None else None
        rewards, episode_length = run_episode(env, model, seed=seed)
        trained_rewards_over_time.append(rewards)
        trained_episode_lengths.append(episode_length)
    
    # Run random action episodes if requested
    random_rewards_over_time = []
    random_episode_lengths = []
    
    if include_random:
        print("Running episodes with random actions...")
        for i in tqdm(range(n_runs), desc="Random Action Episodes", unit="episode"):
            # Use different seeds for each run
            seed = random_seed + n_runs + i if random_seed is not None else None
            rewards, episode_length = run_episode(env, None, seed=seed)
            random_rewards_over_time.append(rewards)
            random_episode_lengths.append(episode_length)
    
    # Close environment
    env.close()
    
    # Return results
    results = {
        "trained": {
            "rewards_over_time": trained_rewards_over_time,
            "episode_lengths": trained_episode_lengths
        }
    }
    
    if include_random:
        results["random"] = {
            "rewards_over_time": random_rewards_over_time,
            "episode_lengths": random_episode_lengths
        }
        
    results["config"] = config
    return results

def align_and_pad_rewards(rewards_list: List[List[float]], max_length: Optional[int] = None) -> np.ndarray:
    """
    Align rewards to the same length by padding shorter episodes.
    
    Args:
        rewards_list: List of cumulative reward lists from different episodes
        max_length: Maximum length to pad to (if None, use the longest episode)
        
    Returns:
        Numpy array of shape (n_episodes, max_length) with aligned rewards
    """
    if max_length is None:
        max_length = max(len(rewards) for rewards in rewards_list)
    
    # Pad shorter episodes by repeating the last value
    padded_rewards = []
    for rewards in rewards_list:
        if len(rewards) < max_length:
            # Pad with the last reward value (episode ended)
            padded = rewards + [rewards[-1]] * (max_length - len(rewards))
        else:
            # Truncate if somehow longer
            padded = rewards[:max_length]
        padded_rewards.append(padded)
    
    return np.array(padded_rewards)

def plot_cumulative_rewards(
    results: Dict[str, Any],
    title: str = "Cumulative Reward Over Time",
    filename: str = "cumulative_rewards.png",
    save_dir: str = "results/graphs",
    show_std: bool = True,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot cumulative rewards over time for trained and random agents.
    
    Args:
        results: Results dictionary from run_benchmark
        title: Plot title
        filename: Filename to save the plot as
        save_dir: Directory to save the plot in
        show_std: Whether to show standard deviation bands
        figsize: Figure size in inches
    """
    plt.figure(figsize=figsize)
    
    # Get the maximum length across all episodes
    all_rewards = results["trained"]["rewards_over_time"]
    if "random" in results:
        all_rewards += results["random"]["rewards_over_time"]
    max_length = max(len(rewards) for rewards in all_rewards)
    
    # Time steps for x-axis
    time_steps = np.arange(max_length)
    
    # Process trained model results
    trained_rewards = align_and_pad_rewards(results["trained"]["rewards_over_time"], max_length)
    trained_mean = np.mean(trained_rewards, axis=0)
    trained_std = np.std(trained_rewards, axis=0)
    
    # Plot trained model results
    plt.plot(time_steps, trained_mean, label="Trained Model", color="#1f77b4", linewidth=2)
    if show_std:
        plt.fill_between(
            time_steps, 
            trained_mean - trained_std, 
            trained_mean + trained_std, 
            alpha=0.2, 
            color="#1f77b4"
        )
    
    # Process random model results if available
    if "random" in results:
        random_rewards = align_and_pad_rewards(results["random"]["rewards_over_time"], max_length)
        random_mean = np.mean(random_rewards, axis=0)
        random_std = np.std(random_rewards, axis=0)
        
        # Plot random model results
        plt.plot(time_steps, random_mean, label="Random Actions", color="#ff7f0e", linewidth=2)
        if show_std:
            plt.fill_between(
                time_steps, 
                random_mean - random_std, 
                random_mean + random_std, 
                alpha=0.2, 
                color="#ff7f0e"
            )
    
    # Set up plot styling
    plt.grid(True, alpha=0.3)
    plt.title(title, fontsize=14)
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Cumulative Reward", fontsize=12)
    plt.legend(fontsize=10)
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()

def plot_survival_boxplot(
    results: Dict[str, Any],
    title: str = "Episode Duration Comparison",
    filename: str = "survival_boxplot.png",
    save_dir: str = "results/graphs",
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Create a boxplot comparing the episode durations (time survived) between agents.
    
    Args:
        results: Results dictionary from run_benchmark
        title: Plot title
        filename: Filename to save the plot as
        save_dir: Directory to save the plot in
        figsize: Figure size in inches
    """
    # Extract episode lengths for trained model
    trained_lengths = results["trained"]["episode_lengths"]
    
    # Prepare data for boxplot
    data = []
    categories = []
    
    # Add trained model data
    for length in trained_lengths:
        data.append(length)
        categories.append("Trained Model")
    
    # Add random agent data if available
    if "random" in results:
        random_lengths = results["random"]["episode_lengths"]
        for length in random_lengths:
            data.append(length)
            categories.append("Random Actions")
    
    # Create DataFrame for seaborn
    df = pd.DataFrame({
        "Agent Type": categories,
        "Episode Duration (steps)": data
    })
    
    # Set up the figure with higher DPI for better quality
    plt.figure(figsize=figsize, dpi=120)
    
    # Set clean style
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })
    
    # Create a clean boxplot with minimal design
    ax = sns.boxplot(
        x="Agent Type", 
        y="Episode Duration (steps)", 
        data=df,
        width=0.5,
        color='white',
        fliersize=3
    )
    
    # Add individual data points with minimal jitter and larger size
    sns.stripplot(
        x="Agent Type", 
        y="Episode Duration (steps)", 
        data=df,
        color='black',
        size=7,  # Increased from 3 to 7
        alpha=0.6,  # Increased from 0.4 to 0.6 for better visibility
        jitter=True
    )
    
    # Remove colors and simplify
    for i, box in enumerate(ax.artists):
        box.set_edgecolor('black')
        
        # Each box has 6 associated Line2D objects (whiskers, caps, and median)
        for j in range(6*i, 6*(i+1)):
            ax.lines[j].set_color('black')
    
    # Customize appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Add a discrete grid on the y-axis only
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.xaxis.grid(False)
    
    # Customize plot
    plt.title(title, fontsize=14, pad=10)
    plt.xlabel("")  # Remove x-label as it's redundant
    plt.ylabel("Episode Duration (steps)", fontsize=12, labelpad=10)
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def save_benchmark_results(
    results: Dict[str, Any],
    filename: str = "benchmark_results.json",
    save_dir: str = "results/graphs"
) -> None:
    """
    Save benchmark results to a JSON file.
    
    Args:
        results: Results dictionary from run_benchmark
        filename: Filename to save as
        save_dir: Directory to save in
    """
    # Create a copy of results that's JSON serializable
    serializable_results = {}
    
    # Handle trained model results
    serializable_results["trained"] = {
        "mean_episode_length": np.mean(results["trained"]["episode_lengths"]).item(),
        "std_episode_length": np.std(results["trained"]["episode_lengths"]).item(),
        "mean_final_reward": np.mean([rewards[-1] for rewards in results["trained"]["rewards_over_time"]]).item(),
        "std_final_reward": np.std([rewards[-1] for rewards in results["trained"]["rewards_over_time"]]).item()
    }
    
    # Handle random model results if available
    if "random" in results:
        serializable_results["random"] = {
            "mean_episode_length": np.mean(results["random"]["episode_lengths"]).item(),
            "std_episode_length": np.std(results["random"]["episode_lengths"]).item(),
            "mean_final_reward": np.mean([rewards[-1] for rewards in results["random"]["rewards_over_time"]]).item(),
            "std_final_reward": np.std([rewards[-1] for rewards in results["random"]["rewards_over_time"]]).item()
        }
    
    # Add config info
    serializable_results["environment"] = {
        "grid_size": results["config"]["environment"]["grid_size"],
        "n_humans": results["config"]["environment"]["n_humans"],
        "n_infected": results["config"]["environment"]["n_infected"],
        "simulation_time": results["config"]["environment"]["simulation_time"],
        "reward_type": results["config"]["environment"]["reward_type"]
    }
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save to JSON file
    with open(os.path.join(save_dir, filename), 'w') as f:
        json.dump(serializable_results, f, indent=4)

def get_summary_stats(results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Calculate summary statistics from benchmark results.
    
    Args:
        results: Results dictionary from run_benchmark
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {}
    
    # Calculate stats for trained model
    summary["trained"] = {
        "mean_episode_length": float(np.mean(results["trained"]["episode_lengths"])),
        "std_episode_length": float(np.std(results["trained"]["episode_lengths"])),
        "mean_final_reward": float(np.mean([rewards[-1] for rewards in results["trained"]["rewards_over_time"]])),
        "std_final_reward": float(np.std([rewards[-1] for rewards in results["trained"]["rewards_over_time"]]))
    }
    
    # Calculate stats for random agent if available
    if "random" in results:
        summary["random"] = {
            "mean_episode_length": float(np.mean(results["random"]["episode_lengths"])),
            "std_episode_length": float(np.std(results["random"]["episode_lengths"])),
            "mean_final_reward": float(np.mean([rewards[-1] for rewards in results["random"]["rewards_over_time"]])),
            "std_final_reward": float(np.std([rewards[-1] for rewards in results["random"]["rewards_over_time"]]))
        }
    
    return summary 