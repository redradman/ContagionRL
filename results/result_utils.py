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
from utils import Human, STATE_DICT

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
            # Use random actions - sample using environment's RNG for consistency
            # This ensures that the random actions are generated with the same RNG state
            # that is used for environment dynamics
            action = np.array([
                env.np_random.uniform(-1, 1),  # delta_x
                env.np_random.uniform(-1, 1),  # delta_y
                env.np_random.uniform(0, 1)    # adherence
            ], dtype=np.float32)
        
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
            seed = random_seed + i if random_seed is not None else None
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
    Saves two versions: one with individual data points and one without.
    
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
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Set clean style for both plots
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
    
    # FIRST PLOT: Boxplot with individual data points
    plt.figure(figsize=figsize, dpi=120)
    
    # Create a clean boxplot with minimal design
    ax1 = sns.boxplot(
        x="Agent Type", 
        y="Episode Duration (steps)", 
        data=df,
        width=0.5,
        color='white',
        fliersize=3
    )
    
    # Add individual data points with jitter
    sns.stripplot(
        x="Agent Type", 
        y="Episode Duration (steps)", 
        data=df,
        color='black',
        size=5,  
        alpha=0.5,
        jitter=True
    )
    
    # Remove colors and simplify
    for i, box in enumerate(ax1.artists):
        box.set_edgecolor('black')
        
        # Each box has 6 associated Line2D objects (whiskers, caps, and median)
        for j in range(6*i, 6*(i+1)):
            ax1.lines[j].set_color('black')
    
    # Customize appearance
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(0.5)
    ax1.spines['bottom'].set_linewidth(0.5)
    
    # Add a discrete grid on the y-axis only
    ax1.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax1.xaxis.grid(False)
    
    # Customize plot
    plt.title(title, fontsize=14, pad=10)
    plt.xlabel("")  # Remove x-label as it's redundant
    plt.ylabel("Episode Duration (steps)", fontsize=12, labelpad=10)
    
    # Save the figure with points
    plt.tight_layout()
    points_filename = os.path.splitext(filename)[0] + "_with_points" + os.path.splitext(filename)[1]
    plt.savefig(os.path.join(save_dir, points_filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    # SECOND PLOT: Boxplot without individual data points
    plt.figure(figsize=figsize, dpi=120)
    
    # Create a clean boxplot with minimal design (without stripplot)
    ax2 = sns.boxplot(
        x="Agent Type", 
        y="Episode Duration (steps)", 
        data=df,
        width=0.5,
        color='white',
        fliersize=3
    )
    
    # Remove colors and simplify
    for i, box in enumerate(ax2.artists):
        box.set_edgecolor('black')
        
        # Each box has 6 associated Line2D objects (whiskers, caps, and median)
        for j in range(6*i, 6*(i+1)):
            ax2.lines[j].set_color('black')
    
    # Customize appearance
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(0.5)
    ax2.spines['bottom'].set_linewidth(0.5)
    
    # Add a discrete grid on the y-axis only
    ax2.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax2.xaxis.grid(False)
    
    # Customize plot
    plt.title(title, fontsize=14, pad=10)
    plt.xlabel("")  # Remove x-label as it's redundant
    plt.ylabel("Episode Duration (steps)", fontsize=12, labelpad=10)
    
    # Save the figure without points
    plt.tight_layout()
    no_points_filename = os.path.splitext(filename)[0] + "_no_points" + os.path.splitext(filename)[1]
    plt.savefig(os.path.join(save_dir, no_points_filename), dpi=300, bbox_inches='tight')
    plt.close()

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
    # Flatten all rewards across all episodes for overall statistics
    all_trained_rewards = [reward for rewards_list in results["trained"]["rewards_over_time"] for reward in rewards_list]
    
    # Get cumulative rewards for statistical analysis
    # Note: rewards_over_time already contains cumulative rewards
    trained_cumulative_rewards = align_and_pad_rewards(results["trained"]["rewards_over_time"])
    trained_mean_cumulative = np.mean(trained_cumulative_rewards, axis=0)
    trained_std_cumulative = np.std(trained_cumulative_rewards, axis=0, ddof=1)
    
    # Calculate average of mean cumulative and average of std cumulative
    avg_mean_cumulative = float(np.mean(trained_mean_cumulative))
    avg_std_cumulative = float(np.mean(trained_std_cumulative))
    
    summary["trained"] = {
        "mean_episode_length": float(np.mean(results["trained"]["episode_lengths"])),
        "std_episode_length": float(np.std(results["trained"]["episode_lengths"], ddof=1)),
        "mean_reward": float(np.mean(all_trained_rewards)),
        "std_reward": float(np.std(all_trained_rewards, ddof=1)),
        "mean_cumulative_reward": avg_mean_cumulative,
        "std_cumulative_reward": avg_std_cumulative,
        "mean_final_reward": float(np.mean([rewards[-1] for rewards in results["trained"]["rewards_over_time"]])),
        "std_final_reward": float(np.std([rewards[-1] for rewards in results["trained"]["rewards_over_time"]], ddof=1))
    }
    
    # Calculate stats for random agent if available
    if "random" in results:
        # Flatten all rewards across all episodes for overall statistics
        all_random_rewards = [reward for rewards_list in results["random"]["rewards_over_time"] for reward in rewards_list]
        
        # Get cumulative rewards for statistical analysis
        random_cumulative_rewards = align_and_pad_rewards(results["random"]["rewards_over_time"])
        random_mean_cumulative = np.mean(random_cumulative_rewards, axis=0)
        random_std_cumulative = np.std(random_cumulative_rewards, axis=0, ddof=1)
        
        # Calculate average of mean cumulative and average of std cumulative
        avg_mean_cumulative = float(np.mean(random_mean_cumulative))
        avg_std_cumulative = float(np.mean(random_std_cumulative))
        
        summary["random"] = {
            "mean_episode_length": float(np.mean(results["random"]["episode_lengths"])),
            "std_episode_length": float(np.std(results["random"]["episode_lengths"], ddof=1)),
            "mean_reward": float(np.mean(all_random_rewards)),
            "std_reward": float(np.std(all_random_rewards, ddof=1)),
            "mean_cumulative_reward": avg_mean_cumulative,
            "std_cumulative_reward": avg_std_cumulative,
            "mean_final_reward": float(np.mean([rewards[-1] for rewards in results["random"]["rewards_over_time"]])),
            "std_final_reward": float(np.std([rewards[-1] for rewards in results["random"]["rewards_over_time"]], ddof=1))
        }
    
    return summary 

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
    
    # Flatten all rewards across all episodes for trained model
    all_trained_rewards = [reward for rewards_list in results["trained"]["rewards_over_time"] for reward in rewards_list]
    
    # Get cumulative rewards for statistical analysis
    trained_cumulative_rewards = align_and_pad_rewards(results["trained"]["rewards_over_time"])
    trained_mean_cumulative = np.mean(trained_cumulative_rewards, axis=0)
    trained_std_cumulative = np.std(trained_cumulative_rewards, axis=0, ddof=1)
    
    # Calculate average of mean cumulative and average of std cumulative
    avg_mean_cumulative = float(np.mean(trained_mean_cumulative))
    avg_std_cumulative = float(np.mean(trained_std_cumulative))
    
    # Handle trained model results
    serializable_results["trained"] = {
        "mean_episode_length": np.mean(results["trained"]["episode_lengths"]).item(),
        "std_episode_length": np.std(results["trained"]["episode_lengths"], ddof=1).item(),
        "mean_reward": np.mean(all_trained_rewards).item(),
        "std_reward": np.std(all_trained_rewards, ddof=1).item(),
        "mean_cumulative_reward": avg_mean_cumulative,
        "std_cumulative_reward": avg_std_cumulative,
        # Keep the final reward stats for backward compatibility
        "mean_final_reward": np.mean([rewards[-1] for rewards in results["trained"]["rewards_over_time"]]).item(),
        "std_final_reward": np.std([rewards[-1] for rewards in results["trained"]["rewards_over_time"]], ddof=1).item()
    }
    
    # Handle random model results if available
    if "random" in results:
        # Flatten all rewards across all episodes for random model
        all_random_rewards = [reward for rewards_list in results["random"]["rewards_over_time"] for reward in rewards_list]
        
        # Get cumulative rewards for statistical analysis
        random_cumulative_rewards = align_and_pad_rewards(results["random"]["rewards_over_time"])
        random_mean_cumulative = np.mean(random_cumulative_rewards, axis=0)
        random_std_cumulative = np.std(random_cumulative_rewards, axis=0, ddof=1)
        
        # Calculate average of mean cumulative and average of std cumulative
        avg_mean_cumulative = float(np.mean(random_mean_cumulative))
        avg_std_cumulative = float(np.mean(random_std_cumulative))
        
        serializable_results["random"] = {
            "mean_episode_length": np.mean(results["random"]["episode_lengths"]).item(),
            "std_episode_length": np.std(results["random"]["episode_lengths"], ddof=1).item(),
            "mean_reward": np.mean(all_random_rewards).item(),
            "std_reward": np.std(all_random_rewards, ddof=1).item(),
            "mean_cumulative_reward": avg_mean_cumulative,
            "std_cumulative_reward": avg_std_cumulative,
            # Keep the final reward stats for backward compatibility
            "mean_final_reward": np.mean([rewards[-1] for rewards in results["random"]["rewards_over_time"]]).item(),
            "std_final_reward": np.std([rewards[-1] for rewards in results["random"]["rewards_over_time"]], ddof=1).item()
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

def run_exposure_adherence_benchmark(
    model_path: str, 
    n_runs: int, 
    include_random: bool = True,
    random_seed: Optional[int] = 42
) -> Dict[str, Any]:
    """
    Run a benchmark collecting exposure and adherence data for each step.
    
    Args:
        model_path: Path to the trained model
        n_runs: Number of runs to evaluate
        include_random: Whether to include a random agent benchmark
        random_seed: Base seed for reproducibility
        
    Returns:
        Dictionary with benchmark results containing exposure and adherence data
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
    trained_exposure_data = []
    trained_adherence_data = []
    
    print("Running episodes with trained model...")
    for i in tqdm(range(n_runs), desc="Trained Model Episodes", unit="episode"):
        # Use different seeds for each run
        seed = random_seed + i if random_seed is not None else None
        # Reset environment
        obs = env.reset(seed=seed)[0]
        done = False
        
        # Create a list to store data for this episode
        episode_exposures = []
        episode_adherences = []
        
        while not done:
            # Use the trained model
            action, _ = model.predict(obs, deterministic=True)
            
            # Get the agent's adherence from the action
            adherence = action[2]  # Assuming action[2] is the adherence value
            
            # Get agent exposure before taking the step
            # Create a temporary Human object for the agent
            agent_human = None
            if env.agent_state == 0:  # If agent is susceptible
                agent_human = Human(
                    x=env.agent_position[0],
                    y=env.agent_position[1],
                    state=env.agent_state,
                    id=-1
                )
                exposure = env._calculate_total_exposure(agent_human)
            else:
                exposure = 0.0  # Not susceptible, so no exposure
                
            # Store data
            episode_exposures.append(exposure)
            episode_adherences.append(adherence)
            
            # Take a step in the environment
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        
        # Save episode data
        trained_exposure_data.append(episode_exposures)
        trained_adherence_data.append(episode_adherences)
    
    # Run random action episodes if requested
    random_exposure_data = []
    random_adherence_data = []
    
    if include_random:
        print("Running episodes with random actions...")
        for i in tqdm(range(n_runs), desc="Random Action Episodes", unit="episode"):
            # Use different seeds for each run
            seed = random_seed + i if random_seed is not None else None
            # Reset environment
            obs = env.reset(seed=seed)[0]
            done = False
            
            # Create a list to store data for this episode
            episode_exposures = []
            episode_adherences = []
            
            while not done:
                # Use random actions
                action = np.array([
                    env.np_random.uniform(-1, 1),  # delta_x
                    env.np_random.uniform(-1, 1),  # delta_y
                    env.np_random.uniform(0, 1)    # adherence
                ], dtype=np.float32)
                
                # Get the agent's adherence from the action
                adherence = action[2]  # Random adherence value
                
                # Get agent exposure before taking the step
                # Create a temporary Human object for the agent
                agent_human = None
                if env.agent_state == 0:  # If agent is susceptible
                    agent_human = Human(
                        x=env.agent_position[0],
                        y=env.agent_position[1],
                        state=env.agent_state,
                        id=-1
                    )
                    exposure = env._calculate_total_exposure(agent_human)
                else:
                    exposure = 0.0  # Not susceptible, so no exposure
                    
                # Store data
                episode_exposures.append(exposure)
                episode_adherences.append(adherence)
                
                # Take a step in the environment
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            
            # Save episode data
            random_exposure_data.append(episode_exposures)
            random_adherence_data.append(episode_adherences)
    
    # Close environment
    env.close()
    
    # Return results
    results = {
        "trained": {
            "exposure_data": trained_exposure_data,
            "adherence_data": trained_adherence_data
        }
    }
    
    if include_random:
        results["random"] = {
            "exposure_data": random_exposure_data,
            "adherence_data": random_adherence_data
        }
    
    results["config"] = config
    return results

def plot_exposure_adherence_scatterplot(
    results: Dict[str, Any],
    title: str = "Exposure vs. Adherence Comparison",
    filename: str = "exposure_adherence_scatterplot.png",
    save_dir: str = "results/graphs",
    figsize: Tuple[int, int] = (16, 8),
    alpha: float = 0.4,
    include_random: bool = True,
    jitter_amount: float = 0.05
) -> None:
    """
    Create a scatterplot comparing total exposure vs agent adherence for
    trained and random models.
    
    Args:
        results: Results dictionary from run_exposure_adherence_benchmark
        title: Overall plot title
        filename: Filename to save the plot as
        save_dir: Directory to save the plot in
        figsize: Figure size in inches
        alpha: Transparency level for scatter points
        include_random: Whether to include random model data
        jitter_amount: Amount of jitter to add to adherence values for better visibility
    """
    # Create figure with gridspec for main plots and histograms
    fig = plt.figure(figsize=figsize)
    
    if include_random:
        # Adjust the height ratios to accommodate titles and increase spacing between plots
        gs = fig.add_gridspec(3, 4, height_ratios=[0.5, 1, 4], width_ratios=[4, 1, 4, 1],
                            hspace=0.05, wspace=0.3)  # Increased wspace from 0.05 to 0.3
        
        # Title areas
        ax_title1 = fig.add_subplot(gs[0, 0])
        ax_title2 = fig.add_subplot(gs[0, 2])
        
        # Main scatter plots
        ax_scatter1 = fig.add_subplot(gs[2, 0])
        ax_scatter2 = fig.add_subplot(gs[2, 2])
        
        # Histogram for x-axis (middle row)
        ax_histx1 = fig.add_subplot(gs[1, 0], sharex=ax_scatter1)
        ax_histx2 = fig.add_subplot(gs[1, 2], sharex=ax_scatter2)
        
        # Histogram for y-axis (right)
        ax_histy1 = fig.add_subplot(gs[2, 1], sharey=ax_scatter1)
        ax_histy2 = fig.add_subplot(gs[2, 3], sharey=ax_scatter2)
        
        # Turn off axis labels for histograms and title areas
        ax_histx1.tick_params(axis="x", labelbottom=False)
        ax_histx2.tick_params(axis="x", labelbottom=False)
        ax_histy1.tick_params(axis="y", labelleft=False)
        ax_histy2.tick_params(axis="y", labelleft=False)
        
        # Turn off all spines and ticks for title areas
        for ax in [ax_title1, ax_title2]:
            ax.axis('off')
        
        # Set the titles in the title areas
        ax_title1.text(0.5, 0.5, "Trained Model", fontsize=14, ha='center', va='center')
        ax_title2.text(0.5, 0.5, "Random Actions", fontsize=14, ha='center', va='center')
        
        axes_scatter = [ax_scatter1, ax_scatter2]
        axes_histx = [ax_histx1, ax_histx2]
        axes_histy = [ax_histy1, ax_histy2]
    else:
        # Adjust the height ratios to accommodate title
        gs = fig.add_gridspec(3, 2, height_ratios=[0.5, 1, 4], width_ratios=[4, 1],
                           hspace=0.05, wspace=0.15)  # Slightly increased wspace
        
        # Title area
        ax_title = fig.add_subplot(gs[0, 0])
        
        # Main scatter plot
        ax_scatter = fig.add_subplot(gs[2, 0])
        
        # Histogram for x-axis (middle row)
        ax_histx = fig.add_subplot(gs[1, 0], sharex=ax_scatter)
        
        # Histogram for y-axis (right)
        ax_histy = fig.add_subplot(gs[2, 1], sharey=ax_scatter)
        
        # Turn off axis labels for histograms
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)
        
        # Turn off all spines and ticks for title area
        ax_title.axis('off')
        
        # Set the title in the title area
        ax_title.text(0.5, 0.5, "Trained Model", fontsize=14, ha='center', va='center')
        
        axes_scatter = [ax_scatter]
        axes_histx = [ax_histx]
        axes_histy = [ax_histy]
    
    # Plot trained model data
    trained_exposures = [item for sublist in results["trained"]["exposure_data"] for item in sublist]
    trained_adherences = [item for sublist in results["trained"]["adherence_data"] for item in sublist]
    
    # Add jitter to adherence values to better visualize points at 0 and 1
    trained_adherences_jittered = [a + np.random.uniform(-jitter_amount, jitter_amount) for a in trained_adherences]
    
    # Plot scatter for trained model
    axes_scatter[0].scatter(trained_exposures, trained_adherences_jittered, alpha=alpha, c='blue', s=20)
    # Remove the title from scatter plot since we now have a dedicated title area
    axes_scatter[0].set_xlabel("Total Exposure", fontsize=12)
    axes_scatter[0].set_ylabel("Agent Adherence", fontsize=12)
    axes_scatter[0].grid(True, alpha=0.3)
    
    # Slightly expand y-axis to show points at 0 and 1 more clearly
    axes_scatter[0].set_ylim(-0.05, 1.05)
    
    # Set x limits based on exposure range
    x_max = max(trained_exposures) * 1.1 if trained_exposures else 1.0
    axes_scatter[0].set_xlim(0, x_max)
    
    # Plot histograms for trained model
    # Use more bins for a finer-grained view of the distribution
    bin_count = 30
    
    # X-axis histogram
    axes_histx[0].hist(trained_exposures, bins=bin_count, alpha=0.7, color='blue')
    axes_histx[0].set_ylabel('Count', fontsize=10)
    
    # Y-axis histogram - rotated for horizontal orientation
    axes_histy[0].hist(trained_adherences, bins=bin_count, alpha=0.7, color='blue', 
                     orientation='horizontal')
    axes_histy[0].set_xlabel('Count', fontsize=10)
    
    # Add random model data if requested
    if include_random and "random" in results:
        random_exposures = [item for sublist in results["random"]["exposure_data"] for item in sublist]
        random_adherences = [item for sublist in results["random"]["adherence_data"] for item in sublist]
        
        # Add jitter to adherence values
        random_adherences_jittered = [a + np.random.uniform(-jitter_amount, jitter_amount) for a in random_adherences]
        
        # Plot scatter for random model
        axes_scatter[1].scatter(random_exposures, random_adherences_jittered, alpha=alpha, c='orange', s=20)
        # Remove the title from scatter plot since we now have a dedicated title area
        axes_scatter[1].set_xlabel("Total Exposure", fontsize=12)
        axes_scatter[1].set_ylabel("Agent Adherence", fontsize=12)
        axes_scatter[1].grid(True, alpha=0.3)
        
        # Slightly expand y-axis to show points at 0 and 1 more clearly
        axes_scatter[1].set_ylim(-0.05, 1.05)
        
        # Set x limits based on exposure range
        x_max_random = max(random_exposures) * 1.1 if random_exposures else 1.0
        axes_scatter[1].set_xlim(0, x_max_random)
        
        # Plot histograms for random model
        axes_histx[1].hist(random_exposures, bins=bin_count, alpha=0.7, color='orange')
        axes_histx[1].set_ylabel('Count', fontsize=10)
        
        axes_histy[1].hist(random_adherences, bins=bin_count, alpha=0.7, color='orange',
                         orientation='horizontal')
        axes_histy[1].set_xlabel('Count', fontsize=10)
    
    # Set overall title
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Use figure-level tight layout but with padding to prevent overlap
    plt.tight_layout(pad=2.0, h_pad=0.5, w_pad=1.0)
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close() 