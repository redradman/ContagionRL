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
from scipy import stats  # Add this import for statistical tests

# Add the parent directory to the path so we can import environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import SIRSEnvironment
from utils import Human

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
    Collects reward data, episode lengths, and exposure/adherence data.
    
    Args:
        model_path: Path to the trained model
        n_runs: Number of runs to evaluate
        include_random: Whether to include a random agent benchmark
        random_seed: Base seed for reproducibility
        
    Returns:
        Dictionary with benchmark results including exposure and adherence data
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
    trained_exposure_data = []
    trained_adherence_data = []
    
    print("Running episodes with trained model...")
    for i in tqdm(range(n_runs), desc="Trained Model Episodes", unit="episode"):
        # Use different seeds for each run
        seed = random_seed + i if random_seed is not None else None
        
        # Reset environment
        obs = env.reset(seed=seed)[0]
        done = False
        cumulative_rewards = [0]  # Start with 0 reward
        
        # Lists to store exposure and adherence data for this episode
        episode_exposures = []
        episode_adherences = []
        
        while not done:
            # Use the trained model
            action, _ = model.predict(obs, deterministic=True)
            
            # Get the agent's adherence from the action
            adherence = action[2]  # Assuming action[2] is the adherence value
            
            # Get agent exposure before taking the step (if susceptible)
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
            
            # Store exposure and adherence data
            episode_exposures.append(exposure)
            episode_adherences.append(adherence)
            
            # Take a step in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Update cumulative reward
            cumulative_rewards.append(cumulative_rewards[-1] + reward)
            
            done = terminated or truncated
        
        # Save episode data
        trained_rewards_over_time.append(cumulative_rewards)
        trained_episode_lengths.append(len(cumulative_rewards) - 1)  # -1 because we start with 0
        trained_exposure_data.append(episode_exposures)
        trained_adherence_data.append(episode_adherences)
    
    # Run random action episodes if requested
    random_rewards_over_time = []
    random_episode_lengths = []
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
            cumulative_rewards = [0]  # Start with 0 reward
            
            # Lists to store exposure and adherence data for this episode
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
                
                # Get agent exposure before taking the step (if susceptible)
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
                
                # Store exposure and adherence data
                episode_exposures.append(exposure)
                episode_adherences.append(adherence)
                
                # Take a step in the environment
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Update cumulative reward
                cumulative_rewards.append(cumulative_rewards[-1] + reward)
                
                done = terminated or truncated
            
            # Save episode data
            random_rewards_over_time.append(cumulative_rewards)
            random_episode_lengths.append(len(cumulative_rewards) - 1)  # -1 because we start with 0
            random_exposure_data.append(episode_exposures)
            random_adherence_data.append(episode_adherences)
    
    # Close environment
    env.close()
    
    # Return results with both reward and exposure/adherence data
    results = {
        "trained": {
            "rewards_over_time": trained_rewards_over_time,
            "episode_lengths": trained_episode_lengths,
            "exposure_data": trained_exposure_data,
            "adherence_data": trained_adherence_data
        }
    }
    
    if include_random:
        results["random"] = {
            "rewards_over_time": random_rewards_over_time,
            "episode_lengths": random_episode_lengths,
            "exposure_data": random_exposure_data,
            "adherence_data": random_adherence_data
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
    Includes Mann-Whitney U test p-value on the plot.
    
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
    random_lengths = []
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
    
    # Perform Mann-Whitney U test if both trained and random data are available
    if "random" in results and len(trained_lengths) > 0 and len(random_lengths) > 0:
        u_stat, p_value = stats.mannwhitneyu(trained_lengths, random_lengths, alternative='two-sided')
        
        # Add statistical test annotation
        significance = ''
        if p_value < 0.001:
            significance = '***'  # p < 0.001
        elif p_value < 0.01:
            significance = '**'   # p < 0.01
        elif p_value < 0.05:
            significance = '*'    # p < 0.05
            
        # Add the p-value text to the plot
        plt.text(0.5, 0.01, f'Mann-Whitney U Test: p = {p_value:.4f} {significance}', 
                 horizontalalignment='center', 
                 fontsize=10, 
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
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
    
    # Perform Mann-Whitney U test if both trained and random data are available
    if "random" in results and len(trained_lengths) > 0 and len(random_lengths) > 0:
        u_stat, p_value = stats.mannwhitneyu(trained_lengths, random_lengths, alternative='two-sided')
        
        # Add statistical test annotation
        significance = ''
        if p_value < 0.001:
            significance = '***'  # p < 0.001
        elif p_value < 0.01:
            significance = '**'   # p < 0.01
        elif p_value < 0.05:
            significance = '*'    # p < 0.05
            
        # Add the p-value text to the plot
        plt.text(0.5, 0.01, f'Mann-Whitney U Test: p = {p_value:.4f} {significance}', 
                 horizontalalignment='center', 
                 fontsize=10, 
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
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
    summary["trained"] = {
        "mean_episode_length": float(np.mean(results["trained"]["episode_lengths"])),
        "std_episode_length": float(np.std(results["trained"]["episode_lengths"], ddof=1)),
        "mean_final_reward": float(np.mean([rewards[-1] for rewards in results["trained"]["rewards_over_time"]])),
        "std_final_reward": float(np.std([rewards[-1] for rewards in results["trained"]["rewards_over_time"]], ddof=1))
    }
    
    # Calculate stats for random agent if available
    if "random" in results:
        summary["random"] = {
            "mean_episode_length": float(np.mean(results["random"]["episode_lengths"])),
            "std_episode_length": float(np.std(results["random"]["episode_lengths"], ddof=1)),
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
    Includes statistical test results for comparisons between trained and random models.
    
    Args:
        results: Results dictionary from run_benchmark
        filename: Filename to save as
        save_dir: Directory to save in
    """
    # Create a copy of results that's JSON serializable
    serializable_results = {}
    
    # Define conversion function for NumPy types
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return obj
    
    # Extract trained model data for statistical tests
    trained_lengths = results["trained"]["episode_lengths"]
    trained_final_rewards = [rewards[-1] for rewards in results["trained"]["rewards_over_time"]]
    
    # Handle trained model results
    serializable_results["trained"] = {
        "mean_episode_length": float(np.mean(results["trained"]["episode_lengths"])),
        "std_episode_length": float(np.std(results["trained"]["episode_lengths"], ddof=1)),
        "mean_final_reward": float(np.mean(trained_final_rewards)),
        "std_final_reward": float(np.std(trained_final_rewards, ddof=1))
    }
    
    # Handle random model results if available
    if "random" in results:
        # Extract random model data for statistical tests
        random_lengths = results["random"]["episode_lengths"]
        random_final_rewards = [rewards[-1] for rewards in results["random"]["rewards_over_time"]]
        
        serializable_results["random"] = {
            "mean_episode_length": float(np.mean(results["random"]["episode_lengths"])),
            "std_episode_length": float(np.std(results["random"]["episode_lengths"], ddof=1)),
            "mean_final_reward": float(np.mean(random_final_rewards)),
            "std_final_reward": float(np.std(random_final_rewards, ddof=1))
        }
        
        # Add statistical tests comparing trained vs. random
        serializable_results["statistical_tests"] = {}
        
        # Episode length comparison (Mann-Whitney U test)
        if len(trained_lengths) > 0 and len(random_lengths) > 0:
            u_stat, p_value = stats.mannwhitneyu(trained_lengths, random_lengths, alternative='two-sided')
            serializable_results["statistical_tests"]["episode_length"] = {
                "test": "Mann-Whitney U",
                "u_statistic": float(u_stat),
                "p_value": float(p_value),
                "significant_0.05": bool(p_value < 0.05),
                "significant_0.01": bool(p_value < 0.01),
                "significant_0.001": bool(p_value < 0.001)
            }
        
        # Final reward comparison (Mann-Whitney U test)
        if len(trained_final_rewards) > 0 and len(random_final_rewards) > 0:
            u_stat, p_value = stats.mannwhitneyu(trained_final_rewards, random_final_rewards, alternative='two-sided')
            serializable_results["statistical_tests"]["final_reward"] = {
                "test": "Mann-Whitney U",
                "u_statistic": float(u_stat),
                "p_value": float(p_value),
                "significant_0.05": bool(p_value < 0.05),
                "significant_0.01": bool(p_value < 0.01),
                "significant_0.001": bool(p_value < 0.001)
            }
    
    # Add config info
    serializable_results["environment"] = {
        "grid_size": int(results["config"]["environment"]["grid_size"]),
        "n_humans": int(results["config"]["environment"]["n_humans"]),
        "n_infected": int(results["config"]["environment"]["n_infected"]),
        "simulation_time": int(results["config"]["environment"]["simulation_time"]),
        "reward_type": str(results["config"]["environment"]["reward_type"])
    }
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save to JSON file with error handling
    try:
        with open(os.path.join(save_dir, filename), 'w') as f:
            json.dump(serializable_results, f, indent=4, sort_keys=False)
        print(f"Successfully saved benchmark results to {os.path.join(save_dir, filename)}")
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        # Try saving with a simpler approach as a fallback
        try:
            # Apply the conversion function to the entire results structure
            serializable_results = convert_to_serializable(serializable_results)
            
            # Save with a different name to avoid corrupting the original file
            fallback_filename = "fallback_" + filename
            with open(os.path.join(save_dir, fallback_filename), 'w') as f:
                json.dump(serializable_results, f, indent=4)
            print(f"Saved fallback results to {os.path.join(save_dir, fallback_filename)}")
        except Exception as e2:
            print(f"Failed to save fallback JSON file: {e2}")

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

def plot_final_reward_boxplot(
    results: Dict[str, Any],
    title: str = "Final Cumulative Reward Comparison",
    filename: str = "final_reward_boxplot.png",
    save_dir: str = "results/graphs",
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Create a boxplot comparing the final cumulative rewards between agents.
    Saves two versions: one with individual data points and one without.
    Includes Mann-Whitney U test p-value on the plot.
    
    Args:
        results: Results dictionary from run_benchmark
        title: Plot title
        filename: Filename to save the plot as
        save_dir: Directory to save the plot in
        figsize: Figure size in inches
    """
    # Extract final rewards for trained model
    trained_final_rewards = [rewards[-1] for rewards in results["trained"]["rewards_over_time"]]
    
    # Prepare data for boxplot
    data = []
    categories = []
    
    # Add trained model data
    for reward in trained_final_rewards:
        data.append(reward)
        categories.append("Trained Model")
    
    # Add random agent data if available
    random_final_rewards = []
    if "random" in results:
        random_final_rewards = [rewards[-1] for rewards in results["random"]["rewards_over_time"]]
        for reward in random_final_rewards:
            data.append(reward)
            categories.append("Random Actions")
    
    # Create DataFrame for seaborn
    df = pd.DataFrame({
        "Agent Type": categories,
        "Final Cumulative Reward": data
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
        y="Final Cumulative Reward", 
        data=df,
        width=0.5,
        color='white',
        fliersize=3
    )
    
    # Add individual data points with jitter
    sns.stripplot(
        x="Agent Type", 
        y="Final Cumulative Reward", 
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
    
    # Perform Mann-Whitney U test if both trained and random data are available
    if "random" in results and len(trained_final_rewards) > 0 and len(random_final_rewards) > 0:
        u_stat, p_value = stats.mannwhitneyu(trained_final_rewards, random_final_rewards, alternative='two-sided')
        
        # Add statistical test annotation
        significance = ''
        if p_value < 0.001:
            significance = '***'  # p < 0.001
        elif p_value < 0.01:
            significance = '**'   # p < 0.01
        elif p_value < 0.05:
            significance = '*'    # p < 0.05
            
        # Add the p-value text to the plot
        plt.text(0.5, 0.01, f'Mann-Whitney U Test: p = {p_value:.4f} {significance}', 
                 horizontalalignment='center', 
                 fontsize=10, 
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Customize plot
    plt.title(title, fontsize=14, pad=10)
    plt.xlabel("")  # Remove x-label as it's redundant
    plt.ylabel("Final Cumulative Reward", fontsize=12, labelpad=10)
    
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
        y="Final Cumulative Reward", 
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
    
    # Perform Mann-Whitney U test if both trained and random data are available
    if "random" in results and len(trained_final_rewards) > 0 and len(random_final_rewards) > 0:
        u_stat, p_value = stats.mannwhitneyu(trained_final_rewards, random_final_rewards, alternative='two-sided')
        
        # Add statistical test annotation
        significance = ''
        if p_value < 0.001:
            significance = '***'  # p < 0.001
        elif p_value < 0.01:
            significance = '**'   # p < 0.01
        elif p_value < 0.05:
            significance = '*'    # p < 0.05
            
        # Add the p-value text to the plot
        plt.text(0.5, 0.01, f'Mann-Whitney U Test: p = {p_value:.4f} {significance}', 
                 horizontalalignment='center', 
                 fontsize=10, 
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Customize plot
    plt.title(title, fontsize=14, pad=10)
    plt.xlabel("")  # Remove x-label as it's redundant
    plt.ylabel("Final Cumulative Reward", fontsize=12, labelpad=10)
    
    # Save the figure without points
    plt.tight_layout()
    no_points_filename = os.path.splitext(filename)[0] + "_no_points" + os.path.splitext(filename)[1]
    plt.savefig(os.path.join(save_dir, no_points_filename), dpi=300, bbox_inches='tight')
    plt.close() 