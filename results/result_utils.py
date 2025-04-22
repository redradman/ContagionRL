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
import itertools # for pairwise comparisons
import statsmodels.stats.multitest as smm # for multiple comparisons correction

# Add the parent directory to the path so we can import environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import SIRSEnvironment
from utils import Human

# Centralized agent labels dictionary
AGENT_LABELS = {
    "trained": "Trained",
    "random": "Random",
    "stationary": "Stationary",
    "greedy": "Greedy"
}

# Order of agents for plotting
AGENT_PLOT_ORDER = ["stationary", "random", "trained", "greedy"]

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
    Run a benchmark comparing a trained model to baseline agents.
    Collects reward data and episode lengths.
    
    Args:
        model_path: Path to the trained model
        n_runs: Number of runs to evaluate
        include_random: Whether to include random agent benchmarks
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
        
        # Reset environment
        obs = env.reset(seed=seed)[0]
        done = False
        cumulative_rewards = [0]  # Start with 0 reward
        
        while not done:
            # Use the trained model
            action, _ = model.predict(obs, deterministic=True)
            
            # Take a step in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Update cumulative reward
            cumulative_rewards.append(cumulative_rewards[-1] + reward)
            
            done = terminated or truncated
        
        # Save episode data
        trained_rewards_over_time.append(cumulative_rewards)
        trained_episode_lengths.append(len(cumulative_rewards) - 1)  # -1 because we start with 0
    
    # Run random action episodes if requested
    random_rewards_over_time = []
    random_episode_lengths = []
    
    if include_random:
        # Random agent: samples (dx, dy) uniformly from [‑1,1]² and adherence from [0,1]
        print("Running episodes with Random actions...")
        for i in tqdm(range(n_runs), desc="Random Agent Episodes", unit="episode"):
            # Use different seeds for each run
            seed = random_seed + i if random_seed is not None else None
            
            # Reset environment
            obs = env.reset(seed=seed)[0]
            done = False
            cumulative_rewards = [0]  # Start with 0 reward
            
            while not done:
                # Use random actions
                action = np.array([
                    env.np_random.uniform(-1, 1),  # delta_x
                    env.np_random.uniform(-1, 1),  # delta_y
                    env.np_random.uniform(0, 1)    # adherence random from [0,1]
                ], dtype=np.float32)
                
                # Take a step in the environment
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Update cumulative reward
                cumulative_rewards.append(cumulative_rewards[-1] + reward)
                
                done = terminated or truncated
            
            # Save episode data
            random_rewards_over_time.append(cumulative_rewards)
            random_episode_lengths.append(len(cumulative_rewards) - 1)  # -1 because we start with 0
    
    # Run stationary (0 adherence) episodes
    stationary_rewards_over_time = []
    stationary_episode_lengths = []
    
    print("Running episodes with stationary (0 adherence) actions...")
    for i in tqdm(range(n_runs), desc="Stationary Agent Episodes", unit="episode"):
        # Use different seeds for each run
        seed = random_seed + i if random_seed is not None else None
        
        # Reset environment
        obs = env.reset(seed=seed)[0]
        done = False
        cumulative_rewards = [0]  # Start with 0 reward
        
        # Fixed action [0, 0, 0]
        stationary_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        while not done:
            action = stationary_action
            
            # Take a step in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Update cumulative reward
            cumulative_rewards.append(cumulative_rewards[-1] + reward)
            
            done = terminated or truncated
        
        # Save episode data
        stationary_rewards_over_time.append(cumulative_rewards)
        stationary_episode_lengths.append(len(cumulative_rewards) - 1) # -1 because we start with 0

    # --- Greedy Distance Maximizer Agent --- 
    greedy_rewards_over_time = []
    greedy_episode_lengths = []

    print("Running episodes with Greedy Distance Maximizer actions...")
    for i in tqdm(range(n_runs), desc="Greedy Maximizer Episodes", unit="episode"):
        seed = random_seed + i if random_seed is not None else None
        obs = env.reset(seed=seed)[0]
        done = False
        cumulative_rewards = [0]
        adherence = 0.0 # Greedy agent uses 0 adherence

        while not done:
            # --- Greedy Action Logic --- 
            agent_pos = env.agent_position
            infected_humans = [h for h in env.humans if h.state == 1] # STATE_DICT['I'] == 1
            
            # best_action_idx = 0 # Default to stationary
            # max_min_dist = -1.0

            if not infected_humans:
                # If no infected, stay put
                dx, dy = 0.0, 0.0
            else:
                # Find the nearest infected human to the current position
                current_distances = [env._calculate_distance(Human(agent_pos[0], agent_pos[1], 0, -1), h) for h in infected_humans]
                min_current_dist = min(current_distances)
                nearest_infected_idx = current_distances.index(min_current_dist)
                nearest_infected_human = infected_humans[nearest_infected_idx]
                
                # Evaluate potential moves (8 directions + stationary)
                # Using max_movement = 1.0 as scale
                possible_moves = [
                    (0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
                    (0.707, 0.707), (0.707, -0.707), (-0.707, 0.707), (-0.707, -0.707)
                ]
                best_dx, best_dy = 0.0, 0.0
                max_dist_to_nearest = -1.0

                for move_dx, move_dy in possible_moves:
                    # Calculate potential next position
                    next_x = (agent_pos[0] + move_dx) % env.grid_size
                    next_y = (agent_pos[1] + move_dy) % env.grid_size
                    
                    # Calculate distance from potential next pos to the *originally nearest* infected human
                    dist_to_target = env._calculate_distance(Human(next_x, next_y, 0, -1), nearest_infected_human)
                    
                    # Keep track of the move that maximizes distance to this specific target
                    if dist_to_target > max_dist_to_nearest:
                        max_dist_to_nearest = dist_to_target
                        best_dx, best_dy = move_dx, move_dy
                
                dx, dy = best_dx, best_dy
                
            action = np.array([dx, dy, adherence], dtype=np.float32)
            # --- End Greedy Action Logic ---
            
            obs, reward, terminated, truncated, info = env.step(action)
            cumulative_rewards.append(cumulative_rewards[-1] + reward)
            done = terminated or truncated

        greedy_rewards_over_time.append(cumulative_rewards)
        greedy_episode_lengths.append(len(cumulative_rewards) - 1)
    # --- End Greedy Agent --- 
    
    # Close environment
    env.close()
    
    # Return results
    results = {
        "trained": {
            "rewards_over_time": trained_rewards_over_time,
            "episode_lengths": trained_episode_lengths
        },
        "stationary": { 
            "rewards_over_time": stationary_rewards_over_time,
            "episode_lengths": stationary_episode_lengths
        },
        "greedy": {
             "rewards_over_time": greedy_rewards_over_time,
             "episode_lengths": greedy_episode_lengths
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

def plot_survival_boxplot(
    results: Dict[str, Any],
    title: str = "Episode Duration Comparison",
    filename: str = "survival_boxplot.png",
    save_dir: str = "results/graphs",
    figsize: Tuple[int, int] = (10, 6)  # Increased width
) -> None:
    """
    Create a boxplot comparing the episode durations (time survived) between agents.
    Saves two versions: one with individual data points and one without.
    Includes Mann-Whitney U test p-value on the plot for pairwise comparisons.
    
    Args:
        results: Results dictionary from run_benchmark
        title: Plot title
        filename: Filename to save the plot as
        save_dir: Directory to save the plot in
        figsize: Figure size in inches
    """
    agent_types = list(results.keys())
    if "config" in agent_types: agent_types.remove("config") # Remove config key
    
    data = []
    categories = []
    
    # Extract episode lengths and map categories
    category_data = {}
    for agent_type in agent_types:
        if agent_type in results:
            lengths = results[agent_type]["episode_lengths"]
            label = AGENT_LABELS.get(agent_type, agent_type.capitalize())
            category_data[label] = lengths
            for length in lengths:
                data.append(length)
                categories.append(label)
    
    # Order categories for consistent plotting
    ordered_categories = [AGENT_LABELS[k] for k in AGENT_PLOT_ORDER if k in AGENT_LABELS and AGENT_LABELS[k] in category_data]
    
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
    
    # Perform pairwise Mann-Whitney U tests
    pairwise_results = {}
    p_values = []
    comparisons = []
    if len(ordered_categories) > 1:
        for cat1, cat2 in itertools.combinations(ordered_categories, 2):
            data1 = category_data[cat1]
            data2 = category_data[cat2]
            if len(data1) > 0 and len(data2) > 0:
                try:
                    u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                    pairwise_results[(cat1, cat2)] = p_value
                    p_values.append(p_value)
                    comparisons.append((cat1, cat2))
                except ValueError as e:
                    print(f"Warning: Mann-Whitney U test failed for {cat1} vs {cat2}: {e}")
                    pairwise_results[(cat1, cat2)] = np.nan
                    p_values.append(np.nan)
                    comparisons.append((cat1, cat2))
            else:
                pairwise_results[(cat1, cat2)] = np.nan
                p_values.append(np.nan)
                comparisons.append((cat1, cat2))
                
        # Apply multiple comparison correction (Bonferroni)
        if p_values:
             # Filter out NaN p-values before correction if necessary
            valid_p_values = [p for p in p_values if not np.isnan(p)]
            valid_comparisons = [comp for p, comp in zip(p_values, comparisons) if not np.isnan(p)]
            if valid_p_values: # Proceed only if there are valid p-values
                reject, pvals_corrected, _, _ = smm.multipletests(valid_p_values, alpha=0.05, method='bonferroni')
                corrected_results = dict(zip(valid_comparisons, pvals_corrected))
                 # Add back NaN results for comparisons that couldn't be tested
                for comp, p_val in zip(comparisons, p_values):
                    if np.isnan(p_val):
                        corrected_results[comp] = np.nan
            else:
                corrected_results = {comp: np.nan for comp in comparisons} # All NaN if no valid p-values
        else:
            corrected_results = {}

    # --- Function to add annotations --- 
    def add_stats_annotations(ax, plot_type):
        if not corrected_results: return
        
        y_max = df["Episode Duration (steps)"].max()
        y_min = df["Episode Duration (steps)"].min() # Get min value
        y_range = y_max - y_min # Calculate range
         # Handle cases where y_range is zero or very small
        if y_range <= 1e-9: # Use a small threshold instead of == 0 for float safety
            increment = 0.1 * abs(y_max) if abs(y_max) > 1e-9 else 0.1 # Base increment on max value or default
        else:
            # Adjust increment based on the number of comparisons
            num_comparisons = len(corrected_results)
            increment_factor = 0.08 + (num_comparisons - 3) * 0.01 # Base 0.08 for 3 comparisons, add 0.01 per extra
            increment = y_range * max(0.06, increment_factor) # Ensure a minimum increment
            
        current_y = y_max + increment * 0.5
        
        # Define positions for annotations based on ordered categories
        cat_pos = {cat: i for i, cat in enumerate(ordered_categories)}
        
        # Get the trained agent label
        trained_label = AGENT_LABELS.get('trained', 'Trained')
        
        # Only show comparisons between trained agent and others
        for (cat1, cat2), p_corrected in corrected_results.items():
            # Skip if neither is the trained agent or if p-value is NaN
            if trained_label not in (cat1, cat2) or np.isnan(p_corrected):
                continue
            
            # Ensure trained agent is always cat1 for consistency
            if cat2 == trained_label:
                cat1, cat2 = cat2, cat1
            
            pos1 = cat_pos[cat1]
            pos2 = cat_pos[cat2]
            
            # Draw connecting line
            line_x = [pos1, pos1, pos2, pos2]
            line_y = [current_y, current_y + increment * 0.2, current_y + increment * 0.2, current_y]
            ax.plot(line_x, line_y, lw=1.5, c='black')
            
            # Determine significance level
            significance = 'ns' # not significant
            if p_corrected < 0.001: significance = '***'
            elif p_corrected < 0.01: significance = '**'
            elif p_corrected < 0.05: significance = '*'
            
            # Add text annotation
            text_x = (pos1 + pos2) / 2
            text_y = current_y + increment * 0.3
            ax.text(text_x, text_y, significance, ha='center', va='bottom', fontsize=10)
            
            current_y += increment # Move up for the next annotation

        # Adjust y-limit to make space for annotations
        ax.set_ylim(top=current_y + increment * 0.5) # Add a bit more padding at the top

    # --- FIRST PLOT: Boxplot with individual data points --- 
    plt.figure(figsize=figsize, dpi=120)
    
    # Create a clean boxplot with minimal design
    ax1 = sns.boxplot(
        x="Agent Type", 
        y="Episode Duration (steps)", 
        data=df,
        order=ordered_categories, # Use defined order
        width=0.5,
        color='white',
        fliersize=3
    )
    
    # Add individual data points with jitter
    sns.stripplot(
        x="Agent Type", 
        y="Episode Duration (steps)", 
        data=df,
        order=ordered_categories,
        color='black',
        size=5,  
        alpha=0.5,
        jitter=True
    )
    
    # Remove colors and simplify
    for i, box in enumerate(ax1.artists):
        box.set_edgecolor('black')
        for j in range(6*i, 6*(i+1)):
             if j < len(ax1.lines): # Check if line index is valid
                ax1.lines[j].set_color('black')
    
    # Customize appearance
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(0.5)
    ax1.spines['bottom'].set_linewidth(0.5)
    ax1.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax1.xaxis.grid(False)
    
    # Keep x-axis labels horizontal
    plt.xticks(rotation=0)
    
    # Add statistical annotations
    add_stats_annotations(ax1, "with_points")
    
    # Customize plot labels
    plt.title(title, fontsize=14, pad=20) # Increased pad
    plt.xlabel("")  
    plt.ylabel("Episode Duration (steps)", fontsize=12, labelpad=10)
    
    # Save the figure with points
    plt.tight_layout()
    points_filename = os.path.splitext(filename)[0] + "_with_points" + os.path.splitext(filename)[1]
    plt.savefig(os.path.join(save_dir, points_filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- SECOND PLOT: Boxplot without individual data points --- 
    plt.figure(figsize=figsize, dpi=120)
    
    # Create a clean boxplot without stripplot
    ax2 = sns.boxplot(
        x="Agent Type", 
        y="Episode Duration (steps)", 
        data=df,
        order=ordered_categories,
        width=0.5,
        color='white',
        fliersize=3
    )
    
    # Remove colors and simplify
    for i, box in enumerate(ax2.artists):
        box.set_edgecolor('black')
        for j in range(6*i, 6*(i+1)):
            if j < len(ax2.lines):
                ax2.lines[j].set_color('black')
    
    # Customize appearance
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(0.5)
    ax2.spines['bottom'].set_linewidth(0.5)
    ax2.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax2.xaxis.grid(False)
    
    # Keep x-axis labels horizontal
    plt.xticks(rotation=0)
    
    # Add statistical annotations
    add_stats_annotations(ax2, "no_points")

    # Customize plot labels
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel("")  
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
        "mean_final_reward": float(np.mean([rewards[-1] for rewards in results["trained"]["rewards_over_time"] if rewards])),
        "std_final_reward": float(np.std([rewards[-1] for rewards in results["trained"]["rewards_over_time"] if rewards], ddof=1))
    }
    
    # Calculate stats for random agent if available
    if "random" in results:
        summary["random"] = {
            "mean_episode_length": float(np.mean(results["random"]["episode_lengths"])),
            "std_episode_length": float(np.std(results["random"]["episode_lengths"], ddof=1)),
            "mean_final_reward": float(np.mean([rewards[-1] for rewards in results["random"]["rewards_over_time"] if rewards])),
            "std_final_reward": float(np.std([rewards[-1] for rewards in results["random"]["rewards_over_time"] if rewards], ddof=1))
        }
    
    # Calculate stats for stationary agent if available
    if "stationary" in results:
        summary["stationary"] = {
            "mean_episode_length": float(np.mean(results["stationary"]["episode_lengths"])),
            "std_episode_length": float(np.std(results["stationary"]["episode_lengths"], ddof=1)),
            "mean_final_reward": float(np.mean([rewards[-1] for rewards in results["stationary"]["rewards_over_time"] if rewards])),
            "std_final_reward": float(np.std([rewards[-1] for rewards in results["stationary"]["rewards_over_time"] if rewards], ddof=1))
        }
    
    # Calculate stats for greedy agent if available
    if "greedy" in results:
        summary["greedy"] = {
            "mean_episode_length": float(np.mean(results["greedy"]["episode_lengths"])),
            "std_episode_length": float(np.std(results["greedy"]["episode_lengths"], ddof=1)),
            "mean_final_reward": float(np.mean([rewards[-1] for rewards in results["greedy"]["rewards_over_time"] if rewards])),
            "std_final_reward": float(np.std([rewards[-1] for rewards in results["greedy"]["rewards_over_time"] if rewards], ddof=1))
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
            # Handle NaN specifically for JSON compatibility
            if np.isnan(obj):
                return None # Represent NaN as null in JSON
            return float(obj)
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        # Handle boolean types explicitly if they come from numpy
        elif isinstance(obj, np.bool_):
             return bool(obj)
        # Handle None explicitly
        elif obj is None:
             return None
        else:
            # Attempt basic types, raise error otherwise?
            # For now, assume other types are directly serializable
            return obj
    
    # Get summary stats first
    summary_stats = get_summary_stats(results)
    # Use the conversion function immediately for the summary
    serializable_results["summary"] = convert_to_serializable(summary_stats)
    
    # Prepare data for statistical tests
    data_for_tests = {}
    
    # Use the actual agent types present in the results
    agent_keys_present = [k for k in AGENT_LABELS if k in results]
    
    for agent_type in agent_keys_present: # Iterate only over present agents
        label = AGENT_LABELS[agent_type]
        # Check if data exists and is not empty
        lengths_data = results[agent_type].get("episode_lengths", [])
        rewards_data_raw = results[agent_type].get("rewards_over_time", [])
        rewards_data = [r[-1] for r in rewards_data_raw if r] # Get final reward, ensure list not empty
        
        data_for_tests[label] = {
            "lengths": lengths_data if lengths_data else [], # Ensure it's a list
            "rewards": rewards_data if rewards_data else []  # Ensure it's a list
        }

    # Perform pairwise statistical tests
    serializable_results["statistical_tests"] = {}
    agent_keys_for_test = list(data_for_tests.keys())
    
    if len(agent_keys_for_test) > 1:
        for metric in ["lengths", "rewards"]:
            metric_key = "episode_length" if metric == "lengths" else "final_reward"
            serializable_results["statistical_tests"][metric_key] = {}
            
            p_values = []
            comparisons = []
            test_results = {}
            
            for agent1, agent2 in itertools.combinations(agent_keys_for_test, 2):
                data1 = data_for_tests[agent1][metric]
                data2 = data_for_tests[agent2][metric]
                comparison_key = f"{agent1}_vs_{agent2}"
                
                test_data = {
                    "comparison": comparison_key,
                    "test_type": "Mann-Whitney U",
                    "u_statistic": None,
                    "p_value_uncorrected": None,
                    "p_value_bonferroni": None,
                    "significant_0.05_bonferroni": None,
                    "normality_agent1": { "test": "Shapiro-Wilk", "statistic": None, "p_value": None, "is_normal": None },
                    "normality_agent2": { "test": "Shapiro-Wilk", "statistic": None, "p_value": None, "is_normal": None },
                    "variance_equality": { "test": "Levene", "statistic": None, "p_value": None, "equal_variance": None },
                    "permutation": { "test": "Permutation", "observed_difference": None, "p_value": None, "significant_0.05": None }
                }

                current_p_uncorrected = np.nan # Default to NaN

                if len(data1) > 0 and len(data2) > 0:
                    # Mann-Whitney U
                    try:
                        # Check for zero variance which causes ValueError in Mann-Whitney U
                        if len(set(data1)) == 1 and len(set(data2)) == 1 and data1[0] == data2[0]:
                             print(f"Skipping Mann-Whitney U for {comparison_key} ({metric}): Both groups have identical constant values.")
                             # Keep current_p_uncorrected = np.nan
                        else:
                            u_stat, p_val = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                            test_data["u_statistic"] = u_stat
                            test_data["p_value_uncorrected"] = p_val
                            current_p_uncorrected = p_val # Store valid p-value
                    except ValueError as e:
                         print(f"Warning: Mann-Whitney U test failed for {comparison_key} ({metric}): {e}")
                         # Keep current_p_uncorrected as NaN

                    # Shapiro-Wilk (Agent 1)
                    if len(data1) >= 3:
                        try:
                            sw_stat, sw_p = stats.shapiro(data1)
                            test_data["normality_agent1"]["statistic"] = sw_stat
                            test_data["normality_agent1"]["p_value"] = sw_p
                            test_data["normality_agent1"]["is_normal"] = sw_p >= 0.05
                        except (ValueError, stats.FitError) as e: # Catch specific stats errors
                            print(f"Warning: Shapiro test failed for {agent1} ({metric}): {e}")
                    # Shapiro-Wilk (Agent 2)
                    if len(data2) >= 3:
                        try:
                            sw_stat, sw_p = stats.shapiro(data2)
                            test_data["normality_agent2"]["statistic"] = sw_stat
                            test_data["normality_agent2"]["p_value"] = sw_p
                            test_data["normality_agent2"]["is_normal"] = sw_p >= 0.05
                        except (ValueError, stats.FitError) as e:
                            print(f"Warning: Shapiro test failed for {agent2} ({metric}): {e}")
                        
                    # Levene's Test for Variance Equality
                    try:
                        levene_stat, levene_p = stats.levene(data1, data2)
                        test_data["variance_equality"]["statistic"] = levene_stat
                        test_data["variance_equality"]["p_value"] = levene_p
                        test_data["variance_equality"]["equal_variance"] = levene_p >= 0.05
                    except ValueError as e: # Catches zero variance or other issues
                        print(f"Warning: Levene's test failed for {comparison_key} ({metric}): {e}")

                    # Permutation Test
                    try:
                        # Check if means can be calculated (avoid errors with empty lists if checks above fail)
                        if data1 and data2:
                            def diff_of_means(x, y): return np.mean(x) - np.mean(y)
                            observed_diff = diff_of_means(data1, data2)
                            combined = np.concatenate([data1, data2])
                            n_perm = 10000
                            n1 = len(data1)
                            perm_diffs = [diff_of_means(combined_shuffled[:n1], combined_shuffled[n1:]) 
                                          for combined_shuffled in [np.random.permutation(combined) for _ in range(n_perm)]]
                            perm_p = np.sum(np.abs(perm_diffs) >= np.abs(observed_diff)) / n_perm
                            test_data["permutation"]["observed_difference"] = observed_diff
                            test_data["permutation"]["p_value"] = perm_p
                            test_data["permutation"]["significant_0.05"] = perm_p < 0.05
                        else:
                             print(f"Skipping Permutation test for {comparison_key} ({metric}): insufficient data.")
                    except Exception as e:
                        print(f"Warning: Permutation test failed for {comparison_key} ({metric}): {e}")

                else: # Handle case where one or both data lists are empty
                     print(f"Skipping tests for {comparison_key} ({metric}): insufficient data (len(data1)={len(data1)}, len(data2)={len(data2)}).")
                     # Keep current_p_uncorrected as NaN
                     
                # Always append, using NaN if test failed or wasn't run
                p_values.append(current_p_uncorrected)
                comparisons.append(comparison_key)
                test_results[comparison_key] = test_data
                
            # Apply Bonferroni correction
            valid_indices = [i for i, p in enumerate(p_values) if not np.isnan(p)]
            if valid_indices:
                valid_p_values = [p_values[i] for i in valid_indices]
                valid_comparisons = [comparisons[i] for i in valid_indices]
                reject, pvals_corrected, _, _ = smm.multipletests(valid_p_values, alpha=0.05, method='bonferroni')
                
                for i, comparison_key in enumerate(valid_comparisons):
                    test_results[comparison_key]["p_value_bonferroni"] = pvals_corrected[i]
                    test_results[comparison_key]["significant_0.05_bonferroni"] = reject[i]
            
            # Store results
            # Apply conversion immediately to the test_data dict before storing
            serializable_results["statistical_tests"][metric_key] = convert_to_serializable(test_results)

    # Add directional tests (Trained vs Baselines)
    if "Trained" in agent_keys_for_test: # Use filtered keys
        baseline_keys = [k for k in ["Rand-0", "Rand-1", "Stat-0", "Stat-1", "Greedy"] if k in agent_keys_for_test]
        if baseline_keys:
            serializable_results["statistical_tests"]["trained_vs_baselines"] = {}
            for metric in ["lengths", "rewards"]:
                metric_key = "episode_length" if metric == "lengths" else "final_reward"
                directional_results = {}
                p_values_directional = []
                comparisons_directional = []
                
                data_trained = data_for_tests["Trained"][metric]
                
                for baseline in baseline_keys:
                    data_baseline = data_for_tests[baseline][metric]
                    comparison_key = f"Trained_vs_{baseline}"
                    
                    test_data_directional = {
                        "comparison": comparison_key,
                        "test_type": "Mann-Whitney U (one-sided, greater)",
                        "u_statistic": None,
                        "p_value_uncorrected": None,
                        "p_value_bonferroni": None,
                        "significant_0.05_bonferroni": None
                    }
                    
                    current_p_uncorrected_dir = np.nan # Default to NaN

                    if len(data_trained) > 0 and len(data_baseline) > 0:
                        try:
                            # Check for zero variance which causes ValueError 
                            if len(set(data_trained)) == 1 and len(set(data_baseline)) == 1 and data_trained[0] == data_baseline[0]:
                                 print(f"Skipping One-sided Mann-Whitney U for {comparison_key} ({metric}): Both groups have identical constant values.")
                                 # Keep current_p_uncorrected_dir = np.nan
                            else:
                                # Use alternative='greater' for one-sided test (Trained > Baseline)
                                u_stat, p_val = stats.mannwhitneyu(data_trained, data_baseline, alternative='greater')
                                test_data_directional["u_statistic"] = u_stat
                                test_data_directional["p_value_uncorrected"] = p_val
                                current_p_uncorrected_dir = p_val
                        except ValueError as e:
                            print(f"Warning: One-sided Mann-Whitney U test failed for {comparison_key} ({metric}): {e}")
                            # Keep current_p_uncorrected_dir = np.nan
                    else:
                         print(f"Skipping directional test for {comparison_key} ({metric}): insufficient data.")
                         # Keep current_p_uncorrected_dir = np.nan
                         
                    # Always append, using NaN if test failed or wasn't run
                    p_values_directional.append(current_p_uncorrected_dir)
                    comparisons_directional.append(comparison_key)
                    directional_results[comparison_key] = test_data_directional
                    
                # Apply Bonferroni correction specifically to these directional tests
                valid_indices_dir = [i for i, p in enumerate(p_values_directional) if not np.isnan(p)]
                if valid_indices_dir:
                    valid_p_values_dir = [p_values_directional[i] for i in valid_indices_dir]
                    valid_comparisons_dir = [comparisons_directional[i] for i in valid_indices_dir]
                    # Correct across the number of baselines being compared (can be up to 3)
                    reject_dir, pvals_corrected_dir, _, _ = smm.multipletests(valid_p_values_dir, alpha=0.05, method='bonferroni')
                    
                    for i, comparison_key in enumerate(valid_comparisons_dir):
                        directional_results[comparison_key]["p_value_bonferroni"] = pvals_corrected_dir[i]
                        directional_results[comparison_key]["significant_0.05_bonferroni"] = reject_dir[i]
                        
                # Store results for this metric
                serializable_results["statistical_tests"]["trained_vs_baselines"][metric_key] = convert_to_serializable(directional_results)
    
    # Add config info
    if "config" in results:
        env_conf = results["config"].get("environment", {})
        serializable_results["environment_config"] = {
            "grid_size": int(env_conf.get("grid_size", 0)),
            "n_humans": int(env_conf.get("n_humans", 0)),
            "n_infected": int(env_conf.get("n_infected", 0)),
            "simulation_time": int(env_conf.get("simulation_time", 0)),
            "reward_type": str(env_conf.get("reward_type", "N/A"))
    }
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save to JSON file with error handling
    try:
        # The main dictionary should now be fully serializable
        # No need to call convert_to_serializable again here if applied correctly above
        with open(os.path.join(save_dir, filename), 'w') as f:
            json.dump(serializable_results, f, indent=4, sort_keys=False)
        print(f"Successfully saved benchmark results to {os.path.join(save_dir, filename)}")
    except TypeError as e:
        print(f"Error saving JSON file due to TypeError: {e}")
        print("Attempting fallback serialization...")
        # Fallback: Apply conversion one more time on the entire structure
        try:
            serializable_results_fallback = convert_to_serializable(serializable_results)
            fallback_filename = "fallback_" + filename
            with open(os.path.join(save_dir, fallback_filename), 'w') as f:
                json.dump(serializable_results_fallback, f, indent=4)
            print(f"Saved fallback results to {os.path.join(save_dir, fallback_filename)}")
        except Exception as e2:
            print(f"Failed to save fallback JSON file: {e2}")
    except Exception as e:
         print(f"An unexpected error occurred during JSON saving: {e}")

def plot_final_reward_boxplot(
    results: Dict[str, Any],
    title: str = "Final Cumulative Reward Comparison",
    filename: str = "final_reward_boxplot.png",
    save_dir: str = "results/graphs",
    figsize: Tuple[int, int] = (10, 6)  # Increased width
) -> None:
    """
    Create a boxplot comparing the final cumulative rewards between agents.
    Saves two versions: one with individual data points and one without.
    Includes Mann-Whitney U test p-value on the plot for pairwise comparisons.
    
    Args:
        results: Results dictionary from run_benchmark
        title: Plot title
        filename: Filename to save the plot as
        save_dir: Directory to save the plot in
        figsize: Figure size in inches
    """
    agent_types = list(results.keys())
    if "config" in agent_types: agent_types.remove("config") # Remove config key
    
    data = []
    categories = []
    
    # Extract final rewards and map categories
    category_data = {}
    for agent_type in agent_types:
        if agent_type in results:
            # Ensure rewards_over_time exists and is not empty
            if "rewards_over_time" in results[agent_type] and results[agent_type]["rewards_over_time"]:
                final_rewards = [rewards[-1] for rewards in results[agent_type]["rewards_over_time"] if rewards] # Check if reward list is not empty
                label = AGENT_LABELS.get(agent_type, agent_type.capitalize())
                category_data[label] = final_rewards
                for reward_val in final_rewards:
                    data.append(reward_val)
                    categories.append(label)
            else:
                 print(f"Warning: No reward data found for agent type '{agent_type}'")
                 category_data[AGENT_LABELS.get(agent_type, agent_type.capitalize())] = [] # Add empty list if no data

    
    # Order categories for consistent plotting
    ordered_categories = [AGENT_LABELS[k] for k in AGENT_PLOT_ORDER if k in AGENT_LABELS and AGENT_LABELS[k] in category_data]
    
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
    
    # Perform pairwise Mann-Whitney U tests
    pairwise_results = {}
    p_values = []
    comparisons = []
    if len(ordered_categories) > 1:
        for cat1, cat2 in itertools.combinations(ordered_categories, 2):
            data1 = category_data.get(cat1, []) # Safely get data
            data2 = category_data.get(cat2, []) # Safely get data
            
            current_p_uncorrected = np.nan # Default to NaN
            comparison_key = (cat1, cat2) # Use tuple as key
            
            if len(data1) > 0 and len(data2) > 0:
                try:
                     # Check for zero variance which causes ValueError in Mann-Whitney U
                    if len(set(data1)) == 1 and len(set(data2)) == 1 and data1[0] == data2[0]:
                         print(f"Skipping Mann-Whitney U for {cat1} vs {cat2} (Final Reward): Both groups have identical constant values.")
                         # Keep current_p_uncorrected = np.nan
                    else:
                        u_stat, p_val = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                        current_p_uncorrected = p_val
                        pairwise_results[comparison_key] = p_val # Store uncorrected p-value
                except ValueError as e:
                    print(f"Warning: Mann-Whitney U test failed for {cat1} vs {cat2} (Final Reward): {e}")
                    # Keep current_p_uncorrected = np.nan
            else:
                 # Cannot perform test with empty data
                 print(f"Skipping Mann-Whitney U test for {cat1} vs {cat2} (Final Reward): Insufficient data.")
                 # Keep current_p_uncorrected as NaN
            
            # Always append, using NaN if test failed or wasn't run
            p_values.append(current_p_uncorrected)
            comparisons.append(comparison_key)
                
        # Apply multiple comparison correction (Bonferroni)
        corrected_results = {}
        if p_values:
             # Filter out NaN p-values before correction if necessary
            valid_indices = [i for i, p in enumerate(p_values) if not np.isnan(p)]
            valid_p_values = [p_values[i] for i in valid_indices]
            valid_comparisons = [comparisons[i] for i in valid_indices]
            
            if valid_p_values: # Proceed only if there are valid p-values
                reject, pvals_corrected, _, _ = smm.multipletests(valid_p_values, alpha=0.05, method='bonferroni')
                corrected_results_temp = dict(zip(valid_comparisons, pvals_corrected))
                 # Add back NaN results for comparisons that couldn't be tested or failed
                for comp, p_val in zip(comparisons, p_values):
                    corrected_results[comp] = corrected_results_temp.get(comp, np.nan)
            else:
                corrected_results = {comp: np.nan for comp in comparisons} # All NaN if no valid p-values
        # else: corrected_results remains empty

    # --- Function to add annotations --- 
    def add_stats_annotations(ax, plot_type):
        if not corrected_results: return
        
        # Check if dataframe is empty before calculating range
        if df.empty:
            print("Warning: Cannot add annotations to final reward boxplot, DataFrame is empty.")
            return
            
        y_max = df["Final Cumulative Reward"].max()
        y_min = df["Final Cumulative Reward"].min()
        y_range = y_max - y_min
         # Handle cases where y_range is zero or very small
        if y_range <= 1e-9: # Use a small threshold instead of == 0 for float safety
             increment = 0.1 * abs(y_max) if abs(y_max) > 1e-9 else 0.1 # Base increment on max value or default
        else:
            # Adjust increment based on the number of comparisons
            num_comparisons = len(corrected_results)
            increment_factor = 0.08 + (num_comparisons - 3) * 0.01 # Base 0.08 for 3 comparisons, add 0.01 per extra
            increment = y_range * max(0.06, increment_factor) # Ensure a minimum increment
            
        current_y = y_max + increment * 0.5
        
        # Define positions for annotations based on ordered categories
        cat_pos = {cat: i for i, cat in enumerate(ordered_categories)}
        
        # Get the trained agent label
        trained_label = AGENT_LABELS.get('trained', 'Trained')
        
        # Only show comparisons between trained agent and others
        for (cat1, cat2), p_corrected in corrected_results.items():
            # Skip if neither is the trained agent or if p-value is NaN
            if trained_label not in (cat1, cat2) or np.isnan(p_corrected):
                continue
            
            # Ensure trained agent is always cat1 for consistency
            if cat2 == trained_label:
                cat1, cat2 = cat2, cat1
            
            pos1 = cat_pos[cat1]
            pos2 = cat_pos[cat2]
            
            # Draw connecting line
            line_x = [pos1, pos1, pos2, pos2]
            line_y = [current_y, current_y + increment * 0.2, current_y + increment * 0.2, current_y]
            ax.plot(line_x, line_y, lw=1.5, c='black')
            
            # Determine significance level
            significance = 'ns' # not significant
            if p_corrected < 0.001: significance = '***'
            elif p_corrected < 0.01: significance = '**'
            elif p_corrected < 0.05: significance = '*'
            
            # Add text annotation
            text_x = (pos1 + pos2) / 2
            text_y = current_y + increment * 0.3
            ax.text(text_x, text_y, significance, ha='center', va='bottom', fontsize=10)
            
            current_y += increment # Move up for the next annotation

        # Adjust y-limit to make space for annotations
        ax.set_ylim(top=current_y + increment * 0.5) # Add a bit more padding at the top

    # --- FIRST PLOT: Boxplot with individual data points --- 
    plt.figure(figsize=figsize, dpi=120)
    
    # Create a clean boxplot with minimal design
    ax1 = sns.boxplot(
        x="Agent Type", 
        y="Final Cumulative Reward", 
        data=df,
        order=ordered_categories, # Use defined order
        width=0.5,
        color='white',
        fliersize=3
    )
    
    # Add individual data points with jitter
    sns.stripplot(
        x="Agent Type", 
        y="Final Cumulative Reward", 
        data=df,
        order=ordered_categories,
        color='black',
        size=5,  
        alpha=0.5,
        jitter=True
    )
    
    # Remove colors and simplify
    for i, box in enumerate(ax1.artists):
        box.set_edgecolor('black')
        for j in range(6*i, 6*(i+1)):
             if j < len(ax1.lines): # Check if line index is valid
                ax1.lines[j].set_color('black')
    
    # Customize appearance
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(0.5)
    ax1.spines['bottom'].set_linewidth(0.5)
    ax1.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax1.xaxis.grid(False)
    
    # Keep x-axis labels horizontal
    plt.xticks(rotation=0)
    
    # Add statistical annotations
    add_stats_annotations(ax1, "with_points")
    
    # Customize plot labels
    plt.title(title, fontsize=14, pad=20) # Increased pad
    plt.xlabel("")  
    plt.ylabel("Final Cumulative Reward", fontsize=12, labelpad=10)
    
    # Save the figure with points
    plt.tight_layout()
    points_filename = os.path.splitext(filename)[0] + "_with_points" + os.path.splitext(filename)[1]
    plt.savefig(os.path.join(save_dir, points_filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- SECOND PLOT: Boxplot without individual data points --- 
    plt.figure(figsize=figsize, dpi=120)
    
    # Create a clean boxplot without stripplot
    ax2 = sns.boxplot(
        x="Agent Type", 
        y="Final Cumulative Reward", 
        data=df,
        order=ordered_categories,
        width=0.5,
        color='white',
        fliersize=3
    )
    
    # Remove colors and simplify
    for i, box in enumerate(ax2.artists):
        box.set_edgecolor('black')
        for j in range(6*i, 6*(i+1)):
            if j < len(ax2.lines):
                ax2.lines[j].set_color('black')
    
    # Customize appearance
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(0.5)
    ax2.spines['bottom'].set_linewidth(0.5)
    ax2.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax2.xaxis.grid(False)
    
    # Keep x-axis labels horizontal
    plt.xticks(rotation=0)
    
    # Add statistical annotations
    add_stats_annotations(ax2, "no_points")

    # Customize plot labels
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel("")  
    plt.ylabel("Final Cumulative Reward", fontsize=12, labelpad=10)
    
    # Save the figure without points
    plt.tight_layout()
    no_points_filename = os.path.splitext(filename)[0] + "_no_points" + os.path.splitext(filename)[1]
    plt.savefig(os.path.join(save_dir, no_points_filename), dpi=300, bbox_inches='tight')
    plt.close() 