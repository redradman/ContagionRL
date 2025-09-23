#!/usr/bin/env python
import os
import sys
import argparse
import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from textwrap import wrap

# Add the parent directory to the path to access project modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from environment import SIRSDEnvironment, Human

# tueplots styling
from tueplots import bundles
plt.rcParams.update(bundles.neurips2024())
plt.rcParams["figure.dpi"] = 300
sns.set_style("whitegrid")

# Define Movement Types and their labels for plotting
MOVEMENT_TYPES = ["continuous_random", "workplace_home_cycle"]
MOVEMENT_LABELS = {
    "continuous_random": "Random Movement",
    "workplace_home_cycle": "Workplace/Home Cycle",
}
# Order for plotting on the x-axis for the bar plot
PLOT_ORDER_X_AXIS = [MOVEMENT_LABELS[mt] for mt in MOVEMENT_TYPES]
AGENT_ORDER = ['Stationary', 'Random', 'Trained', 'Greedy'] # Order of bars within each group

def load_model_config(model_path: str) -> Dict[str, Any]:
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path} for model {model_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def create_env_from_config(env_config_dict: Dict[str, Any], seed: Optional[int] = None) -> SIRSDEnvironment:
    config_copy = env_config_dict.copy()
    config_copy["render_mode"] = None
    env = SIRSDEnvironment(**config_copy)
    env.reset(seed=seed)
    return env

def calculate_clustering_metrics(env: SIRSDEnvironment) -> Dict[str, float]:
    """
    Calculate spatial clustering metrics for human positions.
    """
    if len(env.humans) == 0:
        return {"spatial_variance": 0.0, "clustering_coefficient": 0.0}
    
    positions = np.array([[h.x, h.y] for h in env.humans])
    
    # Calculate spatial variance (spread of positions)
    center = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - center, axis=1)
    spatial_variance = np.var(distances)
    
    # Calculate clustering coefficient (average distance between all pairs)
    n_humans = len(positions)
    total_distance = 0
    count = 0
    
    for i in range(n_humans):
        for j in range(i+1, n_humans):
            # Handle periodic boundaries
            dx = abs(positions[i][0] - positions[j][0])
            dy = abs(positions[i][1] - positions[j][1])
            
            dx = min(dx, env.grid_size - dx)
            dy = min(dy, env.grid_size - dy)
            
            distance = np.sqrt(dx**2 + dy**2)
            total_distance += distance
            count += 1
    
    avg_distance = total_distance / count if count > 0 else 0
    clustering_coefficient = 1.0 / (1.0 + avg_distance)  # Higher when humans are closer
    
    return {
        "spatial_variance": spatial_variance,
        "clustering_coefficient": clustering_coefficient,
        "avg_pairwise_distance": avg_distance
    }

def run_evaluation_episodes_for_metrics(
    env: SIRSDEnvironment, 
    model: PPO, 
    num_episodes: int,
    base_eval_seed: int 
) -> List[Dict[str, Any]]:
    episode_metrics = []
    for i in range(num_episodes):
        eval_seed_for_run = base_eval_seed + i
        obs, _ = env.reset(seed=eval_seed_for_run)
        done = False
        current_episode_length = 0
        current_cumulative_reward = 0.0
        
        # Track movement-specific metrics
        total_clustering_coef = 0.0
        total_spatial_variance = 0.0
        infection_events = 0
        initial_infected = sum(1 for h in env.humans if h.state == 1)
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            current_episode_length += 1
            current_cumulative_reward += reward
            done = terminated or truncated
            
            # Calculate clustering metrics each timestep
            clustering_metrics = calculate_clustering_metrics(env)
            total_clustering_coef += clustering_metrics["clustering_coefficient"]
            total_spatial_variance += clustering_metrics["spatial_variance"]
            
            # Count new infections
            current_infected = sum(1 for h in env.humans if h.state == 1)
            if current_infected > initial_infected:
                infection_events += (current_infected - initial_infected)
                initial_infected = current_infected
        
        # Calculate averages
        avg_clustering = total_clustering_coef / current_episode_length if current_episode_length > 0 else 0
        avg_spatial_variance = total_spatial_variance / current_episode_length if current_episode_length > 0 else 0
        
        episode_metrics.append({
            "episode_length": current_episode_length,
            "final_reward": current_cumulative_reward,
            "avg_clustering_coefficient": avg_clustering,
            "avg_spatial_variance": avg_spatial_variance,
            "infection_events": infection_events,
            "infection_rate": infection_events / current_episode_length if current_episode_length > 0 else 0
        })
    return episode_metrics

def run_baseline_evaluation(
    env: SIRSDEnvironment, 
    baseline_type: str, 
    num_episodes: int, 
    base_eval_seed: int
) -> List[Dict[str, Any]]:
    episode_metrics = []
    
    for i in range(num_episodes):
        eval_seed_for_run = base_eval_seed + i
        obs, _ = env.reset(seed=eval_seed_for_run)
        done = False
        current_episode_length = 0
        current_cumulative_reward = 0.0
        
        # Track movement-specific metrics
        total_clustering_coef = 0.0
        total_spatial_variance = 0.0
        infection_events = 0
        initial_infected = sum(1 for h in env.humans if h.state == 1)
        
        while not done:
            if baseline_type == "random":
                action = env.action_space.sample()
            elif baseline_type == "stationary":
                action = np.array([0.0, 0.0, 0.0])  # No movement, no adherence change
            elif baseline_type == "greedy":
                # Simple greedy: move away from nearest infected, increase adherence
                agent_pos = np.array(env.agent_position)
                infected_positions = []
                
                for human in env.humans:
                    if human.state == 1:  # Infected
                        human_pos = np.array([human.x, human.y])
                        # Handle periodic boundaries
                        for dx in [-env.grid_size, 0, env.grid_size]:
                            for dy in [-env.grid_size, 0, env.grid_size]:
                                wrapped_pos = human_pos + np.array([dx, dy])
                                infected_positions.append(wrapped_pos)
                
                if infected_positions:
                    infected_positions = np.array(infected_positions)
                    distances = np.linalg.norm(infected_positions - agent_pos, axis=1)
                    nearest_infected = infected_positions[np.argmin(distances)]
                    
                    # Move away from nearest infected
                    direction = agent_pos - nearest_infected
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        movement = direction * 0.5  # Moderate movement speed
                    else:
                        movement = np.array([0.0, 0.0])
                    
                    # Increase adherence
                    adherence_change = 0.1
                else:
                    movement = np.array([0.0, 0.0])
                    adherence_change = 0.0
                
                action = np.array([movement[0], movement[1], adherence_change])
            else:
                raise ValueError(f"Unknown baseline type: {baseline_type}")
                
            obs, reward, terminated, truncated, _ = env.step(action)
            current_episode_length += 1
            current_cumulative_reward += reward
            done = terminated or truncated
            
            # Calculate clustering metrics each timestep
            clustering_metrics = calculate_clustering_metrics(env)
            total_clustering_coef += clustering_metrics["clustering_coefficient"]
            total_spatial_variance += clustering_metrics["spatial_variance"]
            
            # Count new infections
            current_infected = sum(1 for h in env.humans if h.state == 1)
            if current_infected > initial_infected:
                infection_events += (current_infected - initial_infected)
                initial_infected = current_infected
        
        # Calculate averages
        avg_clustering = total_clustering_coef / current_episode_length if current_episode_length > 0 else 0
        avg_spatial_variance = total_spatial_variance / current_episode_length if current_episode_length > 0 else 0
        
        episode_metrics.append({
            "episode_length": current_episode_length,
            "final_reward": current_cumulative_reward,
            "avg_clustering_coefficient": avg_clustering,
            "avg_spatial_variance": avg_spatial_variance,
            "infection_events": infection_events,
            "infection_rate": infection_events / current_episode_length if current_episode_length > 0 else 0
        })
    
    return episode_metrics

def main():
    parser = argparse.ArgumentParser(description="Generate Figure 10 comparing models trained with different movement patterns.")
    parser.add_argument("--model-base", type=str, required=True, help="Prefix for model directories (e.g., 'Fig10'). The script will append '_RandomMove' or '_WorkplaceCycle'.")
    parser.add_argument("--seeds", type=str, default="1,2,3", help="Comma-separated list of seeds for trained models.")
    parser.add_argument("--runs", type=int, default=30, help="Number of evaluation episodes per loaded model.")
    parser.add_argument("--output-dir", type=str, default="figures/", help="Directory to save the figures.")
    parser.add_argument("--eval-seed-base", type=int, default=10000, help="Base seed for evaluation runs (should be different from other figures).")
    
    args = parser.parse_args()

    try:
        train_seeds = [int(s.strip()) for s in args.seeds.split(',')]
    except ValueError:
        print("Error: --seeds must be a comma-separated list of integers.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    all_results_data = []

    for movement_type in MOVEMENT_TYPES:
        print(f"\n=== Processing Movement Type: {movement_type} ===")
        
        # Determine directory naming convention
        if movement_type == "continuous_random":
            movement_str = "RandomMove"
        elif movement_type == "workplace_home_cycle":
            movement_str = "WorkplaceCycle"
        else:
            movement_str = movement_type.replace("_", "")
            
        model_dir_pattern = f"{args.model_base}_{movement_str}"
        
        for train_seed in train_seeds:
            model_dir_name = f"{model_dir_pattern}_seed{train_seed}"
            model_dir_path = os.path.join("logs", model_dir_name)
            
            if not os.path.exists(model_dir_path):
                print(f"Warning: Model directory not found: {model_dir_path}. Skipping.")
                continue
                
            print(f"Processing trained model: {model_dir_name}")
            
            # Load model configuration
            try:
                config = load_model_config(model_dir_path)
                # Handle different config structures
                env_config = config.get("env_config", config.get("environment", {}))
            except FileNotFoundError:
                print(f"Warning: Config not found for {model_dir_name}. Skipping.")
                continue
            
            # Create environment with the same config as training
            env = create_env_from_config(env_config, seed=None)
            
            # Load the trained model
            model_file = os.path.join(model_dir_path, "final_model.zip")
            if not os.path.exists(model_file):
                print(f"Warning: Model file not found: {model_file}. Skipping.")
                continue
                
            model = PPO.load(model_file)
            
            # Run evaluation episodes
            eval_seed_offset = train_seed * 1000
            episode_results = run_evaluation_episodes_for_metrics(
                env, model, args.runs, args.eval_seed_base + eval_seed_offset
            )
            
            # Store results for trained agent
            for episode_result in episode_results:
                all_results_data.append({
                    "movement_type": movement_type,
                    "movement_label": MOVEMENT_LABELS[movement_type],
                    "agent_type": "Trained",
                    "train_seed": train_seed,
                    "final_reward": episode_result["final_reward"],
                    "episode_length": episode_result["episode_length"],
                    "avg_clustering_coefficient": episode_result["avg_clustering_coefficient"],
                    "avg_spatial_variance": episode_result["avg_spatial_variance"],
                    "infection_events": episode_result["infection_events"],
                    "infection_rate": episode_result["infection_rate"]
                })
            
            # Run baseline evaluations for each movement setting
            for baseline_name in ["random", "stationary", "greedy"]:
                baseline_results = run_baseline_evaluation(
                    env, baseline_name, args.runs, 
                    args.eval_seed_base + eval_seed_offset + 10000
                )
                
                baseline_display_name = baseline_name.capitalize()
                
                for episode_result in baseline_results:
                    all_results_data.append({
                        "movement_type": movement_type,
                        "movement_label": MOVEMENT_LABELS[movement_type],
                        "agent_type": baseline_display_name,
                        "train_seed": train_seed,
                        "final_reward": episode_result["final_reward"],
                        "episode_length": episode_result["episode_length"],
                        "avg_clustering_coefficient": episode_result["avg_clustering_coefficient"],
                        "avg_spatial_variance": episode_result["avg_spatial_variance"],
                        "infection_events": episode_result["infection_events"],
                        "infection_rate": episode_result["infection_rate"]
                    })

    # Convert to DataFrame
    df = pd.DataFrame(all_results_data)
    
    if df.empty:
        print("No data collected. Please check that model directories exist and are properly named.")
        return
    
    print(f"\nCollected data from {len(df)} total episodes.")
    print(f"Movement types: {df['movement_label'].unique()}")
    print(f"Agent types: {df['agent_type'].unique()}")

    # Group data by category, agent type, and training seed to get per-seed means (consistent with other figures)
    grouped_reward = df.groupby(['movement_label', 'agent_type', 'train_seed'])['final_reward'].mean().reset_index()
    grouped_episode = df.groupby(['movement_label', 'agent_type', 'train_seed'])['episode_length'].mean().reset_index()
    grouped_infection = df.groupby(['movement_label', 'agent_type', 'train_seed'])['infection_rate'].mean().reset_index()

    # Calculate and print summary statistics with standard deviations from per-seed means
    print("\n" + "="*60)
    print("STANDARD DEVIATION VALUES (from per-seed means)")
    print("="*60)

    reward_summary = grouped_reward.groupby(['movement_label', 'agent_type'])['final_reward'].agg(['mean', 'std']).reset_index()
    episode_summary = grouped_episode.groupby(['movement_label', 'agent_type'])['episode_length'].agg(['mean', 'std']).reset_index()
    infection_summary = grouped_infection.groupby(['movement_label', 'agent_type'])['infection_rate'].agg(['mean', 'std']).reset_index()

    print("\nFinal Reward Standard Deviations:")
    for _, row in reward_summary.iterrows():
        print(f"  {row['movement_label']} - {row['agent_type']}: SD = {row['std']:.3f}")

    print("\nEpisode Length Standard Deviations:")
    for _, row in episode_summary.iterrows():
        print(f"  {row['movement_label']} - {row['agent_type']}: SD = {row['std']:.3f}")

    print("\nInfection Rate Standard Deviations:")
    for _, row in infection_summary.iterrows():
        print(f"  {row['movement_label']} - {row['agent_type']}: SD = {row['std']:.3f}")

    # Generate visualizations with avg reward and episode length on top, infection spread on bottom
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Final Reward Comparison with Standard Deviation
    ax1 = axes[0, 0]
    sns.barplot(
        data=grouped_reward,
        x='movement_label',
        y='final_reward',
        hue='agent_type',
        order=PLOT_ORDER_X_AXIS,
        hue_order=AGENT_ORDER,
        ax=ax1,
        errorbar='sd'
    )
    ax1.set_title('Average Reward by Movement Pattern')
    ax1.set_xlabel('Movement Pattern')
    ax1.set_ylabel('Average Reward')
    ax1.legend(title='Agent Type', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add per-seed mean dots for reward plot
    n_categories = len(PLOT_ORDER_X_AXIS)
    n_agents = len(AGENT_ORDER)
    width = 0.8 / n_agents  # Total width divided by number of agent types
    for i, movement_label in enumerate(PLOT_ORDER_X_AXIS):
        for j, agent_type in enumerate(AGENT_ORDER):
            # Calculate x position for this agent type within this category
            x_pos = i + (j - (n_agents - 1) / 2) * width

            # Get data for this specific combination from grouped data
            subset = grouped_reward[(grouped_reward['movement_label'] == movement_label) &
                                   (grouped_reward['agent_type'] == agent_type)]
            if len(subset) > 0:
                seed_means = subset['final_reward']
                ax1.scatter([x_pos] * len(seed_means), seed_means, color='black', s=80,
                           zorder=10, marker='o', edgecolor='white', linewidth=1.5, alpha=0.8)

    # Add legend entry for per-seed mean dots
    ax1.scatter([], [], color='black', s=80, label='Per-seed Mean', edgecolor='white', linewidth=1.5)

    # Plot 2: Episode Length Comparison with Standard Deviation
    ax2 = axes[0, 1]
    sns.barplot(
        data=grouped_episode,
        x='movement_label',
        y='episode_length',
        hue='agent_type',
        order=PLOT_ORDER_X_AXIS,
        hue_order=AGENT_ORDER,
        ax=ax2,
        errorbar='sd'
    )
    ax2.set_title('Mean Episode Length by Movement Pattern')
    ax2.set_xlabel('Movement Pattern')
    ax2.set_ylabel('Mean Episode Length')
    ax2.legend(title='Agent Type', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add per-seed mean dots for episode length plot
    for i, movement_label in enumerate(PLOT_ORDER_X_AXIS):
        for j, agent_type in enumerate(AGENT_ORDER):
            # Calculate x position for this agent type within this category
            x_pos = i + (j - (n_agents - 1) / 2) * width

            # Get data for this specific combination from grouped data
            subset = grouped_episode[(grouped_episode['movement_label'] == movement_label) &
                                    (grouped_episode['agent_type'] == agent_type)]
            if len(subset) > 0:
                seed_means = subset['episode_length']
                ax2.scatter([x_pos] * len(seed_means), seed_means, color='black', s=80,
                           zorder=10, marker='o', edgecolor='white', linewidth=1.5, alpha=0.8)

    # Add legend entry for per-seed mean dots
    ax2.scatter([], [], color='black', s=80, label='Per-seed Mean', edgecolor='white', linewidth=1.5)

    # Plot 3: Infection Rate Comparison (spans both columns on bottom row)
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    sns.barplot(
        data=grouped_infection,
        x='movement_label',
        y='infection_rate',
        hue='agent_type',
        order=PLOT_ORDER_X_AXIS,
        hue_order=AGENT_ORDER,
        ax=ax3,
        errorbar='sd'
    )
    ax3.set_title('Infection Spread Rate by Movement Pattern')
    ax3.set_xlabel('Movement Pattern')
    ax3.set_ylabel('Infections per Timestep')
    ax3.legend(title='Agent Type', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add per-seed mean dots for infection rate plot
    for i, movement_label in enumerate(PLOT_ORDER_X_AXIS):
        for j, agent_type in enumerate(AGENT_ORDER):
            # Calculate x position for this agent type within this category
            x_pos = i + (j - (n_agents - 1) / 2) * width

            # Get data for this specific combination from grouped data
            subset = grouped_infection[(grouped_infection['movement_label'] == movement_label) &
                                      (grouped_infection['agent_type'] == agent_type)]
            if len(subset) > 0:
                seed_means = subset['infection_rate']
                ax3.scatter([x_pos] * len(seed_means), seed_means, color='black', s=80,
                           zorder=10, marker='o', edgecolor='white', linewidth=1.5, alpha=0.8)

    # Add legend entry for per-seed mean dots
    ax3.scatter([], [], color='black', s=80, label='Per-seed Mean', edgecolor='white', linewidth=1.5)

    # Remove the unused subplot
    fig.delaxes(axes[1, 0])
    fig.delaxes(axes[1, 1])
    
    plt.tight_layout()
    
    # Save figure as PDF (following existing pattern)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    figure_filename = f"figure10_movement_comparison_{timestamp}.pdf"
    figure_path = os.path.join(args.output_dir, figure_filename)
    plt.savefig(figure_path, bbox_inches='tight'); plt.close()
    print(f"Figure saved to {figure_path}")
    
    # Save raw data as CSV
    csv_filename = f"figure10_data_{timestamp}.csv"
    csv_path = os.path.join(args.output_dir, csv_filename)
    df.to_csv(csv_path, index=False)
    print(f"Raw data saved to {csv_path}")
    
    # Statistical analysis and summary table
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS - MOVEMENT PATTERN COMPARISON")
    print("="*80)
    
    # Summary statistics table
    summary_stats = df.groupby(['movement_label', 'agent_type']).agg({
        'final_reward': ['mean', 'std', 'count'],
        'episode_length': ['mean', 'std'],
        'avg_clustering_coefficient': ['mean', 'std'],
        'infection_rate': ['mean', 'std']
    }).round(3)
    
    print("\nSummary Statistics:")
    print(summary_stats)
    
    # Statistical significance tests between movement types
    print("\n" + "-"*60)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("-"*60)
    
    for agent_type in AGENT_ORDER:
        agent_data = df[df['agent_type'] == agent_type]
        
        random_rewards = agent_data[agent_data['movement_type'] == 'continuous_random']['final_reward']
        workplace_rewards = agent_data[agent_data['movement_type'] == 'workplace_home_cycle']['final_reward']
        
        if len(random_rewards) > 0 and len(workplace_rewards) > 0:
            # Mann-Whitney U test
            statistic, p_value = mannwhitneyu(random_rewards, workplace_rewards, alternative='two-sided')
            
            print(f"\n{agent_type} Agent:")
            print(f"  Random Movement:     Mean={random_rewards.mean():.3f}, Std={random_rewards.std():.3f}, N={len(random_rewards)}")
            print(f"  Workplace Movement:  Mean={workplace_rewards.mean():.3f}, Std={workplace_rewards.std():.3f}, N={len(workplace_rewards)}")
            print(f"  Mann-Whitney U test: U={statistic}, p={p_value:.6f}")
            print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
    
    # Analysis of clustering differences
    print("\n" + "-"*60)
    print("MOVEMENT PATTERN IMPACT ANALYSIS")
    print("-"*60)
    
    # Compare clustering coefficients
    random_clustering = df[df['movement_type'] == 'continuous_random']['avg_clustering_coefficient']
    workplace_clustering = df[df['movement_type'] == 'workplace_home_cycle']['avg_clustering_coefficient']
    
    print(f"\nClustering Analysis:")
    print(f"  Random Movement Clustering:     Mean={random_clustering.mean():.4f}, Std={random_clustering.std():.4f}")
    print(f"  Workplace Movement Clustering:  Mean={workplace_clustering.mean():.4f}, Std={workplace_clustering.std():.4f}")
    
    if len(random_clustering) > 0 and len(workplace_clustering) > 0:
        clustering_stat, clustering_p = mannwhitneyu(random_clustering, workplace_clustering, alternative='two-sided')
        print(f"  Clustering Difference Test: U={clustering_stat}, p={clustering_p:.6f}")
        print(f"  Significant clustering difference: {'Yes' if clustering_p < 0.05 else 'No'}")
    

if __name__ == "__main__":
    main()