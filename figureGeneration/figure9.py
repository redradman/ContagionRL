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

# Define Visibility Radius values and their labels for plotting
VISIBILITY_RADIUS_VALUES = [-1, 15]
VISIBILITY_RADIUS_LABELS = {
    -1: r"Full Visibility",
    15: r"Limited Visibility (r=15)",
}
# Order for plotting on the x-axis for the bar plot
PLOT_ORDER_X_AXIS = [VISIBILITY_RADIUS_LABELS[vr] for vr in VISIBILITY_RADIUS_VALUES]
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
        
        # Track additional visibility-specific metrics
        total_visible_humans = 0
        total_observations = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            current_episode_length += 1
            current_cumulative_reward += reward
            done = terminated or truncated
            
            # Count visible humans if using limited visibility
            if env.use_visibility_flag and 'humans_features' in obs:
                humans_features = obs['humans_features']
                features_per_human = len(humans_features) // env.n_humans
                if features_per_human > 6:  # Has visibility flag
                    for h_idx in range(env.n_humans):
                        base_idx = h_idx * features_per_human
                        visibility_flag = humans_features[base_idx]
                        if visibility_flag > 0.5:  # Visible
                            total_visible_humans += 1
                    total_observations += 1
                
        # Calculate average visibility ratio for this episode
        avg_visibility_ratio = total_visible_humans / max(total_observations * env.n_humans, 1)
        
        episode_metrics.append({
            "episode_length": current_episode_length,
            "final_reward": current_cumulative_reward,
            "avg_visibility_ratio": avg_visibility_ratio
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
        
        # Track visibility metrics
        total_visible_humans = 0
        total_observations = 0
        
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
            
            # Count visible humans
            if env.use_visibility_flag and 'humans_features' in obs:
                humans_features = obs['humans_features']
                features_per_human = len(humans_features) // env.n_humans
                if features_per_human > 6:  # Has visibility flag
                    for h_idx in range(env.n_humans):
                        base_idx = h_idx * features_per_human
                        visibility_flag = humans_features[base_idx]
                        if visibility_flag > 0.5:  # Visible
                            total_visible_humans += 1
                    total_observations += 1
        
        avg_visibility_ratio = total_visible_humans / max(total_observations * env.n_humans, 1)
        
        episode_metrics.append({
            "episode_length": current_episode_length,
            "final_reward": current_cumulative_reward,
            "avg_visibility_ratio": avg_visibility_ratio
        })
    
    return episode_metrics

def main():
    parser = argparse.ArgumentParser(description="Generate Figure 9 comparing models trained with different visibility radius settings.")
    parser.add_argument("--model-base", type=str, required=True, help="Prefix for model directories (e.g., 'Fig9'). The script will append '_FullVis' or '_Vis15'.")
    parser.add_argument("--seeds", type=str, default="1,2,3", help="Comma-separated list of seeds for trained models.")
    parser.add_argument("--runs", type=int, default=30, help="Number of evaluation episodes per loaded model.")
    parser.add_argument("--output-dir", type=str, default="figures/", help="Directory to save the figures.")
    parser.add_argument("--eval-seed-base", type=int, default=9000, help="Base seed for evaluation runs (should be different from other figures).")
    
    args = parser.parse_args()

    try:
        train_seeds = [int(s.strip()) for s in args.seeds.split(',')]
    except ValueError:
        print("Error: --seeds must be a comma-separated list of integers.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    all_results_data = []

    for visibility_radius in VISIBILITY_RADIUS_VALUES:
        print(f"\n=== Processing Visibility Radius: {visibility_radius} ===")
        
        # Determine directory naming convention
        visibility_str = "FullVis" if visibility_radius == -1 else f"Vis{visibility_radius}"
        model_dir_pattern = f"{args.model_base}_{visibility_str}"
        
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
                    "visibility_radius": visibility_radius,
                    "visibility_label": VISIBILITY_RADIUS_LABELS[visibility_radius],
                    "agent_type": "Trained",
                    "train_seed": train_seed,
                    "final_reward": episode_result["final_reward"],
                    "episode_length": episode_result["episode_length"],
                    "avg_visibility_ratio": episode_result["avg_visibility_ratio"]
                })
            
            # Run baseline evaluations for each visibility setting
            for baseline_name in ["random", "stationary", "greedy"]:
                baseline_results = run_baseline_evaluation(
                    env, baseline_name, args.runs, 
                    args.eval_seed_base + eval_seed_offset + 10000
                )
                
                baseline_display_name = baseline_name.capitalize()
                
                for episode_result in baseline_results:
                    all_results_data.append({
                        "visibility_radius": visibility_radius,
                        "visibility_label": VISIBILITY_RADIUS_LABELS[visibility_radius],
                        "agent_type": baseline_display_name,
                        "train_seed": train_seed,
                        "final_reward": episode_result["final_reward"],
                        "episode_length": episode_result["episode_length"],
                        "avg_visibility_ratio": episode_result["avg_visibility_ratio"]
                    })

    # Convert to DataFrame
    df = pd.DataFrame(all_results_data)
    
    if df.empty:
        print("No data collected. Please check that model directories exist and are properly named.")
        return
    
    print(f"\nCollected data from {len(df)} total episodes.")
    print(f"Visibility settings: {df['visibility_label'].unique()}")
    print(f"Agent types: {df['agent_type'].unique()}")

    # Group data by category, agent type, and training seed to get per-seed means (consistent with other figures)
    grouped_reward = df.groupby(['visibility_label', 'agent_type', 'train_seed'])['final_reward'].mean().reset_index()
    grouped_episode = df.groupby(['visibility_label', 'agent_type', 'train_seed'])['episode_length'].mean().reset_index()

    # Calculate and print summary statistics with standard deviations from per-seed means
    print("\n" + "="*60)
    print("STANDARD DEVIATION VALUES (from per-seed means)")
    print("="*60)

    reward_summary = grouped_reward.groupby(['visibility_label', 'agent_type'])['final_reward'].agg(['mean', 'std']).reset_index()
    episode_summary = grouped_episode.groupby(['visibility_label', 'agent_type'])['episode_length'].agg(['mean', 'std']).reset_index()

    print("\nFinal Reward Standard Deviations:")
    for _, row in reward_summary.iterrows():
        print(f"  {row['visibility_label']} - {row['agent_type']}: SD = {row['std']:.3f}")

    print("\nEpisode Length Standard Deviations:")
    for _, row in episode_summary.iterrows():
        print(f"  {row['visibility_label']} - {row['agent_type']}: SD = {row['std']:.3f}")

    # Generate visualizations - 2 side-by-side charts as requested
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Final Reward Comparison with Standard Deviation
    ax1 = axes[0]
    sns.barplot(
        data=grouped_reward,
        x='visibility_label',
        y='final_reward',
        hue='agent_type',
        order=PLOT_ORDER_X_AXIS,
        hue_order=AGENT_ORDER,
        ax=ax1,
        errorbar='sd'
    )
    ax1.set_title('Average Reward by Visibility Setting')
    ax1.set_xlabel('Visibility Condition')
    ax1.set_ylabel('Average Reward')
    ax1.legend(title='Agent Type', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add per-seed mean dots for reward plot
    n_categories = len(PLOT_ORDER_X_AXIS)
    n_agents = len(AGENT_ORDER)
    width = 0.8 / n_agents  # Total width divided by number of agent types
    for i, visibility_label in enumerate(PLOT_ORDER_X_AXIS):
        for j, agent_type in enumerate(AGENT_ORDER):
            # Calculate x position for this agent type within this category
            x_pos = i + (j - (n_agents - 1) / 2) * width

            # Get data for this specific combination from grouped data
            subset = grouped_reward[(grouped_reward['visibility_label'] == visibility_label) &
                                   (grouped_reward['agent_type'] == agent_type)]
            if len(subset) > 0:
                seed_means = subset['final_reward']
                ax1.scatter([x_pos] * len(seed_means), seed_means, color='black', s=80,
                           zorder=10, marker='o', edgecolor='white', linewidth=1.5, alpha=0.8)

    # Add legend entry for per-seed mean dots
    ax1.scatter([], [], color='black', s=80, label='Per-seed Mean', edgecolor='white', linewidth=1.5)

    # Plot 2: Episode Length Comparison with Standard Deviation
    ax2 = axes[1]
    sns.barplot(
        data=grouped_episode,
        x='visibility_label',
        y='episode_length',
        hue='agent_type',
        order=PLOT_ORDER_X_AXIS,
        hue_order=AGENT_ORDER,
        ax=ax2,
        errorbar='sd'
    )
    ax2.set_title('Mean Episode Length by Visibility Setting')
    ax2.set_xlabel('Visibility Condition')
    ax2.set_ylabel('Mean Episode Length')
    ax2.legend(title='Agent Type', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add per-seed mean dots for episode length plot
    for i, visibility_label in enumerate(PLOT_ORDER_X_AXIS):
        for j, agent_type in enumerate(AGENT_ORDER):
            # Calculate x position for this agent type within this category
            x_pos = i + (j - (n_agents - 1) / 2) * width

            # Get data for this specific combination from grouped data
            subset = grouped_episode[(grouped_episode['visibility_label'] == visibility_label) &
                                    (grouped_episode['agent_type'] == agent_type)]
            if len(subset) > 0:
                seed_means = subset['episode_length']
                ax2.scatter([x_pos] * len(seed_means), seed_means, color='black', s=80,
                           zorder=10, marker='o', edgecolor='white', linewidth=1.5, alpha=0.8)

    # Add legend entry for per-seed mean dots
    ax2.scatter([], [], color='black', s=80, label='Per-seed Mean', edgecolor='white', linewidth=1.5)
    
    plt.tight_layout()
    
    # Save figure as PDF (following existing pattern)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    figure_filename = f"figure9_visibility_comparison_{timestamp}.pdf"
    figure_path = os.path.join(args.output_dir, figure_filename)
    plt.savefig(figure_path, bbox_inches='tight'); plt.close()
    print(f"Figure saved to {figure_path}")
    
    # Save raw data as CSV
    csv_filename = f"figure9_data_{timestamp}.csv"
    csv_path = os.path.join(args.output_dir, csv_filename)
    df.to_csv(csv_path, index=False)
    print(f"Raw data saved to {csv_path}")
    
    # Statistical analysis and summary table
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS - VISIBILITY RADIUS COMPARISON")
    print("="*80)
    
    # Summary statistics table
    summary_stats = df.groupby(['visibility_label', 'agent_type']).agg({
        'final_reward': ['mean', 'std', 'count'],
        'episode_length': ['mean', 'std'],
        'avg_visibility_ratio': ['mean', 'std']
    }).round(3)
    
    print("\nSummary Statistics:")
    print(summary_stats)
    
    # Statistical significance tests between visibility conditions
    print("\n" + "-"*60)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("-"*60)
    
    for agent_type in AGENT_ORDER:
        agent_data = df[df['agent_type'] == agent_type]
        
        full_vis_rewards = agent_data[agent_data['visibility_radius'] == -1]['final_reward']
        limited_vis_rewards = agent_data[agent_data['visibility_radius'] == 15]['final_reward']
        
        if len(full_vis_rewards) > 0 and len(limited_vis_rewards) > 0:
            # Mann-Whitney U test
            statistic, p_value = mannwhitneyu(full_vis_rewards, limited_vis_rewards, alternative='two-sided')
            
            print(f"\n{agent_type} Agent:")
            print(f"  Full Visibility:    Mean={full_vis_rewards.mean():.3f}, Std={full_vis_rewards.std():.3f}, N={len(full_vis_rewards)}")
            print(f"  Limited Visibility: Mean={limited_vis_rewards.mean():.3f}, Std={limited_vis_rewards.std():.3f}, N={len(limited_vis_rewards)}")
            print(f"  Mann-Whitney U test: U={statistic}, p={p_value:.6f}")
            print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
    

if __name__ == "__main__":
    main()