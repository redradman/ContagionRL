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
from typing import Dict, List, Any, Optional
from scipy import stats
import statsmodels.stats.multitest as smm

# Add the parent directory to the path to access project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import SIRSEnvironment
# utils.Human and utils.STATE_DICT are not directly needed here if only running pre-trained models

# tueplots styling
from tueplots import bundles
plt.rcParams.update(bundles.neurips2024())
plt.rcParams["figure.dpi"] = 300
sns.set_style("whitegrid")

# Define labels and order for these specific reward functions
REWARD_FUNC_LABELS = {
    "alive": "Constant",
    "reduce_infection": "Reduce Infection",
    "alive_reduce_infection": "Constant + Reduce Inf.",
    "potential_field": "Potential Field"
}

# Order for plotting on the x-axis
PLOT_ORDER = [
    REWARD_FUNC_LABELS["alive"],
    REWARD_FUNC_LABELS["reduce_infection"],
    REWARD_FUNC_LABELS["alive_reduce_infection"],
    REWARD_FUNC_LABELS["potential_field"]
]

def load_model_config(model_path: str) -> Dict[str, Any]:
    """Load the configuration file associated with a specific model."""
    config_path = os.path.join(os.path.dirname(model_path), "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path} for model {model_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def create_env_from_config(env_config_dict: Dict[str, Any], seed: Optional[int] = None) -> SIRSEnvironment:
    """Create a SIRS environment from a configuration dictionary."""
    config_copy = env_config_dict.copy()
    config_copy["render_mode"] = None # Ensure no rendering during benchmark
    env = SIRSEnvironment(**config_copy)
    env.reset(seed=seed)
    return env

def run_evaluation_episodes(
    env: SIRSEnvironment, 
    model: PPO, 
    num_episodes: int,
    base_eval_seed: int 
) -> List[int]:
    """Run multiple evaluation episodes for a given model and collect episode lengths."""
    episode_lengths = []
    for i in range(num_episodes):
        eval_seed_for_run = base_eval_seed + i
        obs, _ = env.reset(seed=eval_seed_for_run)
        done = False
        current_episode_length = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            current_episode_length += 1
            done = terminated or truncated
        episode_lengths.append(current_episode_length)
    return episode_lengths

def main():
    parser = argparse.ArgumentParser(description="Generate Figure 2 comparing models trained with different reward functions.")
    parser.add_argument("--pf-base", type=str, required=True, help="Base name for Potential Field models.")
    parser.add_argument("--alive-base", type=str, required=True, help="Base name for Constant (alive) reward models.")
    parser.add_argument("--reduce-base", type=str, required=True, help="Base name for Reduce Infection models.")
    parser.add_argument("--combo-base", type=str, required=True, help="Base name for Constant + Reduce Infection models.")
    
    parser.add_argument("--runs", type=int, default=30, help="Number of evaluation episodes per loaded model (default: 30).")
    parser.add_argument("--seeds", type=str, default="1,2,3", help="Comma-separated list of seeds for trained models (e.g., '1,2,3').")
    parser.add_argument("--output-dir", type=str, default="figures/", help="Directory to save the generated figures (default: figures/).")
    parser.add_argument("--eval-seed", type=int, default=2000, help="Base seed for evaluation runs (default: 2000).")

    args = parser.parse_args()

    try:
        train_seeds = [int(s.strip()) for s in args.seeds.split(',')]
    except ValueError:
        print("Error: --seeds must be a comma-separated list of integers.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    all_results_data = []
    base_env_config_dict = None # To store the common environment config

    model_bases = {
        REWARD_FUNC_LABELS["alive"]: args.alive_base,
        REWARD_FUNC_LABELS["reduce_infection"]: args.reduce_base,
        REWARD_FUNC_LABELS["alive_reduce_infection"]: args.combo_base,
        REWARD_FUNC_LABELS["potential_field"]: args.pf_base,
    }

    # Offset for evaluation seeds per category to ensure uniqueness
    category_seed_offset = 0

    for reward_label, model_base_name in model_bases.items():
        print(f"Processing models for reward function: {reward_label}")
        for train_seed in tqdm(train_seeds, desc=f"Models for {reward_label}"):
            model_dir_name = f"{model_base_name}_seed{train_seed}"
            model_path = os.path.join("logs", model_dir_name, "best_model.zip")

            if not os.path.exists(model_path):
                print(f"Warning: Model file not found for {reward_label} (seed {train_seed}) at {model_path}. Skipping.")
                continue
            
            try:
                if base_env_config_dict is None:
                    model_config = load_model_config(model_path)
                    base_env_config_dict = model_config.get("environment")
                    if base_env_config_dict is None:
                        print(f"Error: 'environment' key missing in config for {model_path}. Cannot proceed.")
                        return
                
                # Create env with a unique seed for this model loading instance to avoid PPO internal state issues
                # This seed is for env creation, not necessarily for the episode runs themselves.
                env_creation_seed = args.eval_seed + category_seed_offset + train_seed
                env = create_env_from_config(base_env_config_dict, seed=env_creation_seed) 
                model = PPO.load(model_path, env=env) # Load model with its environment

                # Define a unique base seed for the sequence of evaluation runs for *this specific model*
                model_eval_run_base_seed = args.eval_seed + category_seed_offset * 100 + train_seed * args.runs

                episode_lengths = run_evaluation_episodes(env, model, args.runs, model_eval_run_base_seed)
                
                for length in episode_lengths:
                    all_results_data.append({
                        "reward_function_label": reward_label,
                        "model_train_seed": train_seed,
                        "episode_length": length
                    })
                env.close()
            except Exception as e:
                print(f"Error processing model {model_path} for {reward_label}: {e}")
        category_seed_offset += 1 # Increment for next category

    if not all_results_data:
        print("No data collected from any models. Exiting.")
        return

    results_df = pd.DataFrame(all_results_data)

    # --- Plotting (Violin + Box Overlay + Stripplot) ---
    plt.figure(figsize=(8, 6)) # Adjusted for potentially more categories or longer labels
    y_metric_col = "episode_length"
    y_label = "Episode Duration (steps)"

    # Filter out categories if they have no data
    plot_order_filtered = [label for label in PLOT_ORDER if label in results_df['reward_function_label'].unique()]

    # 1. Boxplot (outlines)
    ax = sns.boxplot(
        x="reward_function_label", y=y_metric_col, data=results_df, order=plot_order_filtered,
        width=0.6, showfliers=False, saturation=1,
        boxprops=dict(facecolor='none', edgecolor='black'),
        medianprops=dict(color='black'), whiskerprops=dict(color='black'), capprops=dict(color='black')
    )

    # 2. Violin plot (semi-transparent overlay)
    sns.violinplot(
        x="reward_function_label", y=y_metric_col, data=results_df, order=plot_order_filtered,
        width=0.8, inner=None, palette="muted", cut=0, alpha=0.5, ax=ax,
        hue="reward_function_label", legend=False
    )

    # 3. Stripplot (individual points)
    sns.stripplot(
        x="reward_function_label", y=y_metric_col, data=results_df, order=plot_order_filtered,
        color='black', alpha=0.3, jitter=0.2, size=3, ax=ax
    )

    # --- Statistical Annotations ---
    potential_field_data = results_df[results_df['reward_function_label'] == REWARD_FUNC_LABELS['potential_field']][y_metric_col]
    comparisons_data = []
    p_values_uncorrected = []

    for reward_label_to_compare in plot_order_filtered:
        if reward_label_to_compare == REWARD_FUNC_LABELS['potential_field']:
            continue
        
        compare_data = results_df[results_df['reward_function_label'] == reward_label_to_compare][y_metric_col]
        if len(potential_field_data) > 0 and len(compare_data) > 0:
            try:
                if len(set(potential_field_data)) == 1 and len(set(compare_data)) == 1 and potential_field_data.iloc[0] == compare_data.iloc[0]:
                    p_val = 1.0
                else:
                    _, p_val = stats.mannwhitneyu(potential_field_data, compare_data, alternative='two-sided')
                comparisons_data.append((REWARD_FUNC_LABELS['potential_field'], reward_label_to_compare))
                p_values_uncorrected.append(p_val)
            except ValueError as e:
                print(f"Warning: Mann-Whitney U test failed for PF vs {reward_label_to_compare}: {e}")
                comparisons_data.append((REWARD_FUNC_LABELS['potential_field'], reward_label_to_compare))
                p_values_uncorrected.append(np.nan)
        else:
            comparisons_data.append((REWARD_FUNC_LABELS['potential_field'], reward_label_to_compare))
            p_values_uncorrected.append(np.nan)

    if p_values_uncorrected:
        valid_indices = [i for i, p in enumerate(p_values_uncorrected) if not np.isnan(p)]
        valid_p_values = [p_values_uncorrected[i] for i in valid_indices]
        valid_comparisons = [comparisons_data[i] for i in valid_indices]

        if valid_p_values:
            _ , pvals_corrected, _, _ = smm.multipletests(valid_p_values, alpha=0.05, method='bonferroni')
            y_max = results_df[y_metric_col].max()
            y_min = results_df[y_metric_col].min()
            y_range = y_max - y_min
            num_valid_comparisons = len(valid_comparisons)

            if y_range <= 1e-9: 
                increment_base = 0.1 * abs(y_max) if abs(y_max) > 1e-9 else 0.1
            else: 
                increment_base = y_range * 0.08
            
            increment_total_height_factor = 0.1 * num_valid_comparisons
            if y_range > 1e-9: 
                increment = max(increment_base, y_range * increment_total_height_factor / num_valid_comparisons if num_valid_comparisons > 0 else increment_base)
            else: 
                increment = increment_base

            current_y = y_max + increment * 0.5
            cat_pos = {cat: i for i, cat in enumerate(plot_order_filtered)}

            for i, (cat1, cat2) in enumerate(valid_comparisons): # cat1 is always Potential Field here
                p_corrected = pvals_corrected[i]
                pos1 = cat_pos[cat1]
                pos2 = cat_pos[cat2]

                line_x = [pos1, pos1, pos2, pos2]
                line_y = [current_y, current_y + increment * 0.2, current_y + increment * 0.2, current_y]
                ax.plot(line_x, line_y, lw=1.0, c='black')

                significance = 'ns'
                if p_corrected < 0.001: 
                    significance = '***'
                elif p_corrected < 0.01: 
                    significance = '**'
                elif p_corrected < 0.05: 
                    significance = '*'
                
                text_x = (pos1 + pos2) / 2
                text_y = current_y + increment * 0.25
                ax.text(text_x, text_y, significance, ha='center', va='bottom', fontsize=8)
                current_y += increment
            
            if num_valid_comparisons > 0: 
                ax.set_ylim(top=current_y + increment * 0.2)

    title_str = "Comparison of Reward Functions\nEpisode Duration"
    plt.title(title_str, fontsize=10)
    plt.xlabel("Reward Function Configuration", fontsize=9)
    plt.ylabel(y_label, fontsize=9)
    ax.tick_params(axis='x', labelsize=8, rotation=15)
    ax.tick_params(axis='y', labelsize=8)
    
    plt.tight_layout(pad=0.5)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use a more generic name for the figure, as model base names are many
    figure_filename = f"figure2_reward_comparison_{timestamp}.pdf"
    figure_path = os.path.join(args.output_dir, figure_filename)
    
    plt.savefig(figure_path, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {figure_path}")

    csv_filename = f"figure2_data_{timestamp}.csv"
    csv_path = os.path.join(args.output_dir, csv_filename)
    results_df.to_csv(csv_path, index=False)
    print(f"Aggregated data saved to {csv_path}")

if __name__ == "__main__":
    main() 