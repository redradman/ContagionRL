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
import statsmodels.stats.multitest as smm
from scipy.stats import mannwhitneyu
from textwrap import wrap

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
    "max_nearest_distance": "Max Nearest Distance",
    "potential_field": "Potential Field"
}

# Order for plotting on the x-axis
PLOT_ORDER = [
    REWARD_FUNC_LABELS["alive"],
    REWARD_FUNC_LABELS["reduce_infection"],
    REWARD_FUNC_LABELS["alive_reduce_infection"],
    REWARD_FUNC_LABELS["max_nearest_distance"],
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
    parser.add_argument("--model-base", type=str, required=True, help="Base name for all models (e.g., Fig2)")
    parser.add_argument("--runs", type=int, default=30, help="Number of evaluation episodes per loaded model (default: 30).")
    parser.add_argument("--seeds", type=str, default="1,2,3", help="Comma-separated list of seeds for trained models (e.g., '1,2,3').")
    parser.add_argument("--output-dir", type=str, default="figures/", help="Directory to save the generated figures (default: figures/).")
    parser.add_argument("--eval-seed", type=int, default=2000, help="Base seed for evaluation runs (default: 2000).")
    parser.add_argument("--aggregate-seeds", action="store_true", help="Aggregate episode lengths per seed before plotting.")

    args = parser.parse_args()

    try:
        train_seeds = [int(s.strip()) for s in args.seeds.split(',')]
    except ValueError:
        print("Error: --seeds must be a comma-separated list of integers.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    all_results_data = []
    base_env_config_dict = None # To store the common environment config

    # Map reward function labels to their suffixes
    reward_func_suffixes = {
        REWARD_FUNC_LABELS["alive"]: "ConstantReward",
        REWARD_FUNC_LABELS["reduce_infection"]: "ReduceInfReward",
        REWARD_FUNC_LABELS["alive_reduce_infection"]: "ComboReward",
        REWARD_FUNC_LABELS["max_nearest_distance"]: "MaxNearestDistReward",
        REWARD_FUNC_LABELS["potential_field"]: "PotentialFieldReward",
    }

    model_bases = {
        label: f"{args.model_base}_{suffix}"
        for label, suffix in reward_func_suffixes.items()
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

    # Optionally aggregate per-seed means only
    if args.aggregate_seeds:
        results_df = results_df.groupby(['reward_function_label', 'model_train_seed'])['episode_length'].mean().reset_index()

    # --- Plotting (Violin + Box Overlay + Stripplot) ---
    plt.figure(figsize=(12, 6))
    y_metric_col = "episode_length"
    y_label = "Episode Duration (steps)"
    plot_order_filtered = [label for label in PLOT_ORDER if label in results_df['reward_function_label'].unique()]

    ax = sns.boxplot(
        x="reward_function_label", y=y_metric_col, data=results_df, order=plot_order_filtered,
        width=0.6, showfliers=False, saturation=1,
        boxprops=dict(facecolor='none', edgecolor='black', linewidth=2),
        medianprops=dict(color='black', linewidth=2), whiskerprops=dict(color='black', linewidth=2), capprops=dict(color='black', linewidth=2)
    )
    sns.violinplot(
        x="reward_function_label", y=y_metric_col, data=results_df, order=plot_order_filtered,
        width=0.8, inner=None, palette="muted", cut=0, alpha=0.5, ax=ax,
        hue="reward_function_label", legend=False
    )
    sns.stripplot(
        x="reward_function_label", y=y_metric_col, data=results_df, order=plot_order_filtered,
        color='black', alpha=0.3, jitter=0.2, size=5, ax=ax
    )

    # Overlay per-seed means as large black dots
    for i, reward_label in enumerate(plot_order_filtered):
        group = results_df[results_df['reward_function_label'] == reward_label]
        if 'model_train_seed' in group.columns:
            seed_means = group.groupby('model_train_seed')[y_metric_col].mean()
            ax.scatter([i]*len(seed_means), seed_means, color='black', s=120, zorder=10, marker='o', edgecolor='white', linewidth=2, label=None)
    # Add legend entry for per-seed mean dots
    ax.scatter([], [], color='black', s=120, label='Per-seed Mean', edgecolor='white', linewidth=2)
    ax.legend(fontsize=13)

    # --- Directional Mann–Whitney U test vs. Potential Field (Table 1 style) ---
    ref_label = 'Potential Field'
    comparisons = []
    raw_one_sided_pvals = []

    for reward_label in plot_order_filtered:
        if reward_label == ref_label:
            continue

        data1 = results_df[results_df['reward_function_label'] == reward_label]['episode_length']
        data2 = results_df[results_df['reward_function_label'] == ref_label]['episode_length']
        mean1, mean2 = np.mean(data1), np.mean(data2)

        # Two-sided test
        _, p_two = mannwhitneyu(data1, data2, alternative='two-sided')

        # Directional one-sided test
        if mean1 > mean2:
            _, p_one = mannwhitneyu(data1, data2, alternative='greater')
            winner = reward_label
        elif mean2 > mean1:
            _, p_one = mannwhitneyu(data2, data1, alternative='greater')
            winner = ref_label
        else:
            p_one = 1.0
            winner = "--"

        comparisons.append({
            "Reward A": reward_label,
            "Reward B": ref_label,
            "p_two": p_two,
            "p_one_raw": p_one,
            "winner": winner
        })
        raw_one_sided_pvals.append(p_one)

    # Apply Bonferroni correction
    _, p_one_corr, _, _ = smm.multipletests(raw_one_sided_pvals, method='bonferroni')

    # Add corrected values and significance
    def stars(p): return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

    for i, row in enumerate(comparisons):
        row["p_one_corr"] = p_one_corr[i]
        row["sig_two"] = stars(row["p_two"])
        row["sig_one"] = stars(p_one_corr[i])
        if row["sig_one"] == "n.s.":
            row["winner"] = "--"

    print("\nOne-Sided Mann–Whitney U Test Results (vs. Potential Field):")
    print("{:<24} {:<20} {:<12} {:<12} {:<8} {:<12} {:<8} {:<20}".format(
        "Reward A", "Reward B", "p (2-sided)", "p (1-sided)", "Sig (2)", "p (1) Corr", "Sig (1)", "Winner"
    ))
    print("-" * 112)
    for row in comparisons:
        print("{:<24} {:<20} {:<12.5g} {:<12.5g} {:<8} {:<12.5g} {:<8} {:<20}".format(
            row['Reward A'], row['Reward B'], row["p_two"], row["p_one_raw"],
            row["sig_two"], row["p_one_corr"], row["sig_one"], row["winner"]
        ))

    # --- Annotate the plot with significance stars from the one-sided tests ---
    # Find y positions for annotation
    y_max = results_df[y_metric_col].max()
    y_min = results_df[y_metric_col].min()
    y_range = y_max - y_min
    increment = y_range * 0.08 if y_range > 1e-9 else 0.1 * abs(y_max) if abs(y_max) > 1e-9 else 0.1
    current_y = y_max + increment * 0.5
    cat_pos = {cat: i for i, cat in enumerate(plot_order_filtered)}
    annotation_idx = 0
    for row in comparisons:
        if row['sig_one'] in ('*', '**', '***'):
            pos1 = cat_pos[row['Reward A']]
            pos2 = cat_pos[row['Reward B']]
            line_x = [pos1, pos1, pos2, pos2]
            line_y = [current_y, current_y + increment * 0.2, current_y + increment * 0.2, current_y]
            ax.plot(line_x, line_y, lw=1.0, c='black')
            text_x = (pos1 + pos2) / 2
            text_y = current_y + increment * 0.45 + annotation_idx * 0.08
            ax.text(text_x, text_y, row['sig_one'], rotation=0, ha='center', fontsize=13)
            annotation_idx += 1
            current_y += increment
    if len(plot_order_filtered) > 1:
        ax.set_ylim(top=current_y + increment * 0.2)

    plt.xlabel("Reward Function", fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    ax.tick_params(axis='x', labelsize=13, rotation=0)
    ax.tick_params(axis='y', labelsize=13)
    plt.tight_layout(pad=0.5)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    figure_filename = f"figure2_reward_comparison_{timestamp}.pdf"
    figure_path = os.path.join(args.output_dir, figure_filename)
    plt.savefig(figure_path, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {figure_path}")

    # --- Bar plot: Mean and 95% bootstrapped CI for each reward function ---
    means = []
    ci_lows = []
    ci_highs = []
    n_boot = 10000 # Standard number of bootstrap samples
    rng = np.random.default_rng(args.eval_seed) # Use a consistent RNG seed

    for label in plot_order_filtered: # plot_order_filtered should be from figure2.py context
        data = results_df[results_df['reward_function_label'] == label][y_metric_col].values
        mean = np.mean(data)
        if len(data) > 1:
            # Generate bootstrap samples of means
            boot_means = rng.choice(data, size=(n_boot, len(data)), replace=True).mean(axis=1)
            # Calculate 2.5th and 97.5th percentiles for 95% CI
            ci_low_val = np.percentile(boot_means, 2.5)
            ci_high_val = np.percentile(boot_means, 97.5)
        else: # Handle cases with insufficient data for CI
            ci_low_val = ci_high_val = mean
        means.append(mean)
        ci_lows.append(mean - ci_low_val) # Error bar length from mean to lower CI bound
        ci_highs.append(ci_high_val - mean) # Error bar length from mean to upper CI bound

    plt.figure(figsize=(10, 5)) # New figure for the bar plot
    bar_x = np.arange(len(plot_order_filtered))
    plt.bar(bar_x, means, yerr=[ci_lows, ci_highs], capsize=8,
            color=sns.color_palette("muted", n_colors=len(plot_order_filtered)), 
            edgecolor='black', linewidth=1.5)
    
    wrapped_bar_labels = ["\n".join(wrap(l, 15)) for l in plot_order_filtered] # Wrap labels
    plt.xticks(bar_x, wrapped_bar_labels, rotation=0, fontsize=11)
    plt.ylabel("Mean Episode Duration", fontsize=13)
    plt.xlabel("Reward Function", fontsize=13)
    plt.tight_layout(pad=0.5)

    bar_figure_filename = f"figure2_bar_means_{timestamp}.pdf" # Use the same timestamp or a new one
    bar_figure_path = os.path.join(args.output_dir, bar_figure_filename)
    plt.savefig(bar_figure_path, bbox_inches='tight')
    plt.close() # Close the bar plot figure
    print(f"Bar plot of means and 95% CI saved to {bar_figure_path}")

    csv_filename = f"figure2_data_{timestamp}.csv"
    csv_path = os.path.join(args.output_dir, csv_filename)
    results_df.to_csv(csv_path, index=False)
    print(f"Aggregated data saved to {csv_path}")

if __name__ == "__main__":
    main() 