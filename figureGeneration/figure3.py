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
import statsmodels.stats.multitest as smm

# Add the parent directory to the path to access project modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from environment import SIRSEnvironment

# tueplots styling
from tueplots import bundles
plt.rcParams.update(bundles.neurips2024())
plt.rcParams["figure.dpi"] = 300
sns.set_style("whitegrid")

# Define Beta values and their labels for plotting
BETA_VALUES = [0.2, 0.5, 0.7, 0.9]
BETA_LABELS = { # For display on the y-axis of the plot
    0.2: "Beta = 0.2",
    0.5: "Beta = 0.5",
    0.7: "Beta = 0.7",
    0.9: "Beta = 0.9"
}
# Order for plotting on the y-axis for horizontal boxplots
PLOT_ORDER = [BETA_LABELS[b] for b in BETA_VALUES]

def load_model_config(model_path: str) -> Dict[str, Any]:
    config_path = os.path.join(os.path.dirname(model_path), "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path} for model {model_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def create_env_from_config(env_config_dict: Dict[str, Any], seed: Optional[int] = None) -> SIRSEnvironment:
    config_copy = env_config_dict.copy()
    config_copy["render_mode"] = None
    env = SIRSEnvironment(**config_copy)
    env.reset(seed=seed)
    return env

def run_evaluation_episodes_for_metrics(
    env: SIRSEnvironment, 
    model: PPO, 
    num_episodes: int,
    base_eval_seed: int 
) -> List[Dict[str, Any]]:
    """Run multiple evaluation episodes and collect episode lengths and final rewards."""
    episode_metrics = []
    for i in range(num_episodes):
        eval_seed_for_run = base_eval_seed + i
        obs, _ = env.reset(seed=eval_seed_for_run)
        done = False
        current_episode_length = 0
        current_cumulative_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            current_episode_length += 1
            current_cumulative_reward += reward
            done = terminated or truncated
        episode_metrics.append({
            "episode_length": current_episode_length,
            "final_reward": current_cumulative_reward
        })
    return episode_metrics

def main():
    parser = argparse.ArgumentParser(description="Generate Figure 3 comparing models trained with different beta values.")
    parser.add_argument("--model-base-prefix", type=str, required=True, help="Prefix for model directories (e.g., 'Fig3_Beta'). The script will append beta values and seed numbers.")
    parser.add_argument("--seeds", type=str, default="1,2,3", help="Comma-separated list of seeds for trained models.")
    parser.add_argument("--runs", type=int, default=30, help="Number of evaluation episodes per loaded model.")
    parser.add_argument("--output-dir", type=str, default="figures/", help="Directory to save the figures.")
    parser.add_argument("--eval-seed-base", type=int, default=3000, help="Base seed for evaluation runs.")
    
    args = parser.parse_args()

    try:
        train_seeds = [int(s.strip()) for s in args.seeds.split(',')]
    except ValueError:
        print("Error: --seeds must be a comma-separated list of integers.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    all_results_data = []
    base_env_config_dict = None
    category_seed_offset = 0 # To ensure unique eval seeds across categories if needed

    for beta_value in BETA_VALUES:
        beta_label = BETA_LABELS[beta_value]
        beta_str_for_name = str(beta_value).replace('.', 'p')
        model_base_name_for_beta = f"{args.model_base_prefix}{beta_str_for_name}"
        
        print(f"Processing models for {beta_label}")
        for train_seed in tqdm(train_seeds, desc=f"Models for {beta_label}"):
            model_dir_name = f"{model_base_name_for_beta}_seed{train_seed}"
            model_path = os.path.join("logs", model_dir_name, "best_model.zip")

            if not os.path.exists(model_path):
                print(f"Warning: Model file not found for {beta_label} (seed {train_seed}) at {model_path}. Skipping.")
                continue
            
            try:
                model_config = load_model_config(model_path)
                # It's crucial that the env_config loaded here corresponds to the one used for training this model (correct beta)
                current_env_config = model_config.get("environment")
                if current_env_config is None or current_env_config.get('beta') != beta_value:
                    print(f"Warning: Beta value in loaded config ({current_env_config.get('beta') if current_env_config else 'N/A'}) for {model_path} does not match expected beta {beta_value}. Skipping.")
                    continue
                
                env_creation_seed = args.eval_seed_base + category_seed_offset + train_seed 
                env = create_env_from_config(current_env_config, seed=env_creation_seed) 
                model = PPO.load(model_path, env=env)

                model_eval_run_base_seed = args.eval_seed_base + category_seed_offset * 100 + train_seed * args.runs
                eval_metrics_list = run_evaluation_episodes_for_metrics(env, model, args.runs, model_eval_run_base_seed)
                
                for metrics in eval_metrics_list:
                    all_results_data.append({
                        "beta_value_label": beta_label,
                        "beta_value": beta_value,
                        "model_train_seed": train_seed,
                        "episode_length": metrics["episode_length"],
                        "final_reward": metrics["final_reward"]
                    })
                env.close()
            except Exception as e:
                print(f"Error processing model {model_path} for {beta_label}: {e}")
        category_seed_offset += len(BETA_VALUES) # Increment offset for next beta group to ensure unique eval seeds

    if not all_results_data:
        print("No data collected from any models. Exiting.")
        return

    results_df = pd.DataFrame(all_results_data)

    # --- Plotting ---    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) # 1 row, 2 columns. Adjusted figsize for tueplots.

    metrics_to_plot = [
        {"col": "episode_length", "label": "Episode Duration (steps)", "ax_idx": 0},
        {"col": "final_reward", "label": "Final Cumulative Reward", "ax_idx": 1}
    ]

    for metric_info in metrics_to_plot:
        ax = axes[metric_info["ax_idx"]]
        y_col = "beta_value_label" # Categories on y-axis for horizontal plot
        x_col = metric_info["col"]

        sns.boxplot(
            x=x_col, y=y_col, data=results_df, order=PLOT_ORDER, orient="h",
            width=0.6, showfliers=False, saturation=1,
            boxprops=dict(facecolor='none', edgecolor='black'),
            medianprops=dict(color='black'), whiskerprops=dict(color='black'), capprops=dict(color='black'),
            ax=ax
        )
        sns.stripplot(
            x=x_col, y=y_col, data=results_df, order=PLOT_ORDER, orient="h",
            color='black', alpha=0.3, jitter=0.2, size=3, ax=ax
        )

        ax.set_xlabel(metric_info["label"], fontsize=9)
        if metric_info["ax_idx"] == 0: # Only set y-label for the first plot
            ax.set_ylabel("Beta Value", fontsize=9)
        else:
            ax.set_ylabel("") # No y-label for the second plot
        
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

        # Statistical Annotations (comparing each beta to beta=0.2 using per-seed aggregates)
        reference_label = BETA_LABELS[0.2]
        
        # Aggregate reference data by model_train_seed for the current metric (x_col)
        ref_beta_data = results_df[results_df[y_col] == reference_label]
        if 'model_train_seed' in ref_beta_data.columns and not ref_beta_data['model_train_seed'].isnull().all():
            reference_data_for_test = ref_beta_data.groupby('model_train_seed')[x_col].mean()
        else:
            print(f"Warning: 'model_train_seed' column missing/all null for reference Beta ({reference_label}) on metric {x_col}. Using raw data.")
            reference_data_for_test = ref_beta_data[x_col] # Fallback

        comparisons_data_stats = []
        p_values_uncorrected_stats = []

        for beta_label_to_compare in PLOT_ORDER:
            if beta_label_to_compare == reference_label:
                continue
            
            # Aggregate comparison beta data by model_train_seed for the current metric (x_col)
            compare_beta_data = results_df[results_df[y_col] == beta_label_to_compare]
            if 'model_train_seed' in compare_beta_data.columns and not compare_beta_data['model_train_seed'].isnull().all():
                compare_data_for_test = compare_beta_data.groupby('model_train_seed')[x_col].mean()
            else:
                print(f"Warning: 'model_train_seed' column missing/all null for Beta ({beta_label_to_compare}) on metric {x_col}. Using raw data.")
                compare_data_for_test = compare_beta_data[x_col] # Fallback

            if len(reference_data_for_test) > 0 and len(compare_data_for_test) > 0:
                try:
                    # Check for identical constant values on aggregated data
                    if len(set(reference_data_for_test)) == 1 and len(set(compare_data_for_test)) == 1 and reference_data_for_test.iloc[0] == compare_data_for_test.iloc[0]:
                        p_val = 1.0
                        print(f"Skipping Mann-Whitney U for {reference_label} (agg) vs {beta_label_to_compare} (agg) on metric {x_col}: Identical constant values.")
                    else:
                        # Ensure enough data points for the test after aggregation
                        if len(reference_data_for_test) < 2 or len(compare_data_for_test) < 2:
                            print(f"Warning: Not enough data for Mann-Whitney U after aggregation for {reference_label} vs {beta_label_to_compare} on {x_col} (Ref: {len(reference_data_for_test)}, Comp: {len(compare_data_for_test)}). p_val=NaN.")
                            p_val = np.nan
                        else:
                            _, p_val = stats.mannwhitneyu(reference_data_for_test, compare_data_for_test, alternative='two-sided')
                    
                    comparisons_data_stats.append((reference_label, beta_label_to_compare))
                    p_values_uncorrected_stats.append(p_val)
                except ValueError as e:
                    print(f"Warning: Mann-Whitney U test failed for {reference_label} (agg) vs {beta_label_to_compare} (agg) on metric {x_col}: {e}")
                    p_values_uncorrected_stats.append(np.nan)
                    comparisons_data_stats.append((reference_label, beta_label_to_compare)) # Ensure added if exception before append
            else:
                p_values_uncorrected_stats.append(np.nan)
                comparisons_data_stats.append((reference_label, beta_label_to_compare)) 
        
        if p_values_uncorrected_stats:
            valid_indices = [i for i, p in enumerate(p_values_uncorrected_stats) if not np.isnan(p)]
            valid_p_values = [p_values_uncorrected_stats[i] for i in valid_indices]
            # Ensure comparisons_data_stats is filtered consistently with valid_p_values
            valid_comparisons_stats = [comparisons_data_stats[i] for i in valid_indices]

            if valid_p_values:
                _, pvals_corrected, _, _ = smm.multipletests(valid_p_values, alpha=0.05, method='bonferroni')
                x_max_for_annot = results_df[x_col].max()
                x_min_for_annot = results_df[x_col].min()
                x_range_for_annot = x_max_for_annot - x_min_for_annot
                num_valid_comp = len(valid_comparisons_stats)

                if x_range_for_annot <= 1e-9: increment_base_x = 0.1 * abs(x_max_for_annot) if abs(x_max_for_annot) > 1e-9 else 0.1
                else: increment_base_x = x_range_for_annot * 0.08

                increment_total_width_factor = 0.1 * num_valid_comp
                if x_range_for_annot > 1e-9: increment_x = max(increment_base_x, x_range_for_annot * increment_total_width_factor / num_valid_comp if num_valid_comp > 0 else increment_base_x)
                else: increment_x = increment_base_x

                current_x_annot = x_max_for_annot + increment_x * 0.5
                cat_pos_y = {cat: i for i, cat in enumerate(PLOT_ORDER)} # y-axis positions

                for i, (cat1, cat2) in enumerate(valid_comparisons_stats):
                    p_corrected = pvals_corrected[i]
                    pos_y1 = cat_pos_y[cat1]
                    pos_y2 = cat_pos_y[cat2]

                    line_y_coords = [pos_y1, pos_y1, pos_y2, pos_y2]
                    line_x_coords = [current_x_annot, current_x_annot + increment_x * 0.2, current_x_annot + increment_x * 0.2, current_x_annot]
                    ax.plot(line_x_coords, line_y_coords, lw=1.0, c='black')

                    significance = 'ns'
                    if p_corrected < 0.001: significance = '***'
                    elif p_corrected < 0.01: significance = '**'
                    elif p_corrected < 0.05: significance = '*'
                    
                    text_y_coord = (pos_y1 + pos_y2) / 2
                    text_x_coord = current_x_annot + increment_x * 0.25
                    ax.text(text_x_coord, text_y_coord, significance, ha='left', va='center', fontsize=8)
                    current_x_annot += increment_x
                
                if num_valid_comp > 0: ax.set_xlim(right=current_x_annot + increment_x * 0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust rect to make space for suptitle

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    figure_filename = f"figure3_beta_comparison_{timestamp}.pdf"
    figure_path = os.path.join(args.output_dir, figure_filename)
    plt.savefig(figure_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {figure_path}")

    csv_filename = f"figure3_data_{timestamp}.csv"
    csv_path = os.path.join(args.output_dir, csv_filename)
    results_df.to_csv(csv_path, index=False)
    print(f"Aggregated data saved to {csv_path}")

if __name__ == "__main__":
    main() # Call main() directly

