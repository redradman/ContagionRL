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

# Define Grid Size values and their labels for plotting
GRID_SIZE_VALUES = [30, 40, 50, 60]
GRID_SIZE_LABELS = { # For display on the x-axis of the plot
    30: r"Grid $30 \times 30$",
    40: r"Grid $40 \times 40$",
    50: r"Grid $50 \times 50$",
    60: r"Grid $60 \times 60$"
}
# Order for plotting on the x-axis for the bar plot
PLOT_ORDER_X_AXIS = [GRID_SIZE_LABELS[gs] for gs in GRID_SIZE_VALUES]
AGENT_ORDER = ['Stationary', 'Random', 'Trained', 'Greedy'] # Order of bars within each group

def load_model_config(model_path: str) -> Dict[str, Any]:
    config_path = os.path.join(os.path.dirname(model_path), "config.json")
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
    parser = argparse.ArgumentParser(description="Generate Figure 6 comparing models trained with different grid sizes.")
    parser.add_argument("--model-base-prefix", type=str, required=True, help="Prefix for model directories (e.g., 'Fig6'). The script will append '_grid<size>'.")
    parser.add_argument("--seeds", type=str, default="1,2,3", help="Comma-separated list of seeds for trained models.")
    parser.add_argument("--runs", type=int, default=30, help="Number of evaluation episodes per loaded model.")
    parser.add_argument("--output-dir", type=str, default="figures/", help="Directory to save the figures.")
    parser.add_argument("--eval-seed-base", type=int, default=4000, help="Base seed for evaluation runs (should be different from other figures).")
    
    args = parser.parse_args()

    try:
        train_seeds = [int(s.strip()) for s in args.seeds.split(',')]
    except ValueError:
        print("Error: --seeds must be a comma-separated list of integers.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    all_results_data = []
    category_seed_offset = 0

    for grid_size_value in GRID_SIZE_VALUES:
        grid_size_label = GRID_SIZE_LABELS[grid_size_value]
        model_base_name_for_grid = f"{args.model_base_prefix}_grid{grid_size_value}"
        print(f"Processing models for {grid_size_label}")
        for train_seed in tqdm(train_seeds, desc=f"Models for {grid_size_label}"):
            model_dir_name = f"{model_base_name_for_grid}_seed{train_seed}"
            model_path = os.path.join("logs", model_dir_name, "best_model.zip")

            if not os.path.exists(model_path):
                print(f"Warning: Model file not found for {grid_size_label} (seed {train_seed}) at {model_path}. Skipping.")
                continue
            try:
                model_config = load_model_config(model_path)
                current_env_config = model_config.get("environment")
                if current_env_config is None or current_env_config.get('grid_size') != grid_size_value:
                    print(f"Warning: Grid size in loaded config ({current_env_config.get('grid_size') if current_env_config else 'N/A'}) for {model_path} does not match expected grid size {grid_size_value}. Skipping.")
                    continue
                
                env_creation_seed = args.eval_seed_base + category_seed_offset + train_seed 
                env = create_env_from_config(current_env_config, seed=env_creation_seed) 
                model = PPO.load(model_path, env=env)
                model_eval_run_base_seed = args.eval_seed_base + category_seed_offset * 100 + train_seed * args.runs
                
                # --- Trained PPO ---
                eval_metrics_list = run_evaluation_episodes_for_metrics(env, model, args.runs, model_eval_run_base_seed)
                for metrics in eval_metrics_list:
                    all_results_data.append({
                        "grid_size_label": grid_size_label,
                        "grid_size_value": grid_size_value,
                        "model_train_seed": train_seed,
                        "episode_length": metrics["episode_length"],
                        "final_reward": metrics["final_reward"],
                        "agent_type": "Trained"
                    })

                # --- Baselines (run for each trained model's env config to ensure correct grid_size) ---
                baseline_env = create_env_from_config(current_env_config, seed=env_creation_seed + 1) # Use slightly different seed for safety
                # Stationary
                for i in range(args.runs):
                    eval_seed = model_eval_run_base_seed + 10000 + i # Ensure unique eval seeds
                    obs, _ = baseline_env.reset(seed=eval_seed)
                    done = False; ep_len = 0
                    while not done:
                        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                        obs, reward, terminated, truncated, _ = baseline_env.step(action)
                        ep_len += 1; done = terminated or truncated
                    all_results_data.append({"grid_size_label": grid_size_label, "grid_size_value": grid_size_value, "model_train_seed": train_seed, "episode_length": ep_len, "final_reward": None, "agent_type": "Stationary"})
                # Random
                for i in range(args.runs):
                    eval_seed = model_eval_run_base_seed + 20000 + i
                    obs, _ = baseline_env.reset(seed=eval_seed)
                    done = False; ep_len = 0
                    while not done:
                        action = baseline_env.action_space.sample()
                        obs, reward, terminated, truncated, _ = baseline_env.step(action)
                        ep_len += 1; done = terminated or truncated
                    all_results_data.append({"grid_size_label": grid_size_label, "grid_size_value": grid_size_value, "model_train_seed": train_seed, "episode_length": ep_len, "final_reward": None, "agent_type": "Random"})
                # Greedy
                for i in range(args.runs):
                    eval_seed = model_eval_run_base_seed + 30000 + i
                    obs, _ = baseline_env.reset(seed=eval_seed)
                    done = False; ep_len = 0
                    adherence = 1.0 
                    while not done:
                        agent_pos = baseline_env.agent_position
                        infected_humans = [h for h in baseline_env.humans if h.state == 1]
                        if not infected_humans:
                            dx, dy = 0.0, 0.0
                        else:
                            current_distances = [baseline_env._calculate_distance(Human(agent_pos[0], agent_pos[1], 0, -1), h) for h in infected_humans]
                            min_current_dist = min(current_distances)
                            nearest_infected_idx = current_distances.index(min_current_dist)
                            nearest_infected_human = infected_humans[nearest_infected_idx]
                            possible_moves = [(0.0,0.0),(1.0,0.0),(-1.0,0.0),(0.0,1.0),(0.0,-1.0),(0.707,0.707),(0.707,-0.707),(-0.707,0.707),(-0.707,-0.707)]
                            best_dx, best_dy = 0.0, 0.0
                            max_dist_to_nearest = -1.0
                            for move_dx, move_dy in possible_moves:
                                next_x = (agent_pos[0] + move_dx) % baseline_env.grid_size
                                next_y = (agent_pos[1] + move_dy) % baseline_env.grid_size
                                dist_to_target = baseline_env._calculate_distance(Human(next_x, next_y, 0, -1), nearest_infected_human)
                                if dist_to_target > max_dist_to_nearest:
                                    max_dist_to_nearest = dist_to_target
                                    best_dx, best_dy = move_dx, move_dy
                            dx, dy = best_dx, best_dy
                        action = np.array([dx, dy, adherence], dtype=np.float32)
                        obs, reward, terminated, truncated, _ = baseline_env.step(action)
                        ep_len += 1; done = terminated or truncated
                    all_results_data.append({"grid_size_label": grid_size_label, "grid_size_value": grid_size_value, "model_train_seed": train_seed, "episode_length": ep_len, "final_reward": None, "agent_type": "Greedy"})
                
                baseline_env.close()
                env.close()
            except Exception as e:
                print(f"Error processing model {model_path} for {grid_size_label}: {e}")
        category_seed_offset += 1 # Increment for next grid size category

    if not all_results_data:
        print("No data collected from any models. Exiting.")
        return

    results_df = pd.DataFrame(all_results_data)
    
    grid_size_order_from_data = sorted(results_df['grid_size_value'].unique())

    # --- Mann-Whitney U Tests: Trained vs Baselines for each Grid Size ---
    for gs_val in grid_size_order_from_data:
        gs_label_for_print = GRID_SIZE_LABELS.get(gs_val, str(gs_val))
        print(f"\nOne-Sided Mann–Whitney U Test Results (Grid Size = {gs_label_for_print}):")
        comparisons = []
        raw_pvals_one = []
        trained_data_series = results_df[(results_df['grid_size_value'] == gs_val) & (results_df['agent_type'] == 'Trained')]['episode_length']
        for baseline_agent_type in ['Stationary', 'Random', 'Greedy']:
            baseline_data_series = results_df[(results_df['grid_size_value'] == gs_val) & (results_df['agent_type'] == baseline_agent_type)]['episode_length']
            p_two, p_one, mean_t, mean_b = np.nan, np.nan, np.nan, np.nan; winner = "--"
            if trained_data_series.empty or baseline_data_series.empty:
                # print(f"Skipping {baseline_agent_type} for Grid Size = {gs_label_for_print} due to missing data.")
                mean_t = trained_data_series.mean() if not trained_data_series.empty else np.nan
                mean_b = baseline_data_series.mean() if not baseline_data_series.empty else np.nan
            else:
                mean_t, mean_b = trained_data_series.mean(), baseline_data_series.mean()
                p_two = mannwhitneyu(trained_data_series, baseline_data_series, alternative='two-sided').pvalue
                if mean_t > mean_b: p_one = mannwhitneyu(trained_data_series, baseline_data_series, alternative='greater').pvalue; winner = 'Trained'
                elif mean_b > mean_t: p_one = mannwhitneyu(baseline_data_series, trained_data_series, alternative='greater').pvalue; winner = baseline_agent_type
                else: p_one = 1.0
            raw_pvals_one.append(p_one)
            comparisons.append({"Baseline": baseline_agent_type, "p_two": p_two, "p_one_raw": p_one, "mean_t": mean_t, "mean_b": mean_b, "winner_initial": winner})
        valid_pvals_indices = [i for i, pval in enumerate(raw_pvals_one) if not np.isnan(pval)]
        valid_pvals_to_correct = [raw_pvals_one[i] for i in valid_pvals_indices]
        corrected_pvals_subset = [np.nan] * len(valid_pvals_to_correct)
        if valid_pvals_to_correct: _, corrected_pvals_subset, _, _ = multipletests(valid_pvals_to_correct, alpha=0.05, method='bonferroni')
        p_one_corr_full = [np.nan] * len(raw_pvals_one)
        for i, original_idx in enumerate(valid_pvals_indices): p_one_corr_full[original_idx] = corrected_pvals_subset[i]
        def stars(p_val): 
            if np.isnan(p_val): return "N/A"
            return "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        print("{:<12} {:<12} {:<12} {:<8} {:<12} {:<8} {:<10} {:<10}".format("Baseline", "p (2-sided)", "p (1-sided)", "Sig (2)", "p (1) Corr", "Sig (1)", "Winner", "Mean Diff"))
        print("-" * 98)
        for i, row_data in enumerate(comparisons):
            row_data["p_one_corr"] = p_one_corr_full[i]; row_data["sig_two"] = stars(row_data["p_two"]); row_data["sig_one"] = stars(row_data["p_one_corr"])
            final_winner = row_data["winner_initial"] if row_data["sig_one"] not in ["n.s.", "N/A"] else "--"
            mean_diff = row_data["mean_t"] - row_data["mean_b"] if not (np.isnan(row_data["mean_t"]) or np.isnan(row_data["mean_b"])) else np.nan
            p2_str = f"{row_data['p_two']:.4g}" if not np.isnan(row_data['p_two']) else "N/A"; p1raw_str = f"{row_data['p_one_raw']:.4g}" if not np.isnan(row_data['p_one_raw']) else "N/A"
            p1corr_str = f"{row_data['p_one_corr']:.4g}" if not np.isnan(row_data['p_one_corr']) else "N/A"; mdiff_str = f"{mean_diff:.2f}" if not np.isnan(mean_diff) else "N/A"
            print("{:<12} {:<12} {:<12} {:<8} {:<12} {:<8} {:<10} {:<10}".format(row_data["Baseline"], p2_str, p1raw_str, row_data["sig_two"], p1corr_str, row_data["sig_one"], final_winner, mdiff_str))

    # --- Get simulation_time from config.json of first available model ---
    simulation_time = None; found_config = False
    for gs_val in GRID_SIZE_VALUES:
        model_base_name_for_grid = f"{args.model_base_prefix}_grid{gs_val}"
        for train_seed in train_seeds:
            model_dir_name = f"{model_base_name_for_grid}_seed{train_seed}"
            model_path = os.path.join("logs", model_dir_name, "best_model.zip")
            try:
                model_cfg = load_model_config(model_path)
                env_cfg = model_cfg.get("environment", {})
                simulation_time = env_cfg.get("simulation_time") # Assuming this is constant
                if simulation_time is not None: found_config = True; break
            except Exception: continue
        if found_config: break
    if simulation_time is None: print("Warning: Could not load simulation_time. Using default 1000."); simulation_time = 1000

    # Group data for bar plot (mean of per-seed means)
    grouped_for_plot = results_df.groupby(['grid_size_value', 'agent_type', 'model_train_seed'])['episode_length'].mean().reset_index()

    def bootstrap_ci(data, n_resamples=10000, ci=95):
        if len(data) < 2: return (np.nan, np.nan)
        boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_resamples)]
        return np.percentile(boot_means, (100-ci)/2), np.percentile(boot_means, 100-(100-ci)/2)

    bar_plot_data = []
    for gs_val in grid_size_order_from_data:
        for agent_type_val in AGENT_ORDER:
            series_means = grouped_for_plot[(grouped_for_plot['grid_size_value'] == gs_val) & (grouped_for_plot['agent_type'] == agent_type_val)]['episode_length'].values
            if len(series_means) == 0: continue
            overall_mean = np.mean(series_means)
            ci_low, ci_high = bootstrap_ci(series_means)
            bar_plot_data.append({'grid_size_value': gs_val, 'agent_type': agent_type_val, 'mean_episode_length': overall_mean, 'ci_low': ci_low, 'ci_high': ci_high})
    bar_df = pd.DataFrame(bar_plot_data)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    bar_width = 0.18
    x_indices = np.arange(len(grid_size_order_from_data))
    palette = sns.color_palette("Set2", n_colors=len(AGENT_ORDER))

    for i, agent_type_val in enumerate(AGENT_ORDER):
        agent_specific_data = bar_df[bar_df['agent_type'] == agent_type_val]
        # Ensure means and CIs align with grid_size_order_from_data
        means_ordered = [agent_specific_data[agent_specific_data['grid_size_value'] == gs_v]['mean_episode_length'].values[0] if not agent_specific_data[agent_specific_data['grid_size_value'] == gs_v].empty else np.nan for gs_v in grid_size_order_from_data]
        ci_lows_ordered = [agent_specific_data[agent_specific_data['grid_size_value'] == gs_v]['ci_low'].values[0] if not agent_specific_data[agent_specific_data['grid_size_value'] == gs_v].empty else np.nan for gs_v in grid_size_order_from_data]
        ci_highs_ordered = [agent_specific_data[agent_specific_data['grid_size_value'] == gs_v]['ci_high'].values[0] if not agent_specific_data[agent_specific_data['grid_size_value'] == gs_v].empty else np.nan for gs_v in grid_size_order_from_data]
        
        err_bars = [
            [m - l if not (np.isnan(m) or np.isnan(l)) else 0 for m, l in zip(means_ordered, ci_lows_ordered)],
            [h - m if not (np.isnan(m) or np.isnan(h)) else 0 for m, h in zip(means_ordered, ci_highs_ordered)]
        ]
        bar_positions = x_indices + (i - (len(AGENT_ORDER)-1)/2) * bar_width
        ax.bar(bar_positions, means_ordered, width=bar_width, label=agent_type_val, color=palette[i], yerr=err_bars, capsize=4, edgecolor='black', linewidth=0.7)
    
    if simulation_time is not None: ax.axhline(simulation_time, color='red', linestyle='--', linewidth=1.5, alpha=0.8, zorder=2)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([GRID_SIZE_LABELS.get(gs_v, str(gs_v)) for gs_v in grid_size_order_from_data], fontsize=11)
    ax.set_xlabel(r"Grid Size", fontsize=13)
    ax.set_ylabel("Mean Episode Duration (steps)", fontsize=13)
    ax.legend(title="Agent Type", fontsize=11, title_fontsize=12, loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    plt.tight_layout(pad=0.5, rect=[0, 0, 0.85, 1])

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    figure_filename = f"figure6_grouped_bar_{timestamp}.pdf"
    figure_path = os.path.join(args.output_dir, figure_filename)
    plt.savefig(figure_path, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {figure_path}")
    
    csv_filename = f"figure6_data_{timestamp}.csv"
    csv_path = os.path.join(args.output_dir, csv_filename)
    results_df.to_csv(csv_path, index=False)
    print(f"Aggregated data saved to {csv_path}")

if __name__ == "__main__":
    main() 