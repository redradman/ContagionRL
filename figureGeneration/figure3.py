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
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# Add the parent directory to the path to access project modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from environment import SIRSEnvironment, Human

# tueplots styling
from tueplots import bundles
plt.rcParams.update(bundles.neurips2024())
plt.rcParams["figure.dpi"] = 300
sns.set_style("whitegrid")

# Define Beta values and their labels for plotting
BETA_VALUES = [0.2, 0.5, 0.7, 0.9]
BETA_LABELS = { # For display on the y-axis of the plot
    0.2: r"$\beta = 0.2$",
    0.5: r"$\beta = 0.5$",
    0.7: r"$\beta = 0.7$",
    0.9: r"$\beta = 0.9$"
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
    category_seed_offset = 0 # To ensure unique eval seeds across categories if needed

    for beta_value in BETA_VALUES:
        beta_label = BETA_LABELS[beta_value]
        beta_str_for_name = str(beta_value).replace('.', 'p')
        model_base_name_for_beta = f"{args.model_base_prefix}_beta{beta_str_for_name}"
        print(f"Processing models for {beta_label}")
        for train_seed in tqdm(train_seeds, desc=f"Models for {beta_label}"):
            model_dir_name = f"{model_base_name_for_beta}_seed{train_seed}"
            model_path = os.path.join("logs", model_dir_name, "best_model.zip")

            if not os.path.exists(model_path):
                print(f"Warning: Model file not found for {beta_label} (seed {train_seed}) at {model_path}. Skipping.")
                continue
            try:
                model_config = load_model_config(model_path)
                current_env_config = model_config.get("environment")
                if current_env_config is None or current_env_config.get('beta') != beta_value:
                    print(f"Warning: Beta value in loaded config ({current_env_config.get('beta') if current_env_config else 'N/A'}) for {model_path} does not match expected beta {beta_value}. Skipping.")
                    continue
                env_creation_seed = args.eval_seed_base + category_seed_offset + train_seed 
                env = create_env_from_config(current_env_config, seed=env_creation_seed) 
                model = PPO.load(model_path, env=env)
                model_eval_run_base_seed = args.eval_seed_base + category_seed_offset * 100 + train_seed * args.runs
                # --- Trained PPO ---
                eval_metrics_list = run_evaluation_episodes_for_metrics(env, model, args.runs, model_eval_run_base_seed)
                for metrics in eval_metrics_list:
                    all_results_data.append({
                        "beta_value_label": beta_label,
                        "beta_value": beta_value,
                        "model_train_seed": train_seed,
                        "episode_length": metrics["episode_length"],
                        "final_reward": metrics["final_reward"],
                        "agent_type": "Trained"
                    })
                # --- Baselines ---
                # Stationary
                for i in range(args.runs):
                    eval_seed = model_eval_run_base_seed + 1000 + i
                    env.reset(seed=eval_seed)
                    obs = env.reset(seed=eval_seed)[0]
                    done = False
                    ep_len = 0
                    while not done:
                        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                        obs, reward, terminated, truncated, _ = env.step(action)
                        ep_len += 1
                        done = terminated or truncated
                    all_results_data.append({
                        "beta_value_label": beta_label,
                        "beta_value": beta_value,
                        "model_train_seed": train_seed,
                        "episode_length": ep_len,
                        "final_reward": None,
                        "agent_type": "Stationary"
                    })
                # Random
                for i in range(args.runs):
                    eval_seed = model_eval_run_base_seed + 2000 + i
                    env.reset(seed=eval_seed)
                    obs = env.reset(seed=eval_seed)[0]
                    done = False
                    ep_len = 0
                    while not done:
                        action = np.array([
                            env.np_random.uniform(-1, 1),
                            env.np_random.uniform(-1, 1),
                            env.np_random.uniform(0, 1)
                        ], dtype=np.float32)
                        obs, reward, terminated, truncated, _ = env.step(action)
                        ep_len += 1
                        done = terminated or truncated
                    all_results_data.append({
                        "beta_value_label": beta_label,
                        "beta_value": beta_value,
                        "model_train_seed": train_seed,
                        "episode_length": ep_len,
                        "final_reward": None,
                        "agent_type": "Random"
                    })
                # Greedy
                for i in range(args.runs):
                    eval_seed = model_eval_run_base_seed + 3000 + i
                    env.reset(seed=eval_seed)
                    obs = env.reset(seed=eval_seed)[0]
                    done = False
                    ep_len = 0
                    adherence = 1.0
                    while not done:
                        agent_pos = env.agent_position
                        infected_humans = [h for h in env.humans if h.state == 1]
                        if not infected_humans:
                            dx, dy = 0.0, 0.0
                        else:
                            current_distances = [env._calculate_distance(Human(agent_pos[0], agent_pos[1], 0, -1), h) for h in infected_humans]
                            min_current_dist = min(current_distances)
                            nearest_infected_idx = current_distances.index(min_current_dist)
                            nearest_infected_human = infected_humans[nearest_infected_idx]
                            possible_moves = [
                                (0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
                                (0.707, 0.707), (0.707, -0.707), (-0.707, 0.707), (-0.707, -0.707)
                            ]
                            best_dx, best_dy = 0.0, 0.0
                            max_dist_to_nearest = -1.0
                            for move_dx, move_dy in possible_moves:
                                next_x = (agent_pos[0] + move_dx) % env.grid_size
                                next_y = (agent_pos[1] + move_dy) % env.grid_size
                                dist_to_target = env._calculate_distance(Human(next_x, next_y, 0, -1), nearest_infected_human)
                                if dist_to_target > max_dist_to_nearest:
                                    max_dist_to_nearest = dist_to_target
                                    best_dx, best_dy = move_dx, move_dy
                            dx, dy = best_dx, best_dy
                        action = np.array([dx, dy, adherence], dtype=np.float32)
                        obs, reward, terminated, truncated, _ = env.step(action)
                        ep_len += 1
                        done = terminated or truncated
                    all_results_data.append({
                        "beta_value_label": beta_label,
                        "beta_value": beta_value,
                        "model_train_seed": train_seed,
                        "episode_length": ep_len,
                        "final_reward": None,
                        "agent_type": "Greedy"
                    })
                env.close()
            except Exception as e:
                print(f"Error processing model {model_path} for {beta_label}: {e}")
        category_seed_offset += len(BETA_VALUES)

    if not all_results_data:
        print("No data collected from any models. Exiting.")
        return

    # --- Get simulation_time from config.json of first available model ---
    simulation_time = None
    found_config = False
    for beta_value in BETA_VALUES:
        beta_str_for_name = str(beta_value).replace('.', 'p')
        model_base_name_for_beta = f"{args.model_base_prefix}_beta{beta_str_for_name}"
        for train_seed in train_seeds:
            model_dir_name = f"{model_base_name_for_beta}_seed{train_seed}"
            model_path = os.path.join("logs", model_dir_name, "best_model.zip")
            try:
                model_config = load_model_config(model_path)
                env_cfg = model_config.get("environment", {})
                simulation_time = env_cfg.get("simulation_time", None)
                if simulation_time is not None:
                    found_config = True
                    break
            except Exception:
                continue
        if found_config:
            break
    if simulation_time is None:
        print("Warning: Could not load simulation_time from any model config.json, using value from config.py.")

    results_df = pd.DataFrame(all_results_data)

    agent_order = ['Stationary', 'Random', 'Trained', 'Greedy']
    beta_order = sorted(results_df['beta_value'].unique())

    # --- Mann-Whitney U Tests: Trained vs Baselines for each Beta ---
    for beta in beta_order: # beta_order should be defined before this loop
        print(f"\nOne-Sided Mann–Whitney U Test Results (Beta = {beta}):")
        comparisons = []
        raw_pvals_one = []
        
        trained_data_series = results_df[(results_df['beta_value'] == beta) & (results_df['agent_type'] == 'Trained')]['episode_length']
        
        for baseline_agent_type in ['Stationary', 'Random', 'Greedy']:
            baseline_data_series = results_df[(results_df['beta_value'] == beta) & (results_df['agent_type'] == baseline_agent_type)]['episode_length']

            # Initialize metrics with NaN in case of missing data
            p_two, p_one, mean_t, mean_b = np.nan, np.nan, np.nan, np.nan
            winner = "--"

            if trained_data_series.empty or baseline_data_series.empty:
                print(f"Skipping {baseline_agent_type} for Beta = {beta} due to missing data for 'Trained' or '{baseline_agent_type}'.")
                mean_t = trained_data_series.mean() if not trained_data_series.empty else np.nan
                mean_b = baseline_data_series.mean() if not baseline_data_series.empty else np.nan
            else:
                mean_t, mean_b = trained_data_series.mean(), baseline_data_series.mean()
                p_two = mannwhitneyu(trained_data_series, baseline_data_series, alternative='two-sided').pvalue
                if mean_t > mean_b:
                    p_one = mannwhitneyu(trained_data_series, baseline_data_series, alternative='greater').pvalue
                    winner = 'Trained'
                elif mean_b > mean_t: # Only if baseline mean is strictly greater
                    p_one = mannwhitneyu(baseline_data_series, trained_data_series, alternative='greater').pvalue
                    winner = baseline_agent_type
                else: # Means are equal or one of the series was empty (already handled by initial NaNs)
                    p_one = 1.0 # If means are equal, one-sided p-value is not straightforwardly < 0.05
            
            raw_pvals_one.append(p_one) # p_one will be np.nan if data was missing
            comparisons.append({
                "Baseline": baseline_agent_type,
                "p_two": p_two,
                "p_one_raw": p_one,
                "mean_t": mean_t,
                "mean_b": mean_b,
                "winner_initial": winner # Store initial winner before significance check
            })

        # Bonferroni correction for the current beta group's one-sided tests
        # Filter out NaNs for correction, then re-assign
        valid_pvals_indices = [i for i, pval in enumerate(raw_pvals_one) if not np.isnan(pval)]
        valid_pvals_to_correct = [raw_pvals_one[i] for i in valid_pvals_indices]
        
        corrected_pvals_subset = [np.nan] * len(valid_pvals_to_correct) # Default to NaN
        if valid_pvals_to_correct: # Only run if there are valid p-values to correct
            _, corrected_pvals_subset, _, _ = multipletests(valid_pvals_to_correct, alpha=0.05, method='bonferroni')

        p_one_corr_full = [np.nan] * len(raw_pvals_one)
        for i, original_idx in enumerate(valid_pvals_indices):
            p_one_corr_full[original_idx] = corrected_pvals_subset[i]
            
        def stars(p_val):
            if np.isnan(p_val): return "N/A"
            return "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."

        print("{:<12} {:<12} {:<12} {:<8} {:<12} {:<8} {:<10} {:<10}".format(
            "Baseline", "p (2-sided)", "p (1-sided)", "Sig (2)", "p (1) Corr", "Sig (1)", "Winner", "Mean Diff"
        ))
        print("-" * 98)
        for i, row_data in enumerate(comparisons):
            row_data["p_one_corr"] = p_one_corr_full[i]
            row_data["sig_two"] = stars(row_data["p_two"])
            row_data["sig_one"] = stars(row_data["p_one_corr"])
            
            # Determine final winner based on corrected significant p-value
            final_winner = row_data["winner_initial"]
            if row_data["sig_one"] == "n.s." or row_data["sig_one"] == "N/A":
                final_winner = "--"
            
            mean_diff = row_data["mean_t"] - row_data["mean_b"] if not (np.isnan(row_data["mean_t"]) or np.isnan(row_data["mean_b"])) else np.nan
            
            # Format numbers as strings for printing, handling NaNs
            p2_str = f"{row_data['p_two']:.4g}" if not np.isnan(row_data['p_two']) else "N/A"
            p1raw_str = f"{row_data['p_one_raw']:.4g}" if not np.isnan(row_data['p_one_raw']) else "N/A"
            p1corr_str = f"{row_data['p_one_corr']:.4g}" if not np.isnan(row_data['p_one_corr']) else "N/A"
            mdiff_str = f"{mean_diff:.2f}" if not np.isnan(mean_diff) else "N/A"

            print("{:<12} {:<12} {:<12} {:<8} {:<12} {:<8} {:<10} {:<10}".format(
                row_data["Baseline"], p2_str, p1raw_str, row_data["sig_two"],
                p1corr_str, row_data["sig_one"], final_winner, mdiff_str
            ))
    # --- End of Mann-Whitney U Tests ---

    grouped = results_df.groupby(['beta_value', 'agent_type', 'model_train_seed'])['episode_length'].mean().reset_index()

    # --- BOOTSTRAP CI FUNCTION ---
    def bootstrap_ci(data, n_resamples=10000, ci=95):
        if len(data) < 2:
            return (np.nan, np.nan)
        boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_resamples)]
        lower = np.percentile(boot_means, (100 - ci) / 2)
        upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
        return lower, upper

    # --- PREPARE DATA FOR BARPLOT ---
    bar_data = []
    for beta in beta_order:
        for agent in agent_order:
            group = grouped[(grouped['beta_value'] == beta) & (grouped['agent_type'] == agent)]
            means = group['episode_length'].values
            if len(means) == 0:
                continue
            mean = np.mean(means)
            ci_low, ci_high = bootstrap_ci(means)
            bar_data.append({
                'beta_value': beta,
                'agent_type': agent,
                'mean_episode_length': mean,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'n_seeds': len(means)
            })
    bar_df = pd.DataFrame(bar_data)

    # --- PLOT GROUPED BARPLOT ---
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    bar_width = 0.18
    x = np.arange(len(beta_order))
    palette = sns.color_palette("Set2", n_colors=len(agent_order))
    for i, agent in enumerate(agent_order):
        agent_data = bar_df[bar_df['agent_type'] == agent]
        means = [agent_data[agent_data['beta_value'] == beta]['mean_episode_length'].values[0] if not agent_data[agent_data['beta_value'] == beta].empty else np.nan for beta in beta_order]
        ci_lows = [agent_data[agent_data['beta_value'] == beta]['ci_low'].values[0] if not agent_data[agent_data['beta_value'] == beta].empty else np.nan for beta in beta_order]
        ci_highs = [agent_data[agent_data['beta_value'] == beta]['ci_high'].values[0] if not agent_data[agent_data['beta_value'] == beta].empty else np.nan for beta in beta_order]
        err = [
            [mean - ci_low if not np.isnan(mean) and not np.isnan(ci_low) else 0 for mean, ci_low in zip(means, ci_lows)],
            [ci_high - mean if not np.isnan(mean) and not np.isnan(ci_high) else 0 for mean, ci_high in zip(means, ci_highs)]
        ]
        bar_positions = x + (i - (len(agent_order)-1)/2) * bar_width
        ax.bar(bar_positions, means, width=bar_width, label=agent, color=palette[i], yerr=err, capsize=4, edgecolor='black', linewidth=0.7)
    # --- Add simulation_time reference line ---
    ax.axhline(simulation_time, color='red', linestyle='--', linewidth=1.5, alpha=0.8, zorder=2)
    # Add label at the right edge, above the line
    # xlim = ax.get_xlim()
    # ax.text(xlim[1], simulation_time + 5, 'Max Episode Length', color='red', fontsize=9, ha='right', va='bottom', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([BETA_LABELS[b] for b in beta_order], fontsize=9)
    ax.set_xlabel(r"Infection Rate ($\beta$)", fontsize=10)
    ax.set_ylabel("Mean Episode Duration (steps)", fontsize=10)
    ax.legend(title="Agent Type", fontsize=9, title_fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    plt.tight_layout(pad=0.5, rect=[0, 0, 0.85, 1])

    # --- STATISTICAL ANNOTATIONS REMOVED (Cliff's Delta) ---
    # (The section calculating and printing Cliff's delta, and annotating the plot with it, is removed)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    figure_filename = f"figure3_grouped_bar_{timestamp}.pdf"
    figure_path = os.path.join(args.output_dir, figure_filename)
    plt.savefig(figure_path, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {figure_path}")
    csv_filename = f"figure3_data_{timestamp}.csv"
    csv_path = os.path.join(args.output_dir, csv_filename)
    results_df.to_csv(csv_path, index=False)
    print(f"Aggregated data saved to {csv_path}")

if __name__ == "__main__":
    main() # Call main() directly

