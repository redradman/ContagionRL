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

# Define Distance Decay values and their labels for plotting
DISTANCE_DECAY_VALUES = [0.15, 0.3, 0.45]
DISTANCE_DECAY_LABELS = { # For display on the x-axis of the plot
    0.15: r"Distance Decay = 0.15",
    0.30: r"Distance Decay = 0.30", # Use 0.30 for consistency if desired for label
    0.45: r"Distance Decay = 0.45",
}
# Order for plotting on the x-axis for the bar plot
PLOT_ORDER_X_AXIS = [DISTANCE_DECAY_LABELS[dd] for dd in DISTANCE_DECAY_VALUES]
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
    parser = argparse.ArgumentParser(description="Generate Figure 8 comparing models trained with different distance decay values.")
    parser.add_argument("--model-base", type=str, required=True, help="Prefix for model directories (e.g., 'Fig8'). The script will append '_distanceDecay<value>'.")
    parser.add_argument("--seeds", type=str, default="1,2,3", help="Comma-separated list of seeds for trained models.")
    parser.add_argument("--runs", type=int, default=30, help="Number of evaluation episodes per loaded model.")
    parser.add_argument("--output-dir", type=str, default="figures/", help="Directory to save the figures.")
    parser.add_argument("--eval-seed-base", type=int, default=6000, help="Base seed for evaluation runs (should be different from other figures).")
    
    args = parser.parse_args()

    try:
        train_seeds = [int(s.strip()) for s in args.seeds.split(',')]
    except ValueError:
        print("Error: --seeds must be a comma-separated list of integers.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    all_results_data = []
    category_seed_offset = 0

    for dd_value in DISTANCE_DECAY_VALUES:
        dd_label = DISTANCE_DECAY_LABELS[dd_value]
        model_base_name_for_dd = f"{args.model_base}_distanceDecay{dd_value}"
        print(f"Processing models for {dd_label}")
        for train_seed in tqdm(train_seeds, desc=f"Models for {dd_label}"):
            model_dir_name = f"{model_base_name_for_dd}_seed{train_seed}"
            model_path = os.path.join("logs", model_dir_name, "best_model.zip")

            if not os.path.exists(model_path):
                print(f"Warning: Model file not found for {dd_label} (seed {train_seed}) at {model_path}. Skipping.")
                continue
            try:
                model_config = load_model_config(model_path)
                current_env_config = model_config.get("environment")
                config_dd_value = current_env_config.get('distance_decay') if current_env_config else None
                if current_env_config is None or config_dd_value is None or not np.isclose(config_dd_value, dd_value):
                    print(f"Warning: Distance decay in loaded config ({config_dd_value}) for {model_path} does not match expected {dd_value}. Skipping.")
                    continue
                
                env_creation_seed = args.eval_seed_base + category_seed_offset + train_seed 
                env = create_env_from_config(current_env_config, seed=env_creation_seed) 
                model = PPO.load(model_path, env=env)
                model_eval_run_base_seed = args.eval_seed_base + category_seed_offset * 100 + train_seed * args.runs
                
                eval_metrics_list = run_evaluation_episodes_for_metrics(env, model, args.runs, model_eval_run_base_seed)
                for metrics in eval_metrics_list:
                    all_results_data.append({
                        "distance_decay_label": dd_label,
                        "distance_decay_value": dd_value,
                        "model_train_seed": train_seed,
                        "episode_length": metrics["episode_length"],
                        "final_reward": metrics["final_reward"],
                        "agent_type": "Trained"
                    })

                baseline_env = create_env_from_config(current_env_config, seed=env_creation_seed + 1)
                for agent_t in AGENT_ORDER: # Iterate using AGENT_ORDER for consistency
                    if agent_t == "Trained": continue # Skip re-evaluating trained, already done
                    base_offset_seed = 0
                    if agent_t == "Stationary": base_offset_seed = 10000
                    elif agent_t == "Random": base_offset_seed = 20000
                    elif agent_t == "Greedy": base_offset_seed = 30000

                    for i in range(args.runs):
                        eval_seed = model_eval_run_base_seed + base_offset_seed + i
                        obs, _ = baseline_env.reset(seed=eval_seed)
                        done = False; ep_len = 0
                        if agent_t == "Stationary":
                            while not done: action = np.array([0.0,0.0,0.0],dtype=np.float32); obs,r,terminated,truncated,_=baseline_env.step(action); ep_len+=1; done=terminated or truncated
                        elif agent_t == "Random":
                            while not done: action = baseline_env.action_space.sample(); obs,r,terminated,truncated,_=baseline_env.step(action); ep_len+=1; done=terminated or truncated
                        elif agent_t == "Greedy":
                            adherence = 1.0
                            while not done:
                                agent_pos=baseline_env.agent_position; infected_humans=[h for h in baseline_env.humans if h.state==1]
                                if not infected_humans: dx,dy=0.0,0.0
                                else:
                                    current_distances=[baseline_env._calculate_distance(Human(agent_pos[0],agent_pos[1],0,-1),h) for h in infected_humans]
                                    nearest_infected_human=infected_humans[np.argmin(current_distances)]
                                    possible_moves=[(0.0,0.0),(1.0,0.0),(-1.0,0.0),(0.0,1.0),(0.0,-1.0),(0.707,0.707),(0.707,-0.707),(-0.707,0.707),(-0.707,-0.707)]
                                    best_dx,best_dy=0.0,0.0; max_dist_to_nearest=-1.0
                                    for move_dx,move_dy in possible_moves:
                                        next_x=(agent_pos[0]+move_dx)%baseline_env.grid_size; next_y=(agent_pos[1]+move_dy)%baseline_env.grid_size
                                        dist_to_target=baseline_env._calculate_distance(Human(next_x,next_y,0,-1),nearest_infected_human)
                                        if dist_to_target>max_dist_to_nearest: max_dist_to_nearest=dist_to_target; best_dx,best_dy=move_dx,move_dy
                                    dx,dy=best_dx,best_dy
                                action=np.array([dx,dy,adherence],dtype=np.float32)
                                obs,r,terminated,truncated,_=baseline_env.step(action); ep_len+=1; done=terminated or truncated
                        all_results_data.append({"distance_decay_label": dd_label, "distance_decay_value": dd_value, "model_train_seed": train_seed, "episode_length": ep_len, "final_reward": None, "agent_type": agent_t})
                
                baseline_env.close()
                env.close()
            except Exception as e:
                print(f"Error processing model {model_path} for {dd_label}: {e}")
        category_seed_offset += 1

    if not all_results_data: print("No data collected. Exiting."); return
    results_df = pd.DataFrame(all_results_data)
    dd_order_from_data = sorted(results_df['distance_decay_value'].unique())

    for dd_val in dd_order_from_data:
        dd_label_for_print = DISTANCE_DECAY_LABELS.get(dd_val, str(dd_val))
        print(f"\nOne-Sided Mann–Whitney U Test Results (Distance Decay = {dd_label_for_print}):")
        comparisons = []; raw_pvals_one = []
        trained_data = results_df[(results_df['distance_decay_value'] == dd_val) & (results_df['agent_type'] == 'Trained')]['episode_length']
        for baseline_type in AGENT_ORDER:
            if baseline_type == 'Trained': continue
            baseline_data = results_df[(results_df['distance_decay_value'] == dd_val) & (results_df['agent_type'] == baseline_type)]['episode_length']
            p2, p1, mt, mb = np.nan, np.nan, np.nan, np.nan; winner = "--"
            if trained_data.empty or baseline_data.empty:
                mt = trained_data.mean() if not trained_data.empty else np.nan; mb = baseline_data.mean() if not baseline_data.empty else np.nan
            else:
                mt, mb = trained_data.mean(), baseline_data.mean()
                p2 = mannwhitneyu(trained_data, baseline_data, alternative='two-sided').pvalue
                if mt > mb: p1 = mannwhitneyu(trained_data, baseline_data, alternative='greater').pvalue; winner = 'Trained'
                elif mb > mt: p1 = mannwhitneyu(baseline_data, trained_data, alternative='greater').pvalue; winner = baseline_type
                else: p1 = 1.0
            raw_pvals_one.append(p1)
            comparisons.append({"Baseline": baseline_type, "p_two": p2, "p_one_raw": p1, "mean_t": mt, "mean_b": mb, "winner_initial": winner})
        valid_indices = [i for i, p in enumerate(raw_pvals_one) if not np.isnan(p)]
        valid_pvals = [raw_pvals_one[i] for i in valid_indices]
        corrected_p_subset = [np.nan] * len(valid_pvals)
        if valid_pvals: _, corrected_p_subset, _, _ = multipletests(valid_pvals, alpha=0.05, method='bonferroni')
        p_one_corr = [np.nan] * len(raw_pvals_one)
        for i, original_idx in enumerate(valid_indices): p_one_corr[original_idx] = corrected_p_subset[i]
        def stars(p): return "N/A" if np.isnan(p) else ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s.")
        print("{:<12} {:<12} {:<12} {:<8} {:<12} {:<8} {:<10} {:<10}".format("Baseline", "p (2-sided)", "p (1-sided)", "Sig (2)", "p (1) Corr", "Sig (1)", "Winner", "Mean Diff"))
        print("-" * 98)
        for i, row in enumerate(comparisons):
            row["p_one_corr"] = p_one_corr[i]; row["sig_two"] = stars(row["p_two"]); row["sig_one"] = stars(row["p_one_corr"])
            fw = row["winner_initial"] if row["sig_one"] not in ["n.s.", "N/A"] else "--"
            md = row["mean_t"] - row["mean_b"] if not (np.isnan(row["mean_t"]) or np.isnan(row["mean_b"])) else np.nan
            p2s = f"{row['p_two']:.4g}" if not np.isnan(row['p_two']) else "N/A"; p1rs = f"{row['p_one_raw']:.4g}" if not np.isnan(row['p_one_raw']) else "N/A"
            p1cs = f"{row['p_one_corr']:.4g}" if not np.isnan(row['p_one_corr']) else "N/A"; mds = f"{md:.2f}" if not np.isnan(md) else "N/A"
            print("{:<12} {:<12} {:<12} {:<8} {:<12} {:<8} {:<10} {:<10}".format(row["Baseline"], p2s, p1rs, row["sig_two"], p1cs, row["sig_one"], fw, mds))

    simulation_time = None; found_config = False
    for dd_val in DISTANCE_DECAY_VALUES:
        model_base_name_for_dd = f"{args.model_base}_distanceDecay{dd_val}"
        for train_seed in train_seeds:
            model_dir_name = f"{model_base_name_for_dd}_seed{train_seed}"
            model_path = os.path.join("logs", model_dir_name, "best_model.zip")
            try:
                model_cfg = load_model_config(model_path)
                env_cfg = model_cfg.get("environment", {})
                simulation_time = env_cfg.get("simulation_time")
                if simulation_time is not None: found_config = True; break
            except Exception: continue
        if found_config: break
    if simulation_time is None: print("Warning: Could not load simulation_time. Using default 1000."); simulation_time = 1000

    grouped_for_plot = results_df.groupby(['distance_decay_value', 'agent_type', 'model_train_seed'])['episode_length'].mean().reset_index()

    def bootstrap_ci(data, n_resamples=10000, ci=95):
        if len(data) < 2: return (np.nan, np.nan)
        boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_resamples)]
        return np.percentile(boot_means, (100-ci)/2), np.percentile(boot_means, 100-(100-ci)/2)

    bar_plot_data = []
    for dd_val in dd_order_from_data:
        for agent_type_val in AGENT_ORDER:
            series_means = grouped_for_plot[(grouped_for_plot['distance_decay_value'] == dd_val) & (grouped_for_plot['agent_type'] == agent_type_val)]['episode_length'].values
            if len(series_means) == 0: continue
            overall_mean = np.mean(series_means)
            ci_low, ci_high = bootstrap_ci(series_means)
            bar_plot_data.append({'distance_decay_value': dd_val, 'agent_type': agent_type_val, 'mean_episode_length': overall_mean, 'ci_low': ci_low, 'ci_high': ci_high})
    bar_df = pd.DataFrame(bar_plot_data)

    plt.figure(figsize=(10, 6))
    ax = plt.gca(); bar_width = 0.18
    x_indices = np.arange(len(dd_order_from_data))
    palette = sns.color_palette("Set2", n_colors=len(AGENT_ORDER))

    for i, agent_type_val in enumerate(AGENT_ORDER):
        agent_data = bar_df[bar_df['agent_type'] == agent_type_val]
        means_ordered = [agent_data[agent_data['distance_decay_value'] == val]['mean_episode_length'].values[0] if not agent_data[agent_data['distance_decay_value'] == val].empty else np.nan for val in dd_order_from_data]
        ci_lows_ordered = [agent_data[agent_data['distance_decay_value'] == val]['ci_low'].values[0] if not agent_data[agent_data['distance_decay_value'] == val].empty else np.nan for val in dd_order_from_data]
        ci_highs_ordered = [agent_data[agent_data['distance_decay_value'] == val]['ci_high'].values[0] if not agent_data[agent_data['distance_decay_value'] == val].empty else np.nan for val in dd_order_from_data]
        err_bars = [[m - l if not (np.isnan(m) or np.isnan(l)) else 0 for m,l in zip(means_ordered, ci_lows_ordered)], [h - m if not (np.isnan(m) or np.isnan(h)) else 0 for m,h in zip(means_ordered, ci_highs_ordered)]]
        bar_positions = x_indices + (i - (len(AGENT_ORDER)-1)/2) * bar_width
        ax.bar(bar_positions, means_ordered, width=bar_width, label=agent_type_val, color=palette[i], yerr=err_bars, capsize=4, edgecolor='black', linewidth=0.7)
    
    if simulation_time is not None: ax.axhline(simulation_time, color='red', linestyle='--', linewidth=1.5, alpha=0.8, zorder=2)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([DISTANCE_DECAY_LABELS.get(val, str(val)) for val in dd_order_from_data], fontsize=11)
    ax.set_xlabel(r"Distance Decay Factor ($k_d$)", fontsize=13) # Example LaTeX label for distance decay
    ax.set_ylabel("Mean Episode Duration (steps)", fontsize=13)
    ax.legend(title="Agent Type", fontsize=11, title_fontsize=12, loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    plt.tight_layout(pad=0.5, rect=[0, 0, 0.85, 1])

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    figure_filename = f"figure8_grouped_bar_{timestamp}.pdf"
    figure_path = os.path.join(args.output_dir, figure_filename)
    plt.savefig(figure_path, bbox_inches='tight'); plt.close()
    print(f"Figure saved to {figure_path}")
    
    csv_filename = f"figure8_data_{timestamp}.csv"
    csv_path = os.path.join(args.output_dir, csv_filename)
    results_df.to_csv(csv_path, index=False)
    print(f"Aggregated data saved to {csv_path}")

if __name__ == "__main__":
    main() 