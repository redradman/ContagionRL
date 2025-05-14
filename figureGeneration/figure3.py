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
import cliffs_delta

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
        model_base_name_for_beta = f"{args.model_base_prefix}{beta_str_for_name}"
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

    # --- STATISTICAL ANNOTATIONS ---
    # For each beta, compare Trained to each baseline using Cliff's delta
    effect_map = {
        'negligible effect': 'n.s.',
        'small effect': 'S',
        'medium effect': 'M',
        'large effect': 'L'
    }
    summary_rows = []
    for i, beta in enumerate(beta_order):
        trained_group = grouped[(grouped['beta_value'] == beta) & (grouped['agent_type'] == 'Trained')]['episode_length'].values
        for j, agent in enumerate(['Stationary', 'Random', 'Greedy']):
            comp_group = grouped[(grouped['beta_value'] == beta) & (grouped['agent_type'] == agent)]['episode_length'].values
            if len(trained_group) > 0 and len(comp_group) > 0:
                d, _ = cliffs_delta.cliffs_delta(trained_group, comp_group)
                abs_d = abs(d)
                if abs_d < 0.147:
                    effect = 'negligible effect'
                elif abs_d < 0.33:
                    effect = 'small effect'
                elif abs_d < 0.474:
                    effect = 'medium effect'
                else:
                    effect = 'large effect'
                label = effect_map[effect]
                bar_x = x[i] + (agent_order.index(agent) - (len(agent_order)-1)/2) * bar_width
                # Find the correct bar height and error for this bar
                bar_idx = agent_order.index(agent)
                mean = bar_df[(bar_df['beta_value'] == beta) & (bar_df['agent_type'] == agent)]['mean_episode_length'].values
                err = bar_df[(bar_df['beta_value'] == beta) & (bar_df['agent_type'] == agent)]['ci_high'].values
                if len(mean) > 0 and len(err) > 0:
                    bar_top = mean[0] + (err[0] - mean[0])
                else:
                    bar_top = max(np.mean(trained_group), np.mean(comp_group))
                bar_y = bar_top + 8  # 8 units above the top of the error bar
                ax.text(bar_x, bar_y, label, ha='center', va='bottom', fontsize=10, rotation=0)
                summary_rows.append([str(beta), agent, f"{d:.2f}", effect])
    # Print summary table
    print("\nCliff's Delta Effect Size Summary (Trained vs Baselines):")
    print("Beta\tBaseline\tCliff's d\tEffect Size")
    for row in summary_rows:
        print("\t".join(row))
    # (Optional: explain abbreviations in caption/legend: L=Large, M=Medium, S=Small, n.s.=Negligible)

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

