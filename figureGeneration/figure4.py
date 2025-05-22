import os
import sys
import argparse
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO, SAC, TD3, A2C
import gymnasium as gym # Helper for flattening obs for SAC/TD3
from tqdm import tqdm
from scipy.stats import mannwhitneyu
import itertools
import statsmodels.stats.multitest as smm
from textwrap import wrap
# from typing import Dict, List, Any, Optional

# tueplots styling
from tueplots import bundles
plt.rcParams.update(bundles.neurips2024())
plt.rcParams["figure.dpi"] = 300
sns.set_style("whitegrid")

# Add the parent directory to the path to access project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import SIRSDEnvironment

class FlattenObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.flatten_space(env.observation_space)
    def observation(self, observation):
        return gym.spaces.flatten(self.env.observation_space, observation)

def create_env_for_algo(env_config: dict, seed: int, algo: str) -> SIRSDEnvironment:
    config = env_config.copy()
    config["render_mode"] = None
    env = SIRSDEnvironment(**config)
    env.reset(seed=seed)
    if algo in ["sac", "a2c"]:
        env = FlattenObservationWrapper(env)
    return env

def run_episode(env, model=None, agent_type=None) -> int:
    obs, _ = env.reset()
    done = False
    episode_length = 0
    stationary_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        elif agent_type == "random":
            action = env.action_space.sample()
        elif agent_type == "stationary":
            action = stationary_action
        elif agent_type == "greedy":
            agent_pos = env.agent_position
            infected_humans = [h for h in env.humans if h.state == 1]
            adherence = 1.0
            if not infected_humans:
                dx, dy = 0.0, 0.0
            else:
                min_dist = float('inf')
                nearest = None
                for inf_h in infected_humans:
                    dist = env._calculate_distance(type('H', (), {'x': agent_pos[0], 'y': agent_pos[1], 'id': -1, 'state': 0}), inf_h)
                    if dist < min_dist:
                        min_dist = dist
                        nearest = (inf_h.x, inf_h.y)
                if nearest:
                    vec_x = agent_pos[0] - nearest[0]
                    vec_y = agent_pos[1] - nearest[1]
                    norm = np.sqrt(vec_x**2 + vec_y**2)
                    if norm > 1e-6:
                        dx = vec_x / norm
                        dy = vec_y / norm
                    else:
                        dx, dy = env.action_space.sample()[:2]
                else:
                    dx, dy = 0.0, 0.0
            action = np.array([dx, dy, adherence], dtype=np.float32)
        else:
            raise ValueError(f"Unknown agent_type: {agent_type}")
        obs, _, terminated, truncated, _ = env.step(action)
        episode_length += 1
        done = terminated or truncated
    return episode_length

def main():
    parser = argparse.ArgumentParser(description="Generate Figure 4: Compare PPO, SAC, TD3, and baselines.")
    parser.add_argument("--model-base", type=str, required=True, help="Base name for all models (e.g., Fig4)")
    parser.add_argument("--runs", type=int, default=30, help="Number of evaluation episodes per model (default: 30)")
    parser.add_argument("--seeds", type=str, default="1,2,3", help="Comma-separated list of seeds for trained models (e.g., '1,2,3')")
    parser.add_argument("--output-dir", type=str, default="figures/", help="Directory to save the generated figures (default: figures/)")
    parser.add_argument("--eval-seed", type=int, default=3000, help="Base seed for evaluation runs (default: 3000)")
    args = parser.parse_args()

    try:
        train_seeds = [int(s.strip()) for s in args.seeds.split(',')]
    except ValueError:
        print("Error: --seeds must be a comma-separated list of integers.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load env config from one of the models
    def load_env_config(model_path):
        import json
        config_path = os.path.join(os.path.dirname(model_path), "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config["environment"]

    # Prepare agent/algorithm labels
    AGENT_LABELS = {
        "ppo": "PPO",
        "sac": "SAC",
        "a2c": "A2C",
        "random": "Random",
        "stationary": "Stationary",
        "greedy": "Greedy"
    }
    PLOT_ORDER = [AGENT_LABELS[a] for a in ["stationary", "random", "ppo", "sac", "a2c", "greedy"]]

    all_results = []

    # --- Evaluate RL models ---
    for algo in ["ppo", "sac", "a2c"]:
        for seed in tqdm(train_seeds, desc=f"{algo.upper()} seeds", position=0):
            model_dir = f"{args.model_base}_{algo.upper()}_seed{seed}"
            model_path = os.path.join("logs", model_dir, "best_model.zip")
            if not os.path.exists(model_path):
                print(f"Warning: Model not found at {model_path}, skipping.")
                continue
            # Load env config for each algorithm/seed
            base_env_config = load_env_config(model_path)
            env = create_env_for_algo(base_env_config, args.eval_seed + seed, algo)
            if algo == "ppo":
                model = PPO.load(model_path, env=env)
            elif algo == "sac":
                model = SAC.load(model_path, env=env)
            elif algo == "a2c":
                model = A2C.load(model_path, env=env)
            else:
                raise ValueError(f"Unknown algo: {algo}")
            for i in tqdm(range(args.runs), desc=f"{algo.upper()} seed{seed} evals", leave=False, position=1):
                ep_len = run_episode(env, model=model)
                all_results.append({"agent_label": AGENT_LABELS[algo], "train_seed": seed, "episode_length": ep_len})
            env.close()

    # --- Evaluate baselines ---
    # Use PPO env config for baselines (from the first PPO model found)
    base_env_config = None
    for seed in train_seeds:
        model_dir = f"{args.model_base}_PPO_seed{seed}"
        model_path = os.path.join("logs", model_dir, "best_model.zip")
        if os.path.exists(model_path):
            base_env_config = load_env_config(model_path)
            break
    if base_env_config is None:
        print("Error: Could not find any PPO model to extract environment config for baselines.")
        sys.exit(1)

    for agent_type in ["random", "stationary", "greedy"]:
        for seed in tqdm(train_seeds, desc=f"{agent_type.capitalize()} seeds", position=0):
            env = create_env_for_algo(base_env_config, args.eval_seed + 100 + seed, "ppo")  # Use PPO env for baselines
            for i in tqdm(range(args.runs), desc=f"{agent_type.capitalize()} seed{seed} evals", leave=False, position=1):
                ep_len = run_episode(env, model=None, agent_type=agent_type)
                all_results.append({"agent_label": AGENT_LABELS[agent_type], "train_seed": seed, "episode_length": ep_len})
            env.close()

    # --- Plotting with 95% bootstrapped CIs over per-seed means ---
    results_df = pd.DataFrame(all_results)
    plt.figure(figsize=(8, 6))
    y_metric_col = "episode_length"
    y_label = "Episode Duration (steps)"
    plot_order_filtered = [label for label in PLOT_ORDER if label in results_df['agent_label'].unique()]

    # --- NEW: Boxplot per agent, overlay per-seed means as black dots with white outline, and all episode durations as points ---
    ax = plt.gca()
    sns.boxplot(x="agent_label", y=y_metric_col, data=results_df, order=plot_order_filtered, ax=ax, palette="muted", showfliers=True)
    # Overlay all individual episode durations as small black points
    sns.stripplot(x="agent_label", y=y_metric_col, data=results_df, order=plot_order_filtered, ax=ax, color='black', alpha=0.7, jitter=0.25, size=3, zorder=5)
    # Overlay per-seed means as large black dots with white outline
    grouped = results_df.groupby(['agent_label', 'train_seed'])[y_metric_col].mean().reset_index()
    for i, agent in enumerate(plot_order_filtered):
        means = grouped[grouped['agent_label'] == agent][y_metric_col].values
        ax.scatter([i]*len(means), means, color='black', s=120, zorder=10, marker='o', edgecolor='white', linewidth=2, label=None)
    # Add legend entry for per-seed mean dots
    # ax.scatter([], [], color='black', s=120, label='Per-seed Mean', edgecolor='white', linewidth=2)
    # ax.legend(fontsize=13)
    plt.xlabel("Algorithm", fontsize=13)
    plt.ylabel(y_label, fontsize=13)
    ax.tick_params(axis='x', labelsize=11, rotation=0)
    ax.tick_params(axis='y', labelsize=11)
    plt.tight_layout(pad=0.5)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    figure_filename = f"figure4_algo_comparison_{timestamp}.pdf"
    figure_path = os.path.join(args.output_dir, figure_filename)
    plt.savefig(figure_path, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {figure_path}")

    # --- Bar plot: Mean and 95% bootstrapped CI for each algorithm ---
    means = []
    ci_lows = []
    ci_highs = []
    n_boot = 10000 # Standard number of bootstrap samples
    rng = np.random.default_rng(args.eval_seed) # Use a consistent RNG seed from args

    for label in plot_order_filtered: # plot_order_filtered is from figure4.py context
        data = results_df[results_df['agent_label'] == label][y_metric_col].values
        mean = np.mean(data)
        if len(data) > 1:
            boot_means = rng.choice(data, size=(n_boot, len(data)), replace=True).mean(axis=1)
            ci_low_val = np.percentile(boot_means, 2.5)
            ci_high_val = np.percentile(boot_means, 97.5)
        else: 
            ci_low_val = ci_high_val = mean
        means.append(mean)
        ci_lows.append(mean - ci_low_val)
        ci_highs.append(ci_high_val - mean)

    plt.figure(figsize=(8, 5)) # New figure for the bar plot
    bar_x = np.arange(len(plot_order_filtered))
    plt.bar(bar_x, means, yerr=[ci_lows, ci_highs], capsize=8,
            color=sns.color_palette("muted", n_colors=len(plot_order_filtered)), 
            edgecolor='black', linewidth=1.5)
    
    wrapped_bar_labels = ["\n".join(wrap(l, 12)) for l in plot_order_filtered] # Wrap labels
    plt.xticks(bar_x, wrapped_bar_labels, rotation=0, fontsize=11)
    plt.ylabel("Mean Episode Duration", fontsize=13) # Consistent y-axis label
    plt.xlabel("Algorithm", fontsize=13) # Consistent x-axis label
    plt.tight_layout(pad=0.5)

    # Use the same timestamp as the previous plot or generate a new one if preferred
    bar_figure_filename = f"figure4_bar_means_algo_comparison_{timestamp}.pdf" 
    bar_figure_path = os.path.join(args.output_dir, bar_figure_filename)
    plt.savefig(bar_figure_path, bbox_inches='tight')
    plt.close() # Close the bar plot figure
    print(f"Bar plot of means and 95% CI saved to {bar_figure_path}")

    # --- Directional Mann-Whitney U test section: one-sided test in direction of higher mean, Bonferroni correction, and directional advantage field ---
    print("\nPairwise Mann–Whitney U Test Results (Raw + Bonferroni-corrected, Directional)")

    def significance_stars(p):
        if p < 0.001: return '***'
        elif p < 0.01: return '**'
        elif p < 0.05: return '*'
        return 'n.s.'

    # Collect all episode lengths for each agent (across all seeds and runs)
    agents = [label for label in PLOT_ORDER if label in results_df['agent_label'].unique()]
    agent_data = {agent: results_df[results_df["agent_label"] == agent]["episode_length"].values for agent in agents}

    # All pairwise comparisons
    comparisons = list(itertools.combinations(agents, 2))
    comparison_results = []
    raw_pvals_one_sided = []

    for a1, a2 in comparisons:
        data1 = agent_data[a1]
        data2 = agent_data[a2]
        mean1 = np.mean(data1)
        mean2 = np.mean(data2)

        # Two-sided test
        _, p_two = mannwhitneyu(data1, data2, alternative='two-sided')

        # One-sided test in direction of higher mean
        if mean1 > mean2:
            _, p_one = mannwhitneyu(data1, data2, alternative='greater')
            direction = a1
        elif mean2 > mean1:
            _, p_one = mannwhitneyu(data2, data1, alternative='greater')
            direction = a2
        else:
            p_one = 1.0
            direction = "--"

        raw_pvals_one_sided.append(p_one)

        comparison_results.append({
            "Agent A": a1,
            "Agent B": a2,
            "p_two_raw": p_two,
            "p_one_raw": p_one,
            "Direction": direction
        })

    # Bonferroni correction on relevant one-sided tests
    _, pvals_one_corr, _, _ = smm.multipletests(raw_pvals_one_sided, alpha=0.05, method="bonferroni")

    # Add corrected p-values and stars
    for i, row in enumerate(comparison_results):
        row["p_one_corr"] = pvals_one_corr[i]
        row["sig_two"] = significance_stars(row["p_two_raw"])
        row["sig_one"] = significance_stars(pvals_one_corr[i])
        # If not significant in one-sided test, nullify the directional advantage
        if row["sig_one"] == "n.s.":
            row["Direction"] = "--"

    # Print NeurIPS-grade table with directional advantage
    print("{:<12} {:<12} {:<12} {:<12} {:<8} {:<12} {:<8} {:<20}".format(
        "Agent A", "Agent B", "p (2-sided)", "p (1-sided)", "Sig (2)", "p (1) Corr", "Sig (1)", "Directional Advantage"
    ))
    print("-" * 100)
    for row in comparison_results:
        print("{:<12} {:<12} {:<12.4g} {:<12.4g} {:<8} {:<12.4g} {:<8} {:<20}".format(
            row["Agent A"], row["Agent B"], row["p_two_raw"], row["p_one_raw"], row["sig_two"],
            row["p_one_corr"], row["sig_one"], row["Direction"]
        ))

    # Load all logs
    all_logs = []
    for algo in ["PPO", "SAC", "A2C"]:
        for seed in train_seeds:
            log_path = f"logs/{args.model_base}_{algo}_seed{seed}/training_returns.csv"
            if os.path.exists(log_path):
                df = pd.read_csv(log_path)
                df["algorithm"] = algo
                df["seed"] = seed
                all_logs.append(df)
    if not all_logs:
        print("No training logs found for any algorithm/seed.")
        return
    df_all = pd.concat(all_logs, ignore_index=True)

    # Option 1: Line plot with shaded area
    plt.figure(figsize=(8, 5))
    for algo in df_all["algorithm"].unique():
        df_algo = df_all[df_all["algorithm"] == algo]
        grouped = df_algo.groupby("episode")["mean_return"].agg(["mean", "std"])
        plt.plot(grouped.index, grouped["mean"], label=algo)
        plt.fill_between(grouped.index, grouped["mean"] - grouped["std"], grouped["mean"] + grouped["std"], alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel("Mean Return")
    plt.legend()
    plt.title("Mean Return Across Training (Shaded: ±1 SD)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "figure4_training_curve.pdf"), bbox_inches='tight')
    plt.close()

    # Option 2: Boxplot at final episode
    final_returns = []
    for algo in df_all["algorithm"].unique():
        for seed in df_all["seed"].unique():
            df_seed = df_all[(df_all["algorithm"] == algo) & (df_all["seed"] == seed)]
            if not df_seed.empty:
                final_return = df_seed.iloc[-1]["mean_return"]
                final_returns.append({"algorithm": algo, "seed": seed, "final_return": final_return})
    df_final = pd.DataFrame(final_returns)

    plt.figure(figsize=(6, 5))
    sns.boxplot(x="algorithm", y="final_return", data=df_final, palette="muted")
    sns.stripplot(x="algorithm", y="final_return", data=df_final, color='black', alpha=0.5, jitter=0.2)
    plt.xlabel("Algorithm")
    plt.ylabel("Final Mean Return")
    plt.title("Final Performance Across Seeds")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "figure4_final_boxplot.pdf"), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main() 