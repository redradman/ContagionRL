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
# from typing import Dict, List, Any, Optional

# tueplots styling
from tueplots import bundles
plt.rcParams.update(bundles.neurips2024())
plt.rcParams["figure.dpi"] = 300
sns.set_style("whitegrid")

# Add the parent directory to the path to access project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import SIRSEnvironment

class FlattenObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.flatten_space(env.observation_space)
    def observation(self, observation):
        return gym.spaces.flatten(self.env.observation_space, observation)

def create_env_for_algo(env_config: dict, seed: int, algo: str) -> SIRSEnvironment:
    config = env_config.copy()
    config["render_mode"] = None
    env = SIRSEnvironment(**config)
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

    # --- Plotting ---
    results_df = pd.DataFrame(all_results)
    plt.figure(figsize=(8, 6))
    y_metric_col = "episode_length"
    y_label = "Episode Duration (steps)"
    plot_order_filtered = [label for label in PLOT_ORDER if label in results_df['agent_label'].unique()]

    # Barplot with error bars (mean and standard error)
    ax = sns.barplot(
        x="agent_label",
        y=y_metric_col,
        data=results_df,
        order=plot_order_filtered,
        errorbar="se",
        capsize=0.15,
        palette="muted",
        errcolor="black",
        errwidth=1.5
    )
    # Add a thin dashed red line at y=simulation_time
    # if base_env_config is not None and "simulation_time" in base_env_config:
    #     sim_time = base_env_config["simulation_time"]
    #     ax.axhline(sim_time, color='red', linestyle='--', linewidth=1)
    plt.xlabel("Agent/Algorithm", fontsize=9)
    plt.ylabel(y_label, fontsize=9)
    ax.tick_params(axis='x', labelsize=8, rotation=0)
    ax.tick_params(axis='y', labelsize=8)
    plt.tight_layout(pad=0.5)
    # Barplot style: Bars show mean ± standard error across all episodes from 3 training seeds.
    # NeurIPS-style caption (for your paper/figure legend):
    # Figure 4: Episode durations across agents. Each point is the result of one evaluation episode. Boxes show the interquartile range and median across all seeds and episodes. Large dots show the mean for each training seed. This reveals variance in agent behavior and robustness across training seeds.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    figure_filename = f"figure4_algo_comparison_{timestamp}.pdf"
    figure_path = os.path.join(args.output_dir, figure_filename)
    plt.savefig(figure_path, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {figure_path}")

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