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
import cliffs_delta

# Add the parent directory to the path to access project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import SIRSEnvironment

# tueplots styling
from tueplots import bundles
plt.rcParams.update(bundles.neurips2024())
plt.rcParams["figure.dpi"] = 300
sns.set_style("whitegrid")

# Ablation variant names and order (should match train_figure5_models.py)
ABLATION_VARIANTS = [
    "no_magnitude", "no_direction", "no_move", "no_adherence", "no_health", "no_S", "full"
]
VARIANT_LABELS = {
    "full":          "Full Potential Field",
    "no_magnitude":  "No Magnitude",
    "no_direction":  "No Direction",
    "no_move":       "No Movement",
    "no_adherence":  "No Adherence",
    "no_health":     "No Health",
    "no_S":          "No Susceptible Repulsion",
}
PLOT_ORDER = [VARIANT_LABELS[v] for v in ABLATION_VARIANTS]


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

def run_evaluation_episodes(env: SIRSEnvironment, model: PPO, num_episodes: int, base_eval_seed: int) -> List[int]:
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
    parser = argparse.ArgumentParser(description="Generate Figure 5 for ablation study (Potential Field reward ablations).")
    parser.add_argument("--model-base", type=str, required=True, help="Base name for all models (e.g., Fig5)")
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
    base_env_config_dict = None

    # Map ablation variant to label
    ablation_label_map = {v: VARIANT_LABELS[v] for v in ABLATION_VARIANTS}
    model_bases = {label: f"{args.model_base}_{variant}" for variant, label in ablation_label_map.items()}

    category_seed_offset = 0
    for ablation_variant in ABLATION_VARIANTS:
        ablation_label = VARIANT_LABELS[ablation_variant]
        model_base_name = f"{args.model_base}_{ablation_variant}"
        print(f"Processing models for ablation: {ablation_label}")
        for train_seed in tqdm(train_seeds, desc=f"Models for {ablation_label}"):
            model_dir_name = f"{model_base_name}_seed{train_seed}"
            model_path = os.path.join("logs", model_dir_name, "best_model.zip")
            if not os.path.exists(model_path):
                print(f"Warning: Model file not found for {ablation_label} (seed {train_seed}) at {model_path}. Skipping.")
                continue
            try:
                if base_env_config_dict is None:
                    model_config = load_model_config(model_path)
                    base_env_config_dict = model_config.get("environment")
                    if base_env_config_dict is None:
                        print(f"Error: 'environment' key missing in config for {model_path}. Cannot proceed.")
                        return
                env_creation_seed = args.eval_seed + category_seed_offset + train_seed
                env = create_env_from_config(base_env_config_dict, seed=env_creation_seed)
                model = PPO.load(model_path, env=env)
                model_eval_run_base_seed = args.eval_seed + category_seed_offset * 100 + train_seed * args.runs
                episode_lengths = run_evaluation_episodes(env, model, args.runs, model_eval_run_base_seed)
                for length in episode_lengths:
                    all_results_data.append({
                        "ablation_label": ablation_label,
                        "ablation_variant": ablation_variant,
                        "model_train_seed": train_seed,
                        "episode_length": length
                    })
                env.close()
            except Exception as e:
                print(f"Error processing model {model_path} for {ablation_label}: {e}")
        category_seed_offset += 1

    if not all_results_data:
        print("No data collected from any models. Exiting.")
        return

    results_df = pd.DataFrame(all_results_data)

    # --- Plotting (Violin + Box Overlay + Stripplot) ---
    plt.figure(figsize=(10, 6))
    y_metric_col = "episode_length"
    y_label = "Episode Duration (steps)"
    plot_order_filtered = [label for label in PLOT_ORDER if label in results_df['ablation_label'].unique()]

    ax = sns.boxplot(
        x="ablation_label", y=y_metric_col, data=results_df, order=plot_order_filtered,
        width=0.6, showfliers=False, saturation=1,
        boxprops=dict(facecolor='none', edgecolor='black'),
        medianprops=dict(color='black'), whiskerprops=dict(color='black'), capprops=dict(color='black')
    )
    sns.violinplot(
        x="ablation_label", y=y_metric_col, data=results_df, order=plot_order_filtered,
        width=0.8, inner=None, palette="muted", cut=0, alpha=0.5, ax=ax,
        hue="ablation_label", legend=False
    )
    sns.stripplot(
        x="ablation_label", y=y_metric_col, data=results_df, order=plot_order_filtered,
        color='black', alpha=0.3, jitter=0.2, size=3, ax=ax
    )

    # Overlay per-seed means as large black dots
    for i, ablation_label in enumerate(plot_order_filtered):
        group = results_df[results_df['ablation_label'] == ablation_label]
        if 'model_train_seed' in group.columns:
            seed_means = group.groupby('model_train_seed')[y_metric_col].mean()
            ax.scatter([i]*len(seed_means), seed_means, color='black', s=80, zorder=10, marker='o', edgecolor='white', linewidth=1.5, label=None)

    # --- Cliff's delta effect size annotations ---
    full_label = VARIANT_LABELS['full']
    full_data = results_df[results_df['ablation_label'] == full_label]
    full_episodes = full_data[y_metric_col].values
    cat_pos = {cat: i for i, cat in enumerate(plot_order_filtered)}
    y_max = results_df[y_metric_col].max()
    y_min = results_df[y_metric_col].min()
    y_range = y_max - y_min
    increment = y_range * 0.08 if y_range > 1e-9 else 0.1 * abs(y_max) if abs(y_max) > 1e-9 else 0.1
    current_y = y_max + increment * 0.5
    for ablation_label in plot_order_filtered:
        if ablation_label == full_label:
            continue
        compare_data = results_df[results_df['ablation_label'] == ablation_label]
        compare_episodes = compare_data[y_metric_col].values
        if len(full_episodes) > 0 and len(compare_episodes) > 0:
            d, _ = cliffs_delta.cliffs_delta(compare_episodes, full_episodes)
            # Interpret effect size
            abs_d = abs(d)
            if abs_d < 0.147:
                effect = 'negligible effect'
            elif abs_d < 0.33:
                effect = 'small effect'
            elif abs_d < 0.474:
                effect = 'medium effect'
            else:
                effect = 'large effect'
            if effect != 'negligible effect':
                pos1 = cat_pos[full_label]
                pos2 = cat_pos[ablation_label]
                line_x = [pos1, pos1, pos2, pos2]
                line_y = [current_y, current_y + increment * 0.2, current_y + increment * 0.2, current_y]
                ax.plot(line_x, line_y, lw=1.0, c='black')
                text_x = (pos1 + pos2) / 2
                text_y = current_y + increment * 0.25
                ax.text(text_x, text_y, effect, ha='center', va='bottom', fontsize=8)
                current_y += increment
    if len(plot_order_filtered) > 1:
        ax.set_ylim(top=current_y + increment * 0.2)

    plt.xlabel("Ablation Variant", fontsize=9)
    plt.ylabel(y_label, fontsize=9)
    ax.tick_params(axis='x', labelsize=8, rotation=0)
    ax.tick_params(axis='y', labelsize=8)
    plt.tight_layout(pad=0.5)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    figure_filename = f"figure5_ablation_comparison_{timestamp}.pdf"
    figure_path = os.path.join(args.output_dir, figure_filename)
    plt.savefig(figure_path, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {figure_path}")
    csv_filename = f"figure5_data_{timestamp}.csv"
    csv_path = os.path.join(args.output_dir, csv_filename)
    results_df.to_csv(csv_path, index=False)
    print(f"Aggregated data saved to {csv_path}")

if __name__ == "__main__":
    main() 