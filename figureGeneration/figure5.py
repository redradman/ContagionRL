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
from textwrap import wrap
from scipy.stats import mannwhitneyu

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

    # --- Plotting (Box + Violin + Stripplot + Per-seed Means) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    y_metric_col = "episode_length"
    y_label = "Episode Duration (steps)"
    plot_order_filtered = [label for label in PLOT_ORDER if label in results_df['ablation_label'].unique()]

    ax = sns.boxplot(
        x="ablation_label", y=y_metric_col, data=results_df, order=plot_order_filtered,
        width=0.6, showfliers=False, saturation=1,
        boxprops=dict(facecolor='none', edgecolor='black', linewidth=2),
        medianprops=dict(color='black', linewidth=2), whiskerprops=dict(color='black', linewidth=2), capprops=dict(color='black', linewidth=2)
    )
    sns.violinplot(
        x="ablation_label", y=y_metric_col, data=results_df, order=plot_order_filtered,
        width=0.8, inner=None, palette="muted", cut=0, alpha=0.5, ax=ax,
        hue="ablation_label", legend=False
    )
    sns.stripplot(
        x="ablation_label", y=y_metric_col, data=results_df, order=plot_order_filtered,
        color='black', alpha=0.3, jitter=0.2, size=5, ax=ax
    )

    # Overlay per-seed means as large black dots
    for i, ablation_label in enumerate(plot_order_filtered):
        group = results_df[results_df['ablation_label'] == ablation_label]
        if 'model_train_seed' in group.columns:
            seed_means = group.groupby('model_train_seed')[y_metric_col].mean()
            ax.scatter([i]*len(seed_means), seed_means, color='black', s=120, zorder=10, marker='o', edgecolor='white', linewidth=2, label=None)
    # Add legend entry for per-seed mean dots
    ax.scatter([], [], color='black', s=120, label='Per-seed Mean', edgecolor='white', linewidth=2)
    ax.legend(fontsize=11)

    # Improve x-tick label clarity (wrap if long, no rotation)
    ax.set_xticklabels(["\n".join(wrap(l, 12)) for l in plot_order_filtered], fontsize=11)
    ax.tick_params(axis='x', labelsize=11, rotation=0)
    ax.tick_params(axis='y', labelsize=11)
    plt.xlabel("Ablation Variant", fontsize=13)
    plt.ylabel(y_label, fontsize=13)
    plt.tight_layout(pad=0.5)

    # --- Directional Mann–Whitney U test vs. Full Model ---
    ref_label = VARIANT_LABELS["full"]
    ref_data = results_df[results_df["ablation_label"] == ref_label]["episode_length"]

    comparisons = []
    raw_one_sided_pvals = []

    for ablation_label in plot_order_filtered:
        if ablation_label == ref_label:
            continue

        compare_data = results_df[results_df["ablation_label"] == ablation_label]["episode_length"]
        mean1 = np.mean(compare_data)
        mean2 = np.mean(ref_data)

        # Two-sided test
        _, p_two = mannwhitneyu(compare_data, ref_data, alternative='two-sided')

        # One-sided test in the direction of higher mean
        if mean1 > mean2:
            _, p_one = mannwhitneyu(compare_data, ref_data, alternative='greater')
            winner = ablation_label
        elif mean2 > mean1:
            _, p_one = mannwhitneyu(ref_data, compare_data, alternative='greater')
            winner = ref_label
        else:
            p_one = 1.0
            winner = "--"

        comparisons.append({
            "Ablation": ablation_label,
            "Reference": ref_label,
            "p_two": p_two,
            "p_one_raw": p_one,
            "winner": winner
        })
        raw_one_sided_pvals.append(p_one)

    # Apply Bonferroni correction
    _, pvals_one_corr, _, _ = smm.multipletests(raw_one_sided_pvals, alpha=0.05, method="bonferroni")

    def significance_stars(p):
        if p < 0.001: return "***"
        elif p < 0.01: return "**"
        elif p < 0.05: return "*"
        return "n.s."

    for i, row in enumerate(comparisons):
        row["p_one_corr"] = pvals_one_corr[i]
        row["sig_two"] = significance_stars(row["p_two"])
        row["sig_one"] = significance_stars(row["p_one_corr"])
        if row["sig_one"] == "n.s.":
            row["winner"] = "--"

    # --- Add significance annotations to box/violin plot ---
    y_max = results_df[y_metric_col].max()
    increment = (results_df[y_metric_col].max() - results_df[y_metric_col].min()) * 0.08 if len(results_df) > 1 else 0.1 * abs(results_df[y_metric_col].max())
    current_y = y_max + increment * 0.5
    annotation_idx = 0

    for row in comparisons:
        if row['sig_one'] in ('*', '**', '***'):
            try:
                pos1 = plot_order_filtered.index(row['Ablation'])
                pos2 = plot_order_filtered.index(row['Reference'])
            except ValueError:
                continue  # skip if label is not found
            pos1, pos2 = sorted((pos1, pos2))
            line_x = [pos1, pos1, pos2, pos2]
            line_y = [current_y, current_y + increment * 0.3, current_y + increment * 0.3, current_y]
            ax.plot(line_x, line_y, lw=1.0, c='black')
            text_x = (pos1 + pos2) / 2
            text_y = current_y + increment * 0.5 + annotation_idx * 0.12
            ax.text(text_x, text_y, row['sig_one'], rotation=0, ha='center', fontsize=13)
            annotation_idx += 1
            current_y += increment * 1.5
    if len(plot_order_filtered) > 1:
        ax.set_ylim(top=current_y + increment * 0.2)

    # Print Table-style summary
    print("\nOne-Sided Mann–Whitney U Test Results (vs. Full Potential Field):")
    print("{:<26} {:<22} {:<12} {:<12} {:<8} {:<12} {:<8} {:<20}".format(
        "Ablation Variant", "Reference", "p (2-sided)", "p (1-sided)", "Sig (2)", "p (1) Corr", "Sig (1)", "Winner"
    ))
    print("-" * 126) # Adjusted length for new column widths
    for row in comparisons:
        print("{:<26} {:<22} {:<12.4g} {:<12.4g} {:<8} {:<12.4g} {:<8} {:<20}".format(
            row["Ablation"], row["Reference"], row["p_two"], row["p_one_raw"],
            row["sig_two"], row["p_one_corr"], row["sig_one"], row["winner"]
        ))

    # Save the boxplot/violin plot before creating a new figure for the bar plot
    timestamp_box = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    figure_filename_box = f"figure5_ablation_comparison_{timestamp_box}.pdf"
    figure_path_box = os.path.join(args.output_dir, figure_filename_box)
    fig.savefig(figure_path_box, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {figure_path_box}")

    # Ensure n_boot, rng, and timestamp are defined for the bar plot
    n_boot = 10000
    rng = np.random.default_rng(42)

    # --- Bar plot: Mean and 95% bootstrapped CI for each ablation variant ---
    means = []
    ci_lows = []
    ci_highs = []
    for label in plot_order_filtered:
        data = results_df[results_df['ablation_label'] == label][y_metric_col].values
        mean = np.mean(data)
        if len(data) > 1:
            boot_means = rng.choice(data, size=(n_boot, len(data)), replace=True).mean(axis=1)
            ci_low = np.percentile(boot_means, 2.5)
            ci_high = np.percentile(boot_means, 97.5)
        else:
            ci_low = ci_high = mean
        means.append(mean)
        ci_lows.append(mean - ci_low)
        ci_highs.append(ci_high - mean)

    plt.figure(figsize=(10, 5))
    bar_x = np.arange(len(plot_order_filtered))
    bar = plt.bar(bar_x, means, yerr=[ci_lows, ci_highs], capsize=8, color=sns.color_palette("muted", n_colors=len(plot_order_filtered)), edgecolor='black', linewidth=1.5)
    wrapped_bar_labels = ["\n".join(wrap(l, 12)) for l in plot_order_filtered]
    plt.xticks(bar_x, wrapped_bar_labels, rotation=0, fontsize=11)
    plt.ylabel("Mean Episode Duration", fontsize=13)
    plt.xlabel("Ablation Variant", fontsize=13)
    plt.tight_layout(pad=0.5)

    bar_figure_filename = f"figure5_bar_means_{timestamp_box}.pdf"
    bar_figure_path = os.path.join(args.output_dir, bar_figure_filename)
    plt.savefig(bar_figure_path, bbox_inches='tight')
    plt.close()
    print(f"Bar plot of means and 95% CI saved to {bar_figure_path}")

    csv_filename = f"figure5_data_{timestamp_box}.csv"
    csv_path = os.path.join(args.output_dir, csv_filename)
    results_df.to_csv(csv_path, index=False)
    print(f"Aggregated data saved to {csv_path}")

    # Print table of raw mean episode lengths per ablation
    print("\nMean Episode Lengths by Ablation Variant:")
    means = results_df.groupby('ablation_label')[y_metric_col].mean()
    for label in plot_order_filtered:
        print(f"{label}: {means[label]:.2f}")

if __name__ == "__main__":
    main() 