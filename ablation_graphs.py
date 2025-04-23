#!/usr/bin/env python
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Define the order and descriptive names for plotting
# Should match the VARIANT_NAMES in ablation.py
VARIANT_NAMES_ORDER = [
    "Full",                         # full
    "Drop Susceptible Repulsion",   # no_S
    "Drop Magnitude",               # no_magnitude
    "Drop Direction",               # no_direction
    "Drop Movement",                # no_move
    "Drop Adherence",               # no_adherence
    "Drop Health",                  # no_health
]

# Define reward component columns from episodes CSV and their nice names
REWARD_COMPONENT_COLS = {
    "ep_health_rate": "Health Rate",
    "ep_adherence_mean": "Mean Adherence",
    "ep_move_dir": "Mean Move Direction Reward",
    "ep_move_mag": "Mean Move Magnitude Reward",
    "ep_move": "Mean Combined Move Reward",
    "ep_adh_reward": "Mean Adherence Reward"
}


def plot_learning_curves(df, y_metric, y_label, title, output_path, variants_order):
    """
    Plots mean metric vs timesteps with std dev bands for each variant.
    """
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Use lineplot which automatically calculates mean and shows confidence interval (or sd)
    sns.lineplot(
        data=df,
        x='timesteps',
        y=y_metric,
        hue='variant',
        hue_order=variants_order,
        errorbar='sd',  # Show standard deviation band
        linewidth=1.5
    )
    
    plt.title(title, fontsize=16)
    plt.xlabel("Total Timesteps", fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.legend(title='Variant', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved learning curve plot: {output_path}")

def _get_final_episode_data(df):
    """Helper to get the data from the last recorded episode for each run (variant + seed)."""
    # Find the maximum timestep recorded for each variant and seed combination
    last_timestep_indices = df.loc[df.groupby(['variant', 'seed'])['timesteps'].idxmax()]
    return last_timestep_indices

def _get_last_n_episodes_data(df, n=20):
    """Helper to get the data from the last N recorded episodes for each run (variant + seed)."""
    # Ensure the dataframe is sorted by timesteps within each group to get the correct tail
    df_sorted = df.sort_values(by=['variant', 'seed', 'timesteps'])
    # Group by variant and seed, then take the last N rows for each group
    last_n_episodes = df_sorted.groupby(['variant', 'seed']).tail(n)
    return last_n_episodes

def plot_final_distribution(df, y_metric, y_label, title, output_path, variants_order, plot_type='violin'):
    """
    Plots the distribution of the final metric for each variant using violin or box plots,
    considering the last 20 episodes of each run.
    """
    final_perf_df = _get_last_n_episodes_data(df, n=20)
    
    plt.figure(figsize=(14, 7))
    sns.set_style("whitegrid")

    if plot_type == 'violin':
        sns.violinplot(
            data=final_perf_df,
            x='variant',
            y=y_metric,
            order=variants_order,
            inner='quartile', # Show quartiles inside violins
            cut=0 # Don't extend density beyond data range
        )
    elif plot_type == 'box':
        sns.boxplot(
            data=final_perf_df,
            x='variant',
            y=y_metric,
            order=variants_order,
            showfliers=True # Show outliers
        )
    else:
        raise ValueError("plot_type must be 'violin' or 'box'")

    plt.title(title, fontsize=16)
    plt.xlabel("Ablation Variant", fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.xticks(rotation=30, ha='right') # Rotate labels for better readability
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved final distribution plot ({plot_type}): {output_path}")

def plot_final_bars(df, y_metric, y_label, title, output_path, variants_order):
    """
    Plots a bar chart of the mean final metric ± std dev for each variant,
    calculated over the last 20 episodes of each run.
    """
    final_perf_df = _get_last_n_episodes_data(df, n=10)

    # Calculate mean and std dev of the metric across the last 20 episodes FOR EACH VARIANT
    grouped_stats = final_perf_df.groupby('variant')[y_metric].agg(['mean', 'std']).reset_index()
    
    # Ensure the order matches variants_order
    grouped_stats['variant'] = pd.Categorical(grouped_stats['variant'], categories=variants_order, ordered=True)
    grouped_stats = grouped_stats.sort_values('variant')

    plt.figure(figsize=(14, 7))
    sns.set_style("whitegrid")

    plt.bar(
        grouped_stats['variant'],
        grouped_stats['mean'],
        yerr=grouped_stats['std'],
        capsize=5, # Add caps to error bars
        color=sns.color_palette("tab10", len(variants_order)) # Use consistent colors
    )

    plt.title(title, fontsize=16)
    plt.xlabel("Ablation Variant", fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.xticks(rotation=30, ha='right') # Rotate labels for better readability
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved final performance bar chart: {output_path}")

def plot_reward_components(df, title_prefix, output_dir, variants_order):
    """
    Plots the evolution of each reward component over time, overlaying all variants.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for col, nice_name in REWARD_COMPONENT_COLS.items():
        if col not in df.columns:
            print(f"Warning: Reward component column '{col}' not found in episodes data. Skipping plot.")
            continue

        plt.figure(figsize=(12, 7))
        sns.set_style("whitegrid")

        sns.lineplot(
            data=df,
            x='timesteps',
            y=col,
            hue='variant',
            hue_order=variants_order,
            errorbar='sd', # Show standard deviation band
            linewidth=1.5,
            legend=False # Remove individual legends from subplots initially
        )

        plot_title = f"{title_prefix}: {nice_name} Evolution"
        output_path = output_dir / f"reward_component_{col}.png"

        plt.title(plot_title, fontsize=16)
        plt.xlabel("Total Timesteps", fontsize=12)
        plt.ylabel(f"Mean {nice_name}", fontsize=12)
        
        # Create a single legend for the figure
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles)) # Remove duplicate labels for legend
        fig = plt.gcf()
        fig.legend(by_label.values(), by_label.keys(), title='Variant', bbox_to_anchor=(0.98, 0.9), loc='upper left', borderaxespad=0.)

        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved reward component plot: {output_path}")

def plot_smoothed_learning_curve(
    df, 
    y_metric, 
    y_label, 
    title, 
    output_path, 
    variants_order, 
    # window_size=1000 # Original fixed binning parameter
    rolling_window_episodes=50 # New rolling window parameter (number of episodes)
):
    """
    Plots a smoothed learning curve using a rolling average over episodes for each variant/seed.

    Args:
        df (pd.DataFrame): The episodes dataframe containing timesteps, variant, seed, and the y_metric.
        y_metric (str): The column name of the metric to plot (e.g., 'ep_length').
        y_label (str): The label for the y-axis.
        title (str): The title for the plot.
        output_path (Path): The path to save the plot image.
        variants_order (list): The order in which to plot the variants.
        rolling_window_episodes (int): The number of preceding episodes to include in the rolling average.
    """
    print(f"Generating smoothed learning curve for '{y_metric}' with rolling window of {rolling_window_episodes} episodes...")

    # Ensure data is sorted correctly for rolling calculation
    df_sorted = df.sort_values(by=['variant', 'seed', 'timesteps']).copy()

    # Calculate rolling mean and std for the metric *within each variant and seed*
    df_sorted['rolling_mean'] = df_sorted.groupby(['variant', 'seed'])[y_metric].transform(
        lambda x: x.rolling(rolling_window_episodes, min_periods=1).mean()
    )
    df_sorted['rolling_std'] = df_sorted.groupby(['variant', 'seed'])[y_metric].transform(
        lambda x: x.rolling(rolling_window_episodes, min_periods=1).std()
    )
    
    # Now, we need to aggregate these rolling metrics across seeds for plotting
    # We can use seaborn's lineplot for this, as it handles aggregation
    # We plot the rolling_mean, and lineplot will average these means across seeds
    # and can calculate SD or CI of these rolling means.

    # --- Plotting ---    
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    
    # Use lineplot on the calculated rolling means. It will average across seeds.
    sns.lineplot(
        data=df_sorted,
        x='timesteps',
        y='rolling_mean', # Plot the pre-calculated rolling mean
        hue='variant',
        hue_order=variants_order,
        errorbar='sd',  # Show SD of the rolling means across seeds
        # errorbar=('ci', 95), # Alternative: Show 95% CI of the mean rolling mean
        linewidth=1.5,
        # palette=sns.color_palette("tab10", len(variants_order))
    )
    
    # Access the current axes to potentially adjust alpha if needed
    ax = plt.gca()
    # Make the SD bands slightly more transparent
    for collection in ax.collections: # Seaborn uses collections for fill_between
        collection.set_alpha(0.2) # Lower alpha value for more transparency
    
    plt.title(title, fontsize=16)
    plt.xlabel(f"Total Timesteps", fontsize=12) # X-axis is still timesteps
    plt.ylabel(f"{y_label} (Rolling Mean, Window={rolling_window_episodes} eps)", fontsize=12)
    plt.legend(title='Variant', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved rolling average learning curve plot: {output_path}")

def plot_early_late_comparison(df, y_metric, y_label, title, output_path, variants_order, n_episodes=20):
    """
    Plots side-by-side boxplots comparing the first N and last N episodes for each variant.

    Args:
        df (pd.DataFrame): Episodes dataframe with variant, seed, y_metric, and an episode identifier (like episode_id or sorted index).
        y_metric (str): The metric to plot (e.g., 'ep_length').
        y_label (str): Y-axis label.
        title (str): Plot title.
        output_path (Path): Path to save the plot.
        variants_order (list): Order for variants on the x-axis.
        n_episodes (int): Number of early/late episodes to compare.
    """
    print(f"Generating early vs. late comparison plot for '{y_metric}' (n={n_episodes})...")

    # Ensure dataframe is sorted to reliably get head/tail
    # Need a consistent episode ordering within each seed run
    # Assuming 'timesteps' or 'episode_id' can provide this order
    if 'episode_id' in df.columns:
        sort_col = 'episode_id'
    elif 'timesteps' in df.columns:
        sort_col = 'timesteps' # Use timesteps if episode_id is missing
    else:
         print("Warning: Cannot determine episode order for early/late plot. Skipping.")
         return
    df_sorted = df.sort_values(by=['variant', 'seed', sort_col])

    # Get the first N episodes for each group
    early_df = df_sorted.groupby(['variant', 'seed']).head(n_episodes).copy()
    early_df['period'] = 'Early'

    # Get the last N episodes for each group
    late_df = df_sorted.groupby(['variant', 'seed']).tail(n_episodes).copy()
    late_df['period'] = 'Late'

    # Combine the early and late data
    combined_df = pd.concat([early_df, late_df], ignore_index=True)

    # --- Plotting ---
    plt.figure(figsize=(16, 8))
    sns.set_style("whitegrid")

    sns.boxplot(
        data=combined_df,
        x='variant',
        y=y_metric,
        hue='period',
        order=variants_order,
        hue_order=['Early', 'Late'],
        showfliers=True, # Optionally show outliers
        # fliersize=3, # Adjust outlier size if needed
        linewidth=1.0 # Box line width
    )

    plt.title(title, fontsize=16)
    plt.xlabel("Ablation Variant", fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.xticks(rotation=30, ha='right')
    plt.legend(title='Period', loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved early vs. late comparison plot: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate graphs for ablation study results.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="ablation",
        help="Directory containing the results_episodes.csv and results_iterations.csv files."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ablation/graphs",
        help="Directory where the generated graphs will be saved."
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes_csv_path = input_dir / "results_episodes.csv"
    # iterations_csv_path = input_dir / "results_iterations.csv" # Not used currently based on plots requested

    if not episodes_csv_path.is_file():
        print(f"Error: Episodes CSV file not found at {episodes_csv_path}")
        return
    # if not iterations_csv_path.is_file():
    #     print(f"Error: Iterations CSV file not found at {iterations_csv_path}")
    #     return

    print(f"Loading data from {episodes_csv_path}...")
    episodes_df = pd.read_csv(episodes_csv_path)
    # iterations_df = pd.read_csv(iterations_csv_path) # Load if needed later

    # --- Generate Plots ---

    # 1. Learning Curves (Episode Return)
    # plot_learning_curves(
    #     df=episodes_df,
    #     y_metric='ep_return',
    #     y_label='Mean Episode Return',
    #     title='Learning Curves: Mean Episode Return vs Timesteps',
    #     output_path=output_dir / 'learning_curve_return.png',
    #     variants_order=VARIANT_NAMES_ORDER
    # )

    # 1b. Learning Curves (Episode Length)
    # plot_learning_curves(
    #     df=episodes_df,
    #     y_metric='ep_length',
    #     y_label='Mean Episode Length',
    #     title='Learning Curves: Mean Episode Length vs Timesteps',
    #     output_path=output_dir / 'learning_curve_length.png',
    #     variants_order=VARIANT_NAMES_ORDER
    # )

    # 2. Final Performance Distribution (Violin Plot - Return)
    # plot_final_distribution(
    #     df=episodes_df,
    #     y_metric='ep_return',
    #     y_label='Final Episode Return',
    #     title='Distribution of Final Episode Return per Variant',
    #     output_path=output_dir / 'final_distribution_return_violin.png',
    #     variants_order=VARIANT_NAMES_ORDER,
    #     plot_type='violin'
    # )
    
    # 2b. Final Performance Distribution (Box Plot - Return) - Optional alternative
    plot_final_distribution(
        df=episodes_df,
        y_metric='ep_return',
        y_label='Final Episode Return',
        title='Distribution of Final Episode Return per Variant',
        output_path=output_dir / 'final_distribution_return_box.png',
        variants_order=VARIANT_NAMES_ORDER,
        plot_type='box'
    )
    
    # 2c. Final Performance Distribution (Violin Plot - Length)
    # plot_final_distribution(
    #     df=episodes_df,
    #     y_metric='ep_length',
    #     y_label='Final Episode Length',
    #     title='Distribution of Final Episode Length per Variant',
    #     output_path=output_dir / 'final_distribution_length_violin.png',
    #     variants_order=VARIANT_NAMES_ORDER,
    #     plot_type='violin'
    # )
    

    # 3. Final Performance Bar Chart (Return)
    plot_final_bars(
        df=episodes_df,
        y_metric='ep_return',
        y_label='Mean Final Episode Return (±SD)',
        title='Final Performance: Mean Episode Return per Variant',
        output_path=output_dir / 'final_performance_return_bar.png',
        variants_order=VARIANT_NAMES_ORDER
    )

    # 3b. Final Performance Bar Chart (Length)
    plot_final_bars(
        df=episodes_df,
        y_metric='ep_length',
        y_label='Mean Final Episode Length (±SD)',
        title='Final Performance: Mean Episode Length per Variant',
        output_path=output_dir / 'final_performance_length_bar.png',
        variants_order=VARIANT_NAMES_ORDER
    )

    # --- Add Smoothed Learning Curve Plot ---
    plot_smoothed_learning_curve(
        df=episodes_df,
        y_metric='ep_length',
        y_label='Mean Episode Length (±SD)',
        title='Smoothed Learning Curve: Mean Episode Length vs Timesteps',
        output_path=output_dir / 'smoothed_learning_curve_length.png',
        variants_order=VARIANT_NAMES_ORDER,
        rolling_window_episodes=300 # Adjust window size (number of episodes) as needed
    )
    # ----------------------------------------

    # 4. Reward Component Evolution
    # plot_reward_components(
    #     df=episodes_df,
    #     title_prefix='Reward Component Evolution',
    #     output_dir=output_dir / 'reward_components', # Save in a sub-directory
    #     variants_order=VARIANT_NAMES_ORDER
    # )

    # 5. Early vs. Late Comparison
    plot_early_late_comparison(
        df=episodes_df,
        y_metric='ep_length',
        y_label='Episode Length', # Changed label slightly
        title='Early (First 20) vs. Late (Last 20) Episode Length Comparison',
        output_path=output_dir / 'early_late_comparison_length.png',
        variants_order=VARIANT_NAMES_ORDER,
        n_episodes=20 # Specify N=20
    )

    print("\nGraph generation complete.")
    print(f"Graphs saved in: {output_dir.resolve()}")

if __name__ == "__main__":
    main()
