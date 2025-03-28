#!/usr/bin/env python
import os
import sys
import argparse
import datetime
from scipy import stats as scipy_stats

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from result_utils import (
    run_benchmark,
    plot_survival_boxplot,
    save_benchmark_results,
    get_summary_stats,
    plot_exposure_adherence_scatterplot,
    plot_final_reward_boxplot
)

def main():
    """
    Run benchmarks comparing a trained model against random actions.
    Generates boxplots for episode duration and final rewards, and exposure vs adherence scatterplot.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark a trained SIRS model against random actions"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model file (.zip)"
    )
    
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of episodes to run for each agent (default: 10)"
    )
    
    parser.add_argument(
        "--no-random",
        action="store_true",
        help="Skip running the random agent benchmark"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/graphs",
        help="Directory to save the results (default: results/graphs)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title prefix for the plots (default: auto-generated based on model path)"
    )
    
    args = parser.parse_args()
    
    # Create timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract model name from path for automatic title generation
    model_name = os.path.basename(os.path.dirname(args.model_path))
    
    # Generate output filename using model name and timestamp
    output_base = f"{model_name}_{timestamp}"
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Starting benchmark with {args.runs} runs...")
    print(f"Model path: {args.model_path}")
    print(f"Results will be saved to: {args.output_dir}")
    
    # Run benchmark to collect all metrics in a single pass
    print("Running benchmark...")
    results = run_benchmark(
        model_path=args.model_path,
        n_runs=args.runs,
        include_random=not args.no_random,
        random_seed=args.seed
    )
    
    # Generate title prefix if not provided
    title_prefix = args.title
    if title_prefix is None:
        title_prefix = f"{model_name}"
    
    # Generate and save survival boxplot
    boxplot_filename = f"{output_base}_episode_duration_boxplot.png"
    boxplot_title = f"Episode Duration: {title_prefix} vs Random Actions ({args.runs} runs)"
    print(f"Generating episode duration boxplot: {boxplot_filename}")
    
    plot_survival_boxplot(
        results,
        title=boxplot_title,
        filename=boxplot_filename,
        save_dir=args.output_dir
    )
    
    # Generate exposure vs adherence scatterplot using data from the same benchmark run
    scatterplot_filename = f"{output_base}_exposure_adherence.png"
    exp_title = f"Exposure vs Adherence: {title_prefix} ({args.runs} runs)"
    print(f"Generating exposure vs adherence scatterplot: {scatterplot_filename}")
    
    plot_exposure_adherence_scatterplot(
        results,
        title=exp_title,
        filename=scatterplot_filename,
        save_dir=args.output_dir,
        include_random=not args.no_random
    )
    
    # Generate and save final reward boxplot
    final_reward_boxplot_filename = f"{output_base}_final_reward_boxplot.png"
    final_reward_boxplot_title = f"Final Cumulative Reward: {title_prefix} vs Random Actions ({args.runs} runs)"
    print(f"Generating final reward boxplot: {final_reward_boxplot_filename}")
    
    plot_final_reward_boxplot(
        results,
        title=final_reward_boxplot_title,
        filename=final_reward_boxplot_filename,
        save_dir=args.output_dir
    )
    
    # Save benchmark data
    data_filename = f"{output_base}_data.json"
    print(f"Saving benchmark data: {data_filename}")
    
    save_benchmark_results(
        results,
        filename=data_filename,
        save_dir=args.output_dir
    )
    
    # Get and print summary statistics
    summary_stats = get_summary_stats(results)
    
    print("\nBenchmark Summary:")
    print("  Trained Model:")
    print(f"    Mean Episode Duration: {summary_stats['trained']['mean_episode_length']:.2f} steps (±{summary_stats['trained']['std_episode_length']:.2f})")
    print(f"    Mean Final Reward: {summary_stats['trained']['mean_final_reward']:.2f} (±{summary_stats['trained']['std_final_reward']:.2f})")
    
    if not args.no_random and "random" in summary_stats:
        print("  Random Actions:")
        print(f"    Mean Episode Duration: {summary_stats['random']['mean_episode_length']:.2f} steps (±{summary_stats['random']['std_episode_length']:.2f})")
        print(f"    Mean Final Reward: {summary_stats['random']['mean_final_reward']:.2f} (±{summary_stats['random']['std_final_reward']:.2f})")
        
        # Calculate and print statistical test results
        trained_lengths = results["trained"]["episode_lengths"]
        random_lengths = results["random"]["episode_lengths"]
        if len(trained_lengths) > 0 and len(random_lengths) > 0:
            u_stat, p_value = scipy_stats.mannwhitneyu(trained_lengths, random_lengths, alternative='two-sided')
            print("\n  Statistical Tests (Mann-Whitney U):")
            print(f"    Episode Duration: p = {p_value:.4f}", end="")
            if p_value < 0.001:
                print(" (***)")
            elif p_value < 0.01:
                print(" (**)")
            elif p_value < 0.05:
                print(" (*)")
            else:
                print(" (not significant)")
        
        trained_final_rewards = [rewards[-1] for rewards in results["trained"]["rewards_over_time"]]
        random_final_rewards = [rewards[-1] for rewards in results["random"]["rewards_over_time"]]
        if len(trained_final_rewards) > 0 and len(random_final_rewards) > 0:
            u_stat, p_value = scipy_stats.mannwhitneyu(trained_final_rewards, random_final_rewards, alternative='two-sided')
            print(f"    Final Reward: p = {p_value:.4f}", end="")
            if p_value < 0.001:
                print(" (***)")
            elif p_value < 0.01:
                print(" (**)")
            elif p_value < 0.05:
                print(" (*)")
            else:
                print(" (not significant)")
    
    print(f"\nResults saved to {args.output_dir}/")
    print(f"Episode Duration Boxplot: {os.path.join(args.output_dir, boxplot_filename)}")
    print(f"Final Reward Boxplot: {os.path.join(args.output_dir, final_reward_boxplot_filename)}")
    print(f"Exposure vs Adherence Plot: {os.path.join(args.output_dir, scatterplot_filename)}")
    print(f"Data: {os.path.join(args.output_dir, data_filename)}")

if __name__ == "__main__":
    main() 