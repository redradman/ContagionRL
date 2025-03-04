#!/usr/bin/env python
import os
import sys
import argparse
import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from result_utils import (
    run_benchmark,
    plot_cumulative_rewards,
    plot_survival_boxplot,
    save_benchmark_results,
    get_summary_stats
)

def main():
    """
    Run benchmarks comparing a trained model against random actions.
    Generates plots showing cumulative reward over time.
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
        help="Custom title for the plot (default: auto-generated based on model path)"
    )
    
    args = parser.parse_args()
    
    # Create timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract model name from path for automatic title generation
    model_name = os.path.basename(os.path.dirname(args.model_path))
    
    # Generate output filename using model name and timestamp
    output_base = f"{model_name}_{timestamp}"
    
    # Generate title if not provided
    title = args.title
    if title is None:
        title = f"Cumulative Reward: {model_name} vs Random Actions"
    
    # Add number of runs to title
    title = f"{title} ({args.runs} runs)"
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Starting benchmark with {args.runs} runs...")
    print(f"Model path: {args.model_path}")
    print(f"Results will be saved to: {args.output_dir}")
    
    # Run benchmark
    print("Running benchmark...")
    results = run_benchmark(
        model_path=args.model_path,
        n_runs=args.runs,
        include_random=not args.no_random,
        random_seed=args.seed
    )
    
    # Generate and save reward plot
    reward_plot_filename = f"{output_base}_rewards.png"
    print(f"Generating reward plot: {reward_plot_filename}")
    
    plot_cumulative_rewards(
        results,
        title=title,
        filename=reward_plot_filename,
        save_dir=args.output_dir,
        show_std=True
    )
    
    # Generate and save survival boxplot
    boxplot_filename = f"{output_base}_survival_boxplot.png"
    boxplot_title = f"Episode Duration: {model_name} vs Random Actions ({args.runs} runs)"
    print(f"Generating episode duration boxplot: {boxplot_filename}")
    
    plot_survival_boxplot(
        results,
        title=boxplot_title,
        filename=boxplot_filename,
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
    stats = get_summary_stats(results)
    
    print("\nBenchmark Summary:")
    print("  Trained Model:")
    print(f"    Mean Episode Length: {stats['trained']['mean_episode_length']:.2f} steps (±{stats['trained']['std_episode_length']:.2f})")
    print(f"    Mean Final Reward: {stats['trained']['mean_final_reward']:.2f} (±{stats['trained']['std_final_reward']:.2f})")
    
    if not args.no_random and "random" in stats:
        print("  Random Actions:")
        print(f"    Mean Episode Length: {stats['random']['mean_episode_length']:.2f} steps (±{stats['random']['std_episode_length']:.2f})")
        print(f"    Mean Final Reward: {stats['random']['mean_final_reward']:.2f} (±{stats['random']['std_final_reward']:.2f})")
    
    print(f"\nResults saved to {args.output_dir}/")
    print(f"Reward Plot: {os.path.join(args.output_dir, reward_plot_filename)}")
    print(f"Survival Boxplot: {os.path.join(args.output_dir, boxplot_filename)}")
    print(f"Data: {os.path.join(args.output_dir, data_filename)}")

if __name__ == "__main__":
    main() 