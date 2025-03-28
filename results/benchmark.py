#!/usr/bin/env python
import os
import sys
import argparse
import datetime
from scipy import stats as scipy_stats
import numpy as np

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
        
        print("\n  Statistical Tests:")
        print("  -----------------")
        
        # Calculate and print statistical test results
        trained_lengths = results["trained"]["episode_lengths"]
        random_lengths = results["random"]["episode_lengths"]
        if len(trained_lengths) > 0 and len(random_lengths) > 0:
            # Mann-Whitney U test for episode duration
            u_stat, p_value = scipy_stats.mannwhitneyu(trained_lengths, random_lengths, alternative='two-sided')
            print("  Episode Duration:")
            print(f"    Mann-Whitney U: p = {p_value:.4f}", end="")
            if p_value < 0.001:
                print(" (***)")
            elif p_value < 0.01:
                print(" (**)")
            elif p_value < 0.05:
                print(" (*)")
            else:
                print(" (not significant)")
            print(f"      Interpretation: {'Distributions differ significantly' if p_value < 0.05 else 'No significant difference in distributions'}")
            
            # Shapiro-Wilk test for normality
            if len(trained_lengths) >= 3:
                sw_stat_trained, sw_p_trained = scipy_stats.shapiro(trained_lengths)
                print(f"    Shapiro-Wilk (Trained): p = {sw_p_trained:.4f}")
                print(f"      Interpretation: {'Non-normal distribution' if sw_p_trained < 0.05 else 'Normal distribution'}")
            
            if len(random_lengths) >= 3:
                sw_stat_random, sw_p_random = scipy_stats.shapiro(random_lengths)
                print(f"    Shapiro-Wilk (Random): p = {sw_p_random:.4f}")
                print(f"      Interpretation: {'Non-normal distribution' if sw_p_random < 0.05 else 'Normal distribution'}")
            
            # Levene's test for equality of variances
            try:
                levene_stat, levene_p = scipy_stats.levene(trained_lengths, random_lengths)
                print(f"    Levene's Test: p = {levene_p:.4f}")
                print(f"      Interpretation: {'Variances differ significantly' if levene_p < 0.05 else 'Equal variances'}")
            except Exception as e:
                print(f"    Levene's Test: Could not perform test - {e}")
            
            # Permutation test
            try:
                def diff_of_means(x, y):
                    return np.mean(x) - np.mean(y)
                
                observed_diff = diff_of_means(trained_lengths, random_lengths)
                # Combine for permutation
                combined = np.concatenate([trained_lengths, random_lengths])
                n_perm = 10000  # Number of permutations
                n1 = len(trained_lengths)
                
                # Run permutation test
                perm_diffs = []
                for _ in range(n_perm):
                    np.random.shuffle(combined)
                    perm_diffs.append(diff_of_means(combined[:n1], combined[n1:]))
                
                # Calculate permutation p-value
                perm_p = np.sum(np.abs(perm_diffs) >= np.abs(observed_diff)) / n_perm
                
                print(f"    Permutation Test: p = {perm_p:.4f}")
                print(f"      Mean Difference: {observed_diff:.2f}")
                print(f"      Interpretation: {'Difference is unlikely due to chance' if perm_p < 0.05 else 'Difference could be due to chance'}")
            except Exception as e:
                print(f"    Permutation Test: Could not perform test - {e}")
        
        # Final reward statistical tests
        trained_final_rewards = [rewards[-1] for rewards in results["trained"]["rewards_over_time"]]
        random_final_rewards = [rewards[-1] for rewards in results["random"]["rewards_over_time"]]
        if len(trained_final_rewards) > 0 and len(random_final_rewards) > 0:
            # Mann-Whitney U test for final reward
            u_stat, p_value = scipy_stats.mannwhitneyu(trained_final_rewards, random_final_rewards, alternative='two-sided')
            print("\n  Final Reward:")
            print(f"    Mann-Whitney U: p = {p_value:.4f}", end="")
            if p_value < 0.001:
                print(" (***)")
            elif p_value < 0.01:
                print(" (**)")
            elif p_value < 0.05:
                print(" (*)")
            else:
                print(" (not significant)")
            print(f"      Interpretation: {'Distributions differ significantly' if p_value < 0.05 else 'No significant difference in distributions'}")
            
            # Shapiro-Wilk test for normality
            if len(trained_final_rewards) >= 3:
                sw_stat_trained, sw_p_trained = scipy_stats.shapiro(trained_final_rewards)
                print(f"    Shapiro-Wilk (Trained): p = {sw_p_trained:.4f}")
                print(f"      Interpretation: {'Non-normal distribution' if sw_p_trained < 0.05 else 'Normal distribution'}")
            
            if len(random_final_rewards) >= 3:
                sw_stat_random, sw_p_random = scipy_stats.shapiro(random_final_rewards)
                print(f"    Shapiro-Wilk (Random): p = {sw_p_random:.4f}")
                print(f"      Interpretation: {'Non-normal distribution' if sw_p_random < 0.05 else 'Normal distribution'}")
            
            # Levene's test for equality of variances
            try:
                levene_stat, levene_p = scipy_stats.levene(trained_final_rewards, random_final_rewards)
                print(f"    Levene's Test: p = {levene_p:.4f}")
                print(f"      Interpretation: {'Variances differ significantly' if levene_p < 0.05 else 'Equal variances'}")
            except Exception as e:
                print(f"    Levene's Test: Could not perform test - {e}")
            
            # Permutation test
            try:
                def diff_of_means(x, y):
                    return np.mean(x) - np.mean(y)
                
                observed_diff = diff_of_means(trained_final_rewards, random_final_rewards)
                # Combine for permutation
                combined = np.concatenate([trained_final_rewards, random_final_rewards])
                n_perm = 10000  # Number of permutations
                n1 = len(trained_final_rewards)
                
                # Run permutation test
                perm_diffs = []
                for _ in range(n_perm):
                    np.random.shuffle(combined)
                    perm_diffs.append(diff_of_means(combined[:n1], combined[n1:]))
                
                # Calculate permutation p-value
                perm_p = np.sum(np.abs(perm_diffs) >= np.abs(observed_diff)) / n_perm
                
                print(f"    Permutation Test: p = {perm_p:.4f}")
                print(f"      Mean Difference: {observed_diff:.2f}")
                print(f"      Interpretation: {'Difference is unlikely due to chance' if perm_p < 0.05 else 'Difference could be due to chance'}")
            except Exception as e:
                print(f"    Permutation Test: Could not perform test - {e}")
    
    print(f"\nResults saved to {args.output_dir}/")
    print(f"Episode Duration Boxplot: {os.path.join(args.output_dir, boxplot_filename)}")
    print(f"Final Reward Boxplot: {os.path.join(args.output_dir, final_reward_boxplot_filename)}")
    print(f"Exposure vs Adherence Plot: {os.path.join(args.output_dir, scatterplot_filename)}")
    print(f"Data: {os.path.join(args.output_dir, data_filename)}")

if __name__ == "__main__":
    main() 