#!/usr/bin/env python
import os
import sys
import argparse
import datetime
from scipy import stats as scipy_stats
import numpy as np
import json

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from result_utils import (
    run_benchmark,
    plot_survival_boxplot,
    save_benchmark_results,
    get_summary_stats,
    plot_final_reward_boxplot,
    AGENT_LABELS
)

def main():
    """
    Run benchmarks comparing a trained model against random actions.
    Generates boxplots for episode duration and final rewards.
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
        help="Skip running the random agent benchmarks"
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
    boxplot_title = f"Episode Duration: {title_prefix} vs Baseline Agents ({args.runs} runs)"
    print(f"Generating episode duration boxplot: {boxplot_filename}")
    
    plot_survival_boxplot(
        results,
        title=boxplot_title,
        filename=boxplot_filename,
        save_dir=args.output_dir
    )
    
    # Generate and save final reward boxplot
    final_reward_boxplot_filename = f"{output_base}_final_reward_boxplot.png"
    final_reward_boxplot_title = f"Final Cumulative Reward: {title_prefix} vs Baseline Agents ({args.runs} runs)"
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
    print(f"  {AGENT_LABELS['trained']}:")
    print(f"    Mean Episode Duration: {summary_stats['trained']['mean_episode_length']:.2f} steps (±{summary_stats['trained']['std_episode_length']:.2f})")
    print(f"    Mean Final Reward: {summary_stats['trained']['mean_final_reward']:.2f} (±{summary_stats['trained']['std_final_reward']:.2f})")
    
    if not args.no_random:
        if "random_reckless" in summary_stats:
            print(f"  {AGENT_LABELS['random_reckless']}:")
            print(f"    Mean Episode Duration: {summary_stats['random_reckless']['mean_episode_length']:.2f} steps (±{summary_stats['random_reckless']['std_episode_length']:.2f})")
            print(f"    Mean Final Reward: {summary_stats['random_reckless']['mean_final_reward']:.2f} (±{summary_stats['random_reckless']['std_final_reward']:.2f})")
        
        if "random_cautious" in summary_stats:
            print(f"  {AGENT_LABELS['random_cautious']}:")
            print(f"    Mean Episode Duration: {summary_stats['random_cautious']['mean_episode_length']:.2f} steps (±{summary_stats['random_cautious']['std_episode_length']:.2f})")
            print(f"    Mean Final Reward: {summary_stats['random_cautious']['mean_final_reward']:.2f} (±{summary_stats['random_cautious']['std_final_reward']:.2f})")
        
    if "stationary" in summary_stats:
        print(f"  {AGENT_LABELS['stationary']}:")
        print(f"    Mean Episode Duration: {summary_stats['stationary']['mean_episode_length']:.2f} steps (±{summary_stats['stationary']['std_episode_length']:.2f})")
        print(f"    Mean Final Reward: {summary_stats['stationary']['mean_final_reward']:.2f} (±{summary_stats['stationary']['std_final_reward']:.2f})")
        
    if "static_cautious" in summary_stats:
        print(f"  {AGENT_LABELS['static_cautious']}:")
        print(f"    Mean Episode Duration: {summary_stats['static_cautious']['mean_episode_length']:.2f} steps (±{summary_stats['static_cautious']['std_episode_length']:.2f})")
        print(f"    Mean Final Reward: {summary_stats['static_cautious']['mean_final_reward']:.2f} (±{summary_stats['static_cautious']['std_final_reward']:.2f})")
        
    if "greedy" in summary_stats:
        print(f"  {AGENT_LABELS['greedy']}:")
        print(f"    Mean Episode Duration: {summary_stats['greedy']['mean_episode_length']:.2f} steps (±{summary_stats['greedy']['std_episode_length']:.2f})")
        print(f"    Mean Final Reward: {summary_stats['greedy']['mean_final_reward']:.2f} (±{summary_stats['greedy']['std_final_reward']:.2f})")
        
    # Load the detailed results from the JSON for statistical tests
    full_results_path = os.path.join(args.output_dir, data_filename)
    statistical_tests = None
    if os.path.exists(full_results_path):
        try:
            with open(full_results_path, 'r') as f:
                full_results = json.load(f)
            statistical_tests = full_results.get("statistical_tests", None)
        except Exception as e:
            print(f"\nWarning: Could not load or parse {data_filename} for detailed stats: {e}")

    if statistical_tests:
        print("\n  Statistical Comparisons (Mann-Whitney U with Bonferroni Correction):")
        print("  -------------------------------------------------------------------")
        
        # Only show comparisons between trained and other agents
        agent_pairs = [
            (AGENT_LABELS["trained"], AGENT_LABELS["stationary"]),
            (AGENT_LABELS["trained"], AGENT_LABELS["static_cautious"]),
            (AGENT_LABELS["trained"], AGENT_LABELS["random_reckless"]),
            (AGENT_LABELS["trained"], AGENT_LABELS["random_cautious"]),
            (AGENT_LABELS["trained"], AGENT_LABELS["greedy"]),
        ]
        metrics = [
            ("episode_length", "Episode Duration"), 
            ("final_reward", "Final Reward")
        ]
        
        for metric_key, metric_name in metrics:
            print(f"\n  {metric_name}:")
            if metric_key in statistical_tests:
                metric_tests = statistical_tests[metric_key]
                displayed_comparison = False
                for agent1, agent2 in agent_pairs:
                    comparison_key = f"{agent1}_vs_{agent2}"
                    alt_comparison_key = f"{agent2}_vs_{agent1}"
                    
                    comp_data = None
                    if comparison_key in metric_tests:
                        comp_data = metric_tests[comparison_key]
                    elif alt_comparison_key in metric_tests:
                         comp_data = metric_tests[alt_comparison_key]
                         
                    if comp_data:
                        p_corrected = comp_data.get("p_value_bonferroni")
                        u_stat = comp_data.get("u_statistic")
                        
                        if p_corrected is not None:
                            displayed_comparison = True
                            significance = " (not significant)"
                            if p_corrected < 0.001: significance = " (*** p < 0.001)"
                            elif p_corrected < 0.01: significance = " (** p < 0.01)"
                            elif p_corrected < 0.05: significance = " (* p < 0.05)"
                            
                            print(f"    {comparison_key.replace('_', ' ')}: p = {p_corrected:.4f}{significance}")
                        else:
                            pass
                if not displayed_comparison:
                    print(f"    No valid comparisons found for {metric_name}.")
            else:
                print(f"    No statistical tests found for {metric_name}.")
                
    # --- Print Directional Test Results (Trained vs Baselines) --- 
    if statistical_tests and "trained_vs_baselines" in statistical_tests:
        print("\n  Directional Comparisons (Trained > Baseline):")
        print("  ---------------------------------------------")
        directional_tests_results = statistical_tests["trained_vs_baselines"]
        
        for metric_key, metric_name in metrics:
            print(f"\n  {metric_name}:")
            if metric_key in directional_tests_results:
                metric_directional = directional_tests_results[metric_key]
                displayed_directional = False
                for baseline_agent in [AGENT_LABELS["stationary"], AGENT_LABELS["static_cautious"], 
                                     AGENT_LABELS["random_reckless"], AGENT_LABELS["random_cautious"], 
                                     AGENT_LABELS["greedy"]]:
                    comparison_key = f"{AGENT_LABELS['trained']}_vs_{baseline_agent}"
                    if comparison_key in metric_directional:
                        comp_data = metric_directional[comparison_key]
                        p_corrected = comp_data.get("p_value_bonferroni")
                        
                        if p_corrected is not None:
                            displayed_directional = True
                            significance = " (not significantly better)"
                            if comp_data.get("significant_0.05_bonferroni"): # Check the boolean flag
                                 # Determine significance level based on p-value for display
                                if p_corrected < 0.001: significance = " (*** significantly better, p < 0.001)"
                                elif p_corrected < 0.01: significance = " (** significantly better, p < 0.01)"
                                elif p_corrected < 0.05: significance = " (* significantly better, p < 0.05)"
                                else: significance = " (significantly better, p < 0.05)" # Fallback if flag is true but p isn't <0.05 (unlikely with Bonferroni)
                            
                            print(f"    {comparison_key.replace('_', ' ')}: p = {p_corrected:.4f}{significance}")
                if not displayed_directional:
                     print(f"    No valid directional comparisons found for {metric_name}.")
            else:
                 print(f"    No directional tests found for {metric_name}.")
    
    print(f"\nResults saved to {args.output_dir}/")
    print(f"Episode Duration Boxplot: {os.path.join(args.output_dir, boxplot_filename)}")
    print(f"Final Reward Boxplot: {os.path.join(args.output_dir, final_reward_boxplot_filename)}")
    print(f"Data: {os.path.join(args.output_dir, data_filename)}")

if __name__ == "__main__":
    main() 