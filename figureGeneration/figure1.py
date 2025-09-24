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

# Add the parent directory to the path to access project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import SIRSDEnvironment
from environment import Human, STATE_DICT 

# tueplots styling
from tueplots import bundles
plt.rcParams.update(bundles.neurips2024())
plt.rcParams["figure.dpi"] = 300
sns.set_style("whitegrid") 


AGENT_LABELS = {
    "trained": "Trained",
    "random": "Random",
    "stationary": "Stationary",
    "greedy": "Greedy"
}

AGENT_PLOT_ORDER = ["Stationary", "Random", "Trained", "Greedy"]


def load_model_config(model_path: str) -> Dict[str, Any]:
    """Load the configuration file associated with a specific model."""
    config_path = os.path.join(os.path.dirname(model_path), "config.json")
    if not os.path.exists(config_path):
        config_path_alt = os.path.join(model_path, "config.json")
        if os.path.exists(config_path_alt):
            config_path = config_path_alt
        else:
            raise FileNotFoundError(f"Config file not found at {config_path} or {config_path_alt}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def create_env_from_config(env_config_dict: Dict[str, Any], seed: Optional[int] = None) -> SIRSDEnvironment:
    """Create a SIRS environment from a configuration dictionary."""
    config_copy = env_config_dict.copy()
    config_copy["render_mode"] = None
    env = SIRSDEnvironment(**config_copy)
    env.reset(seed=seed)
    return env

def run_single_episode(
    env: SIRSDEnvironment, 
    agent_type: str, 
    model: Optional[PPO] = None, 
    eval_seed: Optional[int] = None
) -> Tuple[float, int]:
    """
    Run a single episode for a given agent type.

    Args:
        env: The SIRS environment instance.
        agent_type: Type of agent ('trained', 'random', 'stationary', 'greedy').
        model: The trained PPO model (only used if agent_type is 'trained').
        eval_seed: Seed for resetting the environment for this specific episode.

    Returns:
        Tuple of (final_cumulative_reward, episode_length).
    """
    obs, _ = env.reset(seed=eval_seed)
    done = False
    cumulative_reward = 0.0
    episode_length = 0

    stationary_action_arr = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    while not done:
        if agent_type == "trained":
            if model is None:
                raise ValueError("Model must be provided for 'trained' agent type.")
            action, _ = model.predict(obs, deterministic=True)
        elif agent_type == "random":
            action = env.action_space.sample()
        elif agent_type == "stationary":
            action = stationary_action_arr
        elif agent_type == "greedy":
            agent_pos = env.agent_position
            infected_humans = [h for h in env.humans if h.state == STATE_DICT['I']]
            adherence = 1.0

            if not infected_humans:
                dx, dy = 0.0, 0.0
            else:
                min_dist_to_infected = float('inf')
                nearest_infected_human_pos = None
                for inf_h in infected_humans:
                    dist = env._calculate_distance(Human(agent_pos[0], agent_pos[1], 0, -1), inf_h)
                    if dist < min_dist_to_infected:
                        min_dist_to_infected = dist
                        nearest_infected_human_pos = (inf_h.x, inf_h.y)
                
                if nearest_infected_human_pos:
                    vec_x = agent_pos[0] - nearest_infected_human_pos[0]
                    vec_y = agent_pos[1] - nearest_infected_human_pos[1]
                    
                    norm = np.sqrt(vec_x**2 + vec_y**2)
                    if norm > 1e-6: # Avoid division by zero
                        dx = vec_x / norm
                        dy = vec_y / norm
                    else:
                        dx, dy = env.action_space.sample()[:2] 
                else:
                    dx, dy = 0.0, 0.0
            
            action = np.array([dx, dy, adherence], dtype=np.float32)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
        episode_length += 1
        done = terminated or truncated
    
    return cumulative_reward, episode_length

def collect_episode_data(
    env: SIRSDEnvironment,
    agent_type: str,
    num_runs: int,
    model: Optional[PPO] = None,
    base_eval_seed: int = 42 
) -> List[Dict[str, Any]]:
    """Collects data for multiple episodes for a given agent."""
    run_data = []
    for i in range(num_runs):
        eval_seed_for_run = base_eval_seed + i
        final_reward, ep_length = run_single_episode(env, agent_type, model, eval_seed=eval_seed_for_run)
        run_data.append({
            "agent_label": AGENT_LABELS.get(agent_type, agent_type.capitalize()),
            "agent_type": agent_type,
            "seed_group": base_eval_seed,
            "run_id": i,
            "final_reward": final_reward,
            "episode_length": ep_length
        })
    return run_data


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 1 comparing trained models against baselines.")
    parser.add_argument(
        "--model-base",
        type=str,
        required=True,
        help="Base name of the trained model directories in logs/ (e.g., 'potential_field_20231027_1200')."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=30,
        help="Number of evaluation episodes per agent/seed combination (default: 30)."
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="1,2,3",
        help="Comma-separated list of seeds for trained models and for running baselines (e.g., '1,2,3')."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures/",
        help="Directory to save the generated figures (default: figures/)."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="episode_length",
        choices=["episode_length", "final_reward"],
        help="Metric to plot ('episode_length' or 'final_reward', default: 'episode_length')."
    )
    parser.add_argument(
        "--eval-seed-base",
        type=int,
        default=100000, # Figure 1 * 100000 - ensures no seed overlap
        help="Base seed for evaluation runs to ensure variety (default: 1000)."
    )
    args = parser.parse_args()


    eval_seeds = [int(s.strip()) for s in args.seeds.split(',')]

    os.makedirs(args.output_dir, exist_ok=True)

    all_results_data = []
    
    print("Evaluating trained models...")
    base_env_config_dict = None
    
    for train_seed in tqdm(eval_seeds, desc="Trained Model Seeds"):
        model_dir_name = f"{args.model_base}_seed{train_seed}"
        model_path = os.path.join("logs", model_dir_name, "best_model.zip")

        if not os.path.exists(model_path):
            print(f"Warning: Model file not found for seed {train_seed} at {model_path}. Trying alternative common name...")
            alt_model_dir_name = f"{args.model_base.capitalize()}_seed{train_seed}"
            alt_model_path = os.path.join("logs", alt_model_dir_name, "best_model.zip")
            if os.path.exists(alt_model_path):
                print(f"Found model at alternative path: {alt_model_path}")
                model_path = alt_model_path
            else:
                print(f"Still not found at {alt_model_path}. Skipping seed {train_seed}.")
                continue
        
        try:
            model_specific_config = load_model_config(model_path) 
            if base_env_config_dict is None:
                 base_env_config_dict = model_specific_config.get("environment")
                 if base_env_config_dict is None:
                     print(f"Error: 'environment' key not found in config for model {model_path}")
                     return 


            env_for_trained = create_env_from_config(base_env_config_dict, seed=train_seed) 
            trained_model = PPO.load(model_path, env=env_for_trained)


            current_model_eval_seed_base = args.eval_seed_base + train_seed * args.runs 

            episode_data_trained = collect_episode_data(
                env_for_trained, "trained", args.runs, model=trained_model, 
                base_eval_seed=current_model_eval_seed_base
            )
            for record in episode_data_trained:
                record['trained_model_seed'] = train_seed 
            all_results_data.extend(episode_data_trained)
            env_for_trained.close()
        except Exception as e:
            print(f"Error processing trained model for seed {train_seed}: {e}")

    if base_env_config_dict is None:
        print("Error: Could not load environment configuration from any trained model. Cannot proceed with baselines.")
        return

    baseline_agents = ["random", "stationary", "greedy"]
    for baseline_agent_type in baseline_agents:
        print(f"Evaluating {AGENT_LABELS[baseline_agent_type]} agent...")
        for current_eval_seed_for_baseline_group in tqdm(eval_seeds, desc=f"{AGENT_LABELS[baseline_agent_type]} Eval Seeds", leave=False):
            env_for_baseline = create_env_from_config(base_env_config_dict, seed=current_eval_seed_for_baseline_group)
            
            baseline_run_sequence_seed_base = args.eval_seed_base + eval_seeds.index(current_eval_seed_for_baseline_group) * args.runs

            episode_data_baseline = collect_episode_data(
                env_for_baseline, baseline_agent_type, args.runs,
                base_eval_seed=baseline_run_sequence_seed_base
            )
            for record in episode_data_baseline:
                record['baseline_env_seed_group'] = current_eval_seed_for_baseline_group
            all_results_data.extend(episode_data_baseline)
            env_for_baseline.close()
            
    if not all_results_data:
        print("No data collected. Exiting.")
        return

    results_df = pd.DataFrame(all_results_data)

    plt.figure(figsize=(7, 5))

    y_metric_col = args.metric
    y_label = "Episode Duration (steps)" if args.metric == "episode_length" else "Final Cumulative Reward"
    
    plot_order_filtered = [label for label in AGENT_PLOT_ORDER if label in results_df['agent_label'].unique()]

    ax = sns.boxplot(
        x="agent_label",
        y=y_metric_col,
        data=results_df,
        order=plot_order_filtered,
        width=0.6,
        showfliers=False,
        saturation=1,
        boxprops=dict(facecolor='none', edgecolor='black'),
        medianprops=dict(color='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black')
    )

    sns.violinplot( 
        x="agent_label",
        y=y_metric_col,
        data=results_df,
        order=plot_order_filtered,
        width=0.8, 
        inner=None,
        ax=ax 
    )
    

    sns.stripplot(x="agent_label", y=y_metric_col, data=results_df, order=plot_order_filtered,
                  color='black', alpha=0.3, jitter=0.2, size=3, ax=ax)



    trained_agent_data = results_df[results_df['agent_label'] == AGENT_LABELS['trained']]
    if 'trained_model_seed' in trained_agent_data.columns and not trained_agent_data['trained_model_seed'].isnull().all():
        trained_data_for_test = trained_agent_data.groupby('trained_model_seed')[y_metric_col].mean()
    else:
        print("Warning: 'trained_model_seed' column is missing or all null for Trained agent. Using raw data for statistical tests.")
        trained_data_for_test = trained_agent_data[y_metric_col]

    comparisons_data = []
    p_values_uncorrected = []

    for baseline_label in plot_order_filtered:
        if baseline_label == AGENT_LABELS['trained']:
            continue
        
        baseline_agent_data = results_df[results_df['agent_label'] == baseline_label]
        
        if 'baseline_env_seed_group' in baseline_agent_data.columns and not baseline_agent_data['baseline_env_seed_group'].isnull().all():
            baseline_data_for_test = baseline_agent_data.groupby('baseline_env_seed_group')[y_metric_col].mean()
        else:
            print(f"Warning: 'baseline_env_seed_group' column is missing or all null for {baseline_label}. Using raw data for statistical tests.")
            baseline_data_for_test = baseline_agent_data[y_metric_col]

        if len(trained_data_for_test) > 0 and len(baseline_data_for_test) > 0:
            try:
                if len(set(trained_data_for_test)) == 1 and len(set(baseline_data_for_test)) == 1 and trained_data_for_test.iloc[0] == baseline_data_for_test.iloc[0]:
                    p_val = 1.0
                    print(f"Skipping Mann-Whitney U for Trained (aggregated) vs {baseline_label} (aggregated): Both groups have identical constant values.")
                else:
                    if len(trained_data_for_test) < 2 or len(baseline_data_for_test) < 2:
                         print(f"Warning: Not enough data points for Mann-Whitney U after aggregation for Trained vs {baseline_label} (Trained: {len(trained_data_for_test)}, Baseline: {len(baseline_data_for_test)}). Assigning p_val=NaN.")
                         p_val = np.nan
                    else:
                         _, p_val = stats.mannwhitneyu(trained_data_for_test, baseline_data_for_test, alternative='two-sided')

                comparisons_data.append((AGENT_LABELS['trained'], baseline_label))
                p_values_uncorrected.append(p_val)
            except ValueError as e:
                print(f"Warning: Mann-Whitney U test failed for Trained (aggregated) vs {baseline_label} (aggregated): {e}")

    if p_values_uncorrected:
        valid_indices = [i for i, p in enumerate(p_values_uncorrected) if not np.isnan(p)]
        valid_p_values = [p_values_uncorrected[i] for i in valid_indices]
        valid_comparisons = [comparisons_data[i] for i in valid_indices]

        if valid_p_values:
            _, pvals_corrected, _, _ = smm.multipletests(valid_p_values, alpha=0.05, method='bonferroni')
            
            y_max = results_df[y_metric_col].max()
            y_min = results_df[y_metric_col].min()
            y_range = y_max - y_min
            
            num_valid_comparisons = len(valid_comparisons)
            if num_valid_comparisons == 0: 
                y_range = abs(y_max) if y_max != 0 else 1.0 

            if y_range <= 1e-9 :
                 increment_base = 0.1 * abs(y_max) if abs(y_max) > 1e-9 else 0.1
            else:
                 increment_base = y_range * 0.08
            
            increment_total_height_factor = 0.1 * num_valid_comparisons
            if y_range > 1e-9:
                 increment = max(increment_base, y_range * increment_total_height_factor / num_valid_comparisons if num_valid_comparisons > 0 else increment_base)
            else:
                 increment = increment_base

            current_y = y_max + increment * 0.5
            
            cat_pos = {cat: i for i, cat in enumerate(plot_order_filtered)}

            for i, (cat1, cat2) in enumerate(valid_comparisons):
                p_corrected = pvals_corrected[i]
                pos1 = cat_pos[cat1]
                pos2 = cat_pos[cat2]

                line_x = [pos1, pos1, pos2, pos2]
                line_y = [current_y, current_y + increment * 0.2, current_y + increment * 0.2, current_y]
                ax.plot(line_x, line_y, lw=1.0, c='black') 

                significance = 'ns'
                if p_corrected < 0.001: 
                        significance = '***'
                elif p_corrected < 0.01: 
                        significance = '**'
                elif p_corrected < 0.05: 
                        significance = '*'
                
                text_x = (pos1 + pos2) / 2
                text_y = current_y + increment * 0.25
                ax.text(text_x, text_y, significance, ha='center', va='bottom', fontsize=8) 
                current_y += increment
            
            if num_valid_comparisons > 0 : 
                 ax.set_ylim(top=current_y + increment*0.2)


    plt.xlabel("Agent Type", fontsize=9)
    plt.ylabel(y_label, fontsize=9)
    ax.tick_params(axis='x', labelsize=8, rotation=0) 
    ax.tick_params(axis='y', labelsize=8)
    

    plt.tight_layout(pad=0.5)

    # Create a unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    figure_filename = f"figure1_{args.model_base}_{args.metric}_{timestamp}.pdf"
    figure_path = os.path.join(args.output_dir, figure_filename)
    
    plt.savefig(figure_path, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {figure_path}")

    # Save the aggregated DataFrame for inspection
    csv_filename = f"figure1_data_{args.model_base}_{args.metric}_{timestamp}.csv"
    csv_path = os.path.join(args.output_dir, csv_filename)
    results_df.to_csv(csv_path, index=False)
    print(f"Aggregated data saved to {csv_path}")


if __name__ == "__main__":
    main() 