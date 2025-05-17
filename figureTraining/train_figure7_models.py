#!/usr/bin/env python
import os
import sys
import argparse
import datetime
import copy # For deep copying configurations

# Add the parent directory to the path to access project modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Imports from the main project
from config import env_config as global_env_config_template
from config import ppo_config as global_ppo_config_template
from config import save_config as global_save_config_template

# Attempt to import the refactored training function
try:
    from train import execute_single_training_run
except ImportError as e:
    print(f"Error importing execute_single_training_run from train.py: {e}")
    print("Please ensure train.py is in the project root and has been refactored.")
    sys.exit(1)

# Seeds to run for each adherence effectiveness configuration
SEEDS_FOR_TRAINING = [1, 2, 3]
# Adherence effectiveness values to test for Figure 7
ADHERENCE_EFFECTIVENESS_VALUES = [0.1, 0.3, 0.5]
REWARD_TYPE_FOR_FIG7 = "potential_field" # Assuming Potential Field reward for these tests

def main_fig7_trainer(args):
    """Main function to orchestrate training for Figure 7 models (varying adherence effectiveness)."""
    
    initial_env_config = copy.deepcopy(global_env_config_template)
    initial_ppo_config = copy.deepcopy(global_ppo_config_template)
    initial_save_config = copy.deepcopy(global_save_config_template)

    wandb_project_for_fig7 = os.getenv("WANDB_PROJECT_FIG7", "sirs-rl-fig7-adherence-effectiveness")

    for adherence_effectiveness in ADHERENCE_EFFECTIVENESS_VALUES:
        print(f"\n=== Training models for Figure 7 (Adherence Effectiveness: {adherence_effectiveness}, Reward: {REWARD_TYPE_FOR_FIG7}) ===")
        
        # Modify a copy of the environment config for the current adherence effectiveness value
        env_config_for_this_run = copy.deepcopy(initial_env_config)
        env_config_for_this_run['adherence_effectiveness'] = adherence_effectiveness
        env_config_for_this_run['reward_type'] = REWARD_TYPE_FOR_FIG7
        env_config_for_this_run['reward_ablation'] = "full" # Ensure full Potential Field

        # Construct base_run_name_for_group for W&B grouping and file naming
        base_run_name_for_group = f"Fig7_adherenceEff{adherence_effectiveness}"
        if args.exp_suffix:
            base_run_name_for_group = f"{base_run_name_for_group}_{args.exp_suffix}"

        for seed_val in SEEDS_FOR_TRAINING:
            seed_specific_run_name = f"{base_run_name_for_group}_seed{seed_val}"
            print(f"--- Preparing training for: {seed_specific_run_name} (Adherence Effectiveness: {adherence_effectiveness}, Seed: {seed_val}) ---")
            
            os.makedirs(initial_save_config["base_log_path"], exist_ok=True)

            execute_single_training_run(
                current_seed=seed_val,
                run_name=seed_specific_run_name, 
                log_path_base=initial_save_config["base_log_path"], 
                effective_env_config=env_config_for_this_run,
                effective_ppo_config=copy.deepcopy(initial_ppo_config),
                effective_save_config=copy.deepcopy(initial_save_config),
                should_record_video_flag=args.record_video,
                use_wandb_flag=not args.no_wandb,
                wandb_offline_flag=args.wandb_offline,
                wandb_project_name=wandb_project_for_fig7,
                wandb_group_name=base_run_name_for_group
            )
            print(f"--- Completed training for: {seed_specific_run_name} ---")
        print(f"=== Finished all seeds for Adherence Effectiveness: {adherence_effectiveness} ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models with varying adherence effectiveness for Figure 7.")
    parser.add_argument("--exp-suffix", type=str, default="", help="Optional suffix for all experiment group names.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb-offline", action="store_true", help="Run W&B in offline mode.")
    parser.add_argument("--record-video", action="store_true", help="Enable video recording of evaluation episodes.")
    
    cli_args = parser.parse_args()

    if not cli_args.no_wandb and cli_args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
        print("Using W&B in offline mode.")

    main_fig7_trainer(cli_args) 