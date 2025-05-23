#!/usr/bin/env python
import os
import sys
import argparse
import copy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config import env_config as global_env_config_template
from config import ppo_config as global_ppo_config_template
from config import save_config as global_save_config_template

try:
    from train import execute_single_training_run
except ImportError as e:
    print(f"Error importing execute_single_training_run from train.py: {e}")
    print("Please ensure train.py is in the project root and has been refactored.")
    sys.exit(1)

SEEDS_FOR_TRAINING = [1, 2, 3]
DISTANCE_DECAY_VALUES = [0.15, 0.3, 0.45]
REWARD_TYPE_FOR_FIG8 = "potential_field"

def main_fig8_trainer(args):
    """Main function to orchestrate training for Figure 8 models (varying distance decay)."""
    
    initial_env_config = copy.deepcopy(global_env_config_template)
    initial_ppo_config = copy.deepcopy(global_ppo_config_template)
    initial_save_config = copy.deepcopy(global_save_config_template)

    wandb_project_for_fig8 = os.getenv("WANDB_PROJECT_FIG8", "sirs-rl-fig8-distancedecay")

    for distance_decay in DISTANCE_DECAY_VALUES:
        print(f"\n=== Training models for Figure 8 (Distance Decay: {distance_decay}, Reward: {REWARD_TYPE_FOR_FIG8}) ===")
        
        env_config_for_this_run = copy.deepcopy(initial_env_config)
        env_config_for_this_run['distance_decay'] = distance_decay
        env_config_for_this_run['reward_type'] = REWARD_TYPE_FOR_FIG8
        env_config_for_this_run['reward_ablation'] = "full"

        base_run_name_for_group = f"Fig8_distanceDecay{distance_decay}"
        if args.exp_suffix:
            base_run_name_for_group = f"{base_run_name_for_group}_{args.exp_suffix}"

        for seed_val in SEEDS_FOR_TRAINING:
            seed_specific_run_name = f"{base_run_name_for_group}_seed{seed_val}"
            print(f"--- Preparing training for: {seed_specific_run_name} (Distance Decay: {distance_decay}, Seed: {seed_val}) ---")
            
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
                wandb_project_name=wandb_project_for_fig8,
                wandb_group_name=base_run_name_for_group
            )
            print(f"--- Completed training for: {seed_specific_run_name} ---")
        print(f"=== Finished all seeds for Distance Decay: {distance_decay} ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models with varying distance decay for Figure 8.")
    parser.add_argument("--exp-suffix", type=str, default="", help="Optional suffix for all experiment group names.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb-offline", action="store_true", help="Run W&B in offline mode.")
    parser.add_argument("--record-video", action="store_true", help="Enable video recording of evaluation episodes.")
    
    cli_args = parser.parse_args()

    if not cli_args.no_wandb and cli_args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
        print("Using W&B in offline mode.")

    main_fig8_trainer(cli_args) 