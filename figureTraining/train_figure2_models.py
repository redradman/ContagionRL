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
    from train import (
        execute_single_training_run,
    )
except ImportError as e:
    print(f"Error importing from train.py: {e}")
    print("Please ensure train.py is in the project root and has been refactored with execute_single_training_run.")
    sys.exit(1)

REWARD_CONFIGURATIONS = [
    {
        "label": "Constant", 
        "reward_type": "constant", 
        "base_exp_name": "Fig2_ConstantReward"
    },
    {
        "label": "ReduceInfection", 
        "reward_type": "reduceInfectionProb", 
        "base_exp_name": "Fig2_ReduceInfReward"
    },
    {
        "label": "ConstantPlusReduceInfection", 
        "reward_type": "reduceInfectionProbwithConstant", 
        "base_exp_name": "Fig2_ComboReward"
    },
    {
        "label": "MaxNearestDistance", 
        "reward_type": "max_nearest_distance", 
        "base_exp_name": "Fig2_MaxNearestDistReward"
    },
    {
        "label": "PotentialField", 
        "reward_type": "potential_field", 
        "base_exp_name": "Fig2_PotentialFieldReward"
    } 
    # Add more configurations if needed
]

SEEDS_FOR_TRAINING = [1, 2, 3]

def main_fig2_trainer(args):
    """Main function to orchestrate training for different reward configurations."""
    
    initial_env_config = copy.deepcopy(global_env_config_template)
    initial_ppo_config = copy.deepcopy(global_ppo_config_template)
    initial_save_config = copy.deepcopy(global_save_config_template)

    wandb_project_for_fig2 = os.getenv("WANDB_PROJECT_FIG2", "sirs-rl-fig2-rewards")

    for config_details in REWARD_CONFIGURATIONS:
        current_reward_label = config_details["label"]
        current_reward_type = config_details["reward_type"]
        base_exp_name_prefix = config_details["base_exp_name"]

        print(f"\n=== Training models for reward type: {current_reward_label} ({current_reward_type}) ===")
        
        env_config_for_this_run = copy.deepcopy(initial_env_config)
        env_config_for_this_run['reward_type'] = current_reward_type
        
        base_run_name_for_group = f"{base_exp_name_prefix}"
        if args.exp_suffix:
            base_run_name_for_group = f"{base_run_name_for_group}_{args.exp_suffix}"

        for seed_val in SEEDS_FOR_TRAINING:
            seed_specific_run_name = f"{base_run_name_for_group}_seed{seed_val}"
            print(f"# Preparing training for: {seed_specific_run_name} (Reward: {current_reward_label}, Seed: {seed_val}) #")
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
                wandb_project_name=wandb_project_for_fig2,
                wandb_group_name=base_run_name_for_group
            )
            print(f"# Completed training for: {seed_specific_run_name} #")
        print(f"=== Finished all seeds for reward type: {current_reward_label} ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models with different reward functions for Figure 2.")
    parser.add_argument("--exp-suffix", type=str, default="", help="Optional suffix for all experiment group names.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb-offline", action="store_true", help="Run W&B in offline mode.")
    parser.add_argument("--record-video", action="store_true", help="Enable video recording of evaluation episodes during these trainings.")
    
    cli_args = parser.parse_args()

    if not cli_args.no_wandb and cli_args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
        print("Using W&B in offline mode.")

    main_fig2_trainer(cli_args) 