#!/usr/bin/env python
import os
import argparse
import json
import csv
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple
import random
import torch
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback
import wandb
from wandb.integration.sb3 import WandbCallback

from environment import SIRSEnvironment
from config import env_config, ppo_config, save_config

# Define constants
ABLATION_VARIANTS = ["full", "no_magnitude", "no_direction", "no_move", "no_adherence", "no_health", "no_S"]
SEEDS = [0, 1, 2]  # Run each variant with 3 different seeds
TIMESTEPS = 524288  # Smaller than full training for ablation study
LOG_INTERVAL = 2048  # Log metrics every 2048 steps

# Mapping from terse keys to descriptive variant names
VARIANT_NAMES = {
    "full":          "Full",
    "no_magnitude":  "Drop Magnitude",
    "no_direction":  "Drop Direction",
    "no_move":       "Drop Movement",
    "no_adherence":  "Drop Adherence",
    "no_health":     "Drop Health",
    "no_S":          "Drop Susceptible Repulsion",
}

# Function to get descriptive name from terse key
def get_variant_name(variant_key):
    return VARIANT_NAMES.get(variant_key, variant_key)

# Create a callback to log iteration metrics
class IterationMetricsCallback(BaseCallback):
    def __init__(self, verbose=0, csv_file=None, log_interval=LOG_INTERVAL):
        super().__init__(verbose)
        self.csv_file = csv_file
        self.log_interval = log_interval
        self.last_log_time = 0
        self.csv_header_written = False
        self.first_update_recorded = False
        
    def _on_step(self) -> bool:
        return True
        
    def _on_rollout_end(self) -> None:
        """
        This method is called after each rollout (collecting n_steps of experience).
        This is when policy updates happen, so it's the best time to log metrics.
        """
        # Log metrics every rollout to capture policy update statistics
        self._log_metrics()
    
    def _on_training_end(self) -> None:
        """Ensure we log metrics at the end of training."""
        self._log_metrics()
            
    def _log_metrics(self):
        if self.csv_file is None:
            return
            
        # Get variant key and convert to descriptive name
        variant_key = ""
        try:
            if hasattr(self.training_env, 'get_attr'):
                variants = self.training_env.get_attr('reward_ablation')
                if variants and len(variants) > 0:
                    variant_key = variants[0]
        except Exception:
            variant_key = "unknown"
            
        # Convert the variant key to the descriptive name
        variant_name = get_variant_name(variant_key)
        
        # Extract policy metrics
        metrics = self._extract_policy_metrics()
        
        # Skip if we don't have any meaningful metrics yet (all zeros)
        if not self.first_update_recorded:
            # Check if we have any non-zero policy metrics (indicating real updates)
            has_real_metrics = any(abs(metrics.get(key, 0.0)) > 1e-6 for key in 
                              ['policy_loss', 'value_loss', 'approx_kl', 'clip_fraction'])
            
            if has_real_metrics:
                self.first_update_recorded = True
            elif self.n_calls == 1:
                # First call with no metrics - write header but skip row
                if not self.csv_header_written:
                    # Check if file exists - only write header if it doesn't
                    file_exists = os.path.exists(self.csv_file)
                    if not file_exists:
                        with open(self.csv_file, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                "variant", "seed", "iteration", "timesteps", 
                                "policy_loss", "value_loss", "entropy_loss", 
                                "approx_kl", "clip_fraction", "explained_variance", 
                                "std", "learning_rate"
                            ])
                    self.csv_header_written = True
                return
            
        # Only record if we have valid metrics or this is the first meaningful update
        # or this is the last iteration of training
        if metrics or self.first_update_recorded or self.num_timesteps >= self.model._total_timesteps:
            # Create row data in exact column order
            row_data = [
                variant_name,                          # variant (descriptive name)
                self.model.seed,                        # seed
                self.n_calls,                           # iteration
                self.model.num_timesteps,               # timesteps
                metrics.get('policy_loss', 0.0),        # policy_loss
                metrics.get('value_loss', 0.0),         # value_loss
                metrics.get('entropy_loss', 0.0),       # entropy_loss
                metrics.get('approx_kl', 0.0),          # approx_kl
                metrics.get('clip_fraction', 0.0),      # clip_fraction
                metrics.get('explained_variance', 0.0), # explained_variance
                metrics.get('std', 0.0),                # std
                metrics.get('learning_rate', 0.0003)    # learning_rate
            ]
            
            # Format float values with at least 6 decimal places for precision
            for i, value in enumerate(row_data):
                if isinstance(value, float):
                    row_data[i] = float(f"{value:.6f}")
            
            # Write header if needed (file doesn't exist)
            if not self.csv_header_written:
                file_exists = os.path.exists(self.csv_file)
                if not file_exists:
                    with open(self.csv_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            "variant", "seed", "iteration", "timesteps", 
                            "policy_loss", "value_loss", "entropy_loss", 
                            "approx_kl", "clip_fraction", "explained_variance", 
                            "std", "learning_rate"
                        ])
                self.csv_header_written = True
            
            # Append row to CSV
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)
    
    def _extract_policy_metrics(self):
        """Extract policy metrics from the model's logger."""
        metrics = {}
        
        # We're looking specifically for these policy diagnostics
        policy_metrics = [
            'policy_loss', 'value_loss', 'explained_variance', 'std', 
            'entropy_loss', 'approx_kl', 'clip_fraction', 'learning_rate'
        ]
        
        # First, try to get metrics from logger.name_to_value directly
        if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
            for name, value in self.model.logger.name_to_value.items():
                # Look for policy gradient loss specifically
                if 'policy_gradient_loss' in name and 'policy_loss' not in metrics:
                    metrics['policy_loss'] = value
                
                # Map other metrics directly
                for metric in policy_metrics:
                    if metric in name and metric not in metrics:
                        metrics[metric] = value
        
        # Second, try to get metrics from model's training information
        if hasattr(self.model, 'train_infos') and self.model.train_infos:
            train_info = self.model.train_infos[-1]  # Get the most recent training info
            for metric in policy_metrics:
                if metric in train_info and metric not in metrics:
                    metrics[metric] = train_info[metric]
                    
            # Map policy_gradient_loss to policy_loss if found
            if 'policy_gradient_loss' in train_info and 'policy_loss' not in metrics:
                metrics['policy_loss'] = train_info['policy_gradient_loss']
        
        # If we get here and still have no metrics, try using PPO-specific attributes
        if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'get_distribution'):
            try:
                # Add policy standard deviation if available
                if hasattr(self.model.policy, 'log_std') and 'std' not in metrics:
                    metrics['std'] = float(np.exp(self.model.policy.log_std.mean().detach().cpu().numpy()))
                
                # Add learning rate if available from optimizer
                if hasattr(self.model, '_learning_rate') and 'learning_rate' not in metrics:
                    metrics['learning_rate'] = float(self.model._learning_rate)
            except Exception:
                # Ignore errors when trying to extract policy metrics
                pass
        
        return metrics

# Create a callback to log episode metrics including reward components
class EpisodeRewardComponentsCallback(BaseCallback):
    def __init__(self, verbose=0, csv_file=None):
        super().__init__(verbose)
        self.csv_file = csv_file
        self.episode_counter = 0  # Track episode IDs
        self.csv_header_written = False
        
        # Create a dictionary to track metrics for active episodes
        self.active_episodes = {}
        # Buffer for completed episodes waiting to be written
        self.completed_episodes = []
        
    def _init_episode_tracking(self, env_idx):
        """Initialize tracking for a new episode"""
        self.active_episodes[env_idx] = {
            'start_timestep': self.num_timesteps,
            'length': 0,
            'return': 0.0,
            'health_steps': 0,
            'adherence_sum': 0.0,
            'r_dir_sum': 0.0,
            'r_mag_sum': 0.0,
            'r_move_sum': 0.0,
            'r_adherence_sum': 0.0
        }
        
    def _on_step(self) -> bool:
        """
        Track reward components for each active episode and detect completed episodes.
        This method is called after each environment step.
        """
        try:
            # Make sure we have tracking initialized for all environments
            if not hasattr(self.training_env, 'get_attr'):
                return True
                
            # Get dones array directly from locals
            dones = self.locals.get('dones', [])
            
            # Get info array for episode length if available
            infos = self.locals.get('infos', [])
            
            # Get reward components and other metrics from each environment
            components = self.training_env.get_attr('reward_components')
            adherence_values = self.training_env.get_attr('agent_adherence')
            agent_states = self.training_env.get_attr('agent_state')
            
            # Process data from each environment
            for env_idx in range(len(dones)):
                # Initialize tracking for this environment if needed
                if env_idx not in self.active_episodes:
                    self._init_episode_tracking(env_idx)
                
                # Get the metrics dict for this environment
                episode = self.active_episodes[env_idx]
                
                # Update length
                episode['length'] += 1
                
                # Track health steps (agent is in susceptible state)
                # STATE_DICT['S'] is 0 in the environment
                if env_idx < len(agent_states) and agent_states[env_idx] == 0:
                    episode['health_steps'] += 1
                
                # Track adherence
                if env_idx < len(adherence_values):
                    episode['adherence_sum'] += adherence_values[env_idx]
                
                # Track reward components
                if env_idx < len(components) and isinstance(components[env_idx], dict):
                    comp = components[env_idx]
                    episode['r_dir_sum'] += comp.get('r_dir', 0.0)
                    episode['r_mag_sum'] += comp.get('r_mag', 0.0)
                    episode['r_move_sum'] += comp.get('r_move', 0.0)
                    episode['r_adherence_sum'] += comp.get('r_adherence', 0.0)
                    episode['return'] += comp.get('total', 0.0)
                
                # Check if episode is done - this is the key change to catch all completions
                if env_idx < len(dones) and dones[env_idx]:
                    # Find episode length from info if available
                    episode_length = None
                    if env_idx < len(infos):
                        info = infos[env_idx]
                        # Check different possible keys for episode length
                        if 'episode' in info and 'l' in info['episode']:
                            episode_length = info['episode']['l']
                        elif 'episode_length' in info:
                            episode_length = info['episode_length']
                    
                    # Finalize this episode immediately
                    self._finalize_episode(env_idx, episode_length)
            
            # Write any completed episodes to CSV
            if self.completed_episodes:
                self._write_completed_episodes()
                # Clear buffer after writing
                self.completed_episodes = []
                
        except Exception as e:
            if self.verbose > 0:
                print(f"Error in _on_step of EpisodeRewardComponentsCallback: {e}")
                import traceback
                traceback.print_exc()
        
        return True
                
    def _finalize_episode(self, env_idx, episode_length=None):
        """
        Process a completed episode and add it to completed_episodes buffer
        """
        if env_idx not in self.active_episodes:
            return
            
        # Get the episode data
        episode = self.active_episodes[env_idx]
        
        # Increment global episode counter
        self.episode_counter += 1
        
        # Use provided episode length if available, otherwise use tracked length
        ep_length = episode_length if episode_length is not None else episode['length']
        
        # Only process if we have actual steps
        if ep_length > 0:
            # Calculate episode metrics
            ep_return = episode['return']
            
            # Calculate rate and averages - use proper episode length for calculations
            health_rate = episode['health_steps'] / ep_length
            adherence_mean = episode['adherence_sum'] / ep_length
            r_dir_mean = episode['r_dir_sum'] / ep_length
            r_mag_mean = episode['r_mag_sum'] / ep_length
            r_move_mean = episode['r_move_sum'] / ep_length
            r_adherence_mean = episode['r_adherence_sum'] / ep_length
            
            # Create a completed episode record
            self.completed_episodes.append({
                'episode_id': self.episode_counter,
                'timesteps': self.num_timesteps,
                'ep_length': ep_length,
                'ep_return': ep_return,
                'ep_health_rate': health_rate,
                'ep_adherence_mean': adherence_mean,
                'ep_move_dir': r_dir_mean,
                'ep_move_mag': r_mag_mean,
                'ep_move': r_move_mean,
                'ep_adh_reward': r_adherence_mean
            })
        
        # Reset tracking for this environment only after finishing
        self._init_episode_tracking(env_idx)
    
    def _on_rollout_end(self) -> None:
        """
        Ensure any remaining completed episodes are written to CSV.
        As a safety net, also check if any active episode has been truncated by rollout end.
        """
        try:
            # As a safety measure, check for any active episodes that might need finalization
            # For example, if the rollout ended in the middle of an episode
            # But normally, all episodes should be finalized in _on_step when dones=True
            for env_idx, episode in list(self.active_episodes.items()):
                if episode['length'] > 0:
                    # We assume the episode continues into the next rollout,
                    # so we don't finalize it unless we have evidence it's done
                    pass
            
            # Write any completed episodes that weren't written yet
            if self.completed_episodes:
                self._write_completed_episodes()
                self.completed_episodes = []
                
        except Exception as e:
            print(f"Error in _on_rollout_end of EpisodeRewardComponentsCallback: {e}")
    
    def _write_completed_episodes(self):
        """Write all completed episodes to CSV"""
        if not self.completed_episodes or self.csv_file is None:
            return
            
        # Get variant key and convert to descriptive name
        variant_key = ""
        try:
            if hasattr(self.training_env, 'get_attr'):
                variants = self.training_env.get_attr('reward_ablation')
                if variants and len(variants) > 0:
                    variant_key = variants[0]
        except Exception:
            variant_key = "unknown"
            
        # Convert the variant key to the descriptive name
        variant_name = get_variant_name(variant_key)
        
        # Process each completed episode
        for episode in self.completed_episodes:
            # Create row data in exact column order
            row_data = [
                variant_name,                  # variant (descriptive name)
                self.model.seed,              # seed
                episode['episode_id'],        # episode_id
                episode['timesteps'],         # timesteps
                episode['ep_length'],         # ep_length
                episode['ep_return'],         # ep_return
                episode['ep_health_rate'],    # ep_health_rate
                episode['ep_adherence_mean'], # ep_adherence_mean
                episode['ep_move_dir'],       # ep_move_dir
                episode['ep_move_mag'],       # ep_move_mag
                episode['ep_move'],           # ep_move
                episode['ep_adh_reward']      # ep_adh_reward
            ]
            
            # Format float values with at least 6 decimal places for precision
            for i, value in enumerate(row_data):
                if isinstance(value, float):
                    row_data[i] = float(f"{value:.6f}")
            
            # Write header if first time (file doesn't exist)
            if not self.csv_header_written:
                file_exists = os.path.exists(self.csv_file)
                if not file_exists:
                    with open(self.csv_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            "variant", "seed", "episode_id", "timesteps", "ep_length", 
                            "ep_return", "ep_health_rate", "ep_adherence_mean", 
                            "ep_move_dir", "ep_move_mag", "ep_move", "ep_adh_reward"
                        ])
                self.csv_header_written = True
            
            # Append row to CSV
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)

def set_global_seeds(seed: int) -> None:
    """
    Set all seeds for reproducibility
    
    Args:
        seed: The seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Set global seed to: {seed}")

def make_env(env_config: Dict[str, Any], seed: int = 0) -> callable:
    """Create a wrapped, monitored SIRS environment."""
    def _init() -> SIRSEnvironment:
        env = SIRSEnvironment(**env_config)
        env.reset(seed=seed)
        return env
    return _init

def initialize_csv_files(episodes_csv, iterations_csv):
    """
    Initialize CSV files with headers if they don't exist.
    This ensures headers are written exactly once at the beginning.
    """
    # Create episodes CSV header if it doesn't exist
    if not os.path.exists(episodes_csv):
        with open(episodes_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "variant", "seed", "episode_id", "timesteps", "ep_length", 
                "ep_return", "ep_health_rate", "ep_adherence_mean", 
                "ep_move_dir", "ep_move_mag", "ep_move", "ep_adh_reward"
            ])
            print(f"Created episodes CSV file with header: {episodes_csv}")
    
    # Create iterations CSV header if it doesn't exist
    if not os.path.exists(iterations_csv):
        with open(iterations_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "variant", "seed", "iteration", "timesteps", 
                "policy_loss", "value_loss", "entropy_loss", 
                "approx_kl", "clip_fraction", "explained_variance", 
                "std", "learning_rate"
            ])
            print(f"Created iterations CSV file with header: {iterations_csv}")

def run_ablation_study(args):
    """
    Run ablation studies for each variant and seed combination
    Using parallel processing to run all seeds for a variant simultaneously
    """
    # Create base directory for ablation results
    ablation_dir = Path(args.output_dir)
    ablation_dir.mkdir(exist_ok=True, parents=True)
    
    # Create results CSV files with required names
    episodes_csv = ablation_dir / "results_episodes.csv"
    iterations_csv = ablation_dir / "results_iterations.csv"
    
    # Initialize CSV files with headers (ensures headers are written exactly once)
    initialize_csv_files(episodes_csv, iterations_csv)
    
    # Create root logging directory
    log_root = ablation_dir / "logs"
    log_root.mkdir(exist_ok=True)
    
    # Create timestamp for consistent run names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Initialize wandb project if enabled
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "ablation_study": True,
                "variants": ABLATION_VARIANTS,
                "seeds": SEEDS,
                "timesteps": TIMESTEPS,
                "environment": env_config,
                "ppo": {k: v for k, v in ppo_config.items() if not isinstance(v, type)}
            },
            name=f"ablation_study_{timestamp}",
            reinit=True
        )
    
    # Import multiprocessing for parallel execution
    import multiprocessing as mp
    
    # Run experiments for each variant - one variant at a time
    for variant in ABLATION_VARIANTS:
        print(f"\n=== Starting training for variant: {variant} with all seeds in parallel ===")
        
        # Create a list to hold all processes
        processes = []
        
        # Start a process for each seed in parallel
        for seed in SEEDS:
            # Create a process for this seed
            process = mp.Process(
                target=train_single_variant_seed,
                args=(variant, seed, timestamp, ablation_dir, log_root, episodes_csv, iterations_csv, args)
            )
            processes.append(process)
            process.start()
            print(f"Started process for {variant} with seed {seed}")
        
        # Wait for all processes for this variant to complete
        for process in processes:
            process.join()
            
        print(f"\n=== Completed all seeds for variant: {variant} ===")
    
    # Generate summary plots and statistics
    generate_summary(episodes_csv, iterations_csv, ablation_dir)

def train_single_variant_seed(variant, seed, timestamp, ablation_dir, log_root, episodes_csv, iterations_csv, args):
    """
    Train a single variant-seed combination
    This function runs in its own process
    """
    try:
        # Create a unique run name for this variant-seed combination
        run_name = f"{variant}_seed{seed}_{timestamp}"
        
        # Create a copy of the environment config with this variant
        var_env_config = env_config.copy()
        var_env_config["reward_ablation"] = variant
        
        # Create log directory for this run
        log_path = log_root / run_name
        log_path.mkdir(exist_ok=True)
        
        # Set seeds for reproducibility
        set_global_seeds(seed)
        
        # Create vectorized environment - FOR ABLATION, USE 1 ENV
        n_envs = 1  # Use just 1 environment for accurate tracking
        env_fns = [make_env(var_env_config, seed=seed + i) for i in range(n_envs)]
        vec_env = SubprocVecEnv(env_fns)
        
        # Create monitor wrapper with proper info
        monitor_path = os.path.join(log_path, "monitor")
        os.makedirs(monitor_path, exist_ok=True)
        
        # Ensure each environment gets an ID included in info dict
        vec_env = VecMonitor(
            vec_env, 
            monitor_path,
            info_keywords=("episode_length",)
        )
        
        # Add VecNormalize last
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
        
        # Create callbacks
        callbacks = []
        
        # Create checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=save_config["save_freq"],
            save_path=log_path,
            name_prefix=f"{variant}_model",
            save_replay_buffer=save_config["save_replay_buffer"],
            save_vecnormalize=True
        )
        callbacks.append(checkpoint_callback)
        
        # Create episode metrics callback
        episode_callback = EpisodeRewardComponentsCallback(
            verbose=1,
            csv_file=str(episodes_csv)
        )
        # Set csv_header_written flag to True since we've already written the header
        episode_callback.csv_header_written = True
        callbacks.append(episode_callback)
        
        # Create iteration metrics callback
        iteration_callback = IterationMetricsCallback(
            verbose=1,
            csv_file=str(iterations_csv),
            log_interval=LOG_INTERVAL
        )
        # Set csv_header_written flag to True since we've already written the header
        iteration_callback.csv_header_written = True
        callbacks.append(iteration_callback)
        
        # Add wandb callback if enabled
        if args.use_wandb:
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                group="ablation_study",
                config={
                    "variant": variant,
                    "seed": seed,
                    "environment": var_env_config,
                    "ppo": {k: v for k, v in ppo_config.items() if not isinstance(v, type)}
                },
                reinit=True
            )
            
            wandb_callback = WandbCallback(
                gradient_save_freq=100000,
                model_save_path=str(log_path),
                verbose=1
            )
            callbacks.append(wandb_callback)
        
        # Create policy kwargs dictionary that's JSON serializable
        policy_kwargs = {}
        if "policy_kwargs" in ppo_config:
            policy_kwargs = ppo_config["policy_kwargs"].copy()
            # Convert activation function to string if it's a type
            if "activation_fn" in policy_kwargs and isinstance(policy_kwargs["activation_fn"], type):
                # Get the name of the activation function
                policy_kwargs["activation_fn"] = policy_kwargs["activation_fn"].__name__
        
        # Create model
        model_kwargs = {k: v for k, v in ppo_config.items() if k not in ["policy_type", "total_timesteps", "n_envs", "policy_kwargs"]}
        if policy_kwargs:
            model_kwargs["policy_kwargs"] = {k: v for k, v in ppo_config["policy_kwargs"].items() 
                                  if not isinstance(v, type)}
        
        # Adjust n_steps for a single environment (batch size should be n_steps * n_envs)
        if "n_steps" in model_kwargs and n_envs == 1 and ppo_config.get("n_envs", 1) > 1:
            # Scale up n_steps to maintain same batch size with fewer environments
            original_n_envs = ppo_config.get("n_envs", 1)
            model_kwargs["n_steps"] = model_kwargs["n_steps"] * original_n_envs
            print(f"[{variant} seed {seed}] Adjusted n_steps to {model_kwargs['n_steps']} for single environment")
        
        model = PPO(
            ppo_config["policy_type"],
            vec_env,
            verbose=save_config["verbose"],
            tensorboard_log=str(log_path / "tensorboard"),
            seed=seed,
            **model_kwargs
        )
        
        # Train model
        print(f"\n=== Training {variant} with seed {seed} ===")
        try:
            model.learn(
                total_timesteps=TIMESTEPS,
                callback=callbacks,
                progress_bar=True,
            )
            print(f"=== Completed training {variant} with seed {seed} ===")
        except KeyboardInterrupt:
            print(f"\nTraining interrupted for {variant} with seed {seed}. Saving model...")
        finally:
            # Save final model
            model.save(os.path.join(log_path, "final_model"))
            vec_env.save(os.path.join(log_path, "vecnormalize.pkl"))
            
            # Save configuration - ensure it's JSON serializable
            config = {
                "variant": variant,
                "seed": seed,
                "timesteps": TIMESTEPS,
                "environment": var_env_config,
                "ppo": {k: str(v) if isinstance(v, type) else v for k, v in ppo_config.items()}
            }
            
            # Convert any Python types to strings
            def make_json_serializable(obj):
                if isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_serializable(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(make_json_serializable(item) for item in obj)
                elif isinstance(obj, type):
                    return obj.__name__
                elif hasattr(obj, '__dict__'):
                    return str(obj)
                else:
                    return obj
            
            config = make_json_serializable(config)
            
            with open(os.path.join(log_path, "config.json"), "w") as f:
                json.dump(config, f, indent=4)
            
            # Close environments
            vec_env.close()
            
            # Close wandb run
            if args.use_wandb:
                wandb.finish()
            
            print(f"=== Saved results for {variant} with seed {seed} ===")
    except Exception as e:
        import traceback
        print(f"Error in training {variant} with seed {seed}: {e}")
        traceback.print_exc()

def generate_summary(episodes_csv, iterations_csv, output_dir):
    """
    Generate summary plots and statistics from the results CSV files
    """
    try:
        # Read results CSV files
        episodes_df = pd.read_csv(episodes_csv)
        iterations_df = pd.read_csv(iterations_csv)
        
        # Generate timesteps vs episode length plot (learning curves)
        plt_path = os.path.join(output_dir, "learning_curves.png")
        generate_learning_curves(iterations_df, plt_path)
        
        # Generate final performance bar chart
        bar_path = os.path.join(output_dir, "final_performance.png")
        generate_performance_bars(episodes_df, bar_path)
        
        # Generate summary statistics
        stats_path = os.path.join(output_dir, "summary_stats.json")
        generate_summary_stats(episodes_df, stats_path)
        
        print(f"\nSummary files generated in {output_dir}")
    except Exception as e:
        print(f"Error generating summary: {e}")

def generate_learning_curves(df, output_path):
    """
    Generate learning curves plot showing episode length vs timesteps for each variant
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        
        # Group by variant and timesteps, calculate mean and std
        grouped = df.groupby(['variant_name', 'total_timesteps'])['ep_len_mean'].agg(['mean', 'std']).reset_index()
        
        # Plot each variant - using descriptive names
        for variant_key in ABLATION_VARIANTS:
            variant_name = get_variant_name(variant_key)
            variant_data = grouped[grouped['variant_name'] == variant_name]
            if not variant_data.empty:
                plt.plot(variant_data['total_timesteps'], variant_data['mean'], label=variant_name)
                plt.fill_between(
                    variant_data['total_timesteps'],
                    variant_data['mean'] - variant_data['std'],
                    variant_data['mean'] + variant_data['std'],
                    alpha=0.2
                )
        
        plt.xlabel('Timesteps')
        plt.ylabel('Episode Length')
        plt.title('Ablation Study: Learning Curves')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    except ImportError:
        print("Could not generate plots: matplotlib or seaborn not installed")

def generate_performance_bars(df, output_path):
    """
    Generate bar chart of final performance for each variant
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Get the maximum timestep for each variant and seed
        max_timesteps = df.groupby(['variant_key', 'seed'])['total_timesteps'].max().reset_index()
        
        # Merge with original data to get final performance metrics
        final_perf = pd.merge(
            df, 
            max_timesteps, 
            on=['variant_key', 'seed', 'total_timesteps']
        )
        
        # Calculate mean and std of ep_len across seeds for each variant
        variant_perf = final_perf.groupby(['variant_key', 'variant_name'])['ep_len'].agg(['mean', 'std']).reset_index()
        
        # Sort by mean performance (descending)
        variant_perf = variant_perf.sort_values('mean', ascending=False)
        
        # Plot bar chart - using descriptive names
        plt.figure(figsize=(14, 8))
        sns.set_style("whitegrid")
        
        bars = plt.bar(variant_perf['variant_name'], variant_perf['mean'], yerr=variant_perf['std'], capsize=10)
        
        plt.xlabel('Reward Ablation Variant')
        plt.ylabel('Episode Length')
        plt.title('Ablation Study: Final Performance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    except ImportError:
        print("Could not generate plots: matplotlib or seaborn not installed")

def generate_summary_stats(df, output_path):
    """
    Generate summary statistics for each variant and save to JSON
    """
    # Get the maximum timestep for each variant and seed
    max_timesteps = df.groupby(['variant_key', 'seed'])['total_timesteps'].max().reset_index()
    
    # Merge with original data to get final performance metrics
    final_perf = pd.merge(
        df, 
        max_timesteps, 
        on=['variant_key', 'seed', 'total_timesteps']
    )
    
    # Calculate statistics for each variant
    stats = {}
    for variant_key in ABLATION_VARIANTS:
        variant_name = get_variant_name(variant_key)
        variant_data = final_perf[final_perf['variant_key'] == variant_key]
        if not variant_data.empty:
            stats[variant_name] = {
                'variant_key': variant_key,
                'mean_episode_length': float(variant_data['ep_len'].mean()),
                'std_episode_length': float(variant_data['ep_len'].std()),
                'mean_episode_reward': float(variant_data['ep_ret'].mean()),
                'std_episode_reward': float(variant_data['ep_ret'].std()),
            }
            
            # Include reward component statistics if available
            component_cols = [col for col in variant_data.columns if col.startswith('mean_')]
            for col in component_cols:
                component_name = col[5:]  # Remove 'mean_' prefix
                stats[variant_name][f'mean_{component_name}'] = float(variant_data[col].mean())
                stats[variant_name][f'std_{component_name}'] = float(variant_data[col].std())
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run ablation studies on potential field reward function")
    parser.add_argument("--output-dir", type=str, default="ablation", help="Directory to save ablation results")
    parser.add_argument("--use-wandb", action="store_true", help="Whether to use Weights & Biases for logging")
    parser.add_argument("--wandb-project", type=str, default="sirs-ablation", help="Wandb project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Wandb entity name")
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS, help="Number of timesteps to train each variant")
    
    args = parser.parse_args()
    
    # Update global timesteps if specified
    if args.timesteps != TIMESTEPS:
        TIMESTEPS = args.timesteps
    
    # Run the ablation study
    run_ablation_study(args)
