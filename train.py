import os
import argparse
import json
from datetime import datetime
from typing import Dict, Any, Callable
import random
import numpy as np
import torch  # Import torch for seeding PyTorch if it's being used

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
import wandb
from wandb.integration.sb3 import WandbCallback
import imageio
from environment import SIRSEnvironment
from config import env_config, ppo_config, save_config

# Add entropy coefficient scheduler
class EntropyCoefCallback(BaseCallback):
    """
    Callback for updating the entropy coefficient during training.
    Allows for scheduled entropy coefficient annealing.
    """
    def __init__(self, initial_value: float, final_value: float = 0.0, schedule_percentage: float = 0.3, verbose: int = 0):
        super().__init__(verbose)
        self.initial_value = initial_value
        self.final_value = final_value
        self.schedule_percentage = schedule_percentage
        self.current_value = initial_value
        
    def _on_step(self) -> bool:
        """
        Update entropy coefficient based on training progress.
        """
        # Calculate progress elapsed (0 to 1)
        if self.model.num_timesteps >= self.model._total_timesteps:
            progress_elapsed = 1.0
        else:
            progress_elapsed = self.model.num_timesteps / self.model._total_timesteps
            
        # Apply scheduling only during the specified percentage of training
        if progress_elapsed >= self.schedule_percentage:
            new_value = self.final_value
        else:
            # Calculate what percentage of the schedule has elapsed
            schedule_progress = min(progress_elapsed / self.schedule_percentage, 1.0)
            
            # Linear interpolation between initial and final values
            new_value = self.initial_value + schedule_progress * (self.final_value - self.initial_value)
        
        # Update the entropy coefficient in the model
        self.model.ent_coef = new_value
        self.current_value = new_value
        
        # Log the current entropy coefficient value
        if self.verbose > 0 and self.n_calls % 100_000 == 0:
            print(f"Current entropy coefficient: {self.current_value:.6f}")
            
        return True

def set_global_seeds(seed: int) -> None:
    """
    Set all seeds for reproducibility.
    
    Args:
        seed: The seed value to use
    """
    # Set Python's random module seed
    random.seed(seed)
    
    # Set NumPy's random generator seed
    np.random.seed(seed)
    
    # Set PyTorch seed (if using neural networks with torch)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        # Make CUDA operations deterministic for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment action space seed
    # This will be handled by SIRSEnvironment.reset(seed=seed)
    
    print(f"Set global seed to: {seed}")

class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env, video_folder, eval_freq=1000, use_wandb=False):
        super().__init__()
        self.eval_env = eval_env
        self.video_folder = video_folder
        self.eval_freq = eval_freq
        self.use_wandb = use_wandb
        os.makedirs(video_folder, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # print("Starting evaluation episode...")
            # Reset the environment and get initial observation
            reset_result = self.eval_env.reset()
            if isinstance(reset_result, tuple):
                obs = reset_result[0]
            else:
                obs = reset_result
            
            # Get underlying env
            env = self.eval_env.envs[0]
            frames = []
            
            # Run episode and collect frames
            done = False
            step_count = 0
            
            # Collect initial frame
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            
            while not done:
                # Get action from the agent
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Step the environment
                obs, reward, terminated, info = self.eval_env.step(action)
                
                # In vectorized env, terminated is a boolean array and info is a list of dicts
                terminated = terminated[0]
                truncated = info[0].get('TimeLimit.truncated', False)
                done = terminated or truncated
                
                # Only add frame if we're not done (to avoid capturing next episode's first frame)
                if not done:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                step_count += 1
            
            # print(f"Episode finished after {step_count} steps")
            
            # Save the video
            if frames:
                video_path = os.path.join(
                    self.video_folder, 
                    f"eval_episode_{self.n_calls}_steps.mp4"
                )
                # Get FPS from environment metadata, raise error if not found
                if "render_fps" not in self.eval_env.envs[0].metadata:
                    raise ValueError("Environment metadata must contain 'render_fps'")
                env_fps = self.eval_env.envs[0].metadata["render_fps"]
                imageio.mimsave(video_path, frames, fps=env_fps)
                
                # Log video to wandb if enabled
                if self.use_wandb:
                    try:
                        # Create a meaningful name for the video
                        episode_name = f"episode_{self.n_calls}"
                        
                        # Log video to wandb - removed fps parameter since it doesn't work with file paths
                        wandb.log({
                            f"videos/{episode_name}": wandb.Video(
                                video_path, 
                                caption=f"Evaluation at {self.n_calls} steps"
                            )
                        }, step=self.n_calls)
                        print(f"Logged evaluation video to wandb at step {self.n_calls}")
                    except Exception as e:
                        # Don't fail if wandb logging fails
                        print(f"Error logging video to wandb: {e}")
            else:
                # print("Warning: No frames were collected during the episode")
                pass
        
        return True

def make_env(env_config: Dict[str, Any], seed: int = 0) -> Callable:
    """Create a wrapped, monitored SIRS environment."""
    def _init() -> SIRSEnvironment:
        env = SIRSEnvironment(**env_config)
        # Set seed for this specific environment instance
        env.reset(seed=seed)
        return env
    return _init

def make_eval_env(env_config: Dict[str, Any], seed: int = 0, record_video: bool = False) -> SIRSEnvironment:
    """Create an evaluation environment with rendering enabled only if record_video is True."""
    eval_config = env_config.copy()
    if record_video:
        eval_config["render_mode"] = "rgb_array"
    else:
        eval_config["render_mode"] = None
    env = SIRSEnvironment(**eval_config)
    env.reset(seed=seed)
    return env

def setup_wandb(config: Dict[str, Any], run_name: str) -> None:
    """Initialize wandb with all configs."""
    # Create wandb settings with increased timeout
    wandb_settings = wandb.Settings(
        init_timeout=120, 
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    )
    
    wandb.init(
        project="sirs-rl",
        name=run_name,
        config={
            "environment": config["environment"],
            "ppo": config["ppo"],
            "save": config["save"]
        },
        settings=wandb_settings
    )

def save_config_with_model(save_path: str, config: Dict[str, Any]) -> None:
    """Save the configuration alongside the model."""
    config_path = os.path.join(save_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

def get_activation_fn(activation_str: str) -> torch.nn.Module:
    """Convert activation function string to PyTorch activation function."""
    # If it's already a PyTorch activation function class, return it
    if isinstance(activation_str, type) and issubclass(activation_str, torch.nn.Module):
        return activation_str
        
    # If it's already a PyTorch activation function instance, return its class
    if isinstance(activation_str, torch.nn.Module):
        return activation_str.__class__
        
    # If it's a string, convert it to the appropriate activation function class
    activation_map = {
        "relu": torch.nn.ReLU,
        "tanh": torch.nn.Tanh,
        "sigmoid": torch.nn.Sigmoid,
        "leaky_relu": torch.nn.LeakyReLU,
        "elu": torch.nn.ELU
    }
    if activation_str.lower() not in activation_map:
        raise ValueError(f"Unsupported activation function: {activation_str}")
    return activation_map[activation_str.lower()]

def main(args):
    # --- Part 1: Setup common to all seeds ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    if "reward_type" not in env_config:
        raise ValueError("reward_type must be specified in env_config for run naming")
    reward_type = env_config["reward_type"]

    # Base name for grouping runs (e.g., for W&B group)
    # This part determines the name prefix before "_seed{current_seed}"
    if args.exp_name:
        base_run_name_for_group = args.exp_name # If exp_name is given, it's the full base
    else:
        base_run_name_for_group = f"{reward_type}_{timestamp}"


    SEEDS = [1, 2, 3]
    should_record_video = args.record_video # Get flag value here, once

    for current_seed in SEEDS:
        print(f"--- Starting run for SEED: {current_seed} ---")
        # --- Part 2: Seed-specific setup ---
        set_global_seeds(current_seed)

        # Create unique run name for this specific seed
        # The user requested name_seed{seed_num}. If base_run_name_for_group already includes a timestamp or exp_name,
        # this appends the seed correctly.
        run_name = f"{base_run_name_for_group}_seed{current_seed}"

        log_path = os.path.join(save_config["base_log_path"], run_name)
        os.makedirs(log_path, exist_ok=True)
        tensorboard_path = os.path.join(log_path, "tensorboard")
        video_folder = os.path.join(log_path, "videos")
        os.makedirs(tensorboard_path, exist_ok=True)
        os.makedirs(video_folder, exist_ok=True)

        use_wandb = not args.no_wandb
        current_wandb_run = None # To store wandb run object
        if use_wandb:
            env_config_for_wandb = env_config.copy()
            env_config_for_wandb["random_seed"] = current_seed
            
            ppo_config_for_wandb = ppo_config.copy()
            if "policy_kwargs" in ppo_config_for_wandb and "activation_fn" in ppo_config_for_wandb["policy_kwargs"]:
                 if isinstance(ppo_config_for_wandb["policy_kwargs"]["activation_fn"], type):
                    ppo_config_for_wandb["policy_kwargs"]["activation_fn"] = ppo_config_for_wandb["policy_kwargs"]["activation_fn"].__name__


            current_wandb_run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "sirs-rl"), # Use environment variable or default
                name=run_name,
                group=base_run_name_for_group,
                config={
                    "environment": env_config_for_wandb,
                    "ppo": ppo_config_for_wandb,
                    "save": save_config,
                    "seed": current_seed
                },
                settings=wandb.Settings(
                    init_timeout=120, 
                    sync_tensorboard=True,
                    # Handle cases where WANDB_DIR might not be set, default to './wandb' relative to script
                    # This helps ensure logs are saved correctly in offline mode per run.
                    # wandb_dir=os.path.join(os.getcwd(), 'wandb', run_name) if args.wandb_offline else None
                ),
                reinit=True
            )
            if args.wandb_offline:
                 print(f"\\nRunning W&B in offline mode for seed {current_seed}. Run 'wandb sync {current_wandb_run.dir}' to sync.")

        # Create vectorized environment with sequential seeds based on current_seed
        # Each env in SubprocVecEnv gets current_seed + its_index
        env_fns = [make_env(env_config, seed=current_seed + i) for i in range(ppo_config["n_envs"])]
        vec_env = SubprocVecEnv(env_fns)
        vec_env = VecMonitor(vec_env, os.path.join(log_path, "monitor"))
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

        eval_env = None
        eval_freq = save_config.get("eval_freq", 0)
        if eval_freq > 0:
            eval_env = make_eval_env(env_config, seed=current_seed, record_video=should_record_video) # Pass flag here
            eval_env = DummyVecEnv([lambda: eval_env])
            eval_env = VecMonitor(eval_env, os.path.join(log_path, "eval"))
            eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
            if vec_env.ret_rms is not None: # Check if ret_rms exists before copying
                 eval_env.ret_rms = vec_env.ret_rms
            eval_env.training = False

        callbacks = []
        checkpoint_callback = CheckpointCallback(
            save_freq=save_config["save_freq"],
            save_path=log_path,
            name_prefix="sirs_model", # The seed is already in log_path
            save_replay_buffer=save_config["save_replay_buffer"],
            save_vecnormalize=True
        )
        callbacks.append(checkpoint_callback)

        if eval_freq > 0 and eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=log_path,
                log_path=log_path,
                eval_freq=eval_freq,
                n_eval_episodes=5, # Explicitly set, though 5 is default
                deterministic=True,
                render=False # EvalCallback itself should not render to screen
            )
            callbacks.append(eval_callback) # Add EvalCallback regardless

            if should_record_video: # Conditionally add VideoRecorderCallback
                video_recorder = VideoRecorderCallback(
                    eval_env,
                    video_folder,
                    eval_freq=eval_freq,
                    use_wandb=use_wandb
                )
                callbacks.append(video_recorder)

        if use_wandb:
            callbacks.append(WandbCallback(
                model_save_path=f"models/{run_name}", # Save model to W&B under run_name
                gradient_save_freq=100_000, # Example, adjust as needed
                verbose=2
            ))

        model_ppo_config = ppo_config.copy()
        save_ppo_config = ppo_config.copy()
        if "policy_kwargs" in save_ppo_config:
            save_ppo_config["policy_kwargs"] = save_ppo_config["policy_kwargs"].copy()
            if "activation_fn" in save_ppo_config["policy_kwargs"]:
                 if isinstance(save_ppo_config["policy_kwargs"]["activation_fn"], type):
                    save_ppo_config["policy_kwargs"]["activation_fn"] = save_ppo_config["policy_kwargs"]["activation_fn"].__name__


        if "ent_coef" in model_ppo_config and isinstance(model_ppo_config["ent_coef"], (float, int)): # Check if scheduling needed
            initial_ent_coef = model_ppo_config["ent_coef"]
            save_ppo_config["ent_coef_initial"] = initial_ent_coef
            save_ppo_config["ent_coef"] = "scheduled"
            
            ent_callback = EntropyCoefCallback(
                initial_value=initial_ent_coef,
                final_value=0.002,  
                schedule_percentage=0.4,  
                verbose=1
            )
            callbacks.append(ent_callback)

        config_to_save = {
            "environment": env_config.copy(),
            "ppo": save_ppo_config,
            "save": save_config.copy(),
            "seed": current_seed # Log the actual seed used for this run
        }
        save_config_with_model(log_path, config_to_save)

        if "policy_kwargs" in model_ppo_config and "activation_fn" in model_ppo_config["policy_kwargs"]:
            activation_str_or_fn = model_ppo_config["policy_kwargs"]["activation_fn"]
            model_ppo_config["policy_kwargs"]["activation_fn"] = get_activation_fn(activation_str_or_fn)
        
        model = PPO(
            model_ppo_config["policy_type"],
            vec_env,
            verbose=save_config["verbose"],
            tensorboard_log=tensorboard_path,
            seed=current_seed, # Use current_seed for the model
            **{k: v for k, v in model_ppo_config.items() if k not in ["policy_type", "total_timesteps", "n_envs"]}
        )

        try:
            print(f"Starting training for {run_name} with {ppo_config['total_timesteps']} timesteps...")
            model.learn(
                total_timesteps=ppo_config["total_timesteps"],
                callback=callbacks,
                progress_bar=True
            )
        except KeyboardInterrupt:
            print(f"\\nTraining interrupted for {run_name}. Saving model...")
        finally:
            model.save(os.path.join(log_path, "final_model"))
            vec_env.save(os.path.join(log_path, "vecnormalize.pkl"))
            if use_wandb and current_wandb_run:
                # Check if there's an active W&B run before finishing
                if wandb.run is not None and wandb.run.id == current_wandb_run.id:
                    wandb.finish()
            
            vec_env.close()
            if eval_env:
                eval_env.close()
            print(f"--- Finished run for SEED: {current_seed} ---")
            # Add a small delay if needed, though usually not necessary
            # import time
            # time.sleep(2) 

    # Original args.seed is now less relevant for the main loop,
    # but other parts of arg parsing remain the same.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a SIRS environment agent using PPO")
    parser.add_argument("--exp-name", type=str, default="", help="Optional experiment name prefix")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging (enabled by default)")
    parser.add_argument("--wandb-offline", action="store_true", help="Run wandb in offline mode to avoid timeout issues")
    parser.add_argument("--config", type=str, default="config.py", help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42, but overridden by internal list [1,2,3])")
    parser.add_argument("--record-video", action="store_true", help="Enable video recording of evaluation episodes.")
    
    args = parser.parse_args()
    
    # Set wandb mode to offline if requested
    if not args.no_wandb and args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
        print("Using wandb in offline mode")
        
    # Import config if custom path provided
    if args.config != 'config.py':
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", args.config)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        env_config = config.env_config
        ppo_config = config.ppo_config
        save_config = config.save_config

    main(args) 