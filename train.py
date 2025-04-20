import os
import argparse
import json
from datetime import datetime
from typing import Dict, Any, Callable
import random
import numpy as np
import torch  # Import torch for seeding PyTorch if it's being used

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
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

def make_eval_env(env_config: Dict[str, Any], seed: int = 0) -> SIRSEnvironment:
    """Create an evaluation environment with rendering enabled."""
    eval_config = env_config.copy()
    eval_config["render_mode"] = "rgb_array"  # Use rgb_array instead of human to avoid window
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
    # Set global seeds for reproducibility
    set_global_seeds(args.seed)
    
    # Create unique run name with reward function type and formatted timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")  # Removed seconds for cleaner names
    
    # Get reward function type from environment config - throw error if not found
    if "reward_type" not in env_config:
        raise ValueError("reward_type must be specified in env_config for run naming")
    reward_type = env_config["reward_type"]
    
    # Create run name with reward function type followed by timestamp
    run_name = f"{reward_type}_{timestamp}"
    
    # Add experiment name prefix if provided
    if args.exp_name:
        run_name = f"{args.exp_name}_{run_name}"
    
    # Add seed to run name if not using the default
    if args.seed != 42:
        run_name = f"{run_name}_seed{args.seed}"

    # Setup logging directory
    log_path = os.path.join(save_config["base_log_path"], run_name)
    os.makedirs(log_path, exist_ok=True)
    
    # Create subdirectories for TensorBoard and videos
    tensorboard_path = os.path.join(log_path, "tensorboard")
    video_folder = os.path.join(log_path, "videos")
    os.makedirs(tensorboard_path, exist_ok=True)
    os.makedirs(video_folder, exist_ok=True)

    # Initialize wandb by default, unless disabled with --no-wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        # Add seed information to wandb config
        env_config_with_seed = env_config.copy()
        env_config_with_seed["random_seed"] = args.seed
        
        setup_wandb({
            "environment": env_config_with_seed,
            "ppo": ppo_config,
            "save": save_config
        }, run_name)
        
        # If using offline mode, print instructions for syncing later
        if args.wandb_offline:
            print("\nRunning wandb in offline mode. To sync later, run:")
            print(f"wandb sync {os.path.join('wandb', f'run-{timestamp}*')}")

    # Create vectorized environment with sequential seeds
    base_seed = args.seed
    env_fns = [make_env(env_config, seed=base_seed + i) for i in range(ppo_config["n_envs"])]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env, os.path.join(log_path, "monitor"))

    # Create evaluation environment if needed
    eval_freq = save_config.get("eval_freq", 0)  # Get eval_freq from config, default to 0
    if eval_freq > 0:
        # eval_env = make_eval_env(env_config, seed=base_seed + 100)  # Different seed for eval
        eval_env = make_eval_env(env_config, seed=base_seed)  # same seed
        eval_env = DummyVecEnv([lambda: eval_env])
        eval_env = VecMonitor(eval_env, os.path.join(log_path, "eval"))

    # Initialize callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_config["save_freq"],
        save_path=log_path,
        name_prefix="sirs_model",
        save_replay_buffer=save_config["save_replay_buffer"],
        save_vecnormalize=True
    )
    callbacks.append(checkpoint_callback)

    # Evaluation and video recording callbacks if requested
    if eval_freq > 0:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=log_path,
            log_path=log_path,
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        video_recorder = VideoRecorderCallback(
            eval_env,
            video_folder,
            eval_freq=eval_freq,
            use_wandb=use_wandb
        )
        callbacks.extend([eval_callback, video_recorder])

    # Wandb callback if wandb is enabled
    if use_wandb:
        callbacks.append(WandbCallback(model_save_path=f"models/{tensorboard_path}",
        gradient_save_freq=100,
        verbose=2))

    # Create a copy of ppo_config for model creation
    model_ppo_config = ppo_config.copy()

    # Create a serializable copy of the config for saving
    save_ppo_config = ppo_config.copy()
    if "policy_kwargs" in save_ppo_config:
        save_ppo_config["policy_kwargs"] = save_ppo_config["policy_kwargs"].copy()
        if "activation_fn" in save_ppo_config["policy_kwargs"]:
            save_ppo_config["policy_kwargs"]["activation_fn"] = "relu"  # Save as string

    # Create entropy coefficient schedule if ent_coef is specified
    if "ent_coef" in model_ppo_config:
        initial_ent_coef = model_ppo_config["ent_coef"]
        # Save the original value in the serialized config
        save_ppo_config["ent_coef_initial"] = initial_ent_coef
        save_ppo_config["ent_coef"] = "scheduled"
        
        # Create entropy callback and add to callbacks list
        ent_callback = EntropyCoefCallback(
            initial_value=initial_ent_coef,
            final_value=0,  # Target value of 0 by the end
            schedule_percentage=0.3,  # Schedule over 30% of training
            verbose=1  # Print updates
        )
        callbacks.append(ent_callback)

    # Save configs with seed information
    config_with_seed = {
        "environment": env_config.copy(),
        "ppo": save_ppo_config,  # Use the serializable version
        "save": save_config.copy(),
        "seed": args.seed
    }
    save_config_with_model(log_path, config_with_seed)

    # Convert activation function string to actual function for model creation
    if "policy_kwargs" in model_ppo_config and "activation_fn" in model_ppo_config["policy_kwargs"]:
        activation_str = model_ppo_config["policy_kwargs"]["activation_fn"]
        model_ppo_config["policy_kwargs"]["activation_fn"] = get_activation_fn(activation_str)
    
    # Create and train the agent
    model = PPO(
        model_ppo_config["policy_type"],
        vec_env,
        verbose=save_config["verbose"],
        tensorboard_log=tensorboard_path,
        seed=args.seed,
        **{k: v for k, v in model_ppo_config.items() if k not in ["policy_type", "total_timesteps", "n_envs"]}
    )

    try:
        model.learn(
            total_timesteps=ppo_config["total_timesteps"],
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        pass
    finally:
        # Save the final model
        model.save(os.path.join(log_path, "final_model"))
        if use_wandb:
            wandb.finish()
        # Clean up
        vec_env.close()
        if eval_freq > 0:
            eval_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a SIRS environment agent using PPO")
    parser.add_argument("--exp-name", type=str, default="", help="Optional experiment name prefix")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging (enabled by default)")
    parser.add_argument("--wandb-offline", action="store_true", help="Run wandb in offline mode to avoid timeout issues")
    parser.add_argument("--config", type=str, default="config.py", help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    
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