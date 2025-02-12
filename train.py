import os
import argparse
import json
from datetime import datetime
from typing import Dict, Any

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
import wandb
from wandb.integration.sb3 import WandbCallback
import imageio
from environment import SIRSEnvironment
from config import env_config, ppo_config, save_config

class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env, video_folder, eval_freq=1000):
        super().__init__()
        self.eval_env = eval_env
        self.video_folder = video_folder
        self.eval_freq = eval_freq
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
            else:
                # print("Warning: No frames were collected during the episode")
                pass
        
        return True

def make_env(env_config: Dict[str, Any], seed: int = 0):
    """Create a wrapped, monitored SIRS environment."""
    def _init():
        env = SIRSEnvironment(**env_config)
        env.reset(seed=seed)
        return env
    return _init

def make_eval_env(env_config: Dict[str, Any], seed: int = 0):
    """Create an evaluation environment with rendering enabled."""
    eval_config = env_config.copy()
    eval_config["render_mode"] = "rgb_array"  # Use rgb_array instead of human to avoid window
    env = SIRSEnvironment(**eval_config)
    env.reset(seed=seed)
    return env

def setup_wandb(config: Dict[str, Any], run_name: str):
    """Initialize wandb with all configs."""
    wandb.init(
        project="sirs-rl",
        name=run_name,
        config={
            "environment": env_config,
            "ppo": ppo_config,
            "save": save_config
        }
    )

def save_config_with_model(save_path: str, config: Dict[str, Any]):
    """Save the configuration alongside the model."""
    config_path = os.path.join(save_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

def main(args):
    # Create unique run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"sirs_ppo_{timestamp}"
    if args.exp_name:
        run_name = f"{args.exp_name}_{run_name}"

    # Setup logging directory
    log_path = os.path.join(save_config["base_log_path"], run_name)
    os.makedirs(log_path, exist_ok=True)
    
    # Create subdirectories for TensorBoard and videos
    tensorboard_path = os.path.join(log_path, "tensorboard")
    video_folder = os.path.join(log_path, "videos")
    os.makedirs(tensorboard_path, exist_ok=True)
    os.makedirs(video_folder, exist_ok=True)

    # Initialize wandb if requested
    if args.use_wandb:
        setup_wandb({
            "environment": env_config,
            "ppo": ppo_config,
            "save": save_config
        }, run_name)

    # Create vectorized environment
    env_fns = [make_env(env_config, seed=i) for i in range(ppo_config["n_envs"])]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env, os.path.join(log_path, "monitor"))

    # Create evaluation environment if needed
    if args.eval_freq > 0:
        eval_env = make_eval_env(env_config, seed=42)
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
    if args.eval_freq > 0:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=log_path,
            log_path=log_path,
            eval_freq=args.eval_freq,
            deterministic=True,
            render=False
        )
        video_recorder = VideoRecorderCallback(
            eval_env,
            video_folder,
            eval_freq=args.eval_freq
        )
        callbacks.extend([eval_callback, video_recorder])

    # Wandb callback if requested
    if args.use_wandb:
        callbacks.append(WandbCallback())

    # Save configs
    save_config_with_model(log_path, {
        "environment": env_config,
        "ppo": ppo_config,
        "save": save_config
    })

    # Create and train the agent
    model = PPO(
        ppo_config["policy_type"],
        vec_env,
        verbose=save_config["verbose"],
        tensorboard_log=tensorboard_path,  # Use the dedicated tensorboard directory
        **{k: v for k, v in ppo_config.items() if k not in ["policy_type", "total_timesteps", "n_envs"]}
    )

    try:
        model.learn(
            total_timesteps=ppo_config["total_timesteps"],
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
    finally:
        # Save the final model
        model.save(os.path.join(log_path, "final_model"))
        if args.use_wandb:
            wandb.finish()
        # Clean up
        vec_env.close()
        if args.eval_freq > 0:
            eval_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a PPO agent for the SIRS environment')
    parser.add_argument('--exp-name', type=str, default='',
                        help='Optional experiment name prefix')
    parser.add_argument('--eval-freq', type=int, default=10000,
                        help='Evaluation frequency in timesteps. Set to 0 to disable.')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--config', type=str, default='config.py',
                        help='Path to config file')
    
    args = parser.parse_args()
    
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