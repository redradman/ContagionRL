import os
import argparse
import json
from stable_baselines3 import PPO
import imageio
from environment import SIRSDEnvironment
import numpy as np

def load_config(model_path: str = None, config_path: str = None):
    """
    Load configuration from either a model path or a direct config.json path.
    
    Args:
        model_path: Path to the trained model (will load config from same directory)
        config_path: Direct path to a config.json file
        
    Returns:
        Dictionary with configuration
    """
    if config_path:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    
    elif model_path:
        config_path = os.path.join(os.path.dirname(model_path), "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    
    else:
        raise ValueError("Either model_path or config_path must be provided")

def record_episode(model, env, video_path: str, deterministic: bool = True, use_random: bool = False):
    """
    Record a single episode using either a trained model or random actions.
    
    Args:
        model: The trained model (can be None if use_random is True)
        env: The environment to run the episode in
        video_path: Path to save the video
        deterministic: Whether to use deterministic actions for the model
        use_random: Whether to use random actions instead of the model
    """
    obs = env.reset()[0]
    frames = []
    done = False
    
    # Collect initial frame
    frame = env.render()
    if frame is not None:
        frames.append(frame)
    
    total_reward = 0
    step_count = 0
    cumulative_reward = 0
    
    while not done:
        if use_random:
            # Use random actions - sample using environment's RNG for consistency
            # This ensures that the random actions are generated with the same RNG state
            # that is used for environment dynamics
            action = np.array([
                env.np_random.uniform(-1, 1),  # delta_x
                env.np_random.uniform(-1, 1),  # delta_y
                env.np_random.uniform(0, 1)    # adherence
            ], dtype=np.float32)
        else:
            # Get action from the model
            action, _ = model.predict(obs, deterministic=deterministic)
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        cumulative_reward = info.get("cumulative_reward", total_reward)  # Use env's cumulative reward if available
        
        # Get frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)
            
        # Print action information
        # dx, dy, adherence = action
        # print(f"Step {step_count}: dx={dx:.2f}, dy={dy:.2f}, adherence={adherence:.2f}, reward={reward:.2f}")
        
        done = terminated or truncated
    
    # Save the video
    if frames:
        imageio.mimsave(video_path, frames, fps=env.metadata["render_fps"])
        print(f"Video saved to {video_path}")
        print(f"Episode finished after {step_count} steps with total reward: {cumulative_reward:.2f}")
    else:
        print("Warning: No frames were collected during the episode")

def main(args):
    # Load the config either from model or direct config file
    if args.config_path:
        config = load_config(config_path=args.config_path)
    else:
        config = load_config(model_path=args.model_path)
    
    env_config = config["environment"]
    
    # Modify config for visualization
    env_config["render_mode"] = "rgb_array"
    
    # Create the environment
    env = SIRSDEnvironment(**env_config)
    
    # Load the model if not using random actions
    model = None
    if not args.random_actions and args.model_path:
        model = PPO.load(args.model_path)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Record episodes
    for i in range(args.num_episodes):
        video_path = os.path.join(args.output_dir, f"episode_{i+1}.mp4")
        record_episode(
            model=model, 
            env=env, 
            video_path=video_path, 
            deterministic=not args.stochastic,
            use_random=args.random_actions
        )
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize a trained PPO agent or random actions')
    parser.add_argument('--model-path', type=str,
                      help='Path to the trained model file')
    parser.add_argument('--config-path', type=str,
                      help='Path to a config.json file (alternative to model-path)')
    parser.add_argument('--output-dir', type=str, default='visualization_videos',
                      help='Directory to save the videos')
    parser.add_argument('--num-episodes', type=int, default=5,
                      help='Number of episodes to record')
    parser.add_argument('--stochastic', action='store_true',
                      help='Use stochastic actions instead of deterministic when using a model')
    parser.add_argument('--random-actions', action='store_true',
                      help='Use random actions instead of a trained model')
    
    args = parser.parse_args()
    
    # Check that at least one of model_path or config_path is provided
    if not args.model_path and not args.config_path:
        parser.error("Either --model-path or --config-path must be provided")
    
    # If using random actions, config-path is required if model-path not provided
    if args.random_actions and not args.model_path and not args.config_path:
        parser.error("When using --random-actions without a model, --config-path must be provided")
    
    main(args)