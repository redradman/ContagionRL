import os
import argparse
import json
from stable_baselines3 import PPO
import imageio
from environment import SIRSEnvironment

def load_config(model_path: str):
    """Load the configuration file associated with the model."""
    config_path = os.path.join(os.path.dirname(model_path), "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def record_episode(model, env, video_path: str, deterministic: bool = True):
    """Record a single episode."""
    obs = env.reset()[0]
    frames = []
    done = False
    
    # Collect initial frame
    frame = env.render()
    if frame is not None:
        frames.append(frame)
    
    total_reward = 0
    step_count = 0
    
    while not done:
        # Get action from the model
        action, _ = model.predict(obs, deterministic=deterministic)
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Get frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)
            
        done = terminated or truncated
    
    # Save the video
    if frames:
        imageio.mimsave(video_path, frames, fps=env.metadata["render_fps"])
        print(f"Video saved to {video_path}")
        print(f"Episode finished after {step_count} steps with total reward: {total_reward:.2f}")
    else:
        print("Warning: No frames were collected during the episode")

def main(args):
    # Load the config
    config = load_config(args.model_path)
    env_config = config["environment"]
    
    # Modify config for visualization
    env_config["render_mode"] = "rgb_array"
    
    # Create the environment
    env = SIRSEnvironment(**env_config)
    
    # Load the model
    model = PPO.load(args.model_path)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Record episodes
    for i in range(args.num_episodes):
        video_path = os.path.join(args.output_dir, f"episode_{i+1}.mp4")
        record_episode(model, env, video_path, deterministic=not args.stochastic)
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize a trained PPO agent')
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to the trained model file')
    parser.add_argument('--output-dir', type=str, default='trained_model_videos',
                      help='Directory to save the videos')
    parser.add_argument('--num-episodes', type=int, default=5,
                      help='Number of episodes to record')
    parser.add_argument('--stochastic', action='store_true',
                      help='Use stochastic actions instead of deterministic')
    
    args = parser.parse_args()
    main(args) 