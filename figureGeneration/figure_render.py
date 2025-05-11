import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the path to access project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import SIRSEnvironment
from config import env_config


def main():
    parser = argparse.ArgumentParser(description="Render a single frame from SIRSEnvironment and save as PDF.")
    parser.add_argument(
        "--output",
        type=str,
        default="figures/environment_render.pdf",
        help="Output PDF file path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for environment reset (default: 42)"
    )
    args = parser.parse_args()

    # Use the default env_config, but ensure render_mode is set
    config = env_config.copy()
    config["render_mode"] = "rgb_array"

    # Create environment
    env = SIRSEnvironment(**config)
    env.reset(seed=args.seed)

    # Take a few random steps to populate the environment with more state diversity
    n_steps = 30
    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset(seed=args.seed)

    # Render a frame after a few steps
    frame = env.render()  # shape: (H, W, 3), dtype: uint8
    if frame is None:
        print("Error: Environment did not return a frame. Check render_mode.")
        return

    # Plot and save as PDF
    plt.figure(figsize=(frame.shape[1] / 100, frame.shape[0] / 100), dpi=100)
    plt.imshow(frame)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(args.output, format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved rendered environment frame to {args.output}")

if __name__ == "__main__":
    main() 