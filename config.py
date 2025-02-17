# Environment Parameters
env_config = {
    "simulation_time": 100,        # Longer episodes for more learning opportunity
    "grid_size": 50,               # A slightly smaller grid for easier navigation
    "n_humans": 50,                # Fewer humans to reduce environmental complexity
    "n_infected": 10,              # Fewer initial infections to avoid overwhelming the agent
    "beta": 0.15,                   # Reduced infection rate
    "initial_agent_adherence": 0.3,# Lower initial adherence to allow the agent more flexibility
    "distance_decay": 0.2,         # Increased decay rate to make distance more important
    "lethality": 0.05,             # Increased to 2% chance of death per step
    "immunity_decay": 0.05,        # Slower immunity decay so recovered remain immune longer
    "recovery_rate": 0.1,          # Increased to 10% chance of recovery per step
    "max_immunity_loss_prob": 0.3, # Lower maximum immunity loss probability
    "adherence_penalty_factor": 4, # Reduced penalty factor for more balanced adherence decisions
    "movement_type": "continuous_random",  # Continuous random movement for humans
    "visibility_radius": 20,       # Moderate visibility for the agent
    "reinfection_count": 8,        # Moderate reinfection count to maintain some infected presence
    "safe_distance": 8,            # Slightly increased safe distance for better distance rewards
    "reward_type": "rewardForState",  # Switch to simpler reward function
    "render_mode": None            # No rendering during training
}

# PPO Hyperparameters
ppo_config = {
    # Network Architecture
    "policy_type": "MultiInputPolicy",
    "policy_kwargs": dict(
        net_arch=dict(
            pi=[512, 256, 128],  # Policy network
            vf=[1024, 512, 512, 256]  # Deeper/wider value network
        ),
        normalize_images=True,
        log_std_init=-1.0,  # Initial std = 0.37
        ortho_init=True
    ),
    
    # PPO specific parameters
    "batch_size": 2048,            # Larger batch for more stable value estimates
    "n_epochs": 10,                 # Reduced to prevent overfitting
    "learning_rate": 5e-5,         # Slower learning for better value estimation
    "gamma": 0.995,                # Slightly higher gamma for better long-term predictions
    "gae_lambda": 0.98,            # Higher lambda for better advantage estimation
    "clip_range": 0.1,             # Smaller clip range for more conservative updates
    "ent_coef": 0.005,             # Lower entropy to focus on value prediction
    "vf_coef": 2.0,                # Increased focus on value function
    "max_grad_norm": 0.5,          # More conservative updates
    
    # Training parameters
    "total_timesteps": 10_000_000,
    "n_envs": 5
}

# For reference:
# - Each episode lasts simulation_time steps (20)
# - With n_envs parallel environments (4), we collect (4 * 20 = 80) steps per set of episodes
# - Total episodes that will be run = total_timesteps / (simulation_time * n_envs)
# In this case: 1000 / (20 * 4) = 12.5 sets of episodes

# Logging and Saving
save_config = {
    "base_log_path": "logs",
    "save_freq": 100_000,  # Save model every n steps
    "save_replay_buffer": True,
    "verbose": 1,
    "eval_freq": 100000000000,  # How often to run evaluation episodes
} 