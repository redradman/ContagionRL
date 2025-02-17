# Environment Parameters
env_config = {
    "simulation_time": 200,        # Longer episodes for more learning opportunity
    "grid_size": 50,               # A slightly smaller grid for easier navigation
    "n_humans": 80,                # Fewer humans to reduce environmental complexity
    "n_infected": 10,              # Fewer initial infections to avoid overwhelming the agent
    "beta": 0.2,                   # Reduced infection rate
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
    "reward_type": "noInfection",  # Switch to simpler reward function
    "render_mode": None            # No rendering during training
}

# PPO Hyperparameters
ppo_config = {
    # Network Architecture
    "policy_type": "MultiInputPolicy",
    "policy_kwargs": dict(
        net_arch=dict(pi=[512, 256, 128], vf=[512, 512, 256]), 
        normalize_images=True,
        log_std_init=-0.5,
        ortho_init=True
    ),
    
    # PPO specific parameters
    "batch_size": 512,            
    "n_epochs": 10,               # Reduced from 10 to prevent overfitting on each batch
    "learning_rate": 1e-4,       # Reduced from 1e-4 for more stable learning
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,           
    "ent_coef": 0.01,          # Reduced from 0.01 to prevent too much exploration
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,        
    
    # Training parameters
    "total_timesteps": 2_000_000,      # Total steps across all episodes and environments
    "n_envs": 8,                  # Number of parallel environments
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
    "eval_freq": 50000,  # How often to run evaluation episodes
} 