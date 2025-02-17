# Environment Parameters
env_config = {
    "simulation_time": 200,        # Longer episodes for more learning opportunity
    "grid_size": 40,               # A slightly smaller grid for easier navigation
    "n_humans": 15,                # Fewer humans to reduce environmental complexity
    "n_infected": 3,               # Fewer initial infections to avoid overwhelming the agent
    "beta": 0.2,                   # Lower infection rate for a milder disease spread
    "initial_agent_adherence": 0.3,# Lower initial adherence to allow the agent more flexibility
    "distance_decay": 0.3,         # Moderate decay rate
    "lethality": 0.05,             # Much lower lethality to reduce abrupt episode endings
    "immunity_decay": 0.05,        # Slower immunity decay so recovered remain immune longer
    "recovery_rate": 0.3,          # Faster recovery rate to help the agent recover quickly if infected
    "max_immunity_loss_prob": 0.5, # Lower maximum immunity loss probability
    "adherence_penalty_factor": 5, # Reduced penalty so that safety measures are not overly punishing
    "movement_type": "continuous_random",  # Continuous random movement for humans
    "visibility_radius": 15,       # Moderate visibility for the agent
    "reinfection_count": 1,        # Minimal reinfections for stability
    "safe_distance": 5,            # Lower threshold for safe distance in reinfection logic
    "reward_type": "balanced",   # Use the new balanced reward function for smoother learning
    "render_mode": None            # No rendering during training
}

# PPO Hyperparameters
ppo_config = {
    # Network Architecture
    "policy_type": "MultiInputPolicy",
    "policy_kwargs": dict(
        net_arch=dict(pi=[128, 64, 128], vf=[128, 64, 128])
    ),
    
    # PPO specific parameters
    "batch_size": 128,            # Increased from 64 for more stable updates
    "n_epochs": 10,               # Reduced from 10 to prevent overfitting on each batch
    "learning_rate": 1e-4,       # Reduced from 1e-4 for more stable learning
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,           
    "ent_coef": 0.005,          # Reduced from 0.01 to prevent too much exploration
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,        
    
    # Training parameters
    "total_timesteps": 2_000_000,      # Total steps across all episodes and environments
    "n_envs": 7,                  # Number of parallel environments
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
    "eval_freq": 70000,  # How often to run evaluation episodes
} 