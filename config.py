# Environment Parameters
env_config = {
    "simulation_time": 100,        # Steps per episode
    "grid_size": 50,
    "n_humans": 30,
    "n_infected": 10,
    "beta": 0.3,
    "initial_agent_adherence": 0.5,
    "distance_decay": 0.15,
    "lethality": 0.4,
    "immunity_decay": 0.1,
    "recovery_rate": 0.2,
    "max_immunity_loss_prob": 1,
    "adherence_penalty_factor": 10,
    "movement_type": "continuous_random",
    "visibility_radius": 8,
    "reinfection_count": 5,
    "safe_distance": 8,
    "reward_type": "increaseDistanceWithInfected",       
    "render_mode": None,  # No rendering during training
}

# PPO Hyperparameters
ppo_config = {
    # Network Architecture
    "policy_type": "MultiInputPolicy",
    "policy_kwargs": dict(
        net_arch=dict(pi=[256, 128, 256], vf=[256, 128, 256])
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