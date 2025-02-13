# Environment Parameters
env_config = {
    "simulation_time": 500,        # Steps per episode
    "grid_size": 50,
    "n_humans": 50,
    "n_infected": 20,
    "beta": 0.2,
    "initial_agent_adherence": 0.5,
    "distance_decay": 0.2,
    "lethality": 0.3,
    "immunity_decay": 0.1,
    "recovery_rate": 0.2,
    "max_immunity_loss_prob": 0.2,
    "adherence_penalty_factor": 2,
    "movement_type": "continuous_random",
    "visibility_radius": 6,
    "reinfection_count": 3,
    "reward_type": "stateBased",       
    "render_mode": None,  # No rendering during training
}

# PPO Hyperparameters
ppo_config = {
    # Network Architecture
    "policy_type": "MultiInputPolicy",
    "policy_kwargs": dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64])
    ),
    
    # PPO specific parameters
    "batch_size": 64,
    "n_epochs": 10,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    
    # Training parameters
    "total_timesteps": 1000000,      # Total steps across all episodes and environments
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
    "save_freq": 20000,  # Save model every n steps
    "save_replay_buffer": True,
    "verbose": 1,
    "eval_freq": 20000,  # How often to run evaluation episodes
} 