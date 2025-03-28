from torch import nn
# Environment Parameters
env_config = {
    "simulation_time": 500,        # Longer episodes for more learning opportunity
    "grid_size": 50,               # A slightly smaller grid for easier navigation
    "n_humans": 40,                # Fewer humans to reduce environmental complexity
    "n_infected": 10,              # Fewer initial infections to avoid overwhelming the agent
    "beta": 0.5,                   # Reduced infection rate
    "initial_agent_adherence": 0,# Lower initial adherence to allow the agent more flexibility
    "distance_decay": 0.3,         # Increased decay rate to make distance more important
    "lethality": 0,             # Increased to 2% chance of death per step
    "immunity_decay": 0.1,        # Slower immunity decay so recovered remain immune longer
    "recovery_rate": 0.1,          # Increased to 10% chance of recovery per step
    "max_immunity_loss_prob": 0.25, # Lower maximum immunity loss probability
    "adherence_penalty_factor": 1, # Reduced penalty factor for more balanced adherence decisions
    "movement_type": "continuous_random",  # Continuous random movement for humans
    "movement_scale": 1,         # Scale factor for non-focal agent movement (0 to 1)
    "visibility_radius": -1,       # DO NOT CHANGE THIS. Fully visilibty is required
    "reinfection_count": 5,        # Moderate reinfection count to maintain some infected presence
    "safe_distance": 10,            # Distance for infected humans at initialization and reinfection
    "init_agent_distance": 5,     # Minimum starting distance ALL humans must be from agent
    "max_distance_for_beta_calculation": 10,  # -1 means no limit (current behavior), >0 limits infection range
    "reward_type": "comprehensive",  # Using our new comprehensive reward function with distance-based crowding
    "render_mode": None            # No rendering during training
}

# PPO Hyperparameters
ppo_config = {
    # Network Architecture
    "policy_type": "MultiInputPolicy",
    "policy_kwargs": dict(
        net_arch=dict(
            pi=[256, 128, 64, 32, 16],  
            vf=[256, 128, 64, 32, 16]  
        ),
        activation_fn=nn.ReLU,  # Explicitly use ReLU activation
        ortho_init=True,        # Use orthogonal initialization for better training stability
        # log_std_init=-1.0,
    ),
    
    # PPO specific parameters
    "batch_size": 2048,            # Larger batch for more stable updates
    # "n_steps": 4096,               # Collect more steps before updating
    "n_epochs": 10,                # Fewer epochs to prevent overfitting
    "learning_rate": 3e-4,         # Reduced learning rate for more stable learning
    "gamma": 0.98,                 # Higher discount factor to focus more on long-term rewards
    "gae_lambda": 0.95,            # Keep same lambda
    # "clip_range": 0.15,            # Reduced clip range for more stable updates
    "ent_coef": 0.005,              # Increased entropy coefficient for better exploration
    # "vf_coef": 1.5,                # Increased value function coefficient for better value estimation
    # "max_grad_norm": 0.5,          # Keep same gradient clipping
    
    # Advanced PPO settings
    "normalize_advantage": True,   # Normalize advantages for more stable training

    # Training parameters
    "total_timesteps": 20_000_000,
    "n_envs": 8                    # Increased parallel environments for more diverse experience
}

# For reference:
# - Each episode lasts simulation_time steps (20)
# - With n_envs parallel environments (4), we collect (4 * 20 = 80) steps per set of episodes
# - Total episodes that will be run = total_timesteps / (simulation_time * n_envs)
# In this case: 1000 / (20 * 4) = 12.5 sets of episodes

# Logging and Saving
save_config = {
    "base_log_path": "logs",
    "save_freq": 250_000,          # Save model every n steps
    "save_replay_buffer": True,
    "verbose": 1,
    "eval_freq": 250_000,          # Evaluate less frequently to save time
} 