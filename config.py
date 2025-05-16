from torch import nn
# Environment Parameters
env_config = {
    "simulation_time": 512,        
    "grid_size": 50,               # the grid is a square. This value sets the size of its length. The grid area is grid_size^2. 
    "n_humans": 40,                
    "n_infected": 10,              
    "beta": 0.5,                   
    "initial_agent_adherence": 0,  # NPI adherence at the beginning
    "distance_decay": 0.3,         
    "lethality": 0,                
    "immunity_loss_prob": 0.25,    
    "recovery_rate": 0.1,          
    "adherence_penalty_factor": 1, 
    "adherence_effectiveness": 0.2,         # Maximum effect of adherence. Lower is better. When adherence is 1, effective beta would be beta_effective = beta * adherence_effectiveness
    "movement_type": "continuous_random",  
    "movement_scale": 1,          
    "visibility_radius": -1,                # -1 means full visibility. Positive values would mean individuals within this distance are visible and others are not. 
    "reinfection_count": 5,        
    "safe_distance": 10,                    
    "init_agent_distance": 5,      
    "max_distance_for_beta_calculation": 10,  
    "reward_type": "potential_field", 
    "reward_ablation": "full",              # Used for ablations, do not change    
    "render_mode": None           
}

# PPO Hyperparameters
ppo_config = {
    # Network Architecture
    "policy_type": "MultiInputPolicy",
    "policy_kwargs": dict(
        net_arch=dict(
            pi=[256, 256, 256, 256],  
            vf=[256, 256, 256, 256]  
        ),
        activation_fn=nn.ReLU,  # Explicitly use ReLU activation
        ortho_init=True,        # Use orthogonal initialization for better training stability
        # log_std_init=-0.6,
    ),
    
    # PPO specific parameters
    "batch_size": 2048,            # Larger batch for more stable updates (>= 512 samples per minibatch)
    "n_steps": 1024,               # Collect more steps before updating
    "n_epochs": 5,                # Fewer epochs to prevent overfitting
    "learning_rate": 3e-4,         # Reduced learning rate for more stable learning
    "gamma": 0.96,                 # Lower discount factor to match shorter episode horizon
    "gae_lambda": 0.95,            # Keep same lambda
    "target_kl": 0.04,             # Looser KL policing to allow larger updates
    "clip_range": 0.2,             # Wider clip range to allow larger policy changes
    # "clip_range_vf": 0.2,          # Add value function clipping to prevent overshooting
    "ent_coef": 0.02,              # Lower entropy coefficient, scheduled decay
    # "vf_coef": 2.0,                # Increased value function coefficient to stabilize value learning
    # "max_grad_norm": 0.5,          # Keep same gradient clipping
    
    # Advanced PPO settings
    "normalize_advantage": True,   # Normalize advantages for more stable training

    # Training parameters
    "total_timesteps": 8_000_000,
    "n_envs": 4                    # Increased parallel environments for more diverse experience
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