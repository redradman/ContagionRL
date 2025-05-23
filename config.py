from torch import nn
# Environment Parameters
env_config = {
    "simulation_time": 512,        
    "grid_size": 50,               
    "n_humans": 40,                
    "n_infected": 10,              
    "beta": 0.5,                   
    "initial_agent_adherence": 0,  
    "distance_decay": 0.3,         
    "lethality": 0,                
    "immunity_loss_prob": 0.25,    
    "recovery_rate": 0.1,          
    "adherence_penalty_factor": 1, 
    "adherence_effectiveness": 0.2,         
    "movement_type": "continuous_random",  
    "movement_scale": 1,          
    "visibility_radius": -1,               
    "reinfection_count": 5,        
    "safe_distance": 10,                    
    "init_agent_distance": 5,      
    "max_distance_for_beta_calculation": 10,  
    "reward_type": "potential_field", 
    "reward_ablation": "full",              
    "render_mode": None           
}

# PPO Hyperparameters
ppo_config = {
    "policy_type": "MultiInputPolicy",
    "policy_kwargs": dict(
        net_arch=dict(
            pi=[256, 256, 256, 256],  
            vf=[256, 256, 256, 256]  
        ),
        activation_fn=nn.ReLU,  
        ortho_init=True,        
    ),
    "batch_size": 2048,          
    "n_steps": 1024,             
    "n_epochs": 5,                
    "learning_rate": 3e-4,        
    "gamma": 0.96,             
    "gae_lambda": 0.95,         
    "target_kl": 0.04,           
    "clip_range": 0.2,            
    "ent_coef": 0.02,             
    "normalize_advantage": True,   
    "total_timesteps": 8_000_000,
    "n_envs": 4                   
}

# Logging and Saving and evals 
save_config = {
    "base_log_path": "logs",
    "save_freq": 250_000,          
    "save_replay_buffer": True,
    "verbose": 1,
    "eval_freq": 250_000,          
} 