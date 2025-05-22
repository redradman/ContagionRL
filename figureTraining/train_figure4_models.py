import os
import sys
import argparse
import datetime
import copy
from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
import gymnasium as gym
from torch import nn
import pandas as pd
import json
# Add wandb imports
import wandb
from wandb.integration.sb3 import WandbCallback

# Add the parent directory to the path to access project modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config import env_config as global_env_config_template
from config import ppo_config as global_ppo_config_template
from config import save_config as global_save_config_template
from environment import SIRSDEnvironment

SEEDS_FOR_TRAINING = [1, 2, 3]
# ALGORITHMS = ["ppo", "sac", "a2c"]
ALGORITHMS = ["a2c"]

# Unified hyperparameter configs for PPO, SAC, and TD3
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

sac_config = {
    "policy_type": "MlpPolicy",
    "policy_kwargs": dict(
        net_arch=dict(
            pi=[256, 256, 256, 256],
            qf=[256, 256, 256, 256]
        ),
        activation_fn=nn.ReLU,
    ),
    "learning_rate": 3e-4,
    "buffer_size": 1_000_000,
    "batch_size": 256,
    "tau": 0.005,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
    "gamma": 0.96,
    "total_timesteps": 8_000_000,
    "n_envs": 4,
}

a2c_config = {
    "policy_type": "MlpPolicy",
    "policy_kwargs": dict(
        net_arch=dict(
            pi=[256, 256, 256, 256],
            vf=[256, 256, 256, 256]
        ),
        activation_fn=nn.ReLU,
        ortho_init=True,
    ),
    "n_steps": 5 * 128,
    "gamma": 0.96,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "learning_rate": 3e-4,
    "use_rms_prop": True,
    "normalize_advantage": True,
    "total_timesteps": 8_000_000,
    "n_envs": 4
}

# Helper to flatten observation space for SAC/TD3
class FlattenObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.flatten_space(env.observation_space)
    def observation(self, observation):
        return gym.spaces.flatten(self.env.observation_space, observation)

def make_env_for_algo(env_config, seed, algo):
    config = copy.deepcopy(env_config)
    config["render_mode"] = None
    def _init():
        env = SIRSDEnvironment(**config)
        env.reset(seed=seed)
        if algo in ["sac", "a2c"]:
            env = FlattenObservationWrapper(env)
        return env
    return _init

class EpisodeReturnLogger(BaseCallback):
    def __init__(self, log_path, algorithm, seed, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.algorithm = algorithm
        self.seed = seed
        self.returns_log = []
        self.episode_counter = 0  # Explicit episode counter

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_return = info["episode"]["r"]
                self.returns_log.append({
                    "algorithm": self.algorithm,
                    "seed": self.seed,
                    "episode": self.episode_counter,
                    "mean_return": ep_return,
                    "timesteps": self.num_timesteps
                })
                self.episode_counter += 1
        return True

    def _on_training_end(self):
        df = pd.DataFrame(self.returns_log)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        df.to_csv(self.log_path, index=False)

# Add entropy coefficient scheduler
class EntropyCoefCallback(BaseCallback):
    """
    Callback for updating the entropy coefficient during training.
    Allows for scheduled entropy coefficient annealing.
    """
    def __init__(self, initial_value: float, final_value: float = 0.0, schedule_percentage: float = 0.3, verbose: int = 0):
        super().__init__(verbose)
        self.initial_value = initial_value
        self.final_value = final_value
        self.schedule_percentage = schedule_percentage
        self.current_value = initial_value
        
    def _on_step(self) -> bool:
        """
        Update entropy coefficient based on training progress.
        """
        if self.model.num_timesteps >= self.model._total_timesteps:
            progress_elapsed = 1.0
        else:
            progress_elapsed = self.model.num_timesteps / self.model._total_timesteps
            
        if progress_elapsed >= self.schedule_percentage:
            new_value = self.final_value
        else:
            schedule_progress = min(progress_elapsed / self.schedule_percentage, 1.0)
            new_value = self.initial_value + schedule_progress * (self.final_value - self.initial_value)
        
        self.model.ent_coef = new_value
        self.current_value = new_value
        
        if self.verbose > 0 and self.n_calls % 100_000 == 0: # Log less frequently
            self.logger.record("train/ent_coef", self.current_value)
            
        return True

def get_activation_fn(act_fn):
    if isinstance(act_fn, str):
        # Accept both 'relu' and 'ReLU'
        act_map = {
            "relu": nn.ReLU,
            "ReLU": nn.ReLU,
            "tanh": nn.Tanh,
            "Tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "Sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU,
            "LeakyReLU": nn.LeakyReLU,
            "elu": nn.ELU,
            "ELU": nn.ELU,
        }
        return act_map[act_fn]
    return act_fn

def save_config_with_model(save_path: str, env_config: dict, algo_config: dict, save_config: dict, seed: int, algo_name: str):
    def make_json_serializable(obj):
        import torch.nn as nn
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(v) for v in obj]
        elif isinstance(obj, type):
            # If it's a type, check if it's a nn.Module subclass
            if issubclass(obj, nn.Module):
                return obj.__name__
            else:
                return obj.__name__
        elif isinstance(obj, nn.Module):
            return obj.__class__.__name__
        else:
            return obj

    config_dict = {
        "environment": env_config,
        algo_name: algo_config,
        "save": save_config,
        "seed": seed
    }
    config_dict = make_json_serializable(config_dict)
    config_path = os.path.join(save_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Train PPO, SAC, and TD3 models for Figure 4.")
    parser.add_argument("--exp-suffix", type=str, default="", help="Optional suffix for experiment group names.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb-offline", action="store_true", help="Run W&B in offline mode.")
    parser.add_argument("--record-video", action="store_true", help="Enable video recording of evaluation episodes during these trainings.")
    args = parser.parse_args()

    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    save_config = copy.deepcopy(global_save_config_template)
    env_config = copy.deepcopy(global_env_config_template)

    for algo in ALGORITHMS:
        base_exp_name = f"Fig4_{algo.upper()}"
        if args.exp_suffix:
            base_exp_name = f"{base_exp_name}_{args.exp_suffix}"
        for seed in SEEDS_FOR_TRAINING:
            run_name = f"{base_exp_name}_seed{seed}"
            print(f"--- Training {algo.upper()} for seed {seed} ---")
            os.makedirs(save_config["base_log_path"], exist_ok=True)

            # W&B setup (mimic train.py)
            use_wandb_flag = not args.no_wandb
            wandb_offline_flag = args.wandb_offline
            wandb_project_name = os.getenv("WANDB_PROJECT_FIG4", "sirs-rl-fig4")
            wandb_group_name = base_exp_name
            current_wandb_run = None
            if use_wandb_flag:
                env_config_for_wandb = env_config.copy()
                env_config_for_wandb["random_seed"] = seed
                if algo == "ppo":
                    algo_config_for_wandb = ppo_config.copy()
                elif algo == "sac":
                    algo_config_for_wandb = sac_config.copy()
                elif algo == "a2c":
                    algo_config_for_wandb = a2c_config.copy()
                if "policy_kwargs" in algo_config_for_wandb and "activation_fn" in algo_config_for_wandb["policy_kwargs"]:
                    act_fn = algo_config_for_wandb["policy_kwargs"]["activation_fn"]
                    if isinstance(act_fn, type):
                        algo_config_for_wandb["policy_kwargs"]["activation_fn"] = act_fn.__name__
                    elif not isinstance(act_fn, str):
                        algo_config_for_wandb["policy_kwargs"]["activation_fn"] = act_fn.__class__.__name__
                current_wandb_run = wandb.init(
                    project=wandb_project_name,
                    name=run_name,
                    group=wandb_group_name,
                    config={
                        "environment": env_config_for_wandb,
                        algo.upper(): algo_config_for_wandb,
                        "save": save_config,
                        "seed": seed
                    },
                    settings=wandb.Settings(init_timeout=120, sync_tensorboard=True),
                    reinit=True
                )
                if wandb_offline_flag:
                    print(f"\nRunning W&B in offline mode for {run_name}. Run 'wandb sync {current_wandb_run.dir}' to sync.")

            # Environment setup
            if algo == "ppo":
                env_fns = [make_env_for_algo(env_config, seed + i, algo) for i in range(ppo_config["n_envs"])]
                vec_env = SubprocVecEnv(env_fns)
                vec_env = VecMonitor(vec_env, os.path.join(save_config["base_log_path"], run_name, "monitor"))
                vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
                model_kwargs = {k: v for k, v in ppo_config.items() if k not in ["policy_type", "total_timesteps", "n_envs"]}
                total_timesteps = ppo_config["total_timesteps"]
                policy_type = ppo_config["policy_type"]
            elif algo == "sac":
                env_fns = [make_env_for_algo(env_config, seed + i, algo) for i in range(sac_config["n_envs"])]
                vec_env = SubprocVecEnv(env_fns)
                vec_env = VecMonitor(vec_env, os.path.join(save_config["base_log_path"], run_name, "monitor"))
                vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
                model_kwargs = {k: v for k, v in sac_config.items() if k not in ["policy_type", "total_timesteps", "n_envs"]}
                total_timesteps = sac_config["total_timesteps"]
                policy_type = sac_config["policy_type"]
            elif algo == "a2c":
                env_fns = [make_env_for_algo(env_config, seed + i, algo) for i in range(a2c_config["n_envs"])]
                vec_env = SubprocVecEnv(env_fns)
                vec_env = VecMonitor(vec_env, os.path.join(save_config["base_log_path"], run_name, "monitor"))
                vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
                model_kwargs = {k: v for k, v in a2c_config.items() if k not in ["policy_type", "total_timesteps", "n_envs"]}
                total_timesteps = a2c_config["total_timesteps"]
                policy_type = a2c_config["policy_type"]
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

            # Before model = PPO(...), SAC(...), TD3(...)
            if "policy_kwargs" in model_kwargs and "activation_fn" in model_kwargs["policy_kwargs"]:
                model_kwargs["policy_kwargs"]["activation_fn"] = get_activation_fn(model_kwargs["policy_kwargs"]["activation_fn"])

            # Model setup
            if algo == "ppo":
                model = PPO(
                    policy=policy_type,
                    env=vec_env,
                    verbose=save_config["verbose"],
                    tensorboard_log=os.path.join(save_config["base_log_path"], run_name, "tensorboard"),
                    seed=seed,
                    **model_kwargs
                )
            elif algo == "sac":
                model = SAC(
                    policy=policy_type,
                    env=vec_env,
                    verbose=save_config["verbose"],
                    tensorboard_log=os.path.join(save_config["base_log_path"], run_name, "tensorboard"),
                    seed=seed,
                    **model_kwargs
                )
            elif algo == "a2c":
                model = A2C(
                    policy=policy_type,
                    env=vec_env,
                    verbose=save_config["verbose"],
                    tensorboard_log=os.path.join(save_config["base_log_path"], run_name, "tensorboard"),
                    seed=seed,
                    **model_kwargs
                )
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

            # Callbacks
            callbacks = []
            checkpoint_callback = CheckpointCallback(
                save_freq=save_config["save_freq"],
                save_path=os.path.join(save_config["base_log_path"], run_name),
                name_prefix=f"sirs_model_{algo}",
                save_replay_buffer=save_config["save_replay_buffer"],
                save_vecnormalize=True
            )
            callbacks.append(checkpoint_callback)
            eval_freq = save_config.get("eval_freq", 0)
            if eval_freq > 0:
                eval_env = SubprocVecEnv([make_env_for_algo(env_config, seed, algo)])
                eval_env = VecMonitor(eval_env, os.path.join(save_config["base_log_path"], run_name, "eval"))
                eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
                eval_callback = EvalCallback(
                    eval_env,
                    best_model_save_path=os.path.join(save_config["base_log_path"], run_name),
                    log_path=os.path.join(save_config["base_log_path"], run_name),
                    eval_freq=eval_freq,
                    n_eval_episodes=10,
                    deterministic=True,
                    render=False
                )
                callbacks.append(eval_callback)
            log_path = os.path.join(save_config["base_log_path"], run_name, "training_returns.csv")
            callbacks.append(EpisodeReturnLogger(log_path, algo.upper(), seed))
            # Add EntropyCoefCallback for PPO only, matching train.py
            if algo == "ppo":
                callbacks.append(EntropyCoefCallback(
                    initial_value=ppo_config["ent_coef"],
                    final_value=0.002,
                    schedule_percentage=0.4,
                    verbose=1
                ))
            # Add WandbCallback if enabled
            if use_wandb_flag:
                callbacks.append(WandbCallback(
                    model_save_path=os.path.join(save_config["base_log_path"], run_name, "wandb_models"),
                    gradient_save_freq=100_000,
                    verbose=2
                ))

            # Training
            try:
                print(f"Starting training for {run_name} with {total_timesteps} timesteps...")
                model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
            except KeyboardInterrupt:
                print(f"\nTraining interrupted for {run_name}. Saving model...")
            finally:
                model.save(os.path.join(save_config["base_log_path"], run_name, f"final_model_{algo}"))
                vec_env.save(os.path.join(save_config["base_log_path"], run_name, "vecnormalize.pkl"))
                vec_env.close()
                # Finish wandb run if started
                if use_wandb_flag and current_wandb_run:
                    if wandb.run is not None and wandb.run.id == current_wandb_run.id:
                        wandb.finish()
                print(f"--- Finished run for {algo.upper()} SEED: {seed} ---")

            # Save config.json for this run
            save_config_with_model(
                os.path.join(save_config["base_log_path"], run_name),
                env_config,
                a2c_config if algo == "a2c" else ppo_config if algo == "ppo" else sac_config if algo == "sac" else None,
                save_config,
                seed,
                algo.upper()
            )

if __name__ == "__main__":
    main() 