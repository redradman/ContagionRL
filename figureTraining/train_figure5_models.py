import os
import sys
import argparse
import copy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from torch import nn
import pandas as pd
import json
import wandb
from wandb.integration.sb3 import WandbCallback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config import env_config as global_env_config_template
from config import ppo_config as global_ppo_config_template
from config import save_config as global_save_config_template
from environment import SIRSDEnvironment

SEEDS_FOR_TRAINING = [1, 2, 3]
ABLATION_VARIANTS = [
    "full", "no_magnitude", "no_direction", "no_move", "no_adherence", "no_health", "no_S"
]
VARIANT_NAMES = {
    "full":          "Full",
    "no_magnitude":  "Drop Magnitude",
    "no_direction":  "Drop Direction",
    "no_move":       "Drop Movement",
    "no_adherence":  "Drop Adherence",
    "no_health":     "Drop Health",
    "no_S":          "Drop Susceptible Repulsion",
}

def make_env_for_ablation(env_config, seed):
    config = copy.deepcopy(env_config)
    config["render_mode"] = None
    def _init():
        env = SIRSDEnvironment(**config)
        env.reset(seed=seed)
        return env
    return _init

class EpisodeReturnLogger(BaseCallback):
    def __init__(self, log_path, ablation, seed, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.ablation = ablation
        self.seed = seed
        self.returns_log = []
        self.episode_counter = 0
    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_return = info["episode"]["r"]
                self.returns_log.append({
                    "ablation": self.ablation,
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

class EntropyCoefCallback(BaseCallback):
    def __init__(self, initial_value: float, final_value: float = 0.0, schedule_percentage: float = 0.3, verbose: int = 0):
        super().__init__(verbose)
        self.initial_value = initial_value
        self.final_value = final_value
        self.schedule_percentage = schedule_percentage
        self.current_value = initial_value
    def _on_step(self) -> bool:
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
        if self.verbose > 0 and self.n_calls % 100_000 == 0:
            self.logger.record("train/ent_coef", self.current_value)
        return True

def make_json_serializable(obj):
    import torch.nn as nn
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, type) and issubclass(obj, nn.Module):
        return obj.__name__
    elif isinstance(obj, nn.Module):
        return obj.__class__.__name__
    else:
        return obj

def save_config_with_model(save_path: str, env_config: dict, algo_config: dict, save_config: dict, seed: int, ablation: str):
    config_dict = {
        "environment": env_config,
        ablation: algo_config,
        "save": save_config,
        "seed": seed
    }
    config_dict = make_json_serializable(config_dict)
    config_path = os.path.join(save_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)

def get_activation_fn(act_fn):
    if isinstance(act_fn, str):
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

def main():
    parser = argparse.ArgumentParser(description="Train PPO models for Figure 5 ablation study.")
    parser.add_argument("--exp-suffix", type=str, default="", help="Optional suffix for experiment group names.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb-offline", action="store_true", help="Run W&B in offline mode.")
    parser.add_argument("--record-video", action="store_true", help="Enable video recording of evaluation episodes during these trainings.")
    parser.add_argument("--ablations", nargs="*", default=ABLATION_VARIANTS, help="Subset of ablation variants to run.")
    args = parser.parse_args()

    save_config = copy.deepcopy(global_save_config_template)
    base_env_config = copy.deepcopy(global_env_config_template)
    ppo_cfg = copy.deepcopy(global_ppo_config_template)

    ablations_to_run = args.ablations if args.ablations else ABLATION_VARIANTS

    for ablation in ablations_to_run:
        base_exp_name = f"Fig5_{ablation}"
        if args.exp_suffix:
            base_exp_name = f"{base_exp_name}_{args.exp_suffix}"
        for seed in SEEDS_FOR_TRAINING:
            run_name = f"{base_exp_name}_seed{seed}"
            print(f"# Training PPO for ablation {ablation} seed {seed} #")
            os.makedirs(save_config["base_log_path"], exist_ok=True)

            # W&B setup
            use_wandb_flag = not args.no_wandb
            wandb_offline_flag = args.wandb_offline
            wandb_project_name = os.getenv("WANDB_PROJECT_FIG5", "sirs-rl-fig5-ablation")
            wandb_group_name = base_exp_name
            current_wandb_run = None
            if use_wandb_flag:
                env_config_for_wandb = base_env_config.copy()
                env_config_for_wandb["random_seed"] = seed
                env_config_for_wandb["reward_ablation"] = ablation
                algo_config_for_wandb = ppo_cfg.copy()
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
                        "ppo": algo_config_for_wandb,
                        "save": save_config,
                        "seed": seed,
                        "ablation": ablation
                    },
                    settings=wandb.Settings(init_timeout=120, sync_tensorboard=True),
                    reinit=True
                )
                if wandb_offline_flag:
                    print(f"\nRunning W&B in offline mode for {run_name}. Run 'wandb sync {current_wandb_run.dir}' to sync.")

            # Environment setup
            env_config = copy.deepcopy(base_env_config)
            env_config["reward_ablation"] = ablation
            env_fns = [make_env_for_ablation(env_config, seed + i) for i in range(ppo_cfg["n_envs"])]
            vec_env = SubprocVecEnv(env_fns)
            vec_env = VecMonitor(vec_env, os.path.join(save_config["base_log_path"], run_name, "monitor"))
            vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
            model_kwargs = {k: v for k, v in ppo_cfg.items() if k not in ["policy_type", "total_timesteps", "n_envs"]}
            total_timesteps = ppo_cfg["total_timesteps"]
            policy_type = ppo_cfg["policy_type"]
            if "policy_kwargs" in model_kwargs and "activation_fn" in model_kwargs["policy_kwargs"]:
                model_kwargs["policy_kwargs"]["activation_fn"] = get_activation_fn(model_kwargs["policy_kwargs"]["activation_fn"])

            # Model setup
            model = PPO(
                policy=policy_type,
                env=vec_env,
                verbose=save_config["verbose"],
                tensorboard_log=os.path.join(save_config["base_log_path"], run_name, "tensorboard"),
                seed=seed,
                **model_kwargs
            )

            # Callbacks
            callbacks = []
            checkpoint_callback = CheckpointCallback(
                save_freq=save_config["save_freq"],
                save_path=os.path.join(save_config["base_log_path"], run_name),
                name_prefix=f"sirs_model_ablation_{ablation}",
                save_replay_buffer=save_config["save_replay_buffer"],
                save_vecnormalize=True
            )
            callbacks.append(checkpoint_callback)
            eval_freq = save_config.get("eval_freq", 0)
            if eval_freq > 0:
                eval_env = SubprocVecEnv([make_env_for_ablation(env_config, seed)])
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
            callbacks.append(EpisodeReturnLogger(log_path, ablation, seed))
            callbacks.append(EntropyCoefCallback(
                initial_value=ppo_cfg["ent_coef"],
                final_value=0.002,
                schedule_percentage=0.4,
                verbose=1
            ))
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
                model.save(os.path.join(save_config["base_log_path"], run_name, f"final_model_ablation_{ablation}"))
                vec_env.save(os.path.join(save_config["base_log_path"], run_name, "vecnormalize.pkl"))
                vec_env.close()
                if use_wandb_flag and current_wandb_run:
                    if wandb.run is not None and wandb.run.id == current_wandb_run.id:
                        wandb.finish()
                print(f"# Finished run for ablation {ablation} SEED: {seed} #")

            # Save config.json for this run
            save_config_with_model(
                os.path.join(save_config["base_log_path"], run_name),
                env_config,
                ppo_cfg,
                save_config,
                seed,
                ablation
            )

if __name__ == "__main__":
    main() 