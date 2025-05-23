import os
import argparse
import json
from datetime import datetime
from typing import Dict, Any, Callable
import random
import numpy as np
import torch 

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
import wandb
from wandb.integration.sb3 import WandbCallback
import imageio
from environment import SIRSDEnvironment
from config import env_config, ppo_config, save_config

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
        
        if self.verbose > 0 and self.n_calls % 100_000 == 0: 
            self.logger.record("train/ent_coef", self.current_value)
            
        return True

def set_global_seeds(seed: int) -> None:
    """
    Set all seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Set global seed to: {seed}")

class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env, video_folder, eval_freq=1000, use_wandb_logging=False):
        super().__init__()
        self.eval_env = eval_env
        self.video_folder = video_folder
        self.eval_freq = eval_freq
        self.use_wandb_logging = use_wandb_logging
        os.makedirs(video_folder, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            obs = self.eval_env.reset()
            
            env = self.eval_env.envs[0]
            frames = []
            done = False
            
            initial_frame = env.render()
            if initial_frame is not None:
                frames.append(initial_frame)
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, info = self.eval_env.step(action)
                
                done = terminated[0] or info[0].get('TimeLimit.truncated', False)
                
                if not done:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
            
            if frames:
                video_path = os.path.join(self.video_folder, f"eval_episode_{self.n_calls}_steps.mp4")
                if "render_fps" not in env.metadata:
                    print("Warning: 'render_fps' not in environment metadata. Using default FPS 10.")
                    env_fps = 10
                else:
                    env_fps = env.metadata["render_fps"]
                imageio.mimsave(video_path, frames, fps=env_fps)
                
                if self.use_wandb_logging and wandb.run is not None:
                    try:
                        wandb.log({
                            f"videos/eval_episode_{self.n_calls}": wandb.Video(video_path, caption=f"Evaluation at {self.n_calls} steps", fps=env_fps)
                        }, step=self.model.num_timesteps)
                        print(f"Logged evaluation video to W&B at step {self.model.num_timesteps}")
                    except Exception as e:
                        print(f"Error logging video to W&B: {e}")
        return True

def make_env(env_config_dict: Dict[str, Any], seed: int = 0) -> Callable:
    """Create a wrapped, monitored SIRS environment."""
    def _init() -> SIRSDEnvironment:
        env = SIRSDEnvironment(**env_config_dict)
        env.reset(seed=seed)
        return env
    return _init

def make_eval_env(env_config_dict: Dict[str, Any], seed: int = 0, record_video: bool = False) -> SIRSDEnvironment:
    """Create an evaluation environment with rendering enabled only if record_video is True."""
    eval_config = env_config_dict.copy()
    if record_video:
        eval_config["render_mode"] = "rgb_array"
    else:
        eval_config["render_mode"] = None
    env = SIRSDEnvironment(**eval_config)
    env.reset(seed=seed)
    return env

def setup_wandb(config: Dict[str, Any], run_name: str) -> None:
    """Initialize wandb with all configs."""
    wandb_settings = wandb.Settings(
        init_timeout=120, 
        sync_tensorboard=True,
    )
    
    wandb.init(
        project="sirs-rl",
        name=run_name,
        config={
            "environment": config["environment"],
            "ppo": config["ppo"],
            "save": config["save"]
        },
        settings=wandb_settings
    )

def save_config_with_model(save_path: str, config_dict: Dict[str, Any]) -> None:
    """Save the configuration alongside the model."""
    config_path = os.path.join(save_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)

def get_activation_fn(activation_str_or_fn: Any) -> torch.nn.Module:
    """Convert activation function string to PyTorch activation function."""
    if isinstance(activation_str_or_fn, type) and issubclass(activation_str_or_fn, torch.nn.Module):
        return activation_str_or_fn
    if isinstance(activation_str_or_fn, torch.nn.Module):
        return activation_str_or_fn.__class__
    if isinstance(activation_str_or_fn, str):
        activation_map = {"relu": torch.nn.ReLU, "tanh": torch.nn.Tanh, "sigmoid": torch.nn.Sigmoid, "leaky_relu": torch.nn.LeakyReLU, "elu": torch.nn.ELU}
        if activation_str_or_fn.lower() not in activation_map:
            raise ValueError(f"Unsupported activation function: {activation_str_or_fn}")
        return activation_map[activation_str_or_fn.lower()]
    raise TypeError(f"activation_fn must be a string or a torch.nn.Module, not {type(activation_str_or_fn)}")

def execute_single_training_run(
    current_seed: int,
    run_name: str,
    log_path_base: str,
    effective_env_config: dict,
    effective_ppo_config: dict,
    effective_save_config: dict,
    should_record_video_flag: bool,
    use_wandb_flag: bool,
    wandb_offline_flag: bool,
    wandb_project_name: str,
    wandb_group_name: str
):
    set_global_seeds(current_seed)

    log_path = os.path.join(log_path_base, run_name)
    os.makedirs(log_path, exist_ok=True)
    tensorboard_path = os.path.join(log_path, "tensorboard")
    video_folder = os.path.join(log_path, "videos")
    os.makedirs(tensorboard_path, exist_ok=True)
    os.makedirs(video_folder, exist_ok=True)
    
    current_wandb_run = None
    if use_wandb_flag:
        env_config_for_wandb = effective_env_config.copy()
        env_config_for_wandb["random_seed"] = current_seed
        
        ppo_config_for_wandb = effective_ppo_config.copy()
        if "policy_kwargs" in ppo_config_for_wandb and "activation_fn" in ppo_config_for_wandb["policy_kwargs"]:
            act_fn = ppo_config_for_wandb["policy_kwargs"]["activation_fn"]
            if isinstance(act_fn, type):
                ppo_config_for_wandb["policy_kwargs"]["activation_fn"] = act_fn.__name__
            elif isinstance(act_fn, str):
                pass
            else:
                ppo_config_for_wandb["policy_kwargs"]["activation_fn"] = act_fn.__class__.__name__


        current_wandb_run = wandb.init(
            project=wandb_project_name,
            name=run_name,
            group=wandb_group_name,
            config={
                "environment": env_config_for_wandb,
                "ppo": ppo_config_for_wandb,
                "save": effective_save_config,
                "seed": current_seed
            },
            settings=wandb.Settings(init_timeout=120, sync_tensorboard=True),
            reinit=True
        )
        if wandb_offline_flag:
            print(f"\nRunning W&B in offline mode for {run_name}. Run 'wandb sync {current_wandb_run.dir}' to sync.")

    env_fns = [make_env(effective_env_config, seed=current_seed + i) for i in range(effective_ppo_config["n_envs"])]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env, os.path.join(log_path, "monitor"))
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    eval_env = None
    eval_freq = effective_save_config.get("eval_freq", 0)
    if eval_freq > 0:
        eval_env = make_eval_env(effective_env_config, seed=current_seed, record_video=should_record_video_flag)
        eval_env = DummyVecEnv([lambda: eval_env])
        eval_env = VecMonitor(eval_env, os.path.join(log_path, "eval"))
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
        if vec_env.ret_rms is not None:
            eval_env.ret_rms = vec_env.ret_rms
        eval_env.training = False

    callbacks = []
    checkpoint_callback = CheckpointCallback(
        save_freq=effective_save_config["save_freq"], save_path=log_path, name_prefix="sirs_model",
        save_replay_buffer=effective_save_config["save_replay_buffer"], save_vecnormalize=True
    )
    callbacks.append(checkpoint_callback)

    if eval_freq > 0 and eval_env is not None:
        eval_callback = EvalCallback(
            eval_env, best_model_save_path=log_path, log_path=log_path, eval_freq=eval_freq,
            n_eval_episodes=20, deterministic=True, render=False
        )
        callbacks.append(eval_callback)
        if should_record_video_flag:
            video_recorder = VideoRecorderCallback(
                eval_env, video_folder, eval_freq=eval_freq, use_wandb_logging=use_wandb_flag
            )
            callbacks.append(video_recorder)

    if use_wandb_flag:
        callbacks.append(WandbCallback(
            model_save_path=os.path.join(log_path, "wandb_models"),
            gradient_save_freq=effective_ppo_config.get("gradient_save_freq", 100_000),
            verbose=2
        ))

    model_ppo_config_run = effective_ppo_config.copy()
    
    if "policy_kwargs" in model_ppo_config_run:
        model_ppo_config_run["policy_kwargs"] = model_ppo_config_run["policy_kwargs"].copy()
        if "activation_fn" in model_ppo_config_run["policy_kwargs"]:
            act_fn_input = model_ppo_config_run["policy_kwargs"]["activation_fn"]
            model_ppo_config_run["policy_kwargs"]["activation_fn"] = get_activation_fn(act_fn_input)

    if "ent_coef" in model_ppo_config_run and isinstance(model_ppo_config_run["ent_coef"], (float, int)):
        ent_callback = EntropyCoefCallback(
            initial_value=model_ppo_config_run["ent_coef"], final_value=0.002, schedule_percentage=0.4, verbose=1
        )
        callbacks.append(ent_callback)
    
    serializable_ppo_config = effective_ppo_config.copy()
    if "policy_kwargs" in serializable_ppo_config and "activation_fn" in serializable_ppo_config["policy_kwargs"]:
        act_fn = serializable_ppo_config["policy_kwargs"]["activation_fn"]
        if isinstance(act_fn, type):
            serializable_ppo_config["policy_kwargs"]["activation_fn"] = act_fn.__name__
        elif not isinstance(act_fn, str):
            serializable_ppo_config["policy_kwargs"]["activation_fn"] = act_fn.__class__.__name__


    config_to_save = {
        "environment": effective_env_config, "ppo": serializable_ppo_config, 
        "save": effective_save_config, "seed": current_seed
    }
    save_config_with_model(log_path, config_to_save)
    
    model = PPO(
        policy=model_ppo_config_run["policy_type"],
        env=vec_env,
        verbose=effective_save_config["verbose"],
        tensorboard_log=tensorboard_path,
        seed=current_seed,
        **{k: v for k, v in model_ppo_config_run.items() if k not in ["policy_type", "total_timesteps", "n_envs"]}
    )

    try:
        print(f"Starting training for {run_name} with {model_ppo_config_run['total_timesteps']} timesteps...")
        model.learn(
            total_timesteps=model_ppo_config_run["total_timesteps"],
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print(f"\nTraining interrupted for {run_name}. Saving model...")
    finally:
        model.save(os.path.join(log_path, "final_model"))
        vec_env.save(os.path.join(log_path, "vecnormalize.pkl"))
        if use_wandb_flag and current_wandb_run:
            if wandb.run is not None and wandb.run.id == current_wandb_run.id:
                wandb.finish()
        vec_env.close()
        if eval_env:
            eval_env.close()
        print(f"# Finished run for SEED: {current_seed} for {run_name} #")

def main(args):
    current_env_config = env_config.copy()
    current_ppo_config = ppo_config.copy()
    current_save_config = save_config.copy()

    if args.config != 'config.py':
        print(f"Loading custom configuration from: {args.config}")
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_config", args.config)
        custom_config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_config_module)
        current_env_config = getattr(custom_config_module, 'env_config', current_env_config)
        current_ppo_config = getattr(custom_config_module, 'ppo_config', current_ppo_config)
        current_save_config = getattr(custom_config_module, 'save_config', current_save_config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    reward_type_from_config = current_env_config.get("reward_type", "unknown_reward_type") 
    
    if args.exp_name:
        base_run_name_for_group = args.exp_name
    else:
        base_run_name_for_group = f"{reward_type_from_config}_{timestamp}"

    SEEDS_TO_RUN = [1, 2, 3]
    
    os.makedirs(current_save_config["base_log_path"], exist_ok=True)

    for seed_val in SEEDS_TO_RUN:
        seed_specific_run_name = f"{base_run_name_for_group}_seed{seed_val}"
        print(f"# Preparing training for: {seed_specific_run_name} (Seed: {seed_val}) #")
        
        execute_single_training_run(
            current_seed=seed_val,
            run_name=seed_specific_run_name,
            log_path_base=current_save_config["base_log_path"],
            effective_env_config=current_env_config,
            effective_ppo_config=current_ppo_config,
            effective_save_config=current_save_config,
            should_record_video_flag=args.record_video,
            use_wandb_flag=not args.no_wandb,
            wandb_offline_flag=args.wandb_offline,
            wandb_project_name=os.getenv("WANDB_PROJECT", "sirs-rl"),
            wandb_group_name=base_run_name_for_group
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a SIRS environment agent using PPO.")
    parser.add_argument("--exp-name", type=str, default="", help="Optional experiment name prefix for grouping runs.")
    parser.add_argument("--config", type=str, default="config.py", help="Path to Python config file (e.g., 'config.py').")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb-offline", action="store_true", help="Run W&B in offline mode.")
    parser.add_argument("--record-video", action="store_true", help="Enable video recording of evaluation episodes.")
    
    cli_args = parser.parse_args()
    
    if not cli_args.no_wandb and cli_args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
        print("Using W&B in offline mode.")
        
    main(cli_args) 