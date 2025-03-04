# ContagionRL
SIRS model and RL agent research

## Usage Guide

### Training (train.py)

Train a new agent with default settings:
```bash
python train.py
```

Common training options:
```bash
# Add experiment name prefix to the logs
python train.py --exp-name my_experiment

# Enable Weights & Biases logging
python train.py --use-wandb

# Use a custom config file
python train.py --config my_config.py

# Set a specific random seed for reproducibility
python train.py --seed 12345
```

#### Reproducibility

For reproducible experiments, use the `--seed` argument to set a specific random seed:

```bash
python train.py --seed 12345
```

This ensures consistent results across runs by:
- Setting global random seeds (Python's random module, NumPy, PyTorch)
- Using sequential seeds derived from the base seed for vectorized environments
- Storing the seed value in configuration files and model metadata
- Including the seed in the run name when not using the default seed (42)

Without specifying a seed, each run will have different random initializations.

### Weights & Biases Integration

Track your training experiments with Weights & Biases:

```bash
# Basic wandb usage
python train.py --use-wandb

# Use offline mode to avoid timeout issues
python train.py --use-wandb --wandb-offline
```

#### Offline Mode

If you encounter timeout issues with wandb, use offline mode to store data locally:

```bash
python train.py --use-wandb --wandb-offline
```

This stores all data in the `wandb/` directory without requiring an internet connection during training.

#### Syncing Offline Runs

After training with offline mode, sync your data to the wandb servers:

```bash
# Sync a specific run using its timestamp ID
./sync_wandb.py --run-id 20250303_184216

# Sync all offline runs at once
./sync_wandb.py --all
```

#### Troubleshooting wandb Issues

If you encounter connection issues with wandb:

1. Use offline mode as described above
2. Check your internet connection and firewall settings
3. Try refreshing your credentials: `wandb login --relogin`
4. Update wandb: `pip install wandb --upgrade`
5. Increase timeout duration by modifying `setup_wandb()` in train.py

### Visualization (visualize.py)

Visualize a trained model:
```bash
python visualize.py --model-path logs/your_run_name/final_model.zip
```

Common visualization options:
```bash
# Record more episodes (default is 5)
python visualize.py --model-path logs/your_run_name/final_model.zip --num-episodes 10

# Use stochastic actions instead of deterministic
python visualize.py --model-path logs/your_run_name/final_model.zip --stochastic

# Specify custom output directory for videos
python visualize.py --model-path logs/your_run_name/final_model.zip --output-dir my_videos
```

You can also visualize checkpoints saved during training:
```bash
python visualize.py --model-path logs/your_run_name/sirs_model_20000_steps.zip
```

### Benchmarking (results/benchmark.py)

Compare a trained model against random actions:
```bash
python results/benchmark.py --model-path logs/your_run_name/final_model.zip
```

Common benchmarking options:
```bash
# Run more episodes for more reliable statistics
python results/benchmark.py --model-path logs/your_run_name/final_model.zip --runs 30

# Skip the random agent comparison
python results/benchmark.py --model-path logs/your_run_name/final_model.zip --no-random

# Specify a custom title for the plot
python results/benchmark.py --model-path logs/your_run_name/final_model.zip --title "My Custom Plot Title"
```

For a quick demonstration without needing a trained model:
```bash
python results/quick_test.py
```

See `results/README.md` for more detailed information on benchmarking.

### Model Files

After training, you'll find the following in your log directory (`logs/your_run_name/`):
- `final_model.zip`: The final trained model
- `best_model.zip`: The best model according to evaluation metrics
- `config.json`: Configuration used for training
- `videos/`: Directory containing evaluation videos
- Checkpoint files saved during training (e.g., `sirs_model_20000_steps.zip`)
