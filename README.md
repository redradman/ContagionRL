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
```

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

### Model Files

After training, you'll find the following in your log directory (`logs/your_run_name/`):
- `final_model.zip`: The final trained model
- `best_model.zip`: The best model according to evaluation metrics
- `config.json`: Configuration used for training
- `videos/`: Directory containing evaluation videos
- Checkpoint files saved during training (e.g., `sirs_model_20000_steps.zip`)
