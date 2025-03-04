# Benchmark Tools for ContagionRL

This directory contains tools for benchmarking trained models against random action baselines.

## Benchmark Script

The `benchmark.py` script allows you to compare a trained model's performance against a random agent. It produces plots showing cumulative reward over time with standard deviation bands.

### Basic Usage

```bash
# Basic usage with a trained model
python results/benchmark.py --model-path logs/your_run_name/final_model.zip

# Running with more episodes
python results/benchmark.py --model-path logs/your_run_name/final_model.zip --runs 20

# Skipping the random agent comparison
python results/benchmark.py --model-path logs/your_run_name/final_model.zip --no-random
```

### Command Line Arguments

- `--model-path`: Path to the trained model file (.zip) [required]
- `--runs`: Number of episodes to run for each agent (default: 10)
- `--no-random`: Skip running the random agent benchmark
- `--output-dir`: Directory to save the results (default: results/graphs)
- `--seed`: Random seed for reproducibility (default: 42)
- `--title`: Custom title for the plot (default: auto-generated based on model path)

### Output

The script produces:

1. A PNG plot showing cumulative reward over time with standard deviation bands
2. A JSON file with detailed benchmark data
3. Console output with summary statistics

## Progress Tracking

The benchmark script includes progress bars using `tqdm` to provide real-time feedback on the benchmarking process. This makes it easy to monitor:

- Overall benchmark progress
- Progress through trained model episodes
- Progress through random action episodes

## Results Organization

All benchmark results are saved in the `results/graphs` subdirectory, which includes:

- Plot images (PNG format)
- Benchmark data (JSON format)

## Extending for Additional Metrics

The benchmark framework is designed to be extensible for additional metrics. The `result_utils.py` module provides reusable functions for:

- Loading models and configurations
- Running episodes with trained or random agents
- Collecting performance data
- Generating plots
- Saving results

To add new metrics, you can create additional plotting and analysis functions in `result_utils.py` and then create new benchmark scripts that leverage these utilities.

## Example Results

After running a benchmark, you'll see output like:

```
Benchmark Summary:
  Trained Model:
    Mean Episode Length: 98.50 steps (±3.21)
    Mean Final Reward: 42.78 (±5.64)
  Random Actions:
    Mean Episode Length: 45.30 steps (±12.86)
    Mean Final Reward: 15.24 (±8.72)

Results saved to results/graphs/
Plot: results/graphs/your_run_name_20230304_121542_rewards.png
Data: results/graphs/your_run_name_20230304_121542_data.json
``` 