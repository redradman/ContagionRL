# ContagionRL: A Flexible Platform for Learning in Different Spatial Epidemic Environments
This repository contains the codebase for ContagionRL, a platform introduced in our paper for reinforcement learning in spatial epidemic environments which unifies compartmental epidemiology and agent-based modeling (ABM).
# Requirements
Create an environment with [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html): 
```bash
conda create --name ContagionRL python=3.13.1
```

Activate the environment: 
```
conda activate ContagionRL
```

Install the dependencies: 
```bash 
pip install -r requirements.txt
```
## Licenses
The following libraries are used in ContagionRL. Their licenses and usage details are listed below. Our codebase is intended for release under BSD-3 license.

| Library           | Version | Purpose                                         | License                          |
| ----------------- | ------- | ----------------------------------------------- | -------------------------------- |
| gymnasium         | 1.0.0   | Toolkit for reinforcement learning environments | MIT License                      |
| imageio           | 2.37.0  | Image and video I/O library                     | BSD 2-Clause                     |
| matplotlib        | 3.10.3  | Comprehensive plotting library                  | Matplotlib License (BSD-based)   |
| pandas            | 2.2.3   | Data manipulation and analysis                  | BSD 3-Clause                     |
| scipy             | 1.15.3  | Advanced scientific computations                | BSD 3-Clause                     |
| seaborn           | 0.13.2  | Statistical data visualization                  | BSD 3-Clause                     |
| stable_baselines3 | 2.5.0   | Reinforcement learning algorithms               | MIT License                      |
| statsmodels       | 0.14.4  | Statistical modeling and testing                | BSD 3-Clause                     |
| torch             | 2.6.0   | Deep learning framework                         | BSD 3-Clause                     |
| tqdm              | 4.67.1  | Progress bar utility                            | MPL 2.0 (Mozilla Public License) |
| tueplots          | 0.2.0   | Publication-style plotting presets              | MIT License                      |
| wandb 	    | 0.19.8  | Experiment tracking and visualization           | MIT License                      |

# SIRS+D Environment 
The SIRS+D environment can be found as `SIRSDEnvironment` in `environment.py` file. `SIRSDEnvironment` can be used directly or using registered version with `gym.make`. The `env_config` corresponds to the environmental parameters in Table 3 of the paper.

The `SIRSDEnvironment` can be used in the default epidemic scenario as follows:
```python 
from environment import SIRSDEnvironment
env = SIRSDEnvironment()
```

Critically, to support an extensive range of epidemics, it could be configured and parameterized: 
```python
from environment import SIRSDEnvironment

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

env = SIRSDEnvironment(**env_config)
```


Alternatively, it is also possible to use the registered form of environment directly from `gymnasium`: 
```python
import registerSIRSD # handles the registeration of SIRSDEnvironment
import gymnasium as gym

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

env = gym.make('SIRSD-v0', **env_config)
```
The `registerSIRSD.py` script registers our environment with `gymnasium` under the name `SIRSD-v0`.

# Results
To regenerate our data, figures and tables in the results or additional experiments follow the instructions here. The table below shows a grouping of the figures and tables in our paper. The figures are grouped by topic (and the script `figureX.py` where `X` is an integer) that produces them. 

| Group Name                                                            | Supporting Figures and Tables | Figure Training Script    | Figure Generation Script |
| :-------------------------------------------------------------------- | :---------------------------- | :------------------------ | :----------------------- |
| **Comparison of Reinforcement Learning Algorithms Performance**       | Figure 1, Figure 6, Table 2   | `train_figure4_models.py` | `figure4.py`             |
| **Comparison of Reward Functions**                                    | Figure 2, Figure 7, Table 7   | `train_figure2_models.py` | `figure2.py`             |
| **Potential Field Reward Function Ablation Study**                    | Figure 3, Figure 8, Table 8   | `train_figure5_models.py` | `figure5.py`             |
| **Environmental Parameter Variation: Infection Rate**                 | Figure 9, Table 9             | `train_figure3_models.py` | `figure3.py`             |
| **Environmental Parameter Variation: Population density (Grid Size)** | Figure 10, Table 10           | `train_figure6_models.py` | `figure6.py`             |
| **Environmental Parameter Variation: Adherence Effectiveness**        | Figure 11, Table 11           | `train_figure7_models.py` | `figure7.py`             |
| **Environmental Parameter Variation: Distance Decay**                 | Figure 12, Table 12           | `train_figure8_models.py` | `figure8.py`             |
| **Static render of a single environment step**                 | Figure 5           | _no training needed_ | `figure_render.py`             |


The results section is divided into two parts: 
1. `figureTraining`: Handle the training of the necessary models to make the graphs. For example: `train_figure2_models.py` handles the training of the models needed to make `figure2.py`. By default, all of the models are trained across 3 seeds. 
2. `figureGeneration`: Create graphs based on the train model made by its corresponding training file. Each `figureX.py` produces multiple visualizations. The charts are saved to a `figures` directory. The tables are printed (with pretty print) into the terminal that is running the script. 

*Note: `train_figure1_models.py` and `figure1.py` are legacy scripts that are not part of the current figure generation workflow, and no figures in the paper are derived from them.*

## Recreating results
To recreate the figures in our paper, follow these instructions. Make sure that you have activated the Conda environment, [see requirements](#Requirements).
**First, train the models:** 
```python
python train_figure[X]_models.py
```
In training you may be asked to log into [W&B](https://wandb.ai/). You can skip this using `--no-wandb` flag.

**Then, recreate the figures/tables:** 
```python
python figure[X].py --model-base Fig[X] --runs 100
```
The above command would create a new folder called `figures/`, which will contain the figure(s). The tables (if applicable) will be printed to the terminal output.
- Be sure to replace `X` with an integer based on the grouping tables above. The value for `X` should be the same for both of the scripts. 
- `--model-base` flag provides a reference to which trained models in `logs` directory should be used.
- `--runs 100` is used to do 100 inferences per seed per model (as done in the paper). 
