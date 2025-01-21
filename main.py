# In your main simulation file
from utils.visualization import EnvironmentRenderer
from environment import Environment
import random

# Initialize environment parameters
grid_size = 20
n_sick_humans = 5
n_healthy_humans = 30
n_viruses = 0
max_timesteps = 10
lethality = 0.1

# Create environment
env = Environment(
    grid_size=grid_size,
    n_sick_humans=n_sick_humans,
    n_healthy_humans=n_healthy_humans,
    n_viruses=n_viruses,
    max_timesteps=max_timesteps,
    lethality=lethality
)

# Initialize renderer
renderer = EnvironmentRenderer(save_dir='simulation_renders')

# Reset environment
env.reset()

# During simulation
for timestep in range(max_timesteps):
    # Create empty actions dictionary for all humans
    actions = {human.id: (random.randint(0, 4), random.randint(0, 10)) 
              for human in env.humans if human.alive}
    
    # Your simulation step
    observations, rewards, done = env.step(actions)
    
    # Render and save frame
    renderer.render_frame(env, timestep, show=False, save=True)
    
    if done:
        break

# Create animation after simulation
renderer.create_animation(output_file='simulation_run.gif')