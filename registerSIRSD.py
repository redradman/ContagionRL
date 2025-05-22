from gymnasium.envs.registration import register
# Register the environment with Gymnasium
register(
    id='SIRSD-v0',
    entry_point='environment:SIRSDEnvironment', # Assuming the file is named environment.py
    # You can set a default max_episode_steps,
    # it can be overridden during env.make() as well but it is not necessary and the passed config would handle this
    # max_episode_steps=1000, # Default, will be set by simulation_time in __init__
)