from gymnasium.envs.registration import register
# Register the environment with Gymnasium
register(
    id='SIRSD-v0',
    entry_point='environment:SIRSDEnvironment',
)