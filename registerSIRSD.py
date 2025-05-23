from gymnasium.envs.registration import register

register(
    id='SIRSD-v0',
    entry_point='environment:SIRSDEnvironment',
)