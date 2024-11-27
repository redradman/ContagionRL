import numpy as np
from agents import Human
from replay_buffer import ReplayBuffer
class Environment:
    def __init__(self, grid_size: int, n_sick_humans: int, n_healthy_humans: int, n_viruses: int):
        self.grid_size = grid_size
        self.n_sick_humans = n_sick_humans
        self.n_healthy_humans = n_healthy_humans
        self.n_viruses = n_viruses

        self.replay_buffer = ReplayBuffer(max_size=10000, batch_size=32)
        self.humans = []

        for _ in range(n_sick_humans):
            self.humans.append(Human(np.random.randint(-grid_size, grid_size), np.random.randint(-grid_size, grid_size), is_infected=True))
        
        for _ in range(n_healthy_humans):
            self.humans.append(Human(np.random.randint(-grid_size, grid_size), np.random.randint(-grid_size, grid_size)))

