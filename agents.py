
import numpy as np

class Human:
    id_counter = 1
    def __init__(self, x, y, is_infected = False, npi_adherence = 0.5, immunity_duration = 10, observation_radius = 10):
        # ID
        self.id = Human.id_counter
        Human.id_counter += 1
        # Position things
        self.x = x
        self.y = y
        # SIRS things
        self.state = "I" if is_infected else "S"
        self.npi_adherence = npi_adherence
        self.immunity_duration = immunity_duration
        # RL things
        self.cumulative_reward = 0
        self.observation_radius = observation_radius
        self.policy = None
    
    def observe(self, humans: list):
        neighbors = []
        for human in humans:
            if self.id != human.id:
                distance = np.abs(self.x - human.x) + np.abs(self.y - human.y)
                if distance <= self.observation_radius:
                    neighbors.append(human)
        return neighbors

    

