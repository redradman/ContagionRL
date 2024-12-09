# import numpy as np
import random
import matplotlib.pyplot as plt
from agents import Human
from replay_buffer import ReplayBuffer

class Environment:
    def __init__(
        self,
        grid_size: int,
        n_sick_humans: int,
        n_healthy_humans: int,
        n_viruses: int, # not used yet
        max_timesteps: int = 1000,
        lethality: float = 0.1,
        transition_probs=None,
    ):
        self.grid_size = grid_size
        self.n_sick_humans = n_sick_humans
        self.n_healthy_humans = n_healthy_humans
        self.n_viruses = n_viruses
        self.max_timesteps = max_timesteps
        self.lethality = lethality
        self.transition_probs = transition_probs or {
            'I_to_R': 0.1,
            'R_to_S': 0.05
        }

        self.replay_buffer = ReplayBuffer(max_size=10000, batch_size=32)
        self.humans = []
        self.timestep = 0

    def reset(self):
        self.humans = []
        half_grid = self.grid_size // 2
        # Initialize sick humans
        for _ in range(self.n_sick_humans):
            x = random.randint(-half_grid, half_grid)
            y = random.randint(-half_grid, half_grid)
            human = Human(
                x=x,
                y=y,
                is_infected=True,
                grid_size=self.grid_size,
                lethality=self.lethality,
                transition_probs=self.transition_probs
            )
            self.humans.append(human)
        # Initialize healthy humans
        for _ in range(self.n_healthy_humans):
            x = random.randint(-half_grid, half_grid)
            y = random.randint(-half_grid, half_grid)
            human = Human(
                x=x,
                y=y,
                is_infected=False,
                grid_size=self.grid_size,
                lethality=self.lethality,
                transition_probs=self.transition_probs
            )
            self.humans.append(human)
        self.timestep = 0

    def step(self, actions):
        """
        :param actions: dict mapping human id to action
        """
        # Agents take actions
        for human in self.humans:
            if human.alive:
                available_actions = human.get_action_space()
                action = actions.get(human.id, random.choice(available_actions))
                if action in available_actions:
                    human.take_action(action)
        # Handle infections
        for human in self.humans:
            if human.alive and human.state == 'I':
                for other in self.humans:
                    if other.id != human.id and other.alive:
                        human.infect(other)
        # Update agent states and reward
        for human in self.humans:
            previous_state = human.state
            human.update_state()
            if not human.alive:
                human.cumulative_reward -= 1  # Penalty already applied in update_state
            elif previous_state != 'I' and human.state == 'I':
                human.times_infected +=1
            else:
                human.cumulative_reward += 1  # Reward for surviving this timestep
        self.timestep += 1
        done = self.timestep >= self.max_timesteps or all(not h.alive for h in self.humans)
        # Collect observations and rewards
        observations = {human.id: human.observe(self.humans) for human in self.humans if human.alive}
        rewards = {human.id: human.cumulative_reward for human in self.humans}
        return observations, rewards, done

    def render(self, filename=None):
        plt.figure(figsize=(6,6))
        half_grid = self.grid_size // 2
        plt.xlim(-half_grid - 1, half_grid + 1)
        plt.ylim(-half_grid - 1, half_grid + 1)
        for human in self.humans:
            if human.alive:
                if human.state == 'S':
                    color = 'blue'
                elif human.state == 'I':
                    color = 'red'
                elif human.state == 'R':
                    color = 'green'
                plt.scatter(human.x, human.y, c=color, s=100)
        plt.grid(True)
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
        plt.close()

