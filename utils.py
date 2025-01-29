import random
import numpy as np
from typing import Tuple

# State definitions
STATE_DICT = {
    'S': 0,  # Susceptible
    'I': 1,  # Infected
    'R': 2,  # Recovered
    'D': 3   # Dead
}

class ReplayBuffer:
    def __init__(self, max_size, batch_size):
        self.buffer = []
        self.max_size = max_size
        self.batch_size = batch_size

    def store(self, state, joint_action, reward, next_state):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append((state, joint_action, reward, next_state))

    def sample(self):
        return random.sample(self.buffer, self.batch_size)


######## Movement handler for humans in the environment ########
class MovementHandler:
    def __init__(self, grid_size: int, movement_type: str = "stationary"):
        """
        Initialize movement handler
        
        Args:
            grid_size: Size of the environment grid
            movement_type: One of ["stationary", "discrete_random", "continuous_random"]
        """
        self.grid_size = grid_size
        self.movement_type = movement_type
        
        # Validate movement type
        valid_types = ["stationary", "discrete_random", "continuous_random"]
        if movement_type not in valid_types:
            raise ValueError(f"Movement type must be one of {valid_types}")

    def get_new_position(self, x: int, y: int, rng: np.random.Generator) -> Tuple[int, int]:
        """
        Get new position based on current position and movement type
        
        Args:
            x: Current x position
            y: Current y position
            rng: Random number generator for reproducibility
            
        Returns:
            Tuple of (new_x, new_y)
        """
        if self.movement_type == "stationary":
            return self._stationary_move(x, y)
        elif self.movement_type == "discrete_random":
            return self._discrete_random_move(x, y, rng)
        else:  # continuous_random
            return self._continuous_random_move(x, y, rng)

    def _stationary_move(self, x: int, y: int) -> Tuple[int, int]:
        """Humans don't move"""
        return x, y

    def _discrete_random_move(self, x: int, y: int, rng: np.random.Generator) -> Tuple[int, int]:
        """
        Random movement in discrete steps (-1, 0, 1) for both x and y
        """
        dx = rng.integers(-1, 2)  # Random integer from [-1, 0, 1]
        dy = rng.integers(-1, 2)  # Random integer from [-1, 0, 1]
        
        # Ensure we stay within bounds
        new_x = max(0, min(x + dx, self.grid_size - 1))
        new_y = max(0, min(y + dy, self.grid_size - 1))
        
        return new_x, new_y

    def _continuous_random_move(self, x: int, y: int, rng: np.random.Generator) -> Tuple[int, int]:
        """
        Random movement in continuous steps [-1, 1] for both x and y
        """
        dx = rng.uniform(-1, 1)
        dy = rng.uniform(-1, 1)
        
        # Ensure we stay within bounds
        new_x = max(0, min(round(x + dx), self.grid_size - 1))
        new_y = max(0, min(round(y + dy), self.grid_size - 1))
        
        return new_x, new_y


######## Human class ########
class Human:
    def __init__(self, id: int, x: int, y: int, state: int = STATE_DICT['S']):
        """
        Initialize a human in the SIRS model
        state: integer representing state (0: Susceptible, 1: Infected, 2: Recovered, 3: Dead)
        """
        self.id = id
        self.x = x
        self.y = y
        self.state = state
        self.time_in_state = 0
        assert self.state in STATE_DICT.values() # make sure no invalid state is passed in

    def move(self, new_x: int, new_y: int, grid_size: int):
        """Move human to new position within grid bounds"""
        self.x = max(0, min(new_x, grid_size))
        self.y = max(0, min(new_y, grid_size))

    def update_state(self, new_state: int):
        """Update state and reset time counter"""
        self.state = new_state
        self.time_in_state = 0