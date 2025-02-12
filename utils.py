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
    def __init__(self, grid_size: int, movement_type: str = "stationary", rounding_digits: int = 2):
        """
        Initialize movement handler
        
        Args:
            grid_size: Size of the environment grid
            movement_type: One of ["stationary", "discrete_random", "continuous_random"]
        """
        self.grid_size = grid_size
        self.movement_type = movement_type
        self.rounding_digits = rounding_digits
        
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
        elif self.movement_type == "continuous_random":  # continuous_random
            return self._continuous_random_move(x, y, rng)
        else:
            raise ValueError(f"Invalid movement type: {self.movement_type}")

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
        new_x = (x + dx) % self.grid_size
        new_y = (y + dy) % self.grid_size
        
        return new_x, new_y

    def _continuous_random_move(self, x: int, y: int, rng: np.random.Generator) -> Tuple[int, int]:
        """
        Random movement in continuous steps [-1, 1] for both x and y
        """
        dx = rng.uniform(-1, 1)
        dy = rng.uniform(-1, 1)
        
        # Ensure we stay within bounds
        new_x = (x + dx) % self.grid_size
        new_y = (y + dy) % self.grid_size
        
        return round(new_x, self.rounding_digits), round(new_y, self.rounding_digits)


######## Human class ########
class Human:
    _next_id = 1  # Class variable to keep track of the next available ID
    ## id 0 or negative is reserved for the agent
    def __init__(self, x: int, y: int, state: int = STATE_DICT['S'], id: int = None, time_in_state: int = 0):
        """
        Initialize a human in the SIRS model
        Args:
            x: x position
            y: y position
            state: integer representing state (0: Susceptible, 1: Infected, 2: Recovered, 3: Dead)
            id: optional manual ID assignment. If None, auto-assigns next available positive ID.
                Negative IDs are reserved for special cases (e.g., agent)
            time_in_state: time spent in current state
        """
        if id is None:
            self.id = Human._next_id
            Human._next_id += 1
        else:
            self.id = id  # Manual ID assignment (used for agent)
            
        self.x = x
        self.y = y
        self.state = state
        self.time_in_state = time_in_state
        assert self.state in STATE_DICT.values() # make sure no invalid state is passed in

    def move(self, new_x: int, new_y: int, grid_size: int):
        """Move human to new position within grid bounds"""
        assert new_x >= 0 and new_x <= grid_size, "new_x is out of bounds"
        assert new_y >= 0 and new_y <= grid_size, "new_y is out of bounds"
        self.x = new_x
        self.y = new_y

    def update_state(self, new_state: int):
        """Update state and reset time counter"""
        self.state = new_state
        self.time_in_state = 0