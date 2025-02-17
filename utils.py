import random
import numpy as np
from typing import Tuple
import math

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
            movement_type: One of ["stationary", "discrete_random", "continuous_random", "circular_formation"]
        """
        self.grid_size = grid_size
        self.movement_type = movement_type
        self.rounding_digits = rounding_digits
        
        # Movement parameters for continuous random
        self.momentum = 0.8  # How much previous velocity affects current velocity
        self.max_acceleration = 0.2  # Maximum change in velocity per step
        self.max_velocity = 1.0  # Maximum velocity magnitude
        
        # Store velocities for continuous random movement
        self.velocities = {}  # Dictionary to store velocities for each human
        
        # Validate movement type
        valid_types = ["stationary", "discrete_random", "continuous_random", "circular_formation"]
        if movement_type not in valid_types:
            raise ValueError(f"Movement type must be one of {valid_types}")

    def initialize_positions(self, n_humans: int, rng: np.random.Generator, n_infected: int = 0, safe_distance: float = 0) -> list:
        """
        Initialize positions for all humans based on movement type
        
        Args:
            n_humans: Number of humans to place
            rng: Random number generator for reproducibility
            n_infected: Number of humans that will be infected (for safe distance initialization)
            safe_distance: Minimum distance required between agent and infected humans
            
        Returns:
            List of tuples (x, y) for each human's initial position
        """
        if self.movement_type == "circular_formation":
            return self._initialize_circular_positions(n_humans)
        else:
            positions = self._initialize_random_positions(n_humans, rng)
            
            # If we need to ensure safe distance for infected humans
            if n_infected > 0 and safe_distance > 0:
                agent_pos = (self.grid_size // 2, self.grid_size // 2)  # Agent starts at center
                
                # Separate positions into infected and non-infected
                infected_positions = []
                non_infected_positions = []
                
                # Keep trying to find valid positions for infected humans
                while len(infected_positions) < n_infected:
                    for pos in positions:
                        # Calculate minimum distance considering periodic boundaries
                        min_dist = float('inf')
                        for dx in [-self.grid_size, 0, self.grid_size]:
                            for dy in [-self.grid_size, 0, self.grid_size]:
                                wrapped_pos = (pos[0] + dx, pos[1] + dy)
                                dist = math.sqrt((wrapped_pos[0] - agent_pos[0])**2 + 
                                               (wrapped_pos[1] - agent_pos[1])**2)
                                min_dist = min(min_dist, dist)
                        
                        if min_dist >= safe_distance and len(infected_positions) < n_infected:
                            infected_positions.append(pos)
                        else:
                            non_infected_positions.append(pos)
                    
                    # If we don't have enough infected positions, generate more
                    if len(infected_positions) < n_infected:
                        new_positions = self._initialize_random_positions(n_infected - len(infected_positions), rng)
                        positions.extend(new_positions)
                
                # Combine positions with infected first (to maintain order)
                positions = infected_positions + non_infected_positions
            
            # Initialize velocities for continuous random movement
            if self.movement_type == "continuous_random":
                for i in range(n_humans):
                    # Initialize random velocities
                    vx = rng.uniform(-self.max_velocity, self.max_velocity)
                    vy = rng.uniform(-self.max_velocity, self.max_velocity)
                    # Normalize if speed is too high
                    speed = math.sqrt(vx**2 + vy**2)
                    if speed > self.max_velocity:
                        vx = (vx / speed) * self.max_velocity
                        vy = (vy / speed) * self.max_velocity
                    self.velocities[i] = [vx, vy]
            
            return positions

    def _initialize_random_positions(self, n_humans: int, rng: np.random.Generator) -> list:
        """Initialize random positions for humans"""
        positions = set()
        result = []
        
        while len(positions) < n_humans:
            x = rng.uniform(0, self.grid_size)
            y = rng.uniform(0, self.grid_size)
            x = round(x, self.rounding_digits)
            y = round(y, self.rounding_digits)
            if (x, y) not in positions:
                positions.add((x, y))
                result.append((x, y))
        
        return result

    def _initialize_circular_positions(self, n_humans: int) -> list:
        """
        Initialize positions in a circular formation with equal dispersion.
        Humans are placed at equal angles around the circle, with a random initial rotation
        to avoid always starting at the same positions.
        """
        positions = []
        center_x = self.grid_size // 2
        center_y = self.grid_size // 2
        radius = 20  # Fixed radius for consistent spacing
        
        # Add a random initial rotation to avoid always starting at the same point
        initial_angle = np.random.uniform(0, 2 * np.pi)
        
        # Calculate angle between each human (equal spacing)
        angle_step = 2 * np.pi / n_humans
        
        for i in range(n_humans):
            # Calculate angle for this human (with random initial offset)
            angle = initial_angle + i * angle_step
            
            # Convert polar coordinates to Cartesian
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            
            # Ensure positions stay within bounds and round them
            x = round(x % self.grid_size, self.rounding_digits)
            y = round(y % self.grid_size, self.rounding_digits)
            positions.append((x, y))
        
        return positions

    def _continuous_random_move(self, x: float, y: float, human_id: int, rng: np.random.Generator) -> Tuple[float, float]:
        """
        Random movement in continuous steps with momentum and smooth acceleration
        """
        if human_id not in self.velocities:
            # Initialize velocity if not exists
            vx = rng.uniform(-self.max_velocity, self.max_velocity)
            vy = rng.uniform(-self.max_velocity, self.max_velocity)
            self.velocities[human_id] = [vx, vy]
        
        # Get current velocity
        vx, vy = self.velocities[human_id]
        
        # Add random acceleration
        ax = rng.uniform(-self.max_acceleration, self.max_acceleration)
        ay = rng.uniform(-self.max_acceleration, self.max_acceleration)
        
        # Update velocity with momentum
        vx = self.momentum * vx + (1 - self.momentum) * ax
        vy = self.momentum * vy + (1 - self.momentum) * ay
        
        # Limit velocity magnitude
        speed = math.sqrt(vx**2 + vy**2)
        if speed > self.max_velocity:
            vx = (vx / speed) * self.max_velocity
            vy = (vy / speed) * self.max_velocity
        
        # Store new velocity
        self.velocities[human_id] = [vx, vy]
        
        # Update position
        new_x = (x + vx) % self.grid_size
        new_y = (y + vy) % self.grid_size
        
        return round(new_x, self.rounding_digits), round(new_y, self.rounding_digits)

    def get_new_position(self, x: float, y: float, rng: np.random.Generator, human_id: int = None) -> Tuple[float, float]:
        """
        Get new position based on current position and movement type
        
        Args:
            x: Current x position
            y: Current y position
            rng: Random number generator for reproducibility
            human_id: ID of the human (needed for continuous random movement)
            
        Returns:
            Tuple of (new_x, new_y)
        """
        if self.movement_type == "stationary":
            return self._stationary_move(x, y)
        elif self.movement_type == "discrete_random":
            return self._discrete_random_move(x, y, rng)
        elif self.movement_type == "continuous_random":
            if human_id is None:
                raise ValueError("human_id is required for continuous random movement")
            return self._continuous_random_move(x, y, human_id, rng)
        elif self.movement_type == "circular_formation":
            return self._circular_formation_move(x, y, rng)
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

    def _circular_formation_move(self, x: int, y: int, rng: np.random.Generator) -> Tuple[int, int]:
        """
        Maintain position in a circular formation around the center agent.
        The agent is assumed to be at (grid_size//2, grid_size//2).
        Each human maintains their relative angle to ensure equal dispersion.
        """
        # Center point (agent's position)
        center_x = self.grid_size // 2
        center_y = self.grid_size // 2
        
        # Calculate current angle of the human relative to center
        dx = x - center_x
        dy = y - center_y
        current_angle = np.arctan2(dy, dx)
        
        # Fixed radius for consistent spacing
        radius = 20
        
        # Calculate new position maintaining the same angle
        new_x = center_x + radius * np.cos(current_angle)
        new_y = center_y + radius * np.sin(current_angle)
        
        # Ensure we stay within bounds and round to specified digits
        new_x = round(new_x % self.grid_size, self.rounding_digits)
        new_y = round(new_y % self.grid_size, self.rounding_digits)
        
        return new_x, new_y


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