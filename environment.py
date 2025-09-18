import gymnasium as gym
import numpy as np
from typing import Optional, List, Tuple
import math
import matplotlib.pyplot as plt

STATE_DICT = {
    'S': 0,  # Susceptible
    'I': 1,  # Infected
    'R': 2,  # Recovered
    'D': 3   # Dead
}

######## Movement handler for humans in the environment ########
class MovementHandler:
    def __init__(self, grid_size: int, movement_type: str = "stationary", rounding_digits: int = 2, movement_scale: float = 1.0):
        """
        Initialize movement handler
        
        Args:
            grid_size: Size of the environment grid
            movement_type: One of ["stationary", "discrete_random", "continuous_random", "circular_formation", "workplace_home_cycle"]
            rounding_digits: Number of digits to round position coordinates to
            movement_scale: Scale factor for movement of non-focal agents (0 to 1, where 0 is no movement and 1 is full movement)
        """
        self.grid_size = grid_size
        self.movement_type = movement_type
        self.rounding_digits = rounding_digits
        self.movement_scale = max(0.0, min(1.0, movement_scale))
        self.momentum = 0.8 
        self.max_acceleration = 0.2 
        self.max_velocity = 1.0
        self.velocities = {}
        valid_types = ["stationary", "discrete_random", "continuous_random", "circular_formation", "workplace_home_cycle"]
        if movement_type not in valid_types:
            raise ValueError(f"Movement type must be one of {valid_types}")
        
        # Workplace/home cycle tracking (only used when movement_type == "workplace_home_cycle")
        self.home_locations = {}  # {human_id: (x_home, y_home)}
        self.work_locations = {}  # {human_id: (x_work, y_work)}
        self.current_targets = {}  # {human_id: 'home' or 'work'}
        self.stay_timers = {}  # {human_id: countdown_int}
        self.movement_states = {}  # {human_id: 'at_home', 'going_to_work', 'at_work', 'going_home'}
        self.movement_patterns = {}  # {human_id: 'workplace_cycle' or 'random'}

    def initialize_positions(self, n_humans: int, rng: np.random.Generator, n_infected: int = 0, safe_distance: float = 0, init_agent_distance: float = 0) -> list:
        """
        Initialize positions for all humans based on movement type
        
        Args:
            n_humans: Number of humans to place
            rng: Random number generator for reproducibility
            n_infected: Number of humans that will be infected (for safe distance initialization)
            safe_distance: Minimum distance required between agent and infected humans
            init_agent_distance: Minimum distance ALL humans should be from the agent at initialization
            
        Returns:
            List of tuples (x, y) for each human's initial position
        """
        if self.movement_type == "circular_formation":
            return self._initialize_circular_positions(n_humans, init_agent_distance)
        else:
            # Get initial random positions
            all_positions = self._initialize_random_positions(n_humans, rng)
            agent_pos = (self.grid_size // 2, self.grid_size // 2)  # Agent starts at center
            
            # If init_agent_distance is specified, ensure all humans maintain that minimum distance
            if init_agent_distance > 0:
                valid_positions = []
                
                # Filter positions to keep only those that maintain init_agent_distance
                for pos in all_positions:
                    # Calculate minimum distance considering periodic boundaries
                    min_dist = float('inf')
                    for dx in [-self.grid_size, 0, self.grid_size]:
                        for dy in [-self.grid_size, 0, self.grid_size]:
                            wrapped_pos = (pos[0] + dx, pos[1] + dy)
                            dist = math.sqrt((wrapped_pos[0] - agent_pos[0])**2 + 
                                           (wrapped_pos[1] - agent_pos[1])**2)
                            min_dist = min(min_dist, dist)
                    
                    if min_dist >= init_agent_distance:
                        valid_positions.append(pos)
                
                # Keep generating positions until we have enough valid ones
                while len(valid_positions) < n_humans:
                    # Generate a new batch of random positions
                    new_batch_size = min(n_humans * 2, n_humans * 10)  # Generate more to increase chances
                    new_positions = self._initialize_random_positions(new_batch_size, rng)
                    
                    # Filter for valid positions
                    for pos in new_positions:
                        min_dist = float('inf')
                        for dx in [-self.grid_size, 0, self.grid_size]:
                            for dy in [-self.grid_size, 0, self.grid_size]:
                                wrapped_pos = (pos[0] + dx, pos[1] + dy)
                                dist = math.sqrt((wrapped_pos[0] - agent_pos[0])**2 + 
                                               (wrapped_pos[1] - agent_pos[1])**2)
                                min_dist = min(min_dist, dist)
                        
                        if min_dist >= init_agent_distance:
                            valid_positions.append(pos)
                            
                            # If we have enough, break out
                            if len(valid_positions) >= n_humans:
                                break
                
                # Take only what we need
                all_positions = valid_positions[:n_humans]
            
            # Now handle the safe_distance requirement for infected humans if needed
            positions = all_positions
            if n_infected > 0 and safe_distance > 0:                
                # Separate positions for infected and non-infected humans
                infected_positions = []
                non_infected_positions = []
                
                # Safe distance check is only needed if it's greater than init_agent_distance
                if safe_distance > init_agent_distance:
                    # Try to find enough positions for infected humans
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
                    
                    while len(infected_positions) < n_infected:
                        new_positions = self._initialize_random_positions(n_infected - len(infected_positions), rng)
                        
                        for pos in new_positions:
                            min_dist = float('inf')
                            for dx in [-self.grid_size, 0, self.grid_size]:
                                for dy in [-self.grid_size, 0, self.grid_size]:
                                    wrapped_pos = (pos[0] + dx, pos[1] + dy)
                                    dist = math.sqrt((wrapped_pos[0] - agent_pos[0])**2 + 
                                                   (wrapped_pos[1] - agent_pos[1])**2)
                                    min_dist = min(min_dist, dist)
                            
                            if min_dist >= safe_distance:
                                infected_positions.append(pos)
                                if len(infected_positions) >= n_infected:
                                    break
                else:
                    infected_positions = positions[:n_infected]
                    non_infected_positions = positions[n_infected:]
                
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
            
            # Initialize workplace/home cycle data
            elif self.movement_type == "workplace_home_cycle":
                self._initialize_workplace_home_cycle(n_humans, rng, positions)
            
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

    def _initialize_circular_positions(self, n_humans: int, init_agent_distance: float = 0) -> list:
        """
        Initialize positions in a circular formation with equal dispersion.
        Humans are placed at equal angles around the circle, with a random initial rotation
        to avoid always starting at the same positions.
        
        Args:
            n_humans: Number of humans to place
            init_agent_distance: Minimum distance from agent. If > 0, radius will be at least this value.
        """
        positions = []
        center_x = self.grid_size // 2
        center_y = self.grid_size // 2
        
        # Use the larger of the default radius (20) or init_agent_distance
        radius = max(20, init_agent_distance) if init_agent_distance > 0 else 20
        
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
        
        # Sample base random influence from Normal(0, sigma^2) aiming for [-1, 1]
        sigma = 1.0 / 3.0 # Std dev so 3*sigma = 1.0
        rand_x = rng.normal(loc=0.0, scale=sigma)
        rand_y = rng.normal(loc=0.0, scale=sigma)

        # Clip the influence to strictly enforce the [-1, 1] range
        rand_x = np.clip(rand_x, -1.0, 1.0)
        rand_y = np.clip(rand_y, -1.0, 1.0)

        # Scale the clipped influence by max_acceleration to get the actual acceleration
        ax = rand_x * self.max_acceleration
        ay = rand_y * self.max_acceleration
        
        # Update velocity with momentum
        vx = self.momentum * vx + (1 - self.momentum) * ax
        vy = self.momentum * vy + (1 - self.momentum) * ay
        
        # Limit velocity magnitude
        speed = math.sqrt(vx**2 + vy**2)
        if speed > self.max_velocity:
            vx = (vx / speed) * self.max_velocity
            vy = (vy / speed) * self.max_velocity
        
        # Apply movement scaling factor for non-focal agents
        vx = vx * self.movement_scale
        vy = vy * self.movement_scale
        
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
                raise ValueError("human_id is required")
            return self._continuous_random_move(x, y, human_id, rng)
        elif self.movement_type == "circular_formation":
            return self._circular_formation_move(x, y, rng)
        elif self.movement_type == "workplace_home_cycle":
            if human_id is None:
                raise ValueError("human_id is required for workplace_home_cycle movement")
            return self._workplace_home_cycle_move(x, y, human_id, rng)
        else:
            raise ValueError(f"Invalid movement type: {self.movement_type}")

    def _stationary_move(self, x: int, y: int) -> Tuple[int, int]:
        """Humans don't move"""
        return x, y

    def _discrete_random_move(self, x: int, y: int, rng: np.random.Generator) -> Tuple[int, int]:
        """
        Random movement in discrete steps (-1, 0, 1) for both x and y
        """
        dx = rng.integers(-1, 2)  
        dy = rng.integers(-1, 2)
        
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

    def _initialize_workplace_home_cycle(self, n_humans: int, rng: np.random.Generator, positions: list):
        """
        Initialize workplace/home cycle data for humans.
        80% follow workplace cycle (determined by human ID), 20% do random movement.
        Workplace: Northwest area, Home: Southwest area, both with radius 15.
        """
        # Define workplace and residential zones with radius 15
        # Workplace: Northwest area (downtown)
        work_center_x = self.grid_size * 0.25  # Northwest quadrant
        work_center_y = self.grid_size * 0.75  # Northwest quadrant  
        work_radius = 15
        
        # Homes: Southwest area (residential)
        home_center_x = self.grid_size * 0.25  # Southwest quadrant
        home_center_y = self.grid_size * 0.25  # Southwest quadrant
        home_radius = 15
        
        for i in range(n_humans):
            # Use human ID to deterministically assign 80% to workplace cycle
            # This ensures consistent assignment across runs with same seed
            human_id_hash = hash(str(i)) % 100  # Convert ID to 0-99 range
            
            if human_id_hash < 80:  # 80% workplace cycle
                self.movement_patterns[i] = 'workplace_cycle'
                
                # Assign home location within southwest residential area
                home_angle = rng.uniform(0, 2 * math.pi)
                home_dist = rng.uniform(0, home_radius)
                home_x = home_center_x + home_dist * math.cos(home_angle)
                home_y = home_center_y + home_dist * math.sin(home_angle)
                
                # Clamp to grid bounds
                home_x = max(0, min(self.grid_size - 1, home_x))
                home_y = max(0, min(self.grid_size - 1, home_y))
                self.home_locations[i] = (round(home_x, self.rounding_digits), 
                                        round(home_y, self.rounding_digits))
                
                # Assign work location within northwest downtown area
                work_angle = rng.uniform(0, 2 * math.pi)
                work_dist = rng.uniform(0, work_radius)
                work_x = work_center_x + work_dist * math.cos(work_angle)
                work_y = work_center_y + work_dist * math.sin(work_angle)
                
                # Clamp to grid bounds
                work_x = max(0, min(self.grid_size - 1, work_x))
                work_y = max(0, min(self.grid_size - 1, work_y))
                self.work_locations[i] = (round(work_x, self.rounding_digits), 
                                        round(work_y, self.rounding_digits))
                
                # Initialize state: start at home
                self.current_targets[i] = 'home'
                self.movement_states[i] = 'at_home'
                self.stay_timers[i] = rng.integers(15, 26)  # 15-25 timesteps at home initially
                
                # Update position to be at home
                positions[i] = self.home_locations[i]
                
            else:
                # Random movement humans (20%)
                self.movement_patterns[i] = 'random'
                # Initialize velocity for random movement
                vx = rng.uniform(-self.max_velocity, self.max_velocity)
                vy = rng.uniform(-self.max_velocity, self.max_velocity)
                speed = math.sqrt(vx**2 + vy**2)
                if speed > self.max_velocity:
                    vx = (vx / speed) * self.max_velocity
                    vy = (vy / speed) * self.max_velocity
                self.velocities[i] = [vx, vy]

    def _workplace_home_cycle_move(self, x: float, y: float, human_id: int, rng: np.random.Generator) -> Tuple[float, float]:
        """
        Handle workplace/home cycle movement for a specific human.
        """
        # Check if this human uses random movement instead
        if self.movement_patterns.get(human_id) == 'random':
            return self._continuous_random_move(x, y, human_id, rng)
            
        # Get current state for this human
        current_state = self.movement_states.get(human_id, 'at_home')
        stay_timer = self.stay_timers.get(human_id, 0)
        
        # Get target locations
        home_loc = self.home_locations.get(human_id, (x, y))
        work_loc = self.work_locations.get(human_id, (x, y))
        
        if current_state in ['at_home', 'at_work']:
            # Currently at a location
            if stay_timer > 0:
                # Still need to stay, decrement timer
                self.stay_timers[human_id] = stay_timer - 1
                return x, y  # Don't move
            else:
                # Time to switch locations
                if current_state == 'at_home':
                    self.current_targets[human_id] = 'work'
                    self.movement_states[human_id] = 'going_to_work'
                else:  # at_work
                    self.current_targets[human_id] = 'home'
                    self.movement_states[human_id] = 'going_home'
        
        # Now handle movement toward target
        target_loc = work_loc if self.current_targets[human_id] == 'work' else home_loc
        
        # Calculate distance to target
        dx = target_loc[0] - x
        dy = target_loc[1] - y
        
        # Handle periodic boundaries
        if abs(dx) > self.grid_size / 2:
            dx = dx - np.sign(dx) * self.grid_size
        if abs(dy) > self.grid_size / 2:
            dy = dy - np.sign(dy) * self.grid_size
            
        distance = math.sqrt(dx**2 + dy**2)
        
        # Check if we've reached the target
        if distance < 1.0:  # Close enough to target
            if self.current_targets[human_id] == 'work':
                self.movement_states[human_id] = 'at_work'
                self.stay_timers[human_id] = rng.integers(18, 23)  # 18-22 timesteps at work
            else:
                self.movement_states[human_id] = 'at_home'
                self.stay_timers[human_id] = rng.integers(18, 23)  # 18-22 timesteps at home
            
            return target_loc[0], target_loc[1]
        
        # Move toward target with reasonable speed
        movement_speed = 1.0  # Units per timestep
        if distance > 0:
            # Normalize direction and apply speed
            move_x = (dx / distance) * movement_speed
            move_y = (dy / distance) * movement_speed
            
            new_x = x + move_x
            new_y = y + move_y
            
            # Handle periodic boundaries
            new_x = new_x % self.grid_size
            new_y = new_y % self.grid_size
            
            return round(new_x, self.rounding_digits), round(new_y, self.rounding_digits)
        
        return x, y  # Fallback


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

######## SIRS+D Environment class ########
class SIRSDEnvironment(gym.Env):
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 10,
        "render_resolution": (1200, 600), 
    }

    # Color definitions for rendering
    COLORS = {
        'background': '#f8f9fa',      # Light gray background
        'grid_lines': '#dee2e6',      # Subtle grid lines
        'agent': '#fd7e14',           # Vibrant orange for agent
        'agent_border': '#212529',    # Dark border for agent
        'S': '#228be6',              # Bright blue for Susceptible
        'I': '#fa5252',              # Vivid red for Infected
        'R': '#40c057',              # Fresh green for Recovered
        'D': '#868e96',              # Neutral gray for Dead
        'text': '#212529',           # Dark text
        'arrow': '#212529',          # Dark arrow
        'table_bg': '#ffffff',       # White table background
        'table_header_bg': '#e9ecef'  # Light gray table header
    }

    def __init__(
        self,
        simulation_time: int = 1000,
        grid_size: int = 20,
        n_humans: int = 100,
        n_infected: int = 5,
        beta: float = 0.3,
        initial_agent_adherence: float = 0.5,
        distance_decay: float = 0.2,
        lethality: float = 0.1,
        immunity_loss_prob: float = 0.1,
        recovery_rate: float = 0.1,
        adherence_penalty_factor: float = 2,
        adherence_effectiveness: float = 0.2,  # effect of adherence (0.2 = 20% of beta remains at max adherence).
        movement_type: str = "continuous_random",
        movement_scale: float = 1.0, 
        visibility_radius: float = -1,  # -1 means full visibility, >=0 means limited visibility
        rounding_digits: int = 2,
        reinfection_count: int = 3,
        safe_distance: float = 0,  
        init_agent_distance: float = 0,  
        max_distance_for_beta_calculation: float = -1,  # -1 means no limit, >0 means distance threshold
        reward_type: str = "comprehensive",
        reward_ablation: str = "full",  # Ablation variant (for potential field reward ONLY, no effect if not using this reward function): full, no_magnitude, no_direction, no_move, no_adherence, no_health, no_S
        render_mode: Optional[str] = None,
        gamma: float = 0.99 # Added gamma for potential shaping
    ):
        super().__init__()

        # Store gamma for potential shaping
        self.gamma = gamma

        # Initialize frames list for video recording
        self.frames = []

        # Validate parameters
        if visibility_radius < -1:
            raise ValueError("visibility_radius must be -1 (full visibility) or a positive number")
        if initial_agent_adherence < 0 or initial_agent_adherence > 1:
            raise ValueError("initial_agent_adherence must be in [0,1]")
        if beta < 0 or beta > 1:
            raise ValueError("beta must be in [0,1]")
        if lethality < 0 or lethality > 1:
            raise ValueError("lethality must be in [0,1]")
        if immunity_loss_prob < 0 or immunity_loss_prob > 1:
            raise ValueError("immunity_loss_prob must be in [0,1]")
        if recovery_rate < 0 or recovery_rate > 1:
            raise ValueError("recovery_rate must be in [0,1]")
        if initial_agent_adherence < 0 or initial_agent_adherence > 1:
            raise ValueError("Adherence must be in [0,1]")
        if adherence_penalty_factor < 1:
            raise ValueError("adherence_penalty_factor must be 1 or greater")
                
        # Store render mode and initialize rendering variables
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        self.last_action = None

        # Training metrics
        self.cumulative_reward = 0.0
        self.dead_count = 0
        self.infected_count = n_infected

        width, height = self.metadata["render_resolution"]
        self.dpi = 100
        self.figure_size = (width / self.dpi, height / self.dpi)

        # General parameters
        self.simulation_time = simulation_time
        self.counter = 0 # counter for the elapsed simulation time
        self.grid_size = grid_size 
        self.rounding_digits = rounding_digits
        self.reinfection_count = reinfection_count

        # Normalization constants
        self.max_distance = math.sqrt(2) * self.grid_size / 2
        self.max_movement = 1.0  

        # Agent parameters that are handled by the env
        self.agent_position = np.array([self.grid_size//2, self.grid_size//2]) # initial position of the agent
        self.initial_agent_adherence = initial_agent_adherence # NPI adherence
        self.agent_adherence = initial_agent_adherence # NPI adherence
        self.agent_state = STATE_DICT['S'] # initial state of the agent
        self.agent_time_in_state = 0  # Track time in state for agent
        self.adherence_penalty_factor = adherence_penalty_factor

        ##############################
        ####### SIRS parameters 
        ##############################
        self.n_humans = n_humans
        self.n_infected = n_infected
        self.beta = beta # infection rate
        self.distance_decay = distance_decay # distance decay rate
        self.lethality = lethality # lethality rate
        self.immunity_loss_prob = immunity_loss_prob # probability of losing immunity
        self.recovery_rate = recovery_rate # recovery rate
        self.visibility_radius = visibility_radius # Visibility radius restored
        self.adherence_effectiveness = adherence_effectiveness  # Store minimum adherence effectiveness

        ##############################
        ####### Observation and Action spaces
        ##############################
        
        # Calculate features per human based on visibility setting
        self.use_visibility_flag = (visibility_radius >= 0)
        # Features: delta_x, delta_y, dist (3) + state_one_hot (3)
        base_features = 3 + 3 
        features_per_human = base_features + 1 if self.use_visibility_flag else base_features
        
        self.observation_space = gym.spaces.Dict({
            "agent_adherence": gym.spaces.Box(
                low=0,
                high=1,
                shape=(1,),
                dtype=np.float32
            ),
            "humans_features": gym.spaces.Box(
                low=-1, 
                high=1,  
                shape=(self.n_humans * features_per_human,),
                dtype=np.float32
            )
        })

        # Define the action space
        # 1) agent_position: 2 continuous features (x, y)
        # 2) agent_adherence: 1 continuous feature (adherence)
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1, 0], dtype=np.float32), 
            high=np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        # initialize humans list
        self.humans: List[Human] = [] 
        # Movement handler  
        self.movement_handler = MovementHandler(grid_size, movement_type, rounding_digits=self.rounding_digits, movement_scale=movement_scale)
        
        # Store reward type
        self.reward_type = reward_type

        self.safe_distance = safe_distance  # Store the safe distance for reinfection
        self.init_agent_distance = init_agent_distance  # Store the initial distance parameter

        # New parameter
        self.max_distance_for_beta_calculation = max_distance_for_beta_calculation

        # Store reward ablation
        self.reward_ablation = reward_ablation

    ####### TRANSITION FUNCTIONS FOR MOVING BETWEEN S, I, R AND DEAD #######

    def _calculate_distance(self, human1: Human, human2: Human) -> float:
        """
        Calculate the minimum distance between two humans in a periodic grid:
            - Take two humans as input
            - Return the raw distance considering grid wrapping
            - Note: This returns the raw distance, normalization should be done by the caller if needed
        """
        # Calculate direct differences
        dx = abs(human1.x - human2.x)
        dy = abs(human1.y - human2.y)
        
        # Consider wrapping around the grid
        dx = min(dx, self.grid_size - dx)
        dy = min(dy, self.grid_size - dy)
        
        return math.sqrt(dx**2 + dy**2) 

    def _get_neighbors_list(self, current_human: Human) -> List[Human]:
        """
        Returns a list of human neighbors within visibility radius.
        If visibility_radius is -1, returns all humans.
        """
        if self.visibility_radius == -1:
            # Full visibility mode - return all humans except the current one
            return [h for h in self.humans if h.id != current_human.id]
        else:
            # Limited visibility mode - only return humans within visibility radius
            visible_humans = []
            for human in self.humans:
                if human.id == current_human.id:
                    continue  # Skip the current human
                
                distance = self._calculate_distance(current_human, human)
                if distance <= self.visibility_radius:
                    visible_humans.append(human)
            
            return visible_humans

    def _get_infected_list(self, current_human: Human) -> List[Human]:
        """
        Returns a list of infected humans.
        If visibility_radius is -1, returns all infected humans.
        Otherwise, returns only infected humans within visibility radius.
        """
        if self.visibility_radius == -1:
            # Full visibility mode - return all infected humans except the current one
            return [h for h in self.humans if h.state == STATE_DICT['I'] and h.id != current_human.id]
        else:
            # Limited visibility mode - return visible infected humans
            neighbors = self._get_neighbors_list(current_human)
            return [h for h in neighbors if h.state == STATE_DICT['I']]

    def _calculate_total_exposure(self, susceptible: Human) -> float:
        """ 
        Return the total exposure from infected individuals.
        If max_distance_for_beta_calculation is -1, all infected contribute to exposure.
        If max_distance_for_beta_calculation > 0, only infected within this distance contribute.
        """
        infected_list = self._get_infected_list(susceptible)

        total_exposure = 0
        for infected in infected_list:
            distance = self._calculate_distance(susceptible, infected)
            
            # Apply distance threshold if enabled (max_distance_for_beta_calculation > 0)
            if self.max_distance_for_beta_calculation == -1 or distance <= self.max_distance_for_beta_calculation:
                total_exposure += math.exp(-self.distance_decay * distance)

        return total_exposure

    def _calculate_infection_probability(self, susceptible: Human, is_agent: bool = False) -> float:
        """
        Calculate probability of infection based on nearby infected individuals
        If visibility_radius is -1, consider all infected individuals
        """
        total_exposure = self._calculate_total_exposure(susceptible)


        if is_agent:
            # Effective beta is reduced but not eliminated by adherence
            effective_beta = self.beta * (self.adherence_effectiveness + (1 - self.adherence_effectiveness) * (1 - self.agent_adherence))
        else:
            effective_beta = self.beta

        # Use a Poisson model for infection probability: P = 1 - exp(-effective_beta * total_exposure)
        probability = 1 - math.exp(- effective_beta * total_exposure)
        return probability

    def _calculate_recovery_probabilities(self, human: Human) -> float:
        """Calculate recovery probabilities for a human: Transition from I to R"""
        if human.state != STATE_DICT['I']:
            raise ValueError("incorrect call to function: probability of recovery is only applicable to humans in the infected state")
        else:
            return self.recovery_rate
    
    def _calculate_immunity_loss_probability(self, human: Human) -> float:
        """Calculate immunity loss probability for a human: Transition from R to S"""
        if human.state != STATE_DICT['R']:
            raise ValueError("incorrect call to function: probability of immunity loss is only applicable to humans in the recovered state")
        else:
            return self.immunity_loss_prob

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[dict, dict]:
        """Reset the environment to the initial state"""
        super().reset(seed=seed)
        
        # Clear frames list
        self.frames = []
        
        # Reset counter and metrics
        self.counter = 0
        self.cumulative_reward = 0.0
        self.dead_count = 0
        self.infected_count = self.n_infected
        
        self.agent_position = np.array([self.grid_size//2, self.grid_size//2])
        self.agent_adherence = self.initial_agent_adherence
        self.agent_state = STATE_DICT['S']
        self.agent_time_in_state = 0  # Reset agent time in state
        
        # Initialize humans with positions from movement handler
        self.humans = []
        initial_positions = self.movement_handler.initialize_positions(
            self.n_humans, 
            self.np_random,
            n_infected=self.n_infected,
            safe_distance=self.safe_distance,
            init_agent_distance=self.init_agent_distance
        )
        
        # Create humans at the initialized positions
        for i, (x, y) in enumerate(initial_positions):
            self.humans.append(Human(x, y, STATE_DICT['S'], id=i+1))  # id+1 because 0 is reserved

        # First n_infected humans will be infected (they are already at safe distance)
        for i in range(self.n_infected):
            self.humans[i].update_state(STATE_DICT['I'])

        return self._get_observation(), {}

    def _update_agent(self, action: np.ndarray) -> None:
        """
        Update the agent status in the environment.
        Action space is normalized to [-1, 1] for movement and [0, 1] for adherence.
        Ensures agent position stays within grid bounds using periodic boundary conditions.
        """
        dx = np.clip(action[0], -1, 1)
        dy = np.clip(action[1], -1, 1)
        adherence = np.clip(action[2], 0, 1)
        
        # Scale movement from [-1,1] to actual grid movement
        scaled_dx = dx * self.max_movement
        scaled_dy = dy * self.max_movement
        
        new_position = self.agent_position + np.array([scaled_dx, scaled_dy])
        self.agent_position = np.array([
            new_position[0] % self.grid_size,  # wrap x-coordinate
            new_position[1] % self.grid_size   # wrap y-coordinate
        ])
        
        # Update NPI level (clipped to [0,1] range)
        self.agent_adherence = adherence
    
    def _handle_human_stepping(self):
        """Handle the stepping of humans and agent state transitions"""
        # First handle agent state transitions if agent is not dead
        if self.agent_state != STATE_DICT['D']:
            agent_human = Human(
                x=self.agent_position[0],
                y=self.agent_position[1],
                state=self.agent_state,
                id=-1,
                time_in_state=self.agent_time_in_state
            )

            # Increment agent time in state
            self.agent_time_in_state += 1

            if self.agent_state == STATE_DICT['S']:
                # Calculate probability of infection
                p_infection = self._calculate_infection_probability(agent_human, is_agent=True)
                if self.np_random.random() < p_infection:
                    self.agent_state = STATE_DICT['I']
                    self.agent_time_in_state = 0  # Reset time in state on transition
                    self.infected_count += 1
            
            elif self.agent_state == STATE_DICT['I']:
                # Check for death
                if self.np_random.random() < self.lethality:
                    self.agent_state = STATE_DICT['D']
                    self.agent_time_in_state = 0  # Reset time in state on transition
                    self.dead_count += 1
                    self.infected_count -= 1
                # Check for recovery if not dead
                elif self.np_random.random() < self._calculate_recovery_probabilities(agent_human):
                    self.agent_state = STATE_DICT['R']
                    self.agent_time_in_state = 0  # Reset time in state on transition
                    self.infected_count -= 1
            
            elif self.agent_state == STATE_DICT['R']:
                # Check for immunity loss
                p_immunity_loss = self._calculate_immunity_loss_probability(agent_human)
                if self.np_random.random() < p_immunity_loss:
                    self.agent_state = STATE_DICT['S']
                    self.agent_time_in_state = 0  # Reset time in state on transition

        # Now handle human stepping and state transitions
        for human in self.humans:
            # First check if human is dead - dead humans don't move
            if human.state == STATE_DICT['D']:
                continue

            # Only move humans that are not dead
            new_x, new_y = self.movement_handler.get_new_position(
                human.x, 
                human.y, 
                self.np_random,
                human_id=human.id
            )
            human.move(new_x, new_y, self.grid_size)

            human.time_in_state += 1

            if human.state == STATE_DICT['S']:
                # Calculate probability of infection
                p_infection = self._calculate_infection_probability(human, is_agent=False)
                if self.np_random.random() < p_infection:
                    human.update_state(STATE_DICT['I'])
                    self.infected_count += 1

            elif human.state == STATE_DICT['I']:
                # Check for death
                if self.np_random.random() < self.lethality:
                    human.update_state(STATE_DICT['D'])
                    self.dead_count += 1
                    self.infected_count -= 1
                    continue

                # Check for recovery
                if self.np_random.random() < self._calculate_recovery_probabilities(human):
                    human.update_state(STATE_DICT['R'])
                    self.infected_count -= 1

            elif human.state == STATE_DICT['R']:
                # Check for immunity loss
                p_immunity_loss = self._calculate_immunity_loss_probability(human)
                if self.np_random.random() < p_immunity_loss:
                    human.update_state(STATE_DICT['S'])

        # Handle reinfection if needed
        if self.reinfection_count > 0 and self.infected_count == 0:
            # Create a temporary Human object for the agent to use distance calculations
            agent_human = Human(
                x=self.agent_position[0],
                y=self.agent_position[1],
                state=self.agent_state,
                id=-1
            )
            
            # Get list of susceptible humans that are outside the safe distance
            susceptible_humans = [h for h in self.humans 
                                if h.state == STATE_DICT['S'] and 
                                self._calculate_distance(agent_human, h) > self.safe_distance]
            
            n_to_reinfect = min(self.reinfection_count, len(susceptible_humans))
            
            if n_to_reinfect > 0:
                # Select random susceptible humans to reinfect
                reinfected_humans = self.np_random.choice(susceptible_humans, n_to_reinfect, replace=False)
                for human in reinfected_humans:
                    human.update_state(STATE_DICT['I'])
                    self.infected_count += 1

    def _get_observation(self):
        """
        Build and return the observation dict for the agent.
        """
        # Normalize agent position to [0,1] range
        agent_adherence = np.array([self.agent_adherence], dtype=np.float32)  # already in [0,1]
        
        # Create agent infection status indicator (1 if infected, 0 otherwise)

        # Create a temporary human for the agent to reuse existing logic for distance calculations
        agent_human = Human(
            x=self.agent_position[0],
            y=self.agent_position[1],
            state=self.agent_state,
            id=-1  # distinct ID
        )
        
        # Get visible humans if using limited visibility
        if self.use_visibility_flag:
            visible_humans = self._get_neighbors_list(agent_human)
            visible_ids = set(h.id for h in visible_humans)
        
        # Calculate features per human based on visibility setting
        # Features: delta_x, delta_y, dist (3) + state_one_hot (3) [+ visibility_flag (1)]
        base_features = 3 + 3  # Positional + One-hot state features
        features_per_human = base_features + 1 if self.use_visibility_flag else base_features

        # Initialize array for human observations
        humans_features = np.zeros((self.n_humans * features_per_human,), dtype=np.float32)

        # Fill in human observations
        for i, current_human in enumerate(self.humans):
            # Calculate base index for this human's features
            base_idx = i * features_per_human
            
            # Create one-hot encoding for the state: [is_dead, is_susceptible_or_recovered, is_infected]
            state_one_hot = np.zeros(3, dtype=np.float32)
            if current_human.state == STATE_DICT['D']:
                state_one_hot[0] = 1.0
            elif current_human.state == STATE_DICT['S'] or current_human.state == STATE_DICT['R']:
                state_one_hot[1] = 1.0
            else: # Infected
                state_one_hot[2] = 1.0
            
            # Calculate relative position (delta_x, delta_y)
            delta_x = current_human.x - self.agent_position[0]
            delta_y = current_human.y - self.agent_position[1]
            
            # Handle wraparound for periodic boundary conditions
            if abs(delta_x) > self.grid_size / 2:
                delta_x = delta_x - np.sign(delta_x) * self.grid_size
            if abs(delta_y) > self.grid_size / 2:
                delta_y = delta_y - np.sign(delta_y) * self.grid_size
                
            # This represents relative position scaled by grid size
            delta_x_norm = delta_x / self.grid_size * 2
            delta_y_norm = delta_y / self.grid_size * 2
            
            # Calculate distance
            dist = self._calculate_distance(agent_human, current_human)
            dist_norm = dist / self.max_distance
            
            if self.use_visibility_flag:
                # Mode with visibility flag
                visibility_flag = 1.0 if current_human.id in visible_ids else 0.0
                
                if visibility_flag == 0.0:
                    # Invisible human
                    humans_features[base_idx : base_idx + features_per_human] = [0.0, 0.0, 0.0, 0.0] + [0.0, 0.0, 0.0]
                else:
                    # Visible human
                    humans_features[base_idx : base_idx + features_per_human] = [visibility_flag, delta_x_norm, delta_y_norm, dist_norm] + state_one_hot.tolist()
            else:
                # Simple mode without visibility flag in observation per human (all humans visible)
                humans_features[base_idx : base_idx + features_per_human] = [delta_x_norm, delta_y_norm, dist_norm] + state_one_hot.tolist()

        # Compose final observation dict
        obs = {
            "agent_adherence": agent_adherence,
            "humans_features": humans_features
        }
        return obs

    def constant_reward(self):
        if self.agent_state == STATE_DICT['S']:
            return 1
        else:
            return 0
        
    def _calculate_reduceInfectionProb_reward(self):
        """
        Reward function that encourages the agent to reduce the infection probability of the population.
        """
        if self.agent_state != STATE_DICT['S']:
            return 0
        agent_human = Human(
            x=self.agent_position[0],
            y=self.agent_position[1],
            state=self.agent_state,
            id=-1
        )
        infection_prob = self._calculate_infection_probability(agent_human, is_agent=True)
        return (1-infection_prob)**2
    
    def _calculate_reduceInfectionProbwithConstant_reward(self):
        """
        Reward function that encourages the agent to reduce the infection probability of the population.
        """
        if self.agent_state != STATE_DICT['S']:
            return 0
        agent_human = Human(
            x=self.agent_position[0],
            y=self.agent_position[1],
            state=self.agent_state,
            id=-1
        )
        infection_prob = self._calculate_infection_probability(agent_human, is_agent=True)
        return  0.8 * (1-infection_prob)**2 + 0.1 * self.constant_reward()

    def _calculate_potential_field_reward(self):
        """
        Implements a vector-based potential field reward that uses directional forces.
        
        For each neighboring human, calculates a force vector pointing from them to the agent,
        with magnitude based on distance (inverse-square law) and infection status.
        
        Rewards the agent for moving in alignment with the resultant force vector,
        with an optional bonus for matching the suggested movement magnitude.
        
        Core components:
        1. Force vectors from each human based on distance and infection status
        2. Directional alignment between agent movement and suggested direction
        3. Magnitude matching between movement and suggested force (optional)
        4. Base rewards for health status and risk-adjusted adherence
        """
        # Get current ablation variant
        variant = self.reward_ablation

        # # Constants for vector field #
        w_I = 1.0      # strength of infected repulsion
        w_S = 0.5      # weak repulsion from susceptibles
        
        # Apply no_S ablation variant (zero out susceptible weight)
        w_S_local = 0.0 if variant == "no_S" else w_S
        
        p = 1        # inverse-square fall-off  
        beta = 0.25    # weight on magnitude term (25%)
        eps = 1e-8     # small epsilon to avoid division by zero
        
        # Get the agent
        agent_human = Human(
            x=self.agent_position[0],
            y=self.agent_position[1],
            state=self.agent_state,
            id=-1
        )
        
        # 1. Build resultant force vector F = (Fx, Fy)
        Fx = Fy = 0.0
        for j, h in enumerate(self.humans):
            # Skip the agent itself (if it's in the humans list)
            if h.id == -1:
                continue
                
            # Recovered/dead humans give no force
            if h.state not in (STATE_DICT['I'], STATE_DICT['S']):
                continue
                
            # Vector FROM neighbor TO agent (toroidal shortest displacement)
            dx = (agent_human.x - h.x + self.grid_size/2) % self.grid_size - self.grid_size/2
            dy = (agent_human.y - h.y + self.grid_size/2) % self.grid_size - self.grid_size/2
            dist_sq = dx*dx + dy*dy + eps
            
            # Calculate force contribution based on state (using w_S_local for no_S ablation)
            weight = w_I if h.state == STATE_DICT['I'] else w_S_local
            scale = weight / (dist_sq ** (p/2))  # 1/d^p
            Fx += scale * dx
            Fy += scale * dy
        
        # Normalize F to get direction
        F_norm = math.sqrt(Fx*Fx + Fy*Fy) + eps
        F_hat = (Fx / F_norm, Fy / F_norm)
        
        # 2. Get agent's chosen movement a = (dx, dy)
        if self.last_action is not None:
            ax, ay = self.last_action[:2]  # Extract movement direction from action
        else:
            ax, ay = 0, 0  # No movement if no action yet
            
        a_norm = math.sqrt(ax*ax + ay*ay) + eps
        
        # 3. Calculate rewards
        # Direction alignment reward: cos(theta) between action and force
        cos_theta = (ax*F_hat[0] + ay*F_hat[1]) / a_norm
        r_dir = np.clip(cos_theta, -1.0, 1.0)
        
        # Magnitude matching reward (optional)
        r_mag = 1.0 - abs(a_norm - min(F_norm, 1.0))
        r_mag = np.clip(r_mag, -1.0, 1.0)
        
        # Combined movement reward with ablations
        if variant == "no_magnitude":
            r_move = r_dir
        elif variant == "no_direction":
            r_move = r_mag
        elif variant == "no_move":
            r_move = 0.0
        else:  # full or any other
            r_move = (1-beta) * r_dir + beta * r_mag
        
        # 4. Health reward - base reward for being susceptible
        if variant == "no_health":
            health_reward = 0.0
        else:
            health_reward = 1.0 if self.agent_state == STATE_DICT['S'] else 0.0
                    
        # 5. Calculate Risk-Adjusted Adherence Reward
        # infection_prob_raw = self._calculate_infection_probability(agent_human, is_agent=False)
        
        if variant == "no_adherence":
            r_adherence = 0.0
        else:
            r_adherence = self.agent_adherence

        
        # Combine all reward components
        total_reward = 0.1 * health_reward + 0.2 * r_adherence + 0.7 * r_move
        
        # Store components for logging
        self.reward_components = {
            'health': health_reward,
            'r_adherence': r_adherence,
            'r_dir': r_dir,
            'r_mag': r_mag,
            'r_move': r_move,
            'total': total_reward,
            'force_mag': F_norm,
        }
        
        return total_reward

    def _calculate_maximize_nearest_distance_reward(self):
        """
        Reward is 1.0 if the agent is farther than max_distance_for_beta_calculation from the nearest susceptible or infected human (if > 0).
        If closer, the reward decays smoothly (linearly). If max_distance_for_beta_calculation == -1, use normalized distance.
        If there are no relevant humans, return 1.0. If the agent is dead or infected, return 0.0.
        """
        if self.agent_state == STATE_DICT['I']:
            return 0.0

        agent_human = Human(
            x=self.agent_position[0],
            y=self.agent_position[1],
            state=self.agent_state,
            id=-1
        )
        relevant_humans = [h for h in self.humans if h.state in (STATE_DICT['S'], STATE_DICT['I'])]
        if not relevant_humans:
            return 1.0
        distances = [self._calculate_distance(agent_human, h) for h in relevant_humans]
        min_distance = min(distances)


        if min_distance >= self.max_distance_for_beta_calculation:
            return 1.0
        else:
            return max(0.0, min_distance / self.max_distance_for_beta_calculation)

    def _calculate_reward(self):
        """
        Map reward type to the corresponding reward function
        """
        reward_functions = {
            "constant": self.constant_reward,
            "reduceInfectionProb": self._calculate_reduceInfectionProb_reward,
            "reduceInfectionProbwithConstant": self._calculate_reduceInfectionProbwithConstant_reward,
            "potential_field": self._calculate_potential_field_reward,  
            "max_nearest_distance": self._calculate_maximize_nearest_distance_reward, 
        }
        
        # Get reward function based on the specified reward type in config: default to potential field reward
        reward_function = reward_functions.get(self.reward_type, self._calculate_potential_field_reward)
        return reward_function()

    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, bool, dict]:
        """
        Step the environment by one timestep
        Args:
            action: np.ndarray with format [delta_x, delta_y, adherence]
        Returns:
            observation: dict of gym.spaces.Space objects
            reward: float
            terminated: bool
            truncated: bool
            info: dict
        """
        # Store the action for rendering
        self.last_action = action.copy()
        
        # Increment counter
        self.counter += 1
        
        # Store current state before update for terminal reward calculation
        previous_agent_state = self.agent_state
        
        # # Calculate distance to nearest infected BEFORE stepping # 
        self.dist_before_step = self.max_distance # Default if no infected
        agent_human_before = Human(
            x=self.agent_position[0],
            y=self.agent_position[1],
            state=self.agent_state,
            id=-1
        )
        infected_list_before = self._get_infected_list(agent_human_before)
        if infected_list_before:
            distances_before = [self._calculate_distance(agent_human_before, h) for h in infected_list_before]
            self.dist_before_step = min(distances_before)
        # # End distance calculation # 
            
        self._update_agent(action) 
        self._handle_human_stepping()

        observation = self._get_observation()
        reward = self._calculate_reward()
        self.cumulative_reward += reward  # Update cumulative reward
     
        # Check if agent became infected in this step
        became_infected = previous_agent_state == STATE_DICT['S'] and self.agent_state == STATE_DICT['I']
        
        terminated = False
        if self.counter >= self.simulation_time:
            terminated = True

        truncated = False
        # Create temporary agent human for distance calculations
        agent_human = Human(
            x=self.agent_position[0],
            y=self.agent_position[1],
            state=self.agent_state,
            id=-1
        )
        
        # Get count of susceptible humans beyond safe distance
        distant_susceptible_count = sum(1 for h in self.humans 
                                      if h.state == STATE_DICT['S'] and 
                                      self._calculate_distance(agent_human, h) > self.safe_distance)
        

        if (self.agent_state == STATE_DICT['D'] or # Agent died
            became_infected or # Agent became infected
            (self.infected_count == 0 and  # No infected individuals
             (self.reinfection_count == 0 or distant_susceptible_count < self.reinfection_count))):  # Can't reinfect
            truncated = True

        # Store frame if rendering is enabled - moved after truncation check to capture final state
        if self.render_mode is not None:
            frame = self._render_frame()
            if frame is not None:
                self.frames.append(frame)
                # If this is the final frame (episode is ending), add it again to ensure it's visible
                if terminated or truncated:
                    self.frames.append(frame)
        
        info = {
            "cumulative_reward": self.cumulative_reward,
            "truncation_reason": "timeout" if terminated else (
                "agent_death" if self.agent_state == STATE_DICT['D'] else
                "agent_infected" if became_infected else
                "no_infection_possible" if truncated else None
            ),
            "agent_state": [k for k, v in STATE_DICT.items() if v == self.agent_state][0],
            "episode_length": self.counter,
            "healthy_count": sum(1 for h in self.humans if h.state == STATE_DICT['S']),
            "infected_count": sum(1 for h in self.humans if h.state == STATE_DICT['I']),
            "recovered_count": sum(1 for h in self.humans if h.state == STATE_DICT['R']),
            "dead_count": sum(1 for h in self.humans if h.state == STATE_DICT['D']),
            "adherence": float(self.agent_adherence),
            **getattr(self, 'reward_components', {}) 
        }
        return observation, reward, terminated, truncated, info

    def _render_frame(self) -> Optional[np.ndarray]:
        """
        Render the current state of the environment using Matplotlib.
        Returns:
            np.ndarray: RGB array of shape (height, width, 3)
        """
        if self.render_mode is None:
            return None

        width, height = self.metadata["render_resolution"]
        dpi = 100
        figsize = (width / dpi, height / dpi)

        # Create figure with a modern style
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=self.COLORS['background'])
        # Update grid spec with legend on left, plot in middle, and two info panels on right
        gs = plt.GridSpec(2, 4, width_ratios=[0.15, 0.6, 0.25, 0], height_ratios=[0.5, 0.5], figure=fig)
        gs.update(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.05)  # Reduced hspace
        
        # Create legend axis first (on the left)
        ax_legend = fig.add_subplot(gs[:, 0])  # Span both rows
        ax_legend.axis('off')
        
        # Grid subplot in the middle
        ax_grid = fig.add_subplot(gs[:, 1])  # Span both rows
        ax_grid.set_facecolor(self.COLORS['background'])
        
        # Agent info panel (top right)
        ax_agent_info = fig.add_subplot(gs[0, 2])
        ax_agent_info.axis('off')
        
        # Population stats panel (bottom right)
        ax_stats = fig.add_subplot(gs[1, 2])
        ax_stats.axis('off')
        
        # Draw grid with subtle lines
        for i in range(self.grid_size + 1):
            ax_grid.axhline(y=i, color=self.COLORS['grid_lines'], linewidth=0.5, alpha=0.5)
            ax_grid.axvline(x=i, color=self.COLORS['grid_lines'], linewidth=0.5, alpha=0.5)

        # Plot humans by state with enhanced styling
        state_labels = {
            'S': 'Susceptible',
            'I': 'Infectious',
            'R': 'Recovered',
            'D': 'Dead'
        }
        
        for state in ['S', 'I', 'R', 'D']:
            humans_in_state = [h for h in self.humans if h.state == STATE_DICT[state]]
            if humans_in_state:
                x = [h.x for h in humans_in_state]
                y = [h.y for h in humans_in_state]
                ax_grid.scatter(x, y, 
                              c=self.COLORS[state], 
                              s=120,  # Slightly larger markers
                              alpha=0.8, 
                              label=state_labels[state],
                              edgecolors='white',  # White edge for contrast
                              linewidth=1)

        # Plot agent with enhanced appearance
        agent_state_str = [k for k, v in STATE_DICT.items() if v == self.agent_state][0]
        ax_grid.scatter([self.agent_position[0]], [self.agent_position[1]], 
                       c=self.COLORS['agent'], 
                       s=250,  # Larger marker size
                       marker='.',  
                       edgecolors=self.COLORS['agent_border'], 
                       linewidth=2,
                       label='Agent',
                       zorder=5)  # Ensure agent is on top

        # Draw movement vector with improved styling
        if self.last_action is not None:
            dx, dy = self.last_action[:2]
            ax_grid.arrow(self.agent_position[0], self.agent_position[1], 
                         dx, dy, 
                         color=self.COLORS['arrow'], 
                         width=0.05,  # Thicker arrow
                         head_width=0.2,
                         head_length=0.3,
                         alpha=0.8,
                         zorder=4)

        # Set grid properties with modern styling
        ax_grid.set_xlim(-1, self.grid_size + 1)
        ax_grid.set_ylim(-1, self.grid_size + 1)
        
        # Calculate professional tick intervals
        n_intervals = 5  # Use 5 intervals for clean divisions
        major_ticks = np.linspace(0, self.grid_size, n_intervals + 1, dtype=int)
        minor_ticks = np.arange(0, self.grid_size + 1, 1)
        
        # Set major and minor ticks
        ax_grid.set_xticks(major_ticks)
        ax_grid.set_yticks(major_ticks)
        ax_grid.set_xticks(minor_ticks, minor=True)
        ax_grid.set_yticks(minor_ticks, minor=True)
        
        # Style the ticks
        ax_grid.tick_params(which='major', colors=self.COLORS['text'], labelsize=10, length=6)
        ax_grid.tick_params(which='minor', colors=self.COLORS['grid_lines'], labelsize=0, length=3)
        
        # Add axis labels with modern styling
        ax_grid.set_xlabel('X Coordinate', color=self.COLORS['text'], fontsize=10, fontweight='bold')
        ax_grid.set_ylabel('Y Coordinate', color=self.COLORS['text'], fontsize=10, fontweight='bold')
        
        # Add subtle grid for major ticks only
        ax_grid.grid(True, which='major', linestyle='-', alpha=0.3, color=self.COLORS['grid_lines'])
        ax_grid.grid(True, which='minor', linestyle=':', alpha=0.2, color=self.COLORS['grid_lines'])
        
        # Move legend outside the plot
        legend = ax_grid.get_legend()
        if legend is not None:
            legend.remove()  # Remove the old legend if it exists
        lines_labels = ax_grid.get_legend_handles_labels()
        ax_legend.legend(*lines_labels, 
                        loc='center',
                        framealpha=0.95,
                        facecolor='white',
                        edgecolor='none',
                        fontsize=10,
                        borderpad=2)
        
        ax_grid.set_aspect('equal')

        # Calculate state counts
        state_counts = {
            'S': sum(1 for h in self.humans if h.state == STATE_DICT['S']),
            'I': sum(1 for h in self.humans if h.state == STATE_DICT['I']),
            'R': sum(1 for h in self.humans if h.state == STATE_DICT['R']),
            'D': sum(1 for h in self.humans if h.state == STATE_DICT['D'])
        }

        # Create agent info table
        agent_table_data = [
            ['Time', f'{self.counter}/{self.simulation_time}'],
            ['Agent State', state_labels[agent_state_str]],
            ['Position', f'({self.agent_position[0]:.1f}, {self.agent_position[1]:.1f})'],
            ['Movement dx', f'{self.last_action[0]:.2f}' if self.last_action is not None else '0.00'],
            ['Movement dy', f'{self.last_action[1]:.2f}' if self.last_action is not None else '0.00'],
            ['Adherence', f'{self.agent_adherence:.2f}'],
            ['Cumulative Reward', f'{self.cumulative_reward:.2f}']
        ]

        # Create population stats table
        stats_table_data = [
            ['Category', 'Count (Percentage)'],
            ['Susceptible', f'{state_counts["S"]} ({state_counts["S"]/self.n_humans:.1%})'],
            ['Infectious', f'{state_counts["I"]} ({state_counts["I"]/self.n_humans:.1%})'],
            ['Recovered', f'{state_counts["R"]} ({state_counts["R"]/self.n_humans:.1%})'],
            ['Dead', f'{state_counts["D"]} ({state_counts["D"]/self.n_humans:.1%})']
        ]

        # Create agent info table
        agent_table = ax_agent_info.table(cellText=agent_table_data, 
                                        loc='center',
                                        cellLoc='left',
                                        colWidths=[0.45, 0.55])  
        
        # Create population stats table
        stats_table = ax_stats.table(cellText=stats_table_data, 
                                   loc='center',
                                   cellLoc='left',
                                   colWidths=[0.45, 0.55])  
        
        # Style both tables
        for table, is_agent_table in [(agent_table, True), (stats_table, False)]:
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.6)
            
            # Add custom styling to table cells
            for (row, col), cell in table.get_celld().items():
                cell.set_facecolor(self.COLORS['table_bg'])
                cell.set_edgecolor(self.COLORS['grid_lines'])
                cell.set_text_props(color=self.COLORS['text'])
                
                if is_agent_table:
                    # Agent table styling
                    if row == 0:  # Header row
                        cell.set_facecolor(self.COLORS['table_header_bg'])
                        cell.set_text_props(weight='bold')
                    elif row == 1 and col == 1:  # Agent State cell
                        cell.set_text_props(color=self.COLORS[agent_state_str])
                        cell.set_text_props(weight='bold')
                else:
                    # Stats table styling
                    if row == 0:  # Header row
                        cell.set_facecolor(self.COLORS['table_header_bg'])
                        cell.set_text_props(weight='bold')
                        cell.set_text_props(color=self.COLORS['text'])
                    elif row == 1:  # Susceptible row
                        cell.set_text_props(color=self.COLORS['S'])
                        cell.set_text_props(weight='bold')
                    elif row == 2:  # Infectious row
                        cell.set_text_props(color=self.COLORS['I'])
                        cell.set_text_props(weight='bold')
                    elif row == 3:  # Recovered row
                        cell.set_text_props(color=self.COLORS['R'])
                        cell.set_text_props(weight='bold')
                    elif row == 4:  # Dead row
                        cell.set_text_props(color=self.COLORS['D'])
                        cell.set_text_props(weight='bold')
                
                # Add subtle padding
                cell.PAD = 0.05

        # Convert figure to RGB array
        fig.canvas.draw()
        
        # Get the correct buffer from the figure
        buf = np.asarray(fig.canvas.buffer_rgba())
        # Convert RGBA to RGB
        data = buf[:, :, :3]
        
        plt.close(fig)
        return data

    def render(self):
        """
        Render the environment.
        Returns:
            numpy array if mode is 'rgb_array'
        """
        return self._render_frame()

    def close(self):
        """Close the environment and cleanup resources."""
        self.fig = None