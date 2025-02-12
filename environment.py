import gymnasium as gym
import numpy as np
from typing import Optional, List, Tuple
import math
from utils import STATE_DICT, MovementHandler, Human
import matplotlib.pyplot as plt
import cv2

######## SIRS Environment class ########
class SIRSEnvironment(gym.Env):
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 4,
        "render_resolution": (800, 600),  # More reasonable figure size for rendering
    }

    # Color definitions for rendering
    COLORS = {
        'background': 'white',
        'grid_lines': '#cccccc',
        'agent': '#ffa500',         # Orange
        'S': '#1e90ff',           # Dodger Blue
        'I': '#dc143c',            # Crimson
        'R': '#32cd32',            # Lime Green
        'D': '#808080',          # Gray
        'text': 'black',
        'panel_bg': '#f0f0f0'    # Light gray for info panel
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
        immunity_decay: float = 0.1,
        recovery_rate: float = 0.1,
        max_immunity_loss_prob: float = 0.2,
        movement_type: str = "stationary",
        visibility_radius: float = -1,
        rounding_digits: int = 2,
        reinfection_count: int = 3,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # Initialize frames list for video recording
        self.frames = []

        # Validate parameters
        if visibility_radius < -1:
            raise ValueError("visibility_radius must be -1 (full visibility) or a positive number")
        if initial_agent_adherence < 0 or initial_agent_adherence > 1:
            raise ValueError("initial_agent_adherence must be in [0,1]")
        
        # Store render mode and initialize rendering variables
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        self.last_action = None

        # Calculate figure size and DPI for rendering
        width, height = self.metadata["render_resolution"]
        self.dpi = 100
        self.figure_size = (width / self.dpi, height / self.dpi)  # Convert pixels to inches
        
        # General parameters
        self.simulation_time = simulation_time
        self.counter = 0 # counter for the simulation time
        self.grid_size = grid_size # from 0 to grid_size exclusive for both of the x and y axis
        self.rounding_digits = rounding_digits
        self.reinfection_count = reinfection_count

        # Normalization constants
        self.max_distance = math.sqrt(2) * self.grid_size  # Maximum possible distance in the grid
        self.max_movement = 1.0  # Maximum movement in any direction (-1 to 1)
        
        # Agent parameters that are handled by the env
        self.agent_position = np.array([self.grid_size//2, self.grid_size//2]) # initial position of the agent
        self.initial_agent_adherence = initial_agent_adherence # NPI adherence
        self.agent_adherence = initial_agent_adherence # NPI adherence
        self.agent_state = STATE_DICT['S'] # initial state of the agent
        self.agent_time_in_state = 0  # Track time in state for agent

        ##############################
        ####### SIRS parameters 
        ##############################
        self.n_humans = n_humans
        self.n_infected = n_infected
        self.beta = beta # infection rate
        self.distance_decay = distance_decay # distance decay rate
        self.lethality = lethality # lethality rate
        self.immunity_decay = immunity_decay # immunity decay rate
        self.recovery_rate = recovery_rate # recovery rate
        self.max_immunity_loss_prob = max_immunity_loss_prob # maximum immunity loss probability
        self.visibility_radius = visibility_radius # visibility radius

        ##############################
        ####### Observation and Action spaces
        ##############################
        # Flatten the observation space to avoid nested structures
        # Structure:
        # - agent_position (2): [x, y]
        # - agent_adherence (1): [adherence]
        # - humans_continuous (n_humans * 4): [visibility, x, y, distance] for each human
        # - humans_discrete (n_humans): [state] for each human
        
        self.observation_space = gym.spaces.Dict({
            "agent_position": gym.spaces.Box(
                low=0, 
                high=1,
                shape=(2,),
                dtype=np.float32
            ),
            "agent_adherence": gym.spaces.Box(
                low=0,
                high=1,
                shape=(1,),
                dtype=np.float32
            ),
            "humans_continuous": gym.spaces.Box(
                low=0,
                high=1,
                shape=(self.n_humans * 4,),  # [visibility, x, y, distance] for each human
                dtype=np.float32
            ),
            "humans_discrete": gym.spaces.Box(
                low=0,
                high=3,  # S=0, I=1, R=2, D=3
                shape=(self.n_humans,),
                dtype=np.int32
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
        self.movement_handler = MovementHandler(grid_size, movement_type, rounding_digits=self.rounding_digits)
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
        Get list of humans within visibility radius of given position
        If visibility_radius is -1, return all humans
        """
        if self.visibility_radius == -1:
            return [h for h in self.humans if h.id != current_human.id] # return all humans except the one that is being checked
        else: 
            visible_humans = []
            for human in self.humans:
                if human.id == current_human.id:
                    continue # skip the current human

                distance = self._calculate_distance(current_human, human)
                if distance <= self.visibility_radius:
                    visible_humans.append(human)
            
            return visible_humans


    def _get_infected_list(self, current_human: Human) -> List[Human]:
        """
        Return list of infected humans
        If center coordinates are provided, only return infected humans within visibility radius
        """
        neighbors = self._get_neighbors_list(current_human)
        return [h for h in neighbors if h.state == STATE_DICT['I']]

    def _calculate_infection_probability(self, susceptible: Human, is_agent: bool = False) -> float:
        """
        Calculate probability of infection based on nearby infected individuals
        If visibility_radius is -1, consider all infected individuals
        """
        infected_list = self._get_infected_list(susceptible)

        total_exposure = 0
        for infected in infected_list:
            distance = self._calculate_distance(susceptible, infected)
            total_exposure += math.exp(-self.distance_decay * distance)

        if is_agent:
            return min(1,(self.beta / (1 + self.agent_adherence)) * total_exposure)
        else:
            return min(1,(self.beta) * total_exposure)

    def _calculate_recovery_probabilities(self, human: Human) -> float:
        """Calculate recovery probabilities for a human: Transition from I to R"""
        if human.state != STATE_DICT['I']:
            raise ValueError("incorrect call to function: probability of recovery is only applicable to humans in the infected state")
        else:
            recovery_prob = 1 - math.exp(-self.recovery_rate * human.time_in_state)
            return recovery_prob
    
    def _calculate_immunity_loss_probability(self, human: Human) -> float:
        """Calculate immunity loss probability for a human: Transition from R to S"""
        if human.state != STATE_DICT['R']:
            raise ValueError("incorrect call to function: probability of immunity loss is only applicable to humans in the recovered state")
        else:
            return self.max_immunity_loss_prob * (1 - math.exp(-self.immunity_decay * human.time_in_state))

    ##### 

    def reset(self, seed: Optional[int] = None) -> Tuple[dict, dict]:
        """Reset the environment to the initial state"""
        super().reset(seed=seed)
        
        # Clear frames list
        self.frames = []
        
        # Reset counter
        self.counter = 0
        
        self.agent_position = np.array([self.grid_size//2, self.grid_size//2])
        self.agent_adherence = self.initial_agent_adherence
        self.agent_state = STATE_DICT['S']
        self.agent_time_in_state = 0  # Reset agent time in state
        # Initialize humans
        self.humans = []
        positions = set()
        
        # Place humans randomly
        for _ in range(self.n_humans):
            while True:
                x = self.np_random.integers(0, self.grid_size)
                y = self.np_random.integers(0, self.grid_size)
                if (x, y) not in positions: # to ensure the uniqueness of the position
                    positions.add((x, y))
                    break
            
            self.humans.append(Human(x, y, STATE_DICT['S'])) # x and y are positions, init state is S

        # Select random humans to be infected
        initial_infected = self.np_random.choice(self.humans, self.n_infected, replace=False)
        for human in initial_infected:
            human.update_state(STATE_DICT['I'])

        return self._get_observation(), {}

    def _update_agent(self, action: np.ndarray) -> None:
        """
        Update the agent status in the environment.
        Action space is normalized to [-1, 1] for movement and [0, 1] for adherence.
        Ensures agent position stays within grid bounds using periodic boundary conditions.
        """
        # Clip actions to ensure they stay within bounds
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
            # Create a temporary Human object for the agent to use existing transition functions
            agent_human = Human(
                x=self.agent_position[0],
                y=self.agent_position[1],
                state=self.agent_state,
                id=-1,
                time_in_state=self.agent_time_in_state  # Pass the agent's time in state
            )

            # Increment agent time in state
            self.agent_time_in_state += 1

            if self.agent_state == STATE_DICT['S']:
                # Calculate probability of infection
                p_infection = self._calculate_infection_probability(agent_human, is_agent=True)
                if self.np_random.random() < p_infection:
                    self.agent_state = STATE_DICT['I']
                    self.agent_time_in_state = 0  # Reset time in state on transition
            
            elif self.agent_state == STATE_DICT['I']:
                # Check for death
                if self.np_random.random() < self.lethality:
                    self.agent_state = STATE_DICT['D']
                    self.agent_time_in_state = 0  # Reset time in state on transition
                # Check for recovery if not dead
                elif self.np_random.random() < self._calculate_recovery_probabilities(agent_human):
                    self.agent_state = STATE_DICT['R']
                    self.agent_time_in_state = 0  # Reset time in state on transition
            
            elif self.agent_state == STATE_DICT['R']:
                # Check for immunity loss
                p_immunity_loss = self._calculate_immunity_loss_probability(agent_human)
                if self.np_random.random() < p_immunity_loss:
                    self.agent_state = STATE_DICT['S']
                    self.agent_time_in_state = 0  # Reset time in state on transition

        # Now handle human stepping and state transitions
        for human in self.humans:
            new_x, new_y = self.movement_handler.get_new_position(
                human.x, 
                human.y, 
                self.np_random
            )
            human.move(new_x, new_y, self.grid_size)

            human.time_in_state += 1
            if human.state == STATE_DICT['D']:
                continue

            if human.state == STATE_DICT['S']:
                # Calculate probability of infection
                p_infection = self._calculate_infection_probability(human, is_agent=False)
                if self.np_random.random() < p_infection:
                    human.update_state(STATE_DICT['I'])

            elif human.state == STATE_DICT['I']:
                # Check for death
                if self.np_random.random() < self.lethality:
                    human.update_state(STATE_DICT['D'])
                    continue

                # Check for recovery
                if self.np_random.random() < self._calculate_recovery_probabilities(human):
                    human.update_state(STATE_DICT['R'])

            elif human.state == STATE_DICT['R']:
                # Check for immunity loss
                p_immunity_loss = self._calculate_immunity_loss_probability(human)
                if self.np_random.random() < p_immunity_loss:
                    human.update_state(STATE_DICT['S'])

        # Check if there are any infected humans
        infected_count = sum(1 for human in self.humans if human.state == STATE_DICT['I'])
        if infected_count == 0:
            # Get list of dead humans
            dead_humans = [h for h in self.humans if h.state == STATE_DICT['D']]
            if dead_humans:
                # Randomly select humans to reinfect
                n_to_reinfect = min(self.reinfection_count, len(dead_humans))
                if n_to_reinfect > 0:
                    reinfected_humans = self.np_random.choice(dead_humans, n_to_reinfect, replace=False)
                    for human in reinfected_humans:
                        human.update_state(STATE_DICT['I'])
                        human.time_in_state = 0  # Reset time in state for newly infected

    def _get_observation(self):
        """
        Build and return the observation dict for the agent.

        Observation space structure:
        {
            "agent_position": Box(shape=(2,)),           # (x, y) normalized to [0,1]
            "agent_adherence": Box(shape=(1,)),          # [adherence in 0..1]
            "humans_continuous": Box(shape=(n_humans * 4,)),  # [visibility, x, y, distance] for each human
            "humans_discrete": Box(shape=(n_humans,)),   # [state] for each human
        }
        """
        # Normalize agent position to [0,1] range
        agent_position = np.array(self.agent_position, dtype=np.float32) / self.grid_size  # shape=(2,)
        agent_adherence = np.array([self.agent_adherence], dtype=np.float32)  # already in [0,1]

        # Create a temporary human for the agent to reuse existing _get_neighbors_list logic
        agent_human = Human(
            x=self.agent_position[0],
            y=self.agent_position[1],
            state=self.agent_state,
            id=-1  # distinct ID so we don't filter this out in _get_neighbors_list
        )
        
        # Get visible humans
        visible_humans = self._get_neighbors_list(agent_human)
        visible_ids = set(h.id for h in visible_humans)

        # Initialize arrays for human observations
        humans_continuous = np.zeros((self.n_humans * 4,), dtype=np.float32)
        humans_discrete = np.zeros((self.n_humans,), dtype=np.int32)

        # Fill in human observations
        for i, human in enumerate(self.humans):
            base_idx = i * 4  # Index for continuous features
            visibility_flag = 1.0 if human.id in visible_ids else 0.0
            
            if visibility_flag == 0:
                # All values remain 0 for invisible humans
                humans_continuous[base_idx:base_idx + 4] = 0
                humans_discrete[i] = 0
            else:
                # Normalize positions and calculate distance
                x_norm = human.x / self.grid_size
                y_norm = human.y / self.grid_size
                dist = self._calculate_distance(agent_human, human)
                dist_norm = dist / self.max_distance

                # Set continuous features [visibility, x, y, distance]
                humans_continuous[base_idx:base_idx + 4] = [visibility_flag, x_norm, y_norm, dist_norm]
                # Set discrete state
                humans_discrete[i] = human.state

        # Compose final observation dict
        obs = {
            "agent_position": agent_position,
            "agent_adherence": agent_adherence,
            "humans_continuous": humans_continuous,
            "humans_discrete": humans_discrete
        }

        return obs
        

    def _calculate_reward(self):    
        """Calculate the reward for the agent"""
        # TODO: implement reward logic
        return 0

    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, bool, dict]:
        """Take a step in the environment"""
        self.counter += 1
        self.last_action = action.copy()
        
        self._update_agent(action)
        self._handle_human_stepping()
        
        observation = self._get_observation()
        reward = self._calculate_reward()
        
        terminated = False
        if self.counter >= self.simulation_time:
            terminated = True
            
        truncated = False
        if self.agent_state == STATE_DICT['D']:
            truncated = True
            
        # Store frame if rendering is enabled
        if self.render_mode is not None:
            frame = self._render_frame()
            if frame is not None:
                self.frames.append(frame)
        info = {} 
        return observation, reward, terminated, truncated, info

    def _render_frame(self) -> Optional[np.ndarray]:
        """
        Render the current state of the environment directly using numpy arrays.
        Returns:
            np.ndarray: RGB array of shape (height, width, 3)
        """
        if self.render_mode is None:
            return None

        # Get dimensions from metadata
        width, height = self.metadata["render_resolution"]
        
        # Create base white image
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Calculate scaling factors to convert from grid coordinates to pixel coordinates
        scale_x = (width * 0.8) / (self.grid_size + 2)  # Leave 10% margin on each side
        scale_y = (height * 0.8) / (self.grid_size + 2)
        offset_x = width * 0.1   # 10% margin
        offset_y = height * 0.1  # 10% margin

        def grid_to_pixel(x, y):
            """Convert grid coordinates to pixel coordinates"""
            return (int(x * scale_x + offset_x), int(y * scale_y + offset_y))

        # Draw grid lines
        grid_color = np.array([204, 204, 204])  # Light gray
        for i in range(self.grid_size + 1):
            x = int(i * scale_x + offset_x)
            y = int(i * scale_y + offset_y)
            # Vertical lines
            image[int(offset_y):int(height-offset_y), x] = grid_color
            # Horizontal lines
            image[y, int(offset_x):int(width-offset_x)] = grid_color

        # Function to draw a circle
        def draw_circle(center_x, center_y, color, size=5):
            x, y = grid_to_pixel(center_x, center_y)
            y = height - y  # Flip y-coordinate
            
            # Define circle bounds
            x1, x2 = max(0, x - size), min(width, x + size)
            y1, y2 = max(0, y - size), min(height, y + size)
            
            # Create circle mask
            Y, X = np.ogrid[y1-y:y2-y, x1-x:x2-x]
            mask = X*X + Y*Y <= size*size
            
            # Apply color to circle area
            image[y1:y2, x1:x2][mask] = color

        # Draw humans
        for human in self.humans:
            state_str = [k for k, v in STATE_DICT.items() if v == human.state][0]
            color = np.array([int(self.COLORS[state_str].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)])
            draw_circle(human.x, human.y, color)

        # Draw agent (larger circle)
        agent_state_str = [k for k, v in STATE_DICT.items() if v == self.agent_state][0]
        agent_color = np.array([int(self.COLORS['agent'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)])
        draw_circle(self.agent_position[0], self.agent_position[1], agent_color, size=8)

        # Draw movement vector if last action exists
        if self.last_action is not None:
            start_x, start_y = grid_to_pixel(self.agent_position[0], self.agent_position[1])
            start_y = height - start_y  # Flip y-coordinate
            dx, dy = self.last_action[:2]
            end_x = int(start_x + dx * scale_x * 0.5)
            end_y = int(start_y - dy * scale_y * 0.5)  # Subtract because y is flipped
            
            # Draw arrow line
            cv2.line(image, (start_x, start_y), (end_x, end_y), (0, 0, 0), 2)

        # Add text information
        info_text = [
            f"Time: {self.counter}/{self.simulation_time}",
            f"Agent State: {agent_state_str}",
            f"Position: ({self.agent_position[0]:.1f}, {self.agent_position[1]:.1f})",
            f"Adherence: {self.agent_adherence:.2f}",
        ]

        # Add state counts
        state_counts = {
            'S': sum(1 for h in self.humans if h.state == STATE_DICT['S']),
            'I': sum(1 for h in self.humans if h.state == STATE_DICT['I']),
            'R': sum(1 for h in self.humans if h.state == STATE_DICT['R']),
            'D': sum(1 for h in self.humans if h.state == STATE_DICT['D'])
        }
        info_text.append(f"Population: S:{state_counts['S']} I:{state_counts['I']} R:{state_counts['R']} D:{state_counts['D']}")

        # Draw text
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(image, text, (10, y_offset + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return image

    def render(self):
        """
        Render the environment.
        Returns:
            numpy array if mode is 'rgb_array'
        """
        return self._render_frame()

    def close(self):
        """Close the environment and cleanup matplotlib resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

#############################
########## ARCHIVE ##########
#############################

        # Observation and Action spaces
        # self.observation_space = gym.spaces.Dict({
        #     "agent_position": gym.spaces.Box(
        #         low=0, 
        #         high=self.grid_size, 
        #         shape=(2,),  # x, y
        #         dtype=np.float32
        #     ),
        #     "agent_adherence": gym.spaces.Box(
        #         low=0, 
        #         high=1, 
        #         shape=(1,),  # agent adherence
        #         dtype=np.float32
        #     ),
        #     "humans": gym.spaces.Box(
        #         low=np.array([0, 0, 0, 0, 0] * self.n_humans),  # visibility_flag, x, y, distance, state
        #         high=np.array([1, 1, 1, 1, 4] * self.n_humans),  # state goes from 0 to 4 (S=0, I=1, R=2, D=3)
        #         shape=(self.n_humans, 5),  # visibility_flag, x, y, distance, state
        #         dtype=np.float32
        #     )
        # })