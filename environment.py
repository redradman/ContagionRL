import gymnasium as gym
import numpy as np
from typing import Optional, List, Tuple
import math
from utils import STATE_DICT, MovementHandler, Human
import pygame
import pygame.gfxdraw
from pygame import Surface

######## SIRS Environment class ########
class SIRSEnvironment(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
        "render_resolution": (800, 800),  # Window size for rendering
    }

    # Color definitions for rendering
    COLORS = {
        'background': (255, 255, 255),  # White
        'grid_lines': (200, 200, 200),  # Light gray
        'agent': (255, 165, 0),         # Orange
        'S': (30, 144, 255),           # Dodger Blue
        'I': (220, 20, 60),            # Crimson
        'R': (50, 205, 50),            # Lime Green
        'D': (128, 128, 128),          # Gray
        'text': (0, 0, 0),             # Black
        'panel_bg': (240, 240, 240)    # Light gray for info panel
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
        reinfection_count: int = 3,  # Number of humans to reinfect when no infected exist
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # Validate parameters
        if visibility_radius < -1:
            raise ValueError("visibility_radius must be -1 (full visibility) or a positive number")
        if initial_agent_adherence < 0 or initial_agent_adherence > 1:
            raise ValueError("initial_agent_adherence must be in [0,1]")
        
        # Store render mode and initialize rendering variables
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.window_size = self.metadata["render_resolution"]
        self.last_action = None  # Store last action for rendering

        # Calculate cell size for rendering
        self.cell_size = min(
            self.window_size[0] // grid_size,
            self.window_size[1] // grid_size
        )
        # Recalculate window size to make it exact
        self.window_size = (
            self.cell_size * grid_size,
            self.cell_size * grid_size
        )
        
        # Font size for rendering
        self.font_size = max(10, self.cell_size // 2)
        
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
        # 1) Each human has 4 continuous features + 1 discrete state
        #    store the continuous features in a Box( shape=(4,) ) 
        #    and the discrete state in a Discrete(4).
        
        # Continuous: visibility_flag, x, y, distance
        #    - visibility_flag in [0,1]
        #    - x, y in [0, 1], normalized
        #    - distance in [0,1] (normalized distance)
        
        human_continuous_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),     # [flag, x, y, dist]
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32), # values are in [0,1] normalized.
            shape=(4,),
            dtype=np.float32
        )
        
        # Discrete state with 4 categories: S=0, I=1, R=2, D=3
        human_state_space = gym.spaces.Discrete(4)
        
        # Each human in the environment is a Dict of:
        #   { "continuous": Box(...), "state": Discrete(4) }
        human_space = gym.spaces.Dict({
            "continuous": human_continuous_space,
            "state": human_state_space
        })
        
        # replicate that 'human_space' for n humans using a Tuple.
        # So "humans" is a tuple of length n_humans, each a Dict with the above structure.
        humans_tuple_space = gym.spaces.Tuple([human_space] * self.n_humans)
        
        # Finally, combine everything into a top-level Dict,
        # including agent_position, agent_adherence, and the tuple of humans.
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
            "humans": humans_tuple_space
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
        self.agent_position = np.array([self.grid_size//2, self.grid_size//2]) # initial position of the agent
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

        Observation space structure (as defined in __init__):
        {
            "agent_position": Box(shape=(2,)),           # (x, y) normalized to [0,1]
            "agent_adherence": Box(shape=(1,)),          # [adherence in 0..1]
            "humans": Tuple(                             # length = n_humans
                Dict({
                    "continuous": Box(shape=(4,)),       # [visibility_flag, x, y, distance]
                    "state": Discrete(4)                 # in {0=S,1=I,2=R,3=D}
                })
            )
        }
        """
        # Normalize agent position to [0,1] range
        agent_position = np.array(self.agent_position, dtype=np.float32) / self.grid_size  # shape=(2,)
        agent_adherence = np.array([self.agent_adherence], dtype=np.float32)  # already in [0,1]

        # We'll create a temporary "human" for the agent to reuse your existing _get_neighbors_list logic.
        agent_human = Human(
            x=self.agent_position[0],
            y=self.agent_position[1],
            state=self.agent_state,
            id=-1  # distinct ID so we don't filter this out in _get_neighbors_list
        )
        
        # "visible_humans" is a list of Human objects within radius, or all if radius=-1
        visible_humans = self._get_neighbors_list(agent_human)
        visible_ids = set(h.id for h in visible_humans)

        # 3) BUILD THE "humans" TUPLE
        # ---------------------------
        # For each of self.humans, compute:
        #   - visibility_flag in {0,1}
        #   - x, y normalized to [0,1]
        #   - distance: distance from the agent normalized to [0,1]
        #   - state: integer in {0,1,2,3}
        # if the visibility_flag is 0, then we mask the values with 0

        humans_obs_list = []
        for current_human in self.humans:
            visibility_flag = 1.0 if (current_human.id in visible_ids) else 0.0
            
            if visibility_flag == 0: 
                x, y, dist, state_int = 0, 0, 0, 0
                human_obs = {
                    "continuous": np.array([visibility_flag, x, y, dist], dtype=np.float32),
                    "state": state_int
                }
                humans_obs_list.append(human_obs)

            else:
                # Normalize positions to [0,1]
                x_norm = current_human.x / self.grid_size
                y_norm = current_human.y / self.grid_size

                # Calculate and normalize distance to [0,1]
                dist = self._calculate_distance(agent_human, current_human)
                dist_norm = dist / self.max_distance  # Normalize using max possible distance

                # state in {0,1,2,3} (S,I,R,D)
                state_int = current_human.state

                # Build the sub-dict for this human
                human_obs = {
                    "continuous": np.array([visibility_flag, x_norm, y_norm, dist_norm], dtype=np.float32),
                    "state": state_int
                }
                humans_obs_list.append(human_obs)

        # convert the list -> tuple to match the Tuple(...) definition
        humans_obs = tuple(humans_obs_list)

        # 4) COMPOSE FINAL OBSERVATION DICT
        # ---------------------------------
        obs = {
            "agent_position": agent_position,
            "agent_adherence": agent_adherence,
            "humans": humans_obs
        }

        return obs
        

    def _calculate_reward(self):    
        """Calculate the reward for the agent"""
        # TODO: implement reward logic
        return 0

    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, bool, dict]:
        """Take a step in the environment. For more regarding the structure refer to the gymnasium documentation"""

        self.counter += 1
        # Store last action for rendering
        self.last_action = action.copy()
        
        # Update agent and humans
        self._update_agent(action) 
        self._handle_human_stepping()

        # handle observation logic
        observation = self._get_observation()

        # handle reward logic
        reward = self._calculate_reward()
     
        # handle termination logic
        terminated = False
        if self.counter >= self.simulation_time: # terminate if the simulation time is reached
            terminated = True

        # handle truncation logic
        truncated = False
        if self.agent_state == STATE_DICT['D']: # truncate if the agent is dead
            truncated = True

        # handle info logic 
        info = {}

        # Render if needed
        if self.render_mode is not None:
            self._render_frame()

        # return the observation, reward, truncation, termination, info
        return observation, reward, terminated, truncated, info

    def _render_frame(self) -> Optional[np.ndarray]:
        """
        Render the current state of the environment.
        Returns:
            np.ndarray: RGB array if mode is 'rgb_array', None if mode is 'human'
        """
        if self.render_mode is None:
            return None

        # Initialize pygame if not done yet
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("SIRS Environment")
            # Initialize font
            pygame.font.init()
            self.font = pygame.font.SysFont('Arial', self.font_size)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Create the canvas
        canvas = pygame.Surface(self.window_size)
        canvas.fill(self.COLORS['background'])

        # Draw grid lines
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                self.COLORS['grid_lines'],
                (x * self.cell_size, 0),
                (x * self.cell_size, self.window_size[1])
            )
        for y in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                self.COLORS['grid_lines'],
                (0, y * self.cell_size),
                (self.window_size[0], y * self.cell_size)
            )

        # Draw humans
        for human in self.humans:
            x = int(human.x * self.cell_size + self.cell_size/2)
            y = int(human.y * self.cell_size + self.cell_size/2)
            radius = self.cell_size // 4

            # Get color based on state
            state_str = [k for k, v in STATE_DICT.items() if v == human.state][0]
            color = self.COLORS[state_str]

            pygame.draw.circle(canvas, color, (x, y), radius)

        # Draw agent
        agent_x = int(self.agent_position[0] * self.cell_size + self.cell_size/2)
        agent_y = int(self.agent_position[1] * self.cell_size + self.cell_size/2)
        agent_radius = self.cell_size // 3

        # Draw agent with state-colored border
        agent_state_str = [k for k, v in STATE_DICT.items() if v == self.agent_state][0]
        pygame.draw.circle(canvas, self.COLORS[agent_state_str], (agent_x, agent_y), agent_radius)
        pygame.draw.circle(canvas, self.COLORS['agent'], (agent_x, agent_y), agent_radius - 2)

        # Draw movement vector if last action exists
        if self.last_action is not None:
            dx, dy = self.last_action[:2]
            # Scale the movement vector for visibility
            arrow_scale = self.cell_size
            end_x = agent_x + dx * arrow_scale
            end_y = agent_y + dy * arrow_scale
            # Draw arrow
            pygame.draw.line(canvas, self.COLORS['text'], 
                           (agent_x, agent_y), (end_x, end_y), 2)
            # Draw arrowhead
            arrow_size = self.cell_size // 8
            pygame.draw.polygon(canvas, self.COLORS['text'], [
                (end_x, end_y),
                (end_x - arrow_size * np.sign(dx), end_y - arrow_size),
                (end_x - arrow_size * np.sign(dx), end_y + arrow_size)
            ])

        # Draw info panel
        info_surface = self._create_info_panel()
        canvas.blit(info_surface, (10, 10))

        if self.render_mode == "human":
            # Copy canvas to window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def _create_info_panel(self) -> Surface:
        """Create an information panel surface"""
        # Count humans in each state
        state_counts = {
            'S': sum(1 for h in self.humans if h.state == STATE_DICT['S']),
            'I': sum(1 for h in self.humans if h.state == STATE_DICT['I']),
            'R': sum(1 for h in self.humans if h.state == STATE_DICT['R']),
            'D': sum(1 for h in self.humans if h.state == STATE_DICT['D'])
        }

        # Create info text
        info_lines = [
            f"Time: {self.counter}/{self.simulation_time}",
            f"Agent State: {[k for k,v in STATE_DICT.items() if v == self.agent_state][0]}",
            f"Time in State: {self.agent_time_in_state}",
            f"Position: ({self.agent_position[0]:.1f}, {self.agent_position[1]:.1f})",
            f"Adherence: {self.agent_adherence:.2f}",
            "Population:",
            f"  S: {state_counts['S']}",
            f"  I: {state_counts['I']}",
            f"  R: {state_counts['R']}",
            f"  D: {state_counts['D']}"
        ]

        # Add last action if available
        if self.last_action is not None:
            info_lines.extend([
                "Last Action:",
                f"  Move: ({self.last_action[0]:.2f}, {self.last_action[1]:.2f})",
                f"  Adherence: {self.last_action[2]:.2f}"
            ])

        # Create surface for info panel
        line_height = self.font_size + 2
        panel_height = line_height * len(info_lines) + 20
        panel_width = 250
        info_surface = pygame.Surface((panel_width, panel_height))
        info_surface.fill(self.COLORS['panel_bg'])
        info_surface.set_alpha(230)  # Semi-transparent

        # Draw text
        for i, line in enumerate(info_lines):
            text_surface = self.font.render(line, True, self.COLORS['text'])
            info_surface.blit(text_surface, (10, 10 + i * line_height))

        return info_surface

    def render(self):
        """
        Render the environment.
        Returns:
            None if mode is 'human', numpy array if mode is 'rgb_array'
        """
        return self._render_frame()

    def close(self):
        """Close the environment and cleanup pygame resources."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

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