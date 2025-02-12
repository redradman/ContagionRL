import gymnasium as gym
import numpy as np
from typing import Optional, List, Tuple
import math
from utils import STATE_DICT, MovementHandler, Human
import matplotlib.pyplot as plt

######## SIRS Environment class ########
class SIRSEnvironment(gym.Env):
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 4,
        "render_resolution": (1200, 600),  # Width increased to accommodate both panels
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
        # Update grid spec to include space for legend
        gs = plt.GridSpec(1, 3, width_ratios=[0.65, 0.1, 0.25], figure=fig)
        
        # Grid subplot with modern styling
        ax_grid = fig.add_subplot(gs[0])
        ax_grid.set_facecolor(self.COLORS['background'])
        
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
                       s=250,  # Larger agent marker
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
        
        # Create legend axis
        ax_legend = fig.add_subplot(gs[1])
        ax_legend.axis('off')
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
                        fontsize=10)
        
        ax_grid.set_aspect('equal')

        # Info table subplot with modern styling
        ax_info = fig.add_subplot(gs[2])
        ax_info.set_facecolor(self.COLORS['background'])
        ax_info.axis('off')

        # Calculate state counts
        state_counts = {
            'S': sum(1 for h in self.humans if h.state == STATE_DICT['S']),
            'I': sum(1 for h in self.humans if h.state == STATE_DICT['I']),
            'R': sum(1 for h in self.humans if h.state == STATE_DICT['R']),
            'D': sum(1 for h in self.humans if h.state == STATE_DICT['D'])
        }

        # Create info table with modern styling
        table_data = [
            ['Time', f'{self.counter}/{self.simulation_time}'],
            ['Agent State', state_labels[agent_state_str]],
            ['Position', f'({self.agent_position[0]:.1f}, {self.agent_position[1]:.1f})'],
            ['Adherence', f'{self.agent_adherence:.2f}'],
            ['Susceptible', f'{state_counts["S"]} ({state_counts["S"]/self.n_humans:.1%})'],
            ['Infectious', f'{state_counts["I"]} ({state_counts["I"]/self.n_humans:.1%})'],
            ['Recovered', f'{state_counts["R"]} ({state_counts["R"]/self.n_humans:.1%})'],
            ['Dead', f'{state_counts["D"]} ({state_counts["D"]/self.n_humans:.1%})']
        ]

        # Create modern styled table
        table = ax_info.table(cellText=table_data, 
                            loc='center',
                            cellLoc='left',
                            colWidths=[0.35, 0.65])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.6)
        
        # Add custom styling to table cells
        for (row, col), cell in table.get_celld().items():
            cell.set_facecolor(self.COLORS['table_bg'])
            cell.set_edgecolor(self.COLORS['grid_lines'])
            cell.set_text_props(color=self.COLORS['text'])
            if row == 0:  # Header row
                cell.set_facecolor(self.COLORS['table_header_bg'])
                cell.set_text_props(weight='bold')
            # Add subtle padding
            cell.PAD = 0.05

        # Adjust layout
        plt.tight_layout()
        
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