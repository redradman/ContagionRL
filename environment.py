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
        "render_fps": 10,
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
        adherence_penalty_factor: float = 2,
        movement_type: str = "stationary",
        movement_scale: float = 1.0,  # Scale factor for non-focal agent movement (0 to 1)
        visibility_radius: float = -1,  # Restored: -1 means full visibility, >=0 means limited visibility
        rounding_digits: int = 2,
        reinfection_count: int = 3,
        safe_distance: float = 0,  # New parameter for minimum safe distance for reinfection
        reward_type: str = "stateBased",
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
        if beta < 0 or beta > 1:
            raise ValueError("beta must be in [0,1]")
        if lethality < 0 or lethality > 1:
            raise ValueError("lethality must be in [0,1]")
        if immunity_decay < 0 or immunity_decay > 1:
            raise ValueError("immunity_decay must be in [0,1]")
        if recovery_rate < 0 or recovery_rate > 1:
            raise ValueError("recovery_rate must be in [0,1]")
        if initial_agent_adherence < 0 or initial_agent_adherence > 1:
            raise ValueError("Adherence must be in [0,1]")
        if adherence_penalty_factor < 1:
            raise ValueError("adherence_penalty_factor must be 1 or greater")
            
        # error checking for reward type done in the handler function
        
        # Store render mode and initialize rendering variables
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        self.last_action = None

        # Training metrics
        self.cumulative_reward = 0.0
        self.dead_count = 0
        self.infected_count = n_infected

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
        self.adherence_penalty_factor = adherence_penalty_factor

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
        self.visibility_radius = visibility_radius # Visibility radius restored

        ##############################
        ####### Observation and Action spaces
        ##############################
        # Flatten the observation space to avoid nested structures
        # Structure:
        # - agent_position (2): [x, y]
        # - agent_adherence (1): [adherence]
        # - is_agent_infected (1): [1 if infected, 0 otherwise]
        # - humans_features (n_humans * features_per_human):
        #   If visibility_radius == -1, returns all humans
        #   If visibility_radius >= 0, returns humans within visibility radius
        
        # Calculate features per human based on visibility setting
        self.use_visibility_flag = (visibility_radius >= 0)
        features_per_human = 5 if self.use_visibility_flag else 4
        
        self.observation_space = gym.spaces.Dict({
            # "agent_position": gym.spaces.Box(
            #     low=0, 
            #     high=1,
            #     shape=(2,),
            #     dtype=np.float32
            # ),
            "agent_adherence": gym.spaces.Box(
                low=0,
                high=1,
                shape=(1,),
                dtype=np.float32
            ),
            "is_agent_infected": gym.spaces.Box(
                low=0,
                high=1,
                shape=(1,),
                dtype=np.float32
            ),
            "humans_features": gym.spaces.Box(
                low=-1,  # Changed from 0 to -1 to accommodate negative relative positions
                high=1,
                shape=(self.n_humans * features_per_human,),  # Features for each human (with or without visibility flag)
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
        """ Return the total exposure"""
        infected_list = self._get_infected_list(susceptible)

        total_exposure = 0
        for infected in infected_list:
            distance = self._calculate_distance(susceptible, infected)
            total_exposure += math.exp(-self.distance_decay * distance)

        return total_exposure

    def _calculate_infection_probability(self, susceptible: Human, is_agent: bool = False) -> float:
        """
        Calculate probability of infection based on nearby infected individuals
        If visibility_radius is -1, consider all infected individuals
        """
        # Phase 1
        # infected_list = self._get_infected_list(susceptible)

        # total_exposure = 0
        # for infected in infected_list:
        #     distance = self._calculate_distance(susceptible, infected)
        #     total_exposure += math.exp(-self.distance_decay * distance)

        # if is_agent:
        #     return min(1,(self.beta / (1 + 10 * self.agent_adherence)) * total_exposure)
        # else:
        #     return min(1,(self.beta) * total_exposure)

        # Phase 2

        total_exposure = self._calculate_total_exposure(susceptible)
                # infected_list = self._get_infected_list(susceptible)

                # total_exposure = 0
                # for infected in infected_list:
                #     distance = self._calculate_distance(susceptible, infected)
                #     total_exposure += math.exp(-self.distance_decay * distance)

        # Define a minimum effective factor that ensures beta never goes to zero
        min_factor = 0.2  # for example, 20% of beta remains at maximum adherence

        if is_agent:
            # Effective beta is reduced but not eliminated by adherence
            effective_beta = self.beta * (min_factor + (1 - min_factor) * (1 - self.agent_adherence))
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
            # recovery_prob = 1 - math.exp(-self.recovery_rate * human.time_in_state)
            # return recovery_prob
            return self.recovery_rate
    
    def _calculate_immunity_loss_probability(self, human: Human) -> float:
        """Calculate immunity loss probability for a human: Transition from R to S"""
        if human.state != STATE_DICT['R']:
            raise ValueError("incorrect call to function: probability of immunity loss is only applicable to humans in the recovered state")
        else:
            # return self.max_immunity_loss_prob * (1 - math.exp(-self.immunity_decay * human.time_in_state))
            return self.max_immunity_loss_prob

    ##### 

    def reset(self, seed: Optional[int] = None) -> Tuple[dict, dict]:
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
            safe_distance=self.safe_distance
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

        Observation space structure:
        {
            "agent_position": Box(shape=(2,)),           # (x, y) normalized to [0,1]
            "agent_adherence": Box(shape=(1,)),          # [adherence in 0..1]
            "is_agent_infected": Box(shape=(1,)),        # [1 if infected, 0 otherwise]
            "humans_features": Box(shape=(n_humans * features_per_human,)),
                # If visibility_radius == -1: [delta_x, delta_y, distance, is_infected] for each human
                # If visibility_radius >= 0: [visibility, delta_x, delta_y, distance, is_infected] for each human
                # delta_x and delta_y are normalized relative positions from agent to human
        }
        """
        # Normalize agent position to [0,1] range
        # agent_pos = np.array(self.agent_position, dtype=np.float32) / self.grid_size  # shape=(2,)
        agent_adherence = np.array([self.agent_adherence], dtype=np.float32)  # already in [0,1]
        
        # Create agent infection status indicator (1 if infected, 0 otherwise)
        is_agent_infected = np.array([1.0 if self.agent_state == STATE_DICT['I'] else 0.0], dtype=np.float32)

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
        features_per_human = 5 if self.use_visibility_flag else 4

        # Initialize array for human observations
        humans_features = np.zeros((self.n_humans * features_per_human,), dtype=np.float32)

        # Fill in human observations
        for i, current_human in enumerate(self.humans):
            # Calculate base index for this human's features
            base_idx = i * features_per_human
            
            # Set infection status (1 for infected, 0 for all other states)
            is_infected = 1.0 if current_human.state == STATE_DICT['I'] else 0.0
            
            # Calculate relative position (delta_x, delta_y)
            delta_x = current_human.x - self.agent_position[0]
            delta_y = current_human.y - self.agent_position[1]
            
            # Handle wraparound for periodic boundary conditions
            if abs(delta_x) > self.grid_size / 2:
                delta_x = delta_x - np.sign(delta_x) * self.grid_size
            if abs(delta_y) > self.grid_size / 2:
                delta_y = delta_y - np.sign(delta_y) * self.grid_size
                
            # Normalize delta_x and delta_y to [-0.5, 0.5] range
            # This represents relative position scaled by grid size
            delta_x_norm = delta_x / self.grid_size
            delta_y_norm = delta_y / self.grid_size
            
            # Calculate distance
            dist = self._calculate_distance(agent_human, current_human)
            dist_norm = dist / self.max_distance
            
            if self.use_visibility_flag:
                # Mode with visibility flag
                visibility_flag = 1.0 if current_human.id in visible_ids else 0.0
                
                if visibility_flag == 0.0:
                    # Invisible human - set position values to 0, but keep infection status
                    humans_features[base_idx:base_idx + 5] = [0.0, 0.0, 0.0, 0.0, is_infected]
                else:
                    # Visible human - include all information
                    humans_features[base_idx:base_idx + 5] = [visibility_flag, delta_x_norm, delta_y_norm, dist_norm, is_infected]
            else:
                # Simple mode without visibility flag (all humans visible)
                humans_features[base_idx:base_idx + 4] = [delta_x_norm, delta_y_norm, dist_norm, is_infected]

        # Compose final observation dict
        obs = {
            # "agent_position": agent_position,
            "agent_adherence": agent_adherence,
            "is_agent_infected": is_agent_infected,
            "humans_features": humans_features
        }
        return obs

    def _calculate_strategic_avoidance_reward(self):
        """
        Advanced reward function designed to teach strategic avoidance of infected individuals
        with optimal adherence management. Provides rich gradient information for the agent
        to learn effective navigation in the presence of infection risks.
        
        Key components:
        1. Survival incentive - base reward for staying alive and susceptible
        2. Distance management - sophisticated scaling of distance to infected individuals
        3. Adherence optimization - rewards for appropriate adherence based on infection proximity
        4. Risk gradient - continuous evaluation of infection risk
        5. Strategic planning - rewards for positioning that allows escape routes
        """
        # Create temporary agent human for distance calculations
        agent_human = Human(
            x=self.agent_position[0],
            y=self.agent_position[1],
            state=self.agent_state,
            id=-1
        )
        
        # Get infected individuals (respecting visibility if enabled)
        infected_list = self._get_infected_list(agent_human)
        
        # Calculate distances to infected humans if any exist
        if infected_list:
            distances = [self._calculate_distance(agent_human, h) for h in infected_list]
            min_distance = min(distances)
            # avg_distance = sum(distances) / len(distances)
            # Count nearby infected (within 2x safe distance)
            # nearby_infected_count = sum(1 for d in distances if d < self.safe_distance * 2)
        else:
            # Use a large value if no infected are present
            max_distance = math.sqrt(2) * self.grid_size
            min_distance = max_distance
            # avg_distance = max_distance
            # nearby_infected_count = 0
        
        # Calculate current infection probability
        infection_probability = self._calculate_infection_probability(agent_human, is_agent=True)
        
        # 1. Base survival reward [0.2] - small constant reward for staying alive and susceptible
        if self.agent_state == STATE_DICT['S']:
            survival_reward = 0.2
        else:  # Infected, Recovered, or Dead
            survival_reward = 0
            
        # 2. Distance management reward [0 to 0.4] - reward for maximizing distance to infected
        safe_dist = max(1, self.safe_distance)  # Ensure at least 1 for sensible distance scaling
        max_grid_dist = math.sqrt(2) * self.grid_size  # Maximum possible distance
        
        if infected_list:
            # Scale the minimum distance to [0,1] - closer to 1 means better distance management
            norm_dist = min(min_distance / max_grid_dist, 1.0)
            
            # Create a sigmoid-like function centered at safe distance for smooth reward
            # This gives diminishing returns as we go beyond safe distance
            midpoint = safe_dist / max_grid_dist  # Centered at safe distance
            # Shifted sigmoid to create peak reward around the safe distance
            distance_reward = 0.4 * (1 / (1 + math.exp(-10 * (norm_dist - midpoint)))) 
        else:
            # If no infected visible, maximum reward
            distance_reward = 0.4
            
        # 3. Adherence optimization [0 to 0.3] - rewards strategic use of adherence
        ideal_adherence = 0.0  # Default when no infection nearby
        
        if infected_list:
            # Calculate ideal adherence based on proximity to infection
            # Closer to infected → higher ideal adherence
            proximity_factor = 1.0 - min(min_distance / (self.safe_distance * 3), 1.0)
            ideal_adherence = proximity_factor
            
            # Scale adherence reward - max reward when adherence matches ideal
            adherence_diff = abs(self.agent_adherence - ideal_adherence)
            adherence_reward = 0.3 * (1.0 - adherence_diff)
        else:
            # When no infected visible, reward lower adherence (less cost)
            adherence_reward = 0.3 * (1.0 - self.agent_adherence)
            
        # 4. Risk avoidance component [0 to 0.3] - continuous gradient based on infection risk
        # Transform infection probability with decreasing marginal penalties
        # (small risks aren't severely punished, but high risks are)
        risk_reward = 0.3 * (1.0 - np.power(infection_probability, 0.7))
        
        # 5. Exploration incentive [-0.1 to 0] - small cost to encourage movement
        # Only apply when there are no infected nearby to avoid conflicting incentives
        if not infected_list:
            # Check if the agent has moved significantly (at least 10% of grid_size)
            if hasattr(self, 'previous_position'):
                distance_moved = np.sqrt(np.sum(np.square(
                    np.array(self.agent_position) - np.array(self.previous_position)
                )))
                movement_factor = min(distance_moved / (0.1 * self.grid_size), 1.0)
                exploration_cost = -0.1 * (1.0 - movement_factor)
            else:
                exploration_cost = -0.05  # Moderate cost when we don't have a previous position
        else:
            exploration_cost = 0  # No exploration cost when there are infected nearby
            
        # Store current position for next step's movement calculation
        self.previous_position = self.agent_position.copy()
        
        # Combine all components
        combined_reward = (
            survival_reward +     # [0 to 0.2] - Base survival
            distance_reward +     # [0 to 0.4] - Distance management
            adherence_reward +    # [0 to 0.3] - Adherence optimization
            risk_reward +         # [0 to 0.3] - Risk avoidance
            exploration_cost      # [-0.1 to 0] - Exploration incentive
        )
        
        # Final reward has a theoretical range of [-0.1, 1.2]
        # But typical values will stay in [-0.1, 1.0] range
        return combined_reward

    def _calculate_distance_maximization_reward(self):
        """
        Reward function focused on maximizing distance to infected individuals.
        
        This function promotes:
        1. Staying alive and susceptible (not getting infected)
        2. Maximizing distance from infected individuals
        3. Using adherence appropriately based on infection proximity
        """
        # Create temporary agent human for distance calculations
        agent_human = Human(
            x=self.agent_position[0],
            y=self.agent_position[1],
            state=self.agent_state,
            id=-1
        )
        
        # Get infected individuals (respecting visibility if enabled)
        infected_list = self._get_infected_list(agent_human)
        
        # Base survival component - reward for staying susceptible
        if self.agent_state == STATE_DICT['S']:
            survival_bonus = 0.2  # Base reward for staying susceptible
        else:
            survival_bonus = 0  # No survival bonus if infected/dead/recovered

        # Distance component - reward increases with distance from infected
        if infected_list:
            # Calculate distances to all infected
            distances = [self._calculate_distance(agent_human, h) for h in infected_list]
            min_distance = min(distances)
            max_possible_distance = math.sqrt(2) * self.grid_size
            
            # Normalize distance to [0,1] range
            normalized_distance = min(min_distance / max_possible_distance, 1.0)
            
            # Calculate reward based on normalized distance
            # Use a sigmoid function to create stronger gradient at mid-distances
            distance_reward = 0.4 * (1 / (1 + math.exp(-10 * (normalized_distance - 0.3))))
        else:
            # Maximum reward if no infected are present
            distance_reward = 0.4
        
        # Combine rewards
        final_reward = survival_bonus + distance_reward
        
        return final_reward

    def _calculate_weighted_avoidance_reward(self):
        """
        Custom reward function that rewards:
        1. Maximizing distance to nearest infected person (40% weight)
        2. Reducing infection probability (40% weight)
        3. Staying alive bonus (10% for susceptible, 0 otherwise)
        
        Total reward is normalized to range approximately [0, 1]
        """
        # Create temporary agent human for calculations
        agent_human = Human(
            x=self.agent_position[0],
            y=self.agent_position[1],
            state=self.agent_state,
            id=-1
        )

        if self.agent_state != STATE_DICT['S']:
            return 0
        
        if self.agent_state == STATE_DICT['I']:
            return - self.simulation_time / 4 # because we want it to last at least this long
        
        # Get infected individuals
        infected_list = self._get_infected_list(agent_human)
        
        # Component 1: Distance to nearest infected (40% weight)
        if infected_list:
            # Calculate distances to all infected
            distances = [self._calculate_distance(agent_human, h) for h in infected_list]
            min_distance = min(distances)
            max_possible_distance = math.sqrt(2) * self.grid_size
            
            # Normalize distance to [0,1] range
            normalized_distance = min(min_distance / max_possible_distance, 1.0)
            
            # Apply weight of 0.4
            distance_reward = 0.4 * normalized_distance
        else:
            # Maximum reward if no infected are present
            distance_reward = 0.4
        
        # Component 2: Infection probability reduction (40% weight)
        if self.agent_state == STATE_DICT['S']:
            # Calculate current infection probability
            infection_prob = self._calculate_infection_probability(agent_human, is_agent=True)
            
            # Normalize and invert: lower probability = higher reward
            infection_prob_reward = 0.4 * (1.0 - infection_prob)
        else:
            # If already infected/recovered/dead, no reward for this component
            infection_prob_reward = 0.0
        
        # Component 3: Stay alive bonus (10% weight)
        if self.agent_state == STATE_DICT['S']:
            survival_bonus = 0.1  # Fixed bonus for being susceptible
        else:
            survival_bonus = 0.0  # No bonus for other states
        
        # Combine all reward components
        final_reward = distance_reward + infection_prob_reward + survival_bonus
        
        return final_reward

    def _calculate_infection_avoidance_reward(self):
        """
        Reward function focused primarily on minimizing infection probability.
        
        Key components:
        1. Infection probability minimization (70% weight) - higher reward for lower probability
        2. State-based rewards/penalties:
           - Susceptible state: Small positive reward (10%)
           - Infected state: Strong negative reward (-1.0)
           - Recovered state: Neutral reward (0)
           - Dead state: Strong negative reward (-2.0)
        3. Adherence optimization (20% weight) - rewards efficient use of adherence
        
        The function encourages the agent to take actions that minimize infection risk
        while using adherence judiciously.
        """
        # Create temporary agent human for calculations
        agent_human = Human(
            x=self.agent_position[0],
            y=self.agent_position[1],
            state=self.agent_state,
            id=-1
        )
        
        # Component 1: State-based rewards/penalties
        if self.agent_state == STATE_DICT['S']:
            state_reward = 0.1  # Small positive reward for staying susceptible
        else:  # infected 
            state_reward = -1
            return state_reward
            
        # Component 2: Infection probability minimization (70% weight)
        # Calculate current infection probability
        infection_prob = self._calculate_infection_probability(agent_human, is_agent=True)
        
        # Apply a non-linear transformation to the probability
        # This creates a stronger gradient as probability increases
        # (e.g., reducing from 0.5 to 0.4 gives more reward than from 0.2 to 0.1)
        infection_avoidance_reward = 0.7 * (1.0 - math.pow(infection_prob, 0.5))
        
        # Component 3: Adherence optimization (20% weight)
        # Create an adherence cost that increases with higher adherence
        # But make it conditional on infection probability
        if infection_prob > 0.2:  
            # Higher infection risk - adherence should be higher
            ideal_adherence = 0.5 + (infection_prob * 0.5)  # Scales from 0.6 to 1.0 as risk increases
            adherence_diff = abs(self.agent_adherence - ideal_adherence)
            adherence_reward = 0.2 * (1.0 - adherence_diff)
        else:
            # Lower infection risk - lower adherence is fine to reduce cost
            adherence_reward = 0.2 * (1.0 - self.agent_adherence)
        
        # Combine components
        final_reward = state_reward + infection_avoidance_reward + adherence_reward
        
        return final_reward

    def _calculate_minimize_exposure_reward(self):
        """
        Reward function focused on minimizing exposure to infected individuals.
        
        Key components:
        1. Base reward for being susceptible (10%)
        2. Exposure minimization (70%) - rewards lower total exposure to infected individuals
        3. Adherence optimization (20%) - rewards efficient use of adherence based on exposure level
        
        This function encourages the agent to minimize its exposure to infected individuals
        while using adherence strategically based on the current risk level.
        """
        # Create temporary agent human for calculations
        agent_human = Human(
            x=self.agent_position[0],
            y=self.agent_position[1],
            state=self.agent_state,
            id=-1
        )
        
        # Component 1: State-based rewards/penalties
        if self.agent_state == STATE_DICT['S']:
            state_reward = 0.2  # Small positive reward for staying susceptible
        else:  # infected 
            state_reward = -10
            return state_reward
            
        # Component 2: Exposure minimization (70% weight)
        # Calculate total exposure using the existing function
        total_exposure = self._calculate_total_exposure(agent_human)
        
        # Normalize exposure to [0,1] range
        # Assuming max possible exposure is when all infected are at distance 0
        infected_list = self._get_infected_list(agent_human)
        if infected_list:
            max_possible_exposure = len(infected_list)
            normalized_exposure = min(total_exposure / max_possible_exposure, 1.0)
            
            # Invert and apply weight: lower exposure = higher reward
            exposure_reward = 0.6 * (1.0 - normalized_exposure)
        else:
            # Maximum reward if no infected are present
            exposure_reward = 0.6
        
        # Component 3: Adherence optimization (20% weight)
        # Create an adherence cost that increases with higher adherence
        # But make it conditional on infection probability
        if total_exposure > 0.2:  
            # Higher infection risk - adherence should be higher
            ideal_adherence = min(0.5 + (total_exposure * 0.5), 1.0)  # Scales from 0.6 to 1.0 as exposure increases
            adherence_diff = abs(self.agent_adherence - ideal_adherence)
            adherence_reward = 0.2 * (1.0 - adherence_diff)
        else:
            # Lower infection risk - lower adherence is fine to reduce cost
            adherence_reward = 0.2 * (1.0 - self.agent_adherence)
        
        # Combine components
        final_reward = state_reward + exposure_reward + adherence_reward
        
        return final_reward
        multiplicative_reward = exposure_reward/0.6 * adherence_reward/0.2 * state_reward/0.2
        return (final_reward + multiplicative_reward)/2

    def _calculate_strategic_survival_reward(self):
        """
        Enhanced reward function with sharper gradients and more immediate feedback
        to improve explained variance.
        
        Key improvements:
        1. More frequent positive feedback with granular rewards
        2. Sharper distance-based gradients with exponential scaling
        3. More direct adherence feedback based on current risk
        4. Immediate rewards for positive actions
        
        Returns a reward with clearer action-reward correlation
        """
        # Create temporary agent human for calculations
        agent_human = Human(
            x=self.agent_position[0],
            y=self.agent_position[1],
            state=self.agent_state,
            id=-1
        )
        
        # COMPONENT 1: State-based reward (50% of total)
        # Only reward susceptible state, strongly penalize all others
        if self.agent_state != STATE_DICT['S']:
            return -2.0  # Stronger negative reward for non-susceptible states
        
        # Base reward for being susceptible
        base_reward = 0.5
        
        # Get infected individuals
        infected_list = self._get_infected_list(agent_human)
        
        # COMPONENT 2: Distance-based reward (30% of total)
        # Using exponential scaling for sharper gradients
        if infected_list:
            # Calculate distances to all infected
            distances = [self._calculate_distance(agent_human, h) for h in infected_list]
            min_distance = min(distances)
            
            # Calculate safe distance ratio with sharper gradient
            safe_dist = max(1.0, self.safe_distance)
            distance_ratio = min_distance / safe_dist
            
            # Exponential reward function with sharper gradient
            # Scales from ~0 at distance=0 to ~0.3 at distance=safe_distance
            # and approaches 0.3 asymptotically beyond that
            distance_reward = 0.3 * (1.0 - math.exp(-2.0 * distance_ratio))
            
            # Add bonus for being beyond safe distance
            if min_distance > safe_dist:
                distance_reward += 0.1
        else:
            # Maximum reward if no infected are present
            distance_reward = 0.4
        
        # COMPONENT 3: Exposure-based reward (20% of total)
        # Direct feedback on current exposure level
        if infected_list:
            # Calculate total exposure
            total_exposure = self._calculate_total_exposure(agent_human)
            
            # Inverse exponential reward - higher exposure = lower reward
            # Provides sharper gradient for exposure reduction
            exposure_reward = 0.2 * math.exp(-3.0 * total_exposure)
        else:
            # Maximum reward if no infected are present
            exposure_reward = 0.2
        
        # COMPONENT 4: Adherence optimization (20% of total)
        # More direct feedback on adherence decisions
        if infected_list:
            # Calculate infection probability
            infection_prob = self._calculate_infection_probability(agent_human, is_agent=True)
            
            # Determine ideal adherence based on infection probability
            # Higher risk → higher ideal adherence with sharper transition
            if infection_prob > 0.1:  # Significant risk threshold
                ideal_adherence = min(0.8, infection_prob * 2.0)  # Scale up to 0.8 max
                adherence_diff = abs(self.agent_adherence - ideal_adherence)
                
                # Sharper penalty for incorrect adherence when risk is high
                adherence_reward = 0.2 * (1.0 - (adherence_diff * 2.0))  # Doubled penalty
                adherence_reward = max(0.0, adherence_reward)  # Ensure non-negative
            else:
                # Low risk - reward lower adherence with clear gradient
                adherence_reward = 0.2 * (1.0 - self.agent_adherence)
        else:
            # When no infected present, strongly reward lower adherence
            adherence_reward = 0.2 * (1.0 - self.agent_adherence)
        
        # COMPONENT 5: Movement reward (10% of total)
        # Reward effective movement away from infected
        if hasattr(self, 'previous_position') and infected_list:
            # Calculate previous and current minimum distances
            prev_distances = []
            for infected in infected_list:
                prev_human = Human(
                    x=self.previous_position[0],
                    y=self.previous_position[1],
                    state=self.agent_state,
                    id=-1
                )
                prev_distances.append(self._calculate_distance(prev_human, infected))
            
            prev_min_distance = min(prev_distances) if prev_distances else 0
            current_min_distance = min_distance
            
            # Reward increasing distance from infected
            distance_change = current_min_distance - prev_min_distance
            movement_reward = 0.1 * (1.0 / (1.0 + math.exp(-5.0 * distance_change)))
        else:
            movement_reward = 0.05  # Neutral movement reward
        
        # Store current position for next step
        self.previous_position = self.agent_position.copy()
        
        # Combine all reward components
        final_reward = base_reward + distance_reward + exposure_reward + adherence_reward + movement_reward
        
        return final_reward
    
    def _calculate_reduceInfectionProb_reward(self):
        """
        Reward function that encourages the agent to reduce the infection probability of the population.
        """
        # Calculate the infection probability of the agent
        if self.agent_state != STATE_DICT['S']:
            return -5
        agent_human = Human(
            x=self.agent_position[0],
            y=self.agent_position[1],
            state=self.agent_state,
            id=-1
        )
        infection_prob = self._calculate_infection_probability(agent_human, is_agent=True)
        return (1-infection_prob)**2
        


    def _calculate_reward(self):    
        """Calculate the reward based on the selected reward type"""
        reward_functions = {
            "strategicAvoidance": self._calculate_strategic_avoidance_reward,
            "distanceMaximization": self._calculate_distance_maximization_reward,
            "weightedAvoidance": self._calculate_weighted_avoidance_reward,
            "infectionAvoidance": self._calculate_infection_avoidance_reward,
            "minimizeExposure": self._calculate_minimize_exposure_reward,
            "strategicSurvival": self._calculate_strategic_survival_reward, 
            "reduceInfectionProb": self._calculate_reduceInfectionProb_reward
        }
        if self.reward_type not in reward_functions:  
            raise ValueError(f"Invalid reward type: {self.reward_type}")
        
        return reward_functions[self.reward_type]()

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
        previous_state = self.agent_state
        
        self._update_agent(action) 
        self._handle_human_stepping()

        observation = self._get_observation()
        reward = self._calculate_reward()
        self.cumulative_reward += reward  # Update cumulative reward
     
        # Check if agent became infected in this step
        became_infected = previous_state != STATE_DICT['I'] and self.agent_state == STATE_DICT['I']
        
        # # Apply strong negative terminal reward if agent became infected
        # if became_infected:
        #     infection_penalty = - self.simulation_time / 4  # Strong negative reward for becoming infected
        #     # used -50 for penalty in the test_weighted_avoidance 20250305 1546
        #     reward += infection_penalty
        #     self.cumulative_reward += infection_penalty
        #     # Add to info dict for monitoring
        #     self.infection_penalty_applied = True
        
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
        
        # Truncate if:
        # 1. The agent dies, or
        # 2. There are no infected individuals AND either:
        #    a) reinfection is disabled (reinfection_count == 0) or
        #    b) there aren't enough susceptible humans beyond safe distance for reinfection
        if (self.agent_state != STATE_DICT['S'] or
            # self.agent_state == STATE_DICT['D'] or
            # self.agent_state == STATE_DICT['I'] or
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
                "agent_infected" if self.agent_state == STATE_DICT['I'] else
                "no_infection_possible" if truncated else None
            ),
            "agent_state": [k for k, v in STATE_DICT.items() if v == self.agent_state][0],
            "episode_length": self.counter,
            "healthy_count": sum(1 for h in self.humans if h.state == STATE_DICT['S']),
            "infected_count": sum(1 for h in self.humans if h.state == STATE_DICT['I']),
            "recovered_count": sum(1 for h in self.humans if h.state == STATE_DICT['R']),
            "dead_count": sum(1 for h in self.humans if h.state == STATE_DICT['D']),
            "adherence": float(self.agent_adherence),
            "terminal_infection_penalty": became_infected,  # Flag if terminal penalty was applied
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