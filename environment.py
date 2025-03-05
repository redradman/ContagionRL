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
        visibility_radius: float = -1,
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
                high=1,  # Changed to 1 since we're using one-hot encoding
                shape=(self.n_humans * 4,),  # 4 states: [S, I, R, D] for each human
                dtype=np.float32  # Changed to float32 for one-hot vectors
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
            return min(1,(self.beta / (1 + 10 * self.agent_adherence)) * total_exposure)
        else:
            return min(1,(self.beta) * total_exposure)

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
        humans_discrete = np.zeros((self.n_humans * 4,), dtype=np.float32)

        # Fill in human observations
        for i, human in enumerate(self.humans):
            base_idx = i * 4  # Index for continuous features
            visibility_flag = 1.0 if human.id in visible_ids else 0.0
            
            if visibility_flag == 0: 
                # All values remain 0 for invisible humans
                humans_continuous[base_idx:base_idx + 4] = 0
                humans_discrete[base_idx:base_idx + 4] = 0
            else:
                # Normalize positions and calculate distance
                x_norm = human.x / self.grid_size
                y_norm = human.y / self.grid_size
                dist = self._calculate_distance(agent_human, human)
                dist_norm = dist / self.max_distance

                # Set continuous features [visibility, x, y, distance]
                humans_continuous[base_idx:base_idx + 4] = [visibility_flag, x_norm, y_norm, dist_norm]
                # Set discrete state
                humans_discrete[base_idx:base_idx + 4] = [0, 0, 0, 0]
                humans_discrete[base_idx + human.state] = 1

        # Compose final observation dict
        obs = {
            "agent_position": agent_position,
            "agent_adherence": agent_adherence,
            "humans_continuous": humans_continuous,
            "humans_discrete": humans_discrete
        }

        return obs
        

    def _calculate_stateBased_reward(self):
        """Basic health-focused reward"""
        if self.agent_state == STATE_DICT['S']:
            state_reward = 1
        else:  # Infected or Dead or Recovered
            state_reward = 0

        reward = state_reward - self.agent_adherence**2 / self.adherence_penalty_factor
        reward = max(0, reward)
        return reward

    def _calculate_avoidInfection_reward(self):
        """Basic altruistic reward focusing on population health"""
        # make human object for the agent to use existing infection probability calculation
        agent_human = Human(
                x=self.agent_position[0],
                y=self.agent_position[1],
                state=self.agent_state,
                id=-1,
                time_in_state=self.agent_time_in_state  # Pass the agent's time in state
        )
        
        probability_infection = self._calculate_infection_probability(agent_human, is_agent=True)
        avoid_infection_reward = 1 - np.exp(-probability_infection*5) # constant multiplier for the reward here (flag)

        reward = avoid_infection_reward - self.agent_adherence / self.adherence_penalty_factor
        reward = max(0, reward)
        return reward

    def _calculate_reward_for_state(self):
        """
        Additive reward function combining:
        1. State reward: Base reward for being susceptible
        2. Distance reward: Reward for maintaining distance from infected
        3. NPI penalty: Small penalty for high adherence
        Final reward = w1*state_reward + w2*distance_reward - w3*adherence_penalty
        """
        # 1. State reward component (40% of total possible reward)
        if self.agent_state == STATE_DICT['S']:
            state_reward = 1
        else:  # Dead
            state_reward = 0

        # 2. Distance reward component (40% of total possible reward)
        agent_human = Human(
            x=self.agent_position[0],
            y=self.agent_position[1],
            state=self.agent_state,
            id=-1,
            time_in_state=self.agent_time_in_state
        )
        
        # Get list of infected humans
        infected_list = self._get_infected_list(agent_human)
        
        # Calculate distance reward
        distance_reward = 1
        if infected_list:
            # Find distance to closest infected
            distances = [self._calculate_distance(agent_human, infected) for infected in infected_list]
            min_distance = min(distances)
            
            distance_reward = 1 - math.exp(-self.distance_decay * min_distance)
        
        # 3. Calculate adherence penalty (20% of total possible reward)
        # Quadratic penalty that grows faster at higher adherence levels
        adherence_penalty = (self.agent_adherence ** 2) / self.adherence_penalty_factor
        
        # Combine rewards additively with weights
        final_reward = 0.4 * state_reward + 0.4 * distance_reward - 0.2 * adherence_penalty
        # final_reward = state_reward * distance_reward * (1 - adherence_penalty)
        
        return final_reward

    def _calculate_infection_avoidance_reward(self):
        """
        Reward function focused on infection avoidance with balanced components.
        Provides continuous gradient for learning how to avoid infected individuals.
        """
        # Create temporary agent human for distance calculations
        agent_human = Human(
            x=self.agent_position[0],
            y=self.agent_position[1],
            state=self.agent_state,
            id=-1
        )
        
        # Get visible infected individuals
        infected_list = [h for h in self.humans if h.state == STATE_DICT['I'] and 
                      self._calculate_distance(agent_human, h) <= self.visibility_radius]
        
        # Calculate current infection probability (used for risk assessment)
        infection_probability = self._calculate_infection_probability(agent_human, is_agent=True)
        
        # 1. Health state component [-0.6 to 0.6] - core incentive for staying healthy
        if self.agent_state == STATE_DICT['S']:
            health_reward = 0.6  # Significant reward for staying susceptible
        elif self.agent_state == STATE_DICT['I']:
            health_reward = -0.6  # Significant penalty for becoming infected
        elif self.agent_state == STATE_DICT['R']:
            health_reward = 0
        else:  # Dead
            health_reward = -0.6  # Same penalty as infected (we already penalize via episode termination)
        
        # 2. Distance component [0 to 0.3] - incentivize maintaining distance
        if infected_list and self.agent_state != STATE_DICT['D']:  # Not relevant if dead
            # Calculate distances to infected humans
            distances = [self._calculate_distance(agent_human, infected) for infected in infected_list]
            min_distance = min(distances)
            
            # Normalize minimum distance to [0, 1] range using sigmoid-like function
            # This creates a smooth reward curve that increases with distance
            normalized_distance = 1 - np.exp(-self.distance_decay * min_distance)
            distance_reward = 0.3 * normalized_distance
        else:
            # Maximum reward if no infected are visible
            distance_reward = 0.3
            
        # 3. Risk assessment component [0 to 0.3] - reward for avoiding risky areas
        risk_reward = 0.3 * (1 - infection_probability)
        
        # 4. Adherence cost component [-0.2 to 0] - small cost proportional to adherence
        # Using a scaled cost that's gentler at lower adherence levels
        adherence_cost = -0.2 * (self.agent_adherence / self.adherence_penalty_factor)
        
        # Combine all components - total range is guaranteed to be [-0.8, 1.2]
        # Further normalize to ensure reward stays in a consistent range
        combined_reward = health_reward + distance_reward + risk_reward + adherence_cost
        
        # Final scaling to ensure the typical range is closer to [-1, 1]
        # Note: The absolute maximum is still slightly outside this range,
        # but typical values will be well-behaved
        final_reward = combined_reward / 1.2
        
        return final_reward

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
        
        # Get visible infected and susceptible individuals
        infected_list = [h for h in self.humans if h.state == STATE_DICT['I'] and 
                     self._calculate_distance(agent_human, h) <= self.visibility_radius]
        
        # Calculate distances to infected humans if any are visible
        if infected_list:
            distances_to_infected = [self._calculate_distance(agent_human, h) for h in infected_list]
            min_distance = min(distances_to_infected)
            avg_distance = sum(distances_to_infected) / len(distances_to_infected)
            # Count nearby infected (within 2x safe distance)
            nearby_infected_count = sum(1 for d in distances_to_infected if d < self.safe_distance * 2)
        else:
            min_distance = self.visibility_radius  # Maximum possible if no infected visible
            avg_distance = self.visibility_radius
            nearby_infected_count = 0
        
        # Calculate current infection probability
        infection_probability = self._calculate_infection_probability(agent_human, is_agent=True)
        
        # 1. Base survival reward [0.2] - small constant reward for staying alive and susceptible
        if self.agent_state == STATE_DICT['S']:
            survival_reward = 0.2
        else:  # Infected, Recovered, or Dead
            survival_reward = 0
            
        # 2. Distance management reward [0 to 0.4] - sophisticated scaling based on safe distance
        if infected_list:
            # Calculate a smooth distance reward that increases as distance increases
            # Uses a logistic function to create a steeper gradient near the safe distance
            safe_dist = self.safe_distance
            # Normalized distance from 0 to 1, where 1 is at visibility radius
            norm_dist = min(min_distance / self.visibility_radius, 1.0)
            
            # Logistic function centered at safe distance (steeper penalty below safe distance)
            k = 10  # Steepness of the curve
            midpoint = safe_dist / self.visibility_radius  # Centered at safe distance
            distance_reward = 0.4 / (1 + np.exp(-k * (norm_dist - midpoint)))
            
            # Additional penalty for having multiple infected nearby (crowding factor)
            if nearby_infected_count > 1:
                crowd_penalty = 0.1 * min(nearby_infected_count - 1, 3)  # Cap at 0.3 penalty
                distance_reward = max(distance_reward - crowd_penalty, 0)
        else:
            # Maximum reward if no infected are visible
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

    def _calculate_reward(self):    
        """Calculate the reward based on the selected reward type"""
        reward_functions = {
            "stateBased": self._calculate_stateBased_reward,
            "avoidInfection": self._calculate_avoidInfection_reward,
            "rewardForState": self._calculate_reward_for_state,
            "infectionAvoidance": self._calculate_infection_avoidance_reward,
            "strategicAvoidance": self._calculate_strategic_avoidance_reward
        }
        if self.reward_type not in reward_functions:  
            raise ValueError(f"Invalid reward type: {self.reward_type}")
        
        return reward_functions[self.reward_type]()

    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, bool, dict]:
        """Take a step in the environment"""
        self.counter += 1
        self.last_action = action.copy()
        
        # Store current state before update for terminal reward calculation
        previous_state = self.agent_state
        
        self._update_agent(action) 
        self._handle_human_stepping()

        observation = self._get_observation()
        reward = self._calculate_reward()
        self.cumulative_reward += reward  # Update cumulative reward
     
        # Check if agent became infected in this step
        became_infected = previous_state != STATE_DICT['I'] and self.agent_state == STATE_DICT['I']
        
        # Apply strong negative terminal reward if agent became infected
        if became_infected:
            infection_penalty = -5.0  # Strong negative reward for becoming infected
            reward += infection_penalty
            self.cumulative_reward += infection_penalty
            # Add to info dict for monitoring
            self.infection_penalty_applied = True
        
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
        if (self.agent_state == STATE_DICT['D'] or
            self.agent_state == STATE_DICT['I'] or
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

        # Draw visibility radius if it's not -1 (full visibility)
        if self.visibility_radius != -1:
            # Draw circles for all periodic images that might be visible
            # We need to consider the 9 possible positions (including wraparound)
            for dx in [-self.grid_size, 0, self.grid_size]:
                for dy in [-self.grid_size, 0, self.grid_size]:
                    circle = plt.Circle(
                        (self.agent_position[0] + dx, self.agent_position[1] + dy),
                        self.visibility_radius,
                        color=self.COLORS['agent'],
                        fill=False,
                        linestyle='--',
                        alpha=0.3,
                        zorder=4
                    )
                    ax_grid.add_patch(circle)

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