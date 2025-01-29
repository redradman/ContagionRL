import gymnasium as gym
import numpy as np
from typing import Optional, List, Tuple
import math
from utils import STATE_DICT, MovementHandler, Human

######## SIRS Environment class ########
class SIRSEnvironment(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

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
        render_mode: Optional[str] = None, # TODO: add rendering
    ):
        super().__init__()

        # General parameters
        self.simulation_time = simulation_time
        self.counter = 0 # counter for the simulation time
        self.grid_size = grid_size # from 0 to grid_size exclusive for both of the x and y axis

        # Agent parameters that are handled by the env
        self.agent_position = np.array([self.grid_size//2, self.grid_size//2]) # initial position of the agent
        self.initial_agent_adherence = initial_agent_adherence # NPI adherence
        self.agent_adherence = initial_agent_adherence # NPI adherence
        self.agent_state = STATE_DICT['S'] # initial state of the agent

        # SIRS parameters
        self.n_humans = n_humans
        self.n_infected = n_infected
        self.beta = beta # infection rate
        self.distance_decay = distance_decay # distance decay rate
        self.lethality = lethality # lethality rate
        self.immunity_decay = immunity_decay # immunity decay rate
        self.recovery_rate = recovery_rate # recovery rate
        self.max_immunity_loss_prob = max_immunity_loss_prob # maximum immunity loss probability
        self.visibility_radius = visibility_radius # visibility radius

        # Observation and Action spaces
        # not defined yet as it requires careful design
        # Define observation space (will be used by the RL agent later)
        # self.observation_space = gym.spaces.Dict({
        #     "agent_position": gym.spaces.Box(low=0, high=self.grid_size, shape=(2,), dtype=np.int32),
        #     "npi_level": gym.spaces.Box(low=0, high=1, shape=(), dtype=np.float32),
        #     "visible_humans": gym.spaces.Box(low=0, high=self.grid_size, shape=(self.n_humans, 4), dtype=np.float32) # x, y, state, time_in_state (full details of all of the visible humans)
        # })


        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        # initialize humans list
        self.humans: List[Human] = [] 
        # Movement handler  
        self.movement_handler = MovementHandler(grid_size, movement_type)
    ####### TRANSITION FUNCTIONS FOR MOVING BETWEEN S, I, R AND DEAD #######

    def _calculate_distance(self, human1: Human, human2: Human) -> float:
        """
        Calculate the minimum distance between two humans in a periodic grid:
            - Take two humans as input
            - Return the min distance considering grid wrapping
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
        # Initialize humans
        self.humans = []
        positions = set()
        
        # Place humans randomly
        for i in range(self.n_humans):
            while True:
                x = self.np_random.integers(0, self.grid_size)
                y = self.np_random.integers(0, self.grid_size)
                if (x, y) not in positions: # to ensure the uniqueness of the position
                    positions.add((x, y))
                    break
            
            self.humans.append(Human(i, x, y, STATE_DICT['S'])) # i would be the id, x and y are positions, init state is S

        # Select random humans to be infected
        initial_infected = self.np_random.choice(self.humans, self.n_infected, replace=False)
        for human in initial_infected:
            human.update_state(STATE_DICT['I'])

        return self._get_observation(), {}

    def _update_agent(self, action: np.ndarray) -> None:
        """Update the agent status in the environment"""
        self.agent_position = action[:2] # update position of the agent
        self.agent_adherence = action[2] # update NPI level
    
    def _handle_human_stepping(self):
        """Handle the stepping of a human"""
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
                p_immunity_loss = self.max_immunity_loss_prob * (1 - math.exp(-self.immunity_decay * human.time_in_state))
                if self.np_random.random() < p_immunity_loss:
                    human.update_state(STATE_DICT['S'])

    def _get_observation(self):
        """Get the observation for the agent"""
        # TODO: implement observation logic, make sure to specify the observation space in the fields part of the class
        return {}

    def _calculate_reward(self):    
        """Calculate the reward for the agent"""
        # TODO: implement reward logic
        return 0

    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, bool, dict]:
        """Take a step in the environment. For more regarding the structure refer to the gymnasium documentation"""

        self.counter += 1
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
        # can add the check here to check if all the humans are dead or not. or if any infected human exists

        # handle truncation logic
        truncated = False
        if self.agent_state == STATE_DICT['D']: # truncate if the agent is dead
            truncated = True


        # handle info logic 
        info = {}


        # return the observation, reward, truncation, termination, info
        return observation, reward, terminated, truncated, info