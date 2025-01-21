import gymnasium as gym
import numpy as np
from typing import Optional, List, Tuple
import math
from utils import STATE_DICT, ReplayBuffer, MovementHandler, Human

######## SIRS Environment class ########
class SIRSEnvironment(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        grid_size: int = 20,
        n_humans: int = 100,
        n_infected: int = 5,
        beta: float = 0.3,
        adherence: float = 0.5,
        distance_decay: float = 0.2,
        lethality: float = 0.1,
        immunity_decay: float = 0.1,
        recovery_rate: float = 0.1,
        max_immunity_loss_prob: float = 0.2,
        movement_type: str = "stationary",
        visibility_radius: float = -1,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # Store parameters
        self.grid_size = grid_size
        self.agent_position = np.array([self.grid_size//2, self.grid_size//2]) # initial position of the agent
        self.npi_level = 0 # initial NPI level

        # SIRS parameters
        self.n_humans = n_humans
        self.n_infected = n_infected
        self.beta = beta # infection rate
        self.adherence = adherence # NPI adherence
        self.distance_decay = distance_decay # distance decay rate
        self.lethality = lethality # lethality rate
        self.immunity_decay = immunity_decay # immunity decay rate
        self.recovery_rate = recovery_rate # recovery rate
        self.max_immunity_loss_prob = max_immunity_loss_prob # maximum immunity loss probability
        self.visibility_radius = visibility_radius # visibility radius

        # Define observation space (will be used by the RL agent later)
        self.observation_space = gym.spaces.Dict({
            "grid": gym.spaces.Box(
            low=np.array([0.0]),
            high=np.array([float('inf')]),
            dtype=np.float32
        )
        })

        self.humans: List[Human] = []
        self.movement_handler = MovementHandler(grid_size, movement_type)
    ####### TRANSITION FUNCTIONS FOR MOVING BETWEEN S, I, R AND DEAD #######

    def _calculate_infection_probability(self, susceptible: Human, infected_list: List[Human]) -> float:
        """
        Calculate probability of infection based on nearby infected individuals
        If visibility_radius is -1, consider all infected individuals
        """
        total_exposure = 0
        for infected in infected_list:
            if infected.state != STATE_DICT['I']:
                assert infected.state == STATE_DICT['I'], "infected human is not in the infected state"
            else:
                distance = math.sqrt((susceptible.x - infected.x)**2 + (susceptible.y - infected.y)**2)
                
                # Skip if infected is outside the visibility radius (unless radius is -1)
                if self.visibility_radius != -1 and distance > self.visibility_radius:
                    continue
                    
                total_exposure += math.exp(-self.distance_decay * distance)
        
        return min(1,(self.beta / (1 + self.adherence)) * total_exposure)

    def _get_infected_list(self, center_x: Optional[int] = None, center_y: Optional[int] = None) -> List[Human]:
        """
        Return list of infected humans
        If center coordinates are provided, only return infected humans within visibility radius
        """
        if center_x is None or center_y is None or self.visibility_radius == -1:
            return [h for h in self.humans if h.state == STATE_DICT['I']]
            
        infected_list = []
        for human in self.humans:
            if human.state == STATE_DICT['I']:
                distance = math.sqrt((center_x - human.x)**2 + (center_y - human.y)**2)
                if distance <= self.visibility_radius:
                    infected_list.append(human)
                    
        return infected_list

    def _get_visible_humans(self, center_x: int, center_y: int) -> List[Human]:
        """
        Get list of humans within visibility radius of given position
        If visibility_radius is -1, return all humans
        """
        if self.visibility_radius == -1:
            return self.humans
            
        visible_humans = []
        for human in self.humans:
            distance = math.sqrt((center_x - human.x)**2 + (center_y - human.y)**2)
            if distance <= self.visibility_radius:
                visible_humans.append(human)
                
        return visible_humans

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
        self.npi_level = 0
        
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
            
            self.humans.append(Human(x, y, STATE_DICT['S']))

        # Select random humans to be infected
        initial_infected = self.np_random.choice(self.humans, self.n_infected, replace=False)
        for human in initial_infected:
            human.update_state(STATE_DICT['I'])

        return self._get_observation(), {}

    def _apply_action(self, action: np.array[np.float32]):
        """Apply the action to the environment"""
        self.agent_position = action[:2] # update position of the agent
        self.npi_level = action[2] # update NPI level

    def step(self, action: np.array[np.float32]) -> Tuple[dict, float, bool, bool, dict]:
        # Update all humans
        self._apply_action(action)


        for human in self.humans:
            human.time_in_state += 1

            if human.state == STATE_DICT['S']:
                # Calculate probability of infection
                infected_list = self._get_infected_list()
                p_infection = self._calculate_infection_probability(human, infected_list)
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

        # For now, return placeholder values
        return self._get_observation(), 0, False, False, {}