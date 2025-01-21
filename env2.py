import gymnasium as gym
import numpy as np
from typing import Optional, List
import math

# State definitions
STATE_DICT = {
    'S': 0,  # Susceptible
    'I': 1,  # Infected
    'R': 2,  # Recovered
    'D': 3   # Dead
}

class Human:
    def __init__(self, x: int, y: int, state: int = STATE_DICT['S']):
        """
        Initialize a human in the SIRS model
        state: integer representing state (0: Susceptible, 1: Infected, 2: Recovered, 3: Dead)
        """
        self.x = x
        self.y = y
        self.state = state
        self.time_in_state = 0
        assert self.state in STATE_DICT.values() # make sure no invalid state is passed in

    def move(self, new_x: int, new_y: int, grid_size: int):
        """Move human to new position within grid bounds"""
        self.x = max(0, min(new_x, grid_size - 1))
        self.y = max(0, min(new_y, grid_size - 1))

    def update_state(self, new_state: int):
        """Update state and reset time counter"""
        self.state = new_state
        self.time_in_state = 0

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
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # Store parameters
        self.grid_size = grid_size
        self.n_humans = n_humans
        self.n_infected_init = n_infected
        self.beta = beta # infection rate
        self.adherence = adherence # NPI adherence
        self.distance_decay = distance_decay # distance decay rate
        self.lethality = lethality # lethality rate
        self.immunity_decay = immunity_decay # immunity decay rate
        self.recovery_rate = recovery_rate # recovery rate
        self.max_immunity_loss_prob = max_immunity_loss_prob # maximum immunity loss probability

        # Define observation space (will be used by the RL agent later)
        self.observation_space = gym.spaces.Dict({
            "grid": gym.spaces.Box(low=0, high=3, shape=(grid_size, grid_size), dtype=np.int32)
        })

        self.humans: List[Human] = []

    def _calculate_infection_probability(self, susceptible: Human, infected_list: List[Human]) -> float:
        """Calculate probability of infection based on the given formula: Transition from S to I"""
        total_exposure = 0
        for infected in infected_list:
            assert infected.state == STATE_DICT['I'], "infected human is not in the infected state"
            distance = math.sqrt((susceptible.x - infected.x)**2 + (susceptible.y - infected.y)**2)
            total_exposure += math.exp(-self.distance_decay * distance)
        
        return min(1,(self.beta / (1 + self.adherence)) * total_exposure)

    def _get_infected_list(self) -> List[Human]:
        """Return list of infected humans"""
        return [h for h in self.humans if h.state == STATE_DICT['I']]

    def _calculate_recovery_and_death_probabilities(self, human: Human) -> List[float, float]:
        """Calculate recovery and death probabilities for a human: Transition from I to R and from I to D"""
        if human.state != STATE_DICT['I']:
            raise ValueError("incorrect call to function: probability of recovery and death is only applicable to humans in the infected state")
        else:
            recovery_prob = 1 - math.exp(-self.recovery_rate * human.time_in_state)
            death_prob = self.lethality
            return recovery_prob, death_prob
    
    def _calculate_immunity_loss_probability(self, human: Human) -> float:
        """Calculate immunity loss probability for a human: Transition from R to S"""
        if human.state != STATE_DICT['R']:
            raise ValueError("incorrect call to function: probability of immunity loss is only applicable to humans in the recovered state")
        else:
            return self.max_immunity_loss_prob * (1 - math.exp(-self.immunity_decay * human.time_in_state))

