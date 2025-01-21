import gymnasium as gym
import numpy as np
from typing import Optional, List, Tuple
import math

# State definitions
STATE_DICT = {
    'S': 0,  # Susceptible
    'I': 1,  # Infected
    'R': 2,  # Recovered
    'D': 3   # Dead
}

######## Movement handler for humans in the environment ########
class MovementHandler:
    def __init__(self, grid_size: int, movement_type: str = "stationary"):
        """
        Initialize movement handler
        
        Args:
            grid_size: Size of the environment grid
            movement_type: One of ["stationary", "discrete_random", "continuous_random"]
        """
        self.grid_size = grid_size
        self.movement_type = movement_type
        
        # Validate movement type
        valid_types = ["stationary", "discrete_random", "continuous_random"]
        if movement_type not in valid_types:
            raise ValueError(f"Movement type must be one of {valid_types}")

    def get_new_position(self, x: int, y: int, rng: np.random.Generator) -> Tuple[int, int]:
        """
        Get new position based on current position and movement type
        
        Args:
            x: Current x position
            y: Current y position
            rng: Random number generator for reproducibility
            
        Returns:
            Tuple of (new_x, new_y)
        """
        if self.movement_type == "stationary":
            return self._stationary_move(x, y)
        elif self.movement_type == "discrete_random":
            return self._discrete_random_move(x, y, rng)
        else:  # continuous_random
            return self._continuous_random_move(x, y, rng)

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
        new_x = max(0, min(x + dx, self.grid_size - 1))
        new_y = max(0, min(y + dy, self.grid_size - 1))
        
        return new_x, new_y

    def _continuous_random_move(self, x: int, y: int, rng: np.random.Generator) -> Tuple[int, int]:
        """
        Random movement in continuous steps [-1, 1] for both x and y
        """
        dx = rng.uniform(-1, 1)
        dy = rng.uniform(-1, 1)
        
        # Ensure we stay within bounds
        new_x = max(0, min(round(x + dx), self.grid_size - 1))
        new_y = max(0, min(round(y + dy), self.grid_size - 1))
        
        return new_x, new_y


######## Human class ########
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
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # Store parameters
        self.grid_size = grid_size
        self.n_humans = n_humans
        self.n_infected = n_infected
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

    ####### TRANSITION FUNCTIONS FOR MOVING BETWEEN S, I, R AND DEAD #######

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

    def _calculate_recovery_and_death_probabilities(self, human: Human) -> Tuple[float, float]:
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

    ##### 

    def reset(self, seed: Optional[int] = None) -> Tuple[dict, dict]:
        """Reset the environment to the initial state"""
        super().reset(seed=seed)
        
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

    def step(self, action: int) -> Tuple[dict, float, bool, bool, dict]:
        # Update all humans
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
                p_recovery = 1 - math.exp(-self.recovery_rate * human.time_in_state)
                if self.np_random.random() < p_recovery:
                    human.update_state(STATE_DICT['R'])

            elif human.state == STATE_DICT['R']:
                # Check for immunity loss
                p_immunity_loss = self.max_immunity_loss_prob * (1 - math.exp(-self.immunity_decay * human.time_in_state))
                if self.np_random.random() < p_immunity_loss:
                    human.update_state(STATE_DICT['S'])

        # For now, return placeholder values
        return self._get_observation(), 0, False, False, {}