import numpy as np
import random

class Human:
    id_counter = 1
    def __init__(
        self,
        x,
        y,
        is_infected=False,
        npi_adherence=0.5,
        immunity_duration=10,
        observation_radius=10,
        grid_size=10,
        lethality=0.1,
        transition_probs=None,
    ):
        # ID
        self.id = Human.id_counter
        Human.id_counter += 1
        # Position
        self.position = np.array([x, y], dtype=np.float32)
        self.grid_size = grid_size
        # SIRS attributes
        self.state = "I" if is_infected else "S"
        self.npi_adherence = npi_adherence
        self.immunity_duration = immunity_duration
        self.infection_timer = 0
        self.recovery_timer = 0
        self.times_infected = 0
        self.lethality = lethality
        self.alive = True
        # RL attributes
        self.cumulative_reward = 0
        self.observation_radius = observation_radius
        self.policy = None
        # Transition probabilities
        self.transition_probs = transition_probs or {
            'I_to_R': 0.1,
            'R_to_S': 0.05
        }
        self.npi_update_frequency = 10  # Update NPI every 10 timesteps
        self.steps_since_npi_update = 0

    def observe(self, humans):
        neighbors = []
        for human in humans:
            if self.id != human.id and human.alive:
                distance = np.abs(self.position - human.position).sum()
                if distance <= self.observation_radius:
                    neighbors.append({
                        'id': human.id,
                        'x': human.position[0],
                        'y': human.position[1],
                        'state': human.state
                    })
        return neighbors

    def should_update_npi(self):
        return self.steps_since_npi_update >= self.npi_update_frequency
        
    def take_action(self, action):
        """
        Takes a tuple of (movement_action, npi_action) as input
        movement_action: int (0-4) representing Up, Down, Left, Right, Stay
        npi_action: int (0-10) representing NPI adherence levels from 0.0 to 1.0
        """
        movement_action, npi_action = action
        
        # Handle movement
        if movement_action in range(5):  # 0-4
            self._handle_movement(movement_action)
        
        # Handle NPI adjustment
        if npi_action in range(11):  # 0-10
            self._handle_npi_adjustment(npi_action)

    def _handle_movement(self, action):
        if not self.alive:
            return
        # Movement vectors for [Up, Down, Left, Right, Stay]
        movement_vectors = np.array([
            [0, 1],   # Up
            [0, -1],  # Down
            [-1, 0],  # Left
            [1, 0],   # Right
            [0, 0]    # Stay
        ], dtype=np.float32)
        
        # Apply movement
        new_position = self.position + movement_vectors[action]
        half_grid = self.grid_size // 2
        
        # Boundary checks using numpy's clip
        self.position = np.clip(new_position, -half_grid, half_grid)

    def update_state(self):
        if not self.alive:
            return
        if self.state == 'I':
            self.infection_timer += 1
            # Chance to die from infection
            if random.random() < self.lethality:
                self.alive = False
                self.cumulative_reward -= 1  # Penalty for dying
                return
            # Transition to Recovered
            if random.random() < self.transition_probs['I_to_R']:
                self.state = 'R'
                self.infection_timer = 0
                self.recovery_timer = 0
        elif self.state == 'R':
            self.recovery_timer += 1
            # Lose immunity over time
            if random.random() < self.transition_probs['R_to_S']:
                self.state = 'S'

    def infect(self, other):
        if self.state == 'I' and other.state == 'S' and other.alive:
            distance = np.abs(self.position - other.position).sum()
            infection_prob = max(0, (1 - distance / self.observation_radius)) * (1 - other.npi_adherence)
            if random.random() < infection_prob:
                other.state = 'I'
                other.times_infected += 1

    def get_action_space(self):
        """
        Returns tuple of (movement_actions, npi_actions)
        movement_actions: list of valid movement actions
        npi_actions: list of valid NPI actions (always all of them)
        """
        half_grid = self.grid_size // 2
        # Create boolean mask for valid movement actions
        valid_movements = np.array([
            self.position[1] < half_grid,     # Up
            self.position[1] > -half_grid,    # Down
            self.position[0] > -half_grid,    # Left
            self.position[0] < half_grid,     # Right
            True                              # Stay always valid
        ])
        movement_actions = np.where(valid_movements)[0].tolist()
        npi_actions = list(range(11))  # 0-10 representing 0.0-1.0
        
        return (movement_actions, npi_actions)
    
    def _handle_npi_adjustment(self, action):
        """
        Convert action (0-10) to NPI value (0.0-1.0)
        action 0 -> 0.0
        action 1 -> 0.1
        ...
        action 10 -> 1.0
        """
        self.npi_adherence = action * 0.1
    