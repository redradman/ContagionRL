import random

class ReplayBuffer:
    def __init__(self, max_size, batch_size):
        self.buffer = []
        self.max_size = max_size
        self.batch_size = batch_size

    def store(self, state, joint_action, reward, next_state):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append((state, joint_action, reward, next_state))

    def sample(self):
        return random.sample(self.buffer, self.batch_size)