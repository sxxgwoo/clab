import random
from collections import namedtuple
import numpy as np
import torch

Experience = namedtuple("Experience", field_names=["state", "action"])


class ReplayBuffer:
    """
    Reinforcement learning replay buffer for training data
    """

    def __init__(self):
        self.memory = []

    def push(self, state, action):
        """saving an experience tuple"""
        experience = Experience(state, action)
        self.memory.append(experience)

    def sample(self, batch_size):
        """randomly sampling a batch of experiences"""
        tem = random.sample(self.memory, batch_size)
        states, actions = zip(*tem)
        states, actions = np.stack(states), np.stack(actions)
        states, actions = torch.FloatTensor(states), torch.FloatTensor(actions)
        return states, actions

    def __len__(self):
        """return the length of replay buffer"""
        return len(self.memory)


if __name__ == '__main__':
    buffer = ReplayBuffer()
    for i in range(1000):
        buffer.push(np.array([i+1, i+2, i+3]), np.array(4), np.array(5), np.array([6, 7, 8]), np.array(0))
    print(buffer.sample(20))
    print(buffer.__len__())
