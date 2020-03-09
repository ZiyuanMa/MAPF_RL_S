import config
from environment import History, observe
import torch
from torch.utils.data import Dataset, Subset
import numpy as np
import numba as nb


class ReplayBuffer(Dataset):

    def __init__(self):
        self.buffer_size = config.buffer_size
        self.size = 0
        self.history_list = []


    def __getitem__(self, index: int):

        history_idx = 0
        step_idx = index
        while step_idx >= len(self.history_list[history_idx]):
            step_idx -= len(self.history_list[history_idx])
            history_idx += 1

        history = self.history_list[history_idx]

        state, action, _ = history[step_idx]

        done = np.array([1], dtype=np.float32)
        cumu_reward = np.zeros(history.num_agents, dtype=np.float32)
        post_state = np.copy(state)
        for i in range(config.forward_steps):
            if step_idx + i < len(history):
                post_state, _, reward = history[step_idx+i]
                cumu_reward += reward
            else:
                done = np.array([0], dtype=np.float32)
                break
        
        return state, action, cumu_reward, post_state, done, history.num_agents


    def __len__(self):

        return self.size
    
    def push(self, history: History):

        # delete if out of bound
        while self.size >= self.buffer_size:
            self.size -= len(self.history_list[0])
            del self.history_list[0]

        # push
        self.history_list.append(history)
        self.size += history.num_agents

    def clear(self):
        self.size = 0
        self.history_list.clear()


    def sample(self, sample_size):
        indices = np.random.randint(self.size, size=sample_size)

        return Subset(self, indices)

