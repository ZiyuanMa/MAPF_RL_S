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
        # wip

        # find the position
        history_idx = 0
        step_idx = index
        while step_idx >= len(self.history_list[history_idx]):
            step_idx -= len(self.history_list[history_idx])
            history_idx += 1

        self.history_list[history_idx][step_idx]

    
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

    @nb.jit(nopython=True)
    def sample(self):
        indices = np.random.randint(self.size, size=config.batch_size)

        return Subset(self, indices)

