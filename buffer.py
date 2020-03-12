import config
from environment import History, observe
import torch
from torch.utils.data import Dataset, Subset
from torch.nn.utils.rnn import pad_sequence
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

        done = np.ones(history.num_agents, dtype=np.float32)
        cumu_reward = np.zeros(history.num_agents, dtype=np.float32)
        post_state = np.copy(state)
        for i in range(config.forward_steps):
            if step_idx + i < len(history):
                post_state, _, reward = history[step_idx+i]
                cumu_reward += reward
            else:
                if history.done():
                    done = np.zeros(history.num_agents, dtype=np.float32)
                break
        mask = np.zeros(history.num_agents, dtype=np.bool)
        return torch.from_numpy(state), torch.from_numpy(action), torch.from_numpy(cumu_reward), torch.from_numpy(post_state), torch.from_numpy(done), torch.from_numpy(mask)


    def __len__(self):

        return self.size
    
    def push(self, history: History):

        # delete if out of bound
        while self.size >= self.buffer_size:
            self.size -= len(self.history_list[0])
            del self.history_list[0]

        # push
        self.history_list.append(history)
        self.size += len(history)

    def clear(self):
        self.size = 0
        self.history_list.clear()


    def sample(self, sample_size):
        if len(self) < sample_size:
            return None
        indices = np.random.randint(self.size, size=sample_size)

        return Subset(self, indices)


def pad_collate(batch):

    # batch.sort(key= lambda x: x[2], reverse=True)
    (state, action, cumu_reward, post_state, done, mask) = zip(*batch)
    state = pad_sequence(state, batch_first=True)
    action = pad_sequence(action, batch_first=True)
    cumu_reward = pad_sequence(cumu_reward, batch_first=True)
    post_state = pad_sequence(post_state, batch_first=True)
    done = pad_sequence(done, batch_first=True)
    mask = pad_sequence(mask, batch_first=True, padding_value=1)

    return state, action, cumu_reward, post_state, done, mask

if __name__ == '__main__':
    a = [torch.zeros((2,4,4)), torch.zeros((3,4,4))]
    b = pad_sequence(a, batch_first=True)
    print(b.shape)