import config
from environment import History, observe
import torch
from torch.utils.data import Dataset, Subset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import numba as nb


class SumTree:
    def __init__(self, size_index):
        self.depth = size_index
        self.tree = np.zeros(2**(self.depth+1), dtype=np.uint32)
        self.length = 0

    def push(self, value):
        assert self.length < 2**self.depth, 'sum tree out of size'

        ptr = 2**self.depth + self.length

        diff = value - self.tree[ptr]
        self.tree[ptr] = value
        ptr = ptr // 2

        while ptr > 0:
            self.tree[ptr] += diff
            ptr = ptr // 2

        self.length += 1

    def pop(self):

        self.tree = np.roll(self.tree, -1)
        self.tree[-1] = 0

        for depth in reversed(range(self.depth)):
            for idx in range(2**depth, 2**(depth+1)):
                self.tree[idx] = self.tree[idx*2] + self.tree[idx*2+1]

        self.length -= 1
        # self.check()

    def search(self, value):

        assert value <= self.tree[1], 'search out of size'

        idx = 1
        for _ in range(self.depth):
            if value < self.tree[2*idx]:
                idx = 2*idx
            else:
                value -= self.tree[2*idx]
                idx = 2*idx + 1

        return idx - 2**self.depth, value

    def check(self):
        for depth in range(self.depth):
            for idx in range(2**depth, 2**(depth+1)):
                assert self.tree[idx] == self.tree[2*idx] + self.tree[2*idx+1], 'tree mismatch ' + str(idx) + ' ' + str(self.tree[idx]) + ' ' + str(self.tree[2*idx]) + ' ' + str(self.tree[2*idx+1])



        



class ReplayBuffer(Dataset):

    def __init__(self):
        self.buffer_size = config.buffer_size
        self.size = 0
        self.history_list = []
        self.search_tree = SumTree(16)


    def __getitem__(self, index: int):


        history_idx, step_idx = self.search_tree.search(index)

        history = self.history_list[history_idx]

        state, action, _ = history[step_idx]

        done = np.ones(1, dtype=np.float32)
        cumu_reward = np.zeros(history.num_agents, dtype=np.float32)
        post_state = np.copy(state)
        td_steps = np.array([config.TD_steps], dtype=np.float32)
        for i in range(config.TD_steps):
            if step_idx + i < len(history):
                post_state, _, reward = history[step_idx+i]
                cumu_reward += reward * config.gamma ** i
            else:
                if history.done():
                    done = np.zeros(1, dtype=np.float32)
                td_steps = np.array([i], dtype=np.float32)
                break
        mask = np.zeros(history.num_agents, dtype=np.bool)
        return torch.from_numpy(state), torch.from_numpy(action), torch.from_numpy(cumu_reward), torch.from_numpy(post_state), torch.from_numpy(done), torch.from_numpy(mask), torch.from_numpy(td_steps)


    def __len__(self):

        return self.size
    
    def push(self, history: History):

        assert self.size == self.search_tree.tree[1], 'size mismatch '+str(self.size) + ' ' + str(self.search_tree.tree[1])


        # delete if out of bound
        while self.size >= self.buffer_size:
            self.size -= len(self.history_list[0])
            del self.history_list[0]
            self.search_tree.pop()

        # push
        self.history_list.append(history)
        self.size += len(history)
        self.search_tree.push(len(history))

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
    (state, action, cumu_reward, post_state, done, mask, td_steps) = zip(*batch)
    state = pad_sequence(state, batch_first=True)
    action = pad_sequence(action, batch_first=True)
    cumu_reward = pad_sequence(cumu_reward, batch_first=True)
    post_state = pad_sequence(post_state, batch_first=True)
    done = torch.stack(done)
    mask = pad_sequence(mask, batch_first=True, padding_value=1)
    td_steps = torch.stack(td_steps)

    return state, action, cumu_reward, post_state, done, mask, td_steps

if __name__ == '__main__':
    a = torch.Tensor([1,2,3])
    print(torch.pow(2, a))