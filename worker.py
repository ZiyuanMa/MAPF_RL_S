from environment import Environment
from buffer import ReplayBuffer
from model import Network
import config
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.distributions import Categorical
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import torch.multiprocessing as mp 
from tqdm import tqdm
import random


class Play(mp.Process):

    def __init__(self, num_agents, network, train_queue, train_lock):
        ''' self play environment process'''
        super(Play, self).__init__()
        self.train_queue = train_queue
        self.train_lock = train_lock
        self.env = Environment(num_agents=num_agents)
        self.network = network
        self.num_agents = num_agents

    def run(self):

        # create and start caculation processes

        while True:
            
            done = False
            # start one eposide
            while not done:
                self.train_lock.wait()

                # observe
                obs = self.env.joint_observe()

                with torch.no_grad():
                    q_vals = self.network(obs)

                # get results
                actions = choose_action(q_vals)
                
                done = self.env.step(actions, q_vals)

            history = self.env.get_history()
            self.train_queue.put(history)

            self.env.reset()


def choose_action(policy):

    if random.random() < config.greedy_coef:
        # random action
        return np.random.randint(config.action_space, size=policy.size()[0])

    else:
        # greedy action
        return torch.argmax(policy, 1).numpy()
            




class train(mp.Process):
    def __init__(self, train_queue, train_lock, target_net):
        super(train, self).__init__()
        self.train_queue = train_queue
        self.train_lock = train_lock

        self.target_net = target_net
        self.train_net = Network()
        self.train_net.load_state_dict(target_net.state_dict())
        self.train_net.train()
        self.train_net.to(device)
        self.optimizer = torch.optim.AdamW(train_net.parameters())

        self.buffer = ReplayBuffer()

        self.steps = 0

    def run(self):
        while self.steps < config.training_eposide:
            update_steps = 0
            training_lock.set()
            eposide = config.buffer_size
            pbar = tqdm(total=eposide)



            training_lock.clear()
            pbar.close()



# def train(training_queue, training_lock, target_net):
#     # network for training
#     training_net = Network()
#     training_net.load_state_dict(target_net.state_dict())
#     training_net = training_net.float()
#     training_net.train()
#     training_net.to(device)

#     optimizer = torch.optim.AdamW(training_net.parameters())

#     buffer = ReplayBuffer()
#     while True:
#         training_lock.set()
#         eposide = config.buffer_size
#         pbar = tqdm(total=eposide)

#         while eposide > 0:
#             data_tuple = training_queue.get()
#             buffer.push(data_tuple)
#             eposide -= len(data_tuple[1][0])
#             pbar.update(len(data_tuple[1][0]))

#         training_lock.clear()
#         pbar.close()

#         for _ in range(config.optim_steps):
            
#             sample_data = buffer.sample()
#             loader = DataLoader(sample_data, batch_size=config.mini_batch_size, num_workers=4)

#             update_network(training_net, optimizer, loader)

#         target_net.load_state_dict(training_net.state_dict())


# def update_network(training_net, optimizer, loader):

#     loss = 0
#     policy_loss = 0
#     value_loss = 0

#     optimizer.zero_grad()

#     for prev_state, prev_num, state, action_idx, action_prob, state_value, reward in loader:

#         prev_state = prev_state.view(-1, 5, 4, 10, 10)

#         prev_state = prev_state.to(device)
#         prev_num = prev_num.to(device)
#         state = state.to(device)
#         action_idx = action_idx.to(device)
#         action_prob = action_prob.to(device)
#         state_value = state_value.to(device)
#         reward = reward.to(device)

#         training_net.init_lstm(prev_state, prev_num)

#         policy, pred_value = training_net(state)
#         policy = torch.gather(policy, 1, action_idx.unsqueeze(1))

#         adv = reward - state_value

#         ratio = policy/action_prob

#         surrogate1 = ratio * adv
#         surrogate2 = torch.clamp(ratio, 1-config.clip_range , 1+config.clip_range) * adv

#         policy_loss = -torch.min(surrogate1, surrogate2).mean()

#         reward = state_value + (reward - state_value).clamp(min=-config.clip_range, max=config.clip_range)
#         value_loss = ((reward - pred_value) ** 2).mean()

#         entropy_loss = Categorical(policy).entropy()
#         entropy_loss = entropy_loss.mean()

#         l = (policy_loss + config.value_coef*value_loss - config.entropy_coef*entropy_loss)
#         l.backward()

#         optimizer.step()

#         loss += l

#     print('loss: %.4f' % loss)
