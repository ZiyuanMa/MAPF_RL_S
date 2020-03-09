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
                obs = torch.from_numpy(obs)

                with torch.no_grad():
                    q_vals = self.network(obs)

                # get results
                actions = select_action(q_vals)
                
                done = self.env.step(actions)

            history = self.env.get_history()
            self.train_queue.put(history)

            self.env.reset()


def select_action(policy):

    if random.random() < config.greedy_coef:
        # random action
        return np.random.randint(config.action_space, size=policy.size()[0])

    else:
        # greedy action
        return torch.argmax(policy, 1).numpy()
            




class Train(mp.Process):
    def __init__(self, train_queue, train_lock, target_net):
        super(Train, self).__init__()
        self.train_queue = train_queue
        self.train_lock = train_lock

        self.target_net = target_net
        self.train_net = Network()
        self.train_net.load_state_dict(target_net.state_dict())
        self.train_net.train()
        self.train_net.to(device)
        self.optimizer = torch.optim.AdamW(self.train_net.parameters())

        self.buffer = ReplayBuffer()

        self.steps = 0

    def run(self):
        while self.steps < config.training_timesteps//config.checkpoint:
            receive = 10000
            self.train_lock.set()
            pbar = tqdm(total=receive)
            while receive > 0:
                history = self.train_queue.get()
                pbar.update(min(len(history),receive))
                receive -= len(history)
                self.buffer.push(history)

            self.train_lock.clear()
            pbar.close()
            for _ in range(3):
            
                sample_data = self.buffer.sample(config.batch_size)
                loader = DataLoader(sample_data, batch_size=200, num_workers=4)

                update_network(self.train_net, self.target_net, self.optimizer, loader)


def update_network(train_net, target_net, optimizer, loader):

    loss = 0

    for state, action, reward, post_state, done, num_agents in loader:
        state = state.to(device)
        action = action.to(device)
        reward = reward.to(device)
        post_state = post_state.to(device)
        done = done.to(device)
        num_agents = num_agents.to(device)

        train_net.eval()
        with torch.no_grad():
            selected_action = train_net(post_state, num_agents).argmax(dim=1, keepdim=True)
            target = reward + config.gamma**config.forward_steps * target_net(post_state, num_agents).gather(1, selected_action) * done
        
        train_net.train()
        q_vals = train_net(state, num_agents)
        q_val = torch.gather(q_vals, 1, action.unsqueeze(1))

        with torch.no_grad():
            target =  q_val + torch.clamp(target-q_val, -1, 1)

        l = ((q_val - target) ** 2).mean()
        l.backward()

        optimizer.step()
        optimizer.zero_grad()

        loss += l.item()

    print('loss: %.4f' % loss)