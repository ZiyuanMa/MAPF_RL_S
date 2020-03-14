from environment import Environment
from buffer import ReplayBuffer, pad_collate
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
import time


class Play(mp.Process):

    def __init__(self, num_agents, global_net, train_queue):
        ''' self play environment process'''
        super(Play, self).__init__()
        self.train_queue = train_queue

        self.env = Environment(num_agents=num_agents)
        self.global_net = global_net
        self.eval_net = Network()
        self.eval_net.load_state_dict(global_net.state_dict())
        self.eval_net.eval()
        self.num_agents = num_agents

    def run(self):

        # create and start caculation processes
        step = 0
        while True:
            
            done = False
            # start one eposide
            while not done:
                # self.train_lock.wait()

                # observe
                obs = self.env.joint_observe()
                obs = torch.from_numpy(obs)

                with torch.no_grad():
                    q_vals = self.eval_net(obs)

                # get results
                actions = select_action(q_vals)

                done = self.env.step(actions)

            history = self.env.get_history()
            self.train_queue.put(history)

            self.env.reset()

            step += 1
            if step == config.update_steps:
                self.eval_net.load_state_dict(self.global_net.state_dict())
                step = 0


def select_action(policy):

    if random.random() < config.greedy_coef:
        # random action
        return np.random.randint(config.action_space, size=policy.size()[0])

    else:
        # greedy action

        return torch.argmax(policy, 1).numpy()
            




class Train(mp.Process):
    def __init__(self, train_queue, global_net):
        super(Train, self).__init__()
        self.train_queue = train_queue

        self.global_net = global_net
        self.global_net.eval()
        self.global_net.to(device)
        self.train_net = Network()
        self.train_net.load_state_dict(self.global_net.state_dict())
        self.train_net.train()
        self.train_net.to(device)
        self.optimizer = torch.optim.AdamW(self.train_net.parameters())

        self.buffer = ReplayBuffer()

        self.steps = 0

    def run(self):
        while self.steps < config.training_steps:
            # receive = 300000
            # self.train_lock.set()
            # pbar = tqdm(total=receive)
            t = time.time()
            count = 0
            temp_buffer = list()
            while not self.train_queue.empty():
                history = self.train_queue.get_nowait()
                # pbar.update(min(len(history),receive))
                # receive -= len(history)
                count += len(history)
                temp_buffer.append(history)
                if count > 50000:
                    break


            self.buffer.multi_push(temp_buffer)


            print('push: '+str(count))
            if count == 0:
                time.sleep(1)

            # self.train_lock.clear()
            # pbar.close()
            
            sample_data = self.buffer.sample(config.batch_size)
            if sample_data is None:
                continue
            
            loader = DataLoader(sample_data, batch_size=config.mini_batch_size, num_workers=4, collate_fn=pad_collate)
            print('udpate '+str(self.steps+1))
            update_network(self.train_net, self.global_net, self.optimizer, loader)

            self.global_net.load_state_dict(self.train_net.state_dict())

            if self.global_net.training:
                raise RuntimeError('train')

            print('finish udpate '+str(self.steps+1))
            self.steps += 1
            config.greedy_coef *= 0.998
            if self.steps % config.checkpoint == 0:
                print('save model ' + str(self.steps//config.checkpoint))
                torch.save(self.global_net.state_dict(), './model'+str(self.steps//config.checkpoint)+'.pth')
            
            print('time: %.3f' %(time.time()-t))

def update_network(train_net, target_net, optimizer, loader):

    loss = 0

    for state, action, reward, post_state, done, mask, td_steps in loader:
        state = state.to(device)
        action = action.to(device)
        reward = reward.to(device)
        post_state = post_state.to(device)
        done = done.to(device)
        mask = mask.to(device)
        td_steps = td_steps.to(device)

        train_net.eval()
        with torch.no_grad():

            selected_action = train_net(post_state, mask).argmax(dim=2, keepdim=True)
            # print(selected_action.shape)
            # t = torch.squeeze(target_net(post_state, mask).gather(2, selected_action))
            # print(reward.shape)
            # done = done.unsqueeze(2)

            target = reward + torch.pow(config.gamma, td_steps) * torch.squeeze(target_net(post_state, mask).gather(2, selected_action), dim=2) * done
            # print(target.shape)
            # target = torch.masked_select(target, mask==False)
        
        train_net.train()
        q_vals = train_net(state, mask)
        q_val = torch.squeeze(torch.gather(q_vals, 2, action.unsqueeze(2)), dim=2)
        # q_val = torch.masked_select(q_val, mask==False)

        # clip
        # with torch.no_grad():
        #     target =  q_val + torch.clamp(target-q_val, -1, 1)

        l = ((target - q_val) ** 2).mean()
        l.backward()

        optimizer.step()
        optimizer.zero_grad()

        loss += l.item()
    loss /= config.batch_size // config.mini_batch_size
    print('loss: %.4f' % loss)
