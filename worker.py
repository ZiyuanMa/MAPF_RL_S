from environment import Environment
from buffer import ReplayBuffer, pad_collate
from model import Network
from search import CBSSolver
import config
import numpy as np
import torch
from torch.utils.data import DataLoader
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import torch.multiprocessing as mp 
from tqdm import tqdm
import random
import time

action_list = np.array([[0, 0],[-1, 0],[1, 0],[0, -1],[0, 1]], dtype=np.int8)

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
            # print(self.env.map)
            # print(self.env.agents_pos)
            # print(self.env.goals)
            if random.random() < 0.2:
                map = (np.copy(self.env.map)==1).tolist()

                
                temp_agents_pos = np.copy(self.env.agents_pos)
                agents_pos= []
                for pos in temp_agents_pos:
                    agents_pos.append(tuple(pos))

                temp_goals_pos = np.copy(self.env.goals)
                goals_pos = []
                for pos in temp_goals_pos:
                    goals_pos.append(tuple(pos))

                solver  =  CBSSolver(map, agents_pos, goals_pos)
                paths = solver.find_solution()

                if len(paths[0]) == 1:
                    continue

                max_len = max([len(path) for path in paths])

                for path in paths:
                    while len(path) < max_len:
                        path.append(path[-1])

                done = False
                for step in range(1, max_len):
                    actions = []

                    direction = np.asarray(paths[0][step]) - np.asarray(paths[0][step-1])
                    
                    if np.array_equal(direction, action_list[0]):
                        actions.append(0)
                    elif np.array_equal(direction, action_list[1]):
                        actions.append(1)
                    elif np.array_equal(direction, action_list[2]):
                        actions.append(2)
                    elif np.array_equal(direction, action_list[3]):
                        actions.append(3)
                    elif np.array_equal(direction, action_list[4]):
                        actions.append(4)


                    direction = np.asarray(paths[1][step]) - np.asarray(paths[1][step-1])
                    
                    if np.array_equal(direction, action_list[0]):
                        actions.append(0)
                    elif np.array_equal(direction, action_list[1]):
                        actions.append(1)
                    elif np.array_equal(direction, action_list[2]):
                        actions.append(2)
                    elif np.array_equal(direction, action_list[3]):
                        actions.append(3)
                    elif np.array_equal(direction, action_list[4]):
                        actions.append(4)


                    direction = np.asarray(paths[2][step]) - np.asarray(paths[2][step-1])
                    
                    if np.array_equal(direction, action_list[0]):
                        actions.append(0)
                    elif np.array_equal(direction, action_list[1]):
                        actions.append(1)
                    elif np.array_equal(direction, action_list[2]):
                        actions.append(2)
                    elif np.array_equal(direction, action_list[3]):
                        actions.append(3)
                    elif np.array_equal(direction, action_list[4]):
                        actions.append(4)

                    done = self.env.step(actions)

                    # for i, pos in enumerate(np.copy(self.env.agents_pos)):
                    #     if not np.array_equal(pos, np.array(list(paths[i][step]))):
                    #         print(step)
                    #         print(paths[0][step-1:step+1])
                    #         print(paths[1][step-1:step+1])
                    #         print(paths[2][step-1:step+1])
                    #         print(self.env.history.agents_pos[-2:])
                    #         print(actions)
                    #         print(self.env.history.rewards[-1])
                if not done:
                    print(self.env.map)
                    print(paths)
                    print(self.env.history.actions)
                    print(self.env.history.agents_pos)
                    raise RuntimeError('not done')


            else:

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
        self.optimizer = torch.optim.Adam(self.train_net.parameters(), lr=5e-5, weight_decay=1e-6)

        self.buffer = ReplayBuffer()

        self.steps = 0

    def run(self):
        while self.steps < config.training_steps:

            t = time.time()
            count = 0
            temp_buffer = list()
            while count <= 10000:
            # while not self.train_queue.empty():
                history = self.train_queue.get()
                count += len(history)
                temp_buffer.append(history)


            self.buffer.multi_push(temp_buffer)


            print('push: '+str(count))

            
            sample_data = self.buffer.sample(config.batch_size)
            if sample_data is None:
                continue
            
            loader = DataLoader(sample_data, batch_size=config.mini_batch_size, num_workers=4, collate_fn=pad_collate)
            print('udpate '+str(self.steps+1))
            update_network(self.train_net, self.global_net, self.optimizer, loader)


            self.train_net.reset_noise()

            self.global_net.load_state_dict(self.train_net.state_dict())


            print('finish udpate '+str(self.steps+1))
            self.steps += 1

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
        with torch.no_grad():
            target =  q_val + torch.clamp(target-q_val, -1, 1)

        l = ((target - q_val) ** 2).mean()
        l.backward()

        optimizer.step()
        optimizer.zero_grad()

        loss += l.item()
    loss /= config.batch_size // config.mini_batch_size
    print('loss: %.4f' % loss)
