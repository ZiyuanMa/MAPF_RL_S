import torch.nn as nn
from torch.optim import Adam
import math
import os
import random
import time
from collections import deque
from copy import deepcopy

import gym
 
import numpy as np
import torch
import torch.distributions
from torch.nn.functional import softmax, log_softmax

from buffer2 import ReplayBuffer, PrioritizedReplayBuffer
from model2 import Network
from environment2 import Environment
import config
from search import find_path

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def learn(  env, number_timesteps,
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), save_path='./models', save_interval=config.save_interval,
            ob_scale=config.ob_scale, gamma=config.gamma, grad_norm=config.grad_norm, double_q=config.double_q,
            param_noise=config.param_noise, dueling=config.dueling,
            exploration_final_eps=config.exploration_final_eps, batch_size=config.batch_size, train_freq=config.train_freq,
            learning_starts=config.learning_starts, target_network_update_freq=config.target_network_update_freq, buffer_size=config.buffer_size,
            prioritized_replay=config.prioritized_replay, prioritized_replay_alpha=config.prioritized_replay_alpha,
            prioritized_replay_beta0=config.prioritized_replay_beta0, atom_num=config.atom_num, min_value=config.min_value, max_value=config.max_value):
    """
    Papers:
    Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep
    reinforcement learning[J]. Nature, 2015, 518(7540): 529.
    Hessel M, Modayil J, Van Hasselt H, et al. Rainbow: Combining Improvements
    in Deep Reinforcement Learning[J]. 2017.

    Parameters:
    ----------
    double_q (bool): if True double DQN will be used
    param_noise (bool): whether or not to use parameter space noise
    dueling (bool): if True dueling value estimation will be used
    exploration_fraction (float): fraction of entire training period over which
                                  the exploration rate is annealed
    exploration_final_eps (float): final value of random action probability
    batch_size (int): size of a batched sampled from replay buffer for training
    train_freq (int): update the model every `train_freq` steps
    learning_starts (int): how many steps of the model to collect transitions
                           for before learning starts
    target_network_update_freq (int): update the target network every
                                      `target_network_update_freq` steps
    buffer_size (int): size of the replay buffer
    prioritized_replay (bool): if True prioritized replay buffer will be used.
    prioritized_replay_alpha (float): alpha parameter for prioritized replay
    prioritized_replay_beta0 (float): beta parameter for prioritized replay
    atom_num (int): atom number in distributional RL for atom_num > 1
    min_value (float): min value in distributional RL
    max_value (float): max value in distributional RL

    """
    # create network and optimizer
    network = Network(atom_num, dueling)
    # network.encoder.load_state_dict(torch.load('./encoder.pth', map_location=torch.device('cpu')))
    # for param in network.encoder.parameters():
    #     param.requires_grad = False


    # optimizer = Adam(
    #     filter(lambda p: p.requires_grad, network.parameters()),
    #     lr=1e-3, eps=1e-5
    # )

    # create target network
    qnet = network.to(device)
    # qnet.encoder.load_state_dict(torch.load('./encoder.pth'))
    # for param in qnet.encoder.parameters():
    #     param.requires_grad = False


    optimizer = Adam(
        filter(lambda p: p.requires_grad, qnet.parameters()),
        lr=1e-3, eps=1e-5
    )

    tar_qnet = deepcopy(qnet)

    # create replay buffer
    if prioritized_replay:
        buffer = PrioritizedReplayBuffer(buffer_size, device,
                                         prioritized_replay_alpha,
                                         prioritized_replay_beta0)
    else:
        buffer = ReplayBuffer(buffer_size, device)

    generator = _generate(device, env, qnet, ob_scale,
                          number_timesteps, param_noise,
                          exploration_final_eps,
                          atom_num, min_value, max_value)

    if atom_num > 1:
        delta_z = float(max_value - min_value) / (atom_num - 1)
        z_i = torch.linspace(min_value, max_value, atom_num).to(device)

    infos = {'eplenmean': deque(maxlen=100), 'eprewmean': deque(maxlen=100)}
    start_ts = time.time()
    for n_iter in range(1, number_timesteps + 1):

        if prioritized_replay:
            buffer.beta += (1 - prioritized_replay_beta0) / number_timesteps
            
        data = generator.__next__()
        buffer.add(data)


        # update qnet
        if n_iter > learning_starts and n_iter % train_freq == 0:
            b_bt, b_bt_steps, b_obs, b_action, b_reward, b_obs_, b_done, b_steps, *extra = buffer.sample(batch_size)


            if atom_num == 1:
                with torch.no_grad():
                    
                    # choose max q index from next observation
                    if double_q:
                        b_action_ = qnet(b_obs_)[0].argmax(2).unsqueeze(2)
                        b_q_ = (1 - b_done).unsqueeze(2) * tar_qnet(b_obs_).gather(2, b_action_)
                    else:
                        b_q_ = (1 - b_done).unsqueeze(2) * tar_qnet(b_obs_).max(2, keepdim=True)[0]

                b_action = b_action.unsqueeze(2)
                b_q = qnet(b_obs).gather(2, b_action)

                b_reward = b_reward.unsqueeze(2)

                abs_td_error = (b_q[:,0,:] - (b_reward[:,0,:] + (gamma ** b_steps) * b_q_[:,0,:])).abs()

                
                priorities = abs_td_error.detach().cpu().clamp(1e-6).numpy()
                priorities = np.average(priorities, axis=1)

                if extra:
                    extra[0] = extra[0].unsqueeze(2)
                    loss = (extra[0] * huber_loss(abs_td_error)).mean()
                else:
                    loss = huber_loss(abs_td_error).mean()

            else:
                batch_idx = torch.ones(config.num_agents, dtype=torch.long).unsqueeze(0) * torch.arange(batch_size, dtype=torch.long).unsqueeze(1)
                agent_idx = torch.arange(config.num_agents, dtype=torch.long).unsqueeze(0) * torch.ones(batch_size, dtype=torch.long).unsqueeze(1)

                with torch.no_grad():
                    b_dist_ = tar_qnet(b_obs_).exp()
                    b_action_ = (b_dist_ * z_i).sum(-1).argmax(2)
                    b_tzj = ((gamma**b_steps * (1 - b_done) * z_i[None, :]).unsqueeze(1) + b_reward.unsqueeze(2)).clamp(min_value, max_value)
                    b_i = (b_tzj - min_value) / delta_z
                    b_lower = b_i.floor()
                    b_upper = b_i.ceil()
                    b_m = torch.zeros(batch_size, config.num_agents, atom_num).to(device)

                    temp = b_dist_[batch_idx, agent_idx, b_action_, :]

                    b_m.scatter_add_(2, b_lower.long(), temp * (b_upper - b_i))
                    b_m.scatter_add_(2, b_upper.long(), temp * (b_i - b_lower))

                b_q = qnet(b_obs)[batch_idx, agent_idx, b_action, :]

                kl_error = -(b_q * b_m).sum(2)
                # use kl error as priorities as proposed by Rainbow
                priorities = kl_error.detach().cpu().clamp(1e-6).numpy()
                priorities = np.average(priorities, axis=1)
                loss = kl_error.mean()

            optimizer.zero_grad()

            loss.backward()
            if grad_norm is not None:
                nn.utils.clip_grad_norm_(qnet.parameters(), grad_norm)
            optimizer.step()

            for tar_net, net in zip(tar_qnet.parameters(), qnet.parameters()):
                tar_net.data.copy_(0.001*net.data + 0.999*tar_net.data)

            if prioritized_replay:
                buffer.update_priorities(extra[1], priorities)

        # update target net and log
        if n_iter % target_network_update_freq == 0:
            # tar_qnet.load_state_dict(qnet.state_dict())
            # print('{} Iter {} {}'.format('=' * 10, n_iter, '=' * 10))
            print('{} Iter {} {}'.format('=' * 10, n_iter, '=' * 10))
            fps = int(target_network_update_freq / (time.time() - start_ts))
            start_ts = time.time()
            print('FPS {}'.format(fps))

            if n_iter > learning_starts and n_iter % train_freq == 0:
                print('vloss: {:.6f}'.format(loss.item()))

        if save_interval and n_iter % save_interval == 0:
            torch.save(qnet.state_dict(), os.path.join(save_path, '{}.pth'.format(n_iter)))


def _generate(device, env, qnet, ob_scale,
              number_timesteps, param_noise,
            exploration_final_eps,
              atom_num, min_value, max_value):

    """ Generate training batch sample """
    explore_steps = (config.exploration_start_eps-exploration_final_eps) / number_timesteps

    o = env.reset()
    
    # if use imitation learning
    imitation = True if random.random() < config.imitation_ratio else False
    imitation_actions = find_path(env)

    while imitation_actions is None:
        o = env.reset()
        imitation_actions = find_path(env)

    o = torch.from_numpy(o).to(device)

    hidden = None
    epsilon = config.exploration_start_eps
    for n in range(1, number_timesteps + 1):

        if imitation:

            a = imitation_actions.pop(0)
            # print(a)

        else:
            # sample action
            with torch.no_grad():
                ob = o

                # 1 x 3 x 3 x 8 x 8
                if hidden is not None:
                    q, hidden = qnet(ob, hidden)
                else:
                    q, hidden = qnet(ob)
                # 1 x 3 x 5 or 1 x 3 x 5 x atom_num

                a = q.argmax(1).cpu().tolist()

                if random.random() < epsilon:
                    a[0] = np.random.randint(0, config.action_space)


        # take action in env
        o_, r, done, info = env.step(a)
        o_ = torch.from_numpy(o_).to(device)
        # print(r)
        


        # return data and update observation

        yield (o[0,:,:,:], a, r, o_[0,:,:,:], int(done), imitation, info)


        if not done and env.steps < config.max_steps:

            o = o_ 
        else:
            o = env.reset()

            imitation = True if random.random() < config.imitation_ratio else False
            imitation_actions = find_path(env)

            while imitation_actions is None:
                o = env.reset()
                imitation_actions = find_path(env)

            o = torch.from_numpy(o).to(device)

            hidden = None
            # if imitation:
            #     print(env.map)
            #     print(env.agents_pos)
            #     print(env.goals_pos)
            #     print(imitation_actions)
        
        epsilon -= explore_steps
            


def huber_loss(abs_td_error):
    flag = (abs_td_error < 1).float()
    return flag * abs_td_error.pow(2) * 0.5 + (1 - flag) * (abs_td_error - 0.5)


def scale_ob(array, device, scale):
    return torch.from_numpy(array.astype(np.float32) * scale).to(device)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


if __name__ == '__main__':

    # a = np.array([[[1,2],[3,4]], [[5,6],[7,8]]])
    # print(a[[0,1],[0,1],[0,1]])

    env = Environment()
    learn(env, 5000000)