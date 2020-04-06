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

from buffer import ReplayBuffer, PrioritizedReplayBuffer
from model import Network
from environment import Environment
import config
from search import find_path

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def learn(  env, number_timesteps,
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), save_path='./models', save_interval=config.save_interval,
            ob_scale=config.ob_scale, gamma=config.gamma, grad_norm=config.grad_norm, double_q=config.double_q,
            param_noise=config.param_noise, dueling=config.dueling, exploration_fraction=config.exploration_fraction,
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
    network.encoder.load_state_dict(torch.load('./encoder.pth', map_location=torch.device('cpu')))
    network.q.load_state_dict(torch.load('./q.pth', map_location=torch.device('cpu')))
    network.state.load_state_dict(torch.load('./state.pth', map_location=torch.device('cpu')))
    for param in network.encoder.parameters():
        param.requires_grad = False

    optimizer = Adam(
        [
        {"params": network.q.parameters(), "lr": 5e-4},
        {"params": network.state.parameters(), "lr": 5e-4},
        {"params": network.self_attn.parameters(), "lr": 1e-3},
        ],
        lr=1e-3, eps=1e-5
    )

    # create target network
    qnet = network.to(device)
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
                          exploration_fraction, exploration_final_eps,
                          atom_num, min_value, max_value)

    if atom_num > 1:
        delta_z = float(max_value - min_value) / (atom_num - 1)
        z_i = torch.linspace(min_value, max_value, atom_num).to(device)

    infos = {'eplenmean': deque(maxlen=100), 'eprewmean': deque(maxlen=100)}
    start_ts = time.time()
    for n_iter in range(1, number_timesteps + 1):

        if prioritized_replay:
            buffer.beta += (1 - prioritized_replay_beta0) / number_timesteps
            
        *data, info = generator.__next__()
        buffer.add(*data)
        for k, v in info.items():
            infos[k].append(v)

        # update qnet
        if n_iter > learning_starts and n_iter % train_freq == 0:
            b_obs, b_action, b_reward, b_obs_, b_done, b_steps, *extra = buffer.sample(batch_size)

            with torch.autograd.set_detect_anomaly(True):
                if atom_num == 1:
                    with torch.no_grad():
                        
                        # choose max q index from next observation
                        if double_q:
                            b_action_ = qnet(b_obs_).argmax(2).unsqueeze(2)
                            b_q_ = (1 - b_done).unsqueeze(2) * tar_qnet(b_obs_).gather(2, b_action_)
                        else:
                            b_q_ = (1 - b_done).unsqueeze(2) * tar_qnet(b_obs_).max(2, keepdim=True)[0]

                    b_action = b_action.unsqueeze(2)
                    b_q = qnet(b_obs).gather(2, b_action)

                    b_reward = b_reward.unsqueeze(2)
                    abs_td_error = (b_q - (b_reward + (gamma ** b_steps).unsqueeze(2) * b_q_)).abs()

                    
                    priorities = abs_td_error.detach().cpu().clamp(1e-6).numpy()
                    priorities = np.average(np.squeeze(priorities, axis=2), axis=1)

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

                if prioritized_replay:
                    buffer.update_priorities(extra[1], priorities)

        # update target net and log
        if n_iter % target_network_update_freq == 0:
            tar_qnet.load_state_dict(qnet.state_dict())
            # print('{} Iter {} {}'.format('=' * 10, n_iter, '=' * 10))
            print('{} Iter {} {}'.format('=' * 10, n_iter, '=' * 10))
            fps = int(target_network_update_freq / (time.time() - start_ts))
            start_ts = time.time()
            print('FPS {}'.format(fps))
            # print('FPS: ' + str(fps))
            # for k, v in infos.items():
            #     v = (sum(v) / len(v)) if v else float('nan')
            #     print(k)
            #     print(v)
                # logger.info('{}: {:.6f}'.format(k, v))
            if n_iter > learning_starts and n_iter % train_freq == 0:
                print('vloss: {:.6f}'.format(loss.item()))
                # print('loss: '+str(loss.item()))

        if save_interval and n_iter % save_interval == 0:
            torch.save(qnet.state_dict(), os.path.join(save_path, '{}.pth'.format(n_iter)))


def _generate(device, env, qnet, ob_scale,
              number_timesteps, param_noise,
              exploration_fraction, exploration_final_eps,
              atom_num, min_value, max_value):

    """ Generate training batch sample """
    noise_scale = 1e-2
    action_dim = config.action_space
    explore_steps = number_timesteps * exploration_fraction
    imitation_frac = config.imitation_ratio / number_timesteps

    if atom_num > 1:
        vrange = torch.linspace(min_value, max_value, atom_num).to(device)

    o = env.reset()
    
    
    # if use imitation learning
    imitation = True if random.random() < config.imitation_ratio else False
    imitation_actions = find_path(env)

    while imitation_actions is None:
        o = env.reset()
        imitation_actions = find_path(env)

    o = torch.from_numpy(o).to(device)


    infos = dict()
    for n in range(1, number_timesteps + 1):
        epsilon = 1.0 - (1.0 - exploration_final_eps) * n / explore_steps
        epsilon = max(exploration_final_eps, epsilon)

        config.imitation_ratio -= imitation_frac

        if imitation:
            # if not imitation_actions:
            #     print(env.map)
            # print(env.agents_pos)
            # print(env.goals_pos)

            a = imitation_actions.pop(0)
            # print(a)

        else:
            # sample action
            with torch.no_grad():
                ob = o.unsqueeze(0)

                # 1 x 3 x 3 x 8 x 8
                q = qnet(ob)
                # 1 x 3 x 5 or 1 x 3 x 5 x atom_num

                if atom_num > 1:
                    q = (q.exp() * vrange).sum(3)
                    
                if not param_noise:
                    if random.random() < epsilon:
                        a = np.random.randint(0, 5, size=config.num_agents).tolist()
                    else:
                        a = q.argmax(2).cpu().tolist()[0]

                else:
                    # see Appendix C of `https://arxiv.org/abs/1706.01905`
                    raise NotImplementedError('no noise')

                    q_dict = deepcopy(qnet.state_dict())
                    for _, m in qnet.named_modules():
                        if isinstance(m, nn.Linear):
                            std = torch.empty_like(m.weight).fill_(noise_scale)
                            m.weight.data.add_(torch.normal(0, std).to(device))
                            std = torch.empty_like(m.bias).fill_(noise_scale)
                            m.bias.data.add_(torch.normal(0, std).to(device))
                    q_perturb = qnet(ob)

                    if atom_num > 1:
                        q_perturb = (q_perturb.exp() * vrange).sum(2)

                    kl_perturb = ((log_softmax(q, 1) - log_softmax(q_perturb, 1)) *
                                    softmax(q, 1)).sum(-1).mean()

                    kl_explore = -math.log(1 - epsilon + epsilon / action_dim)

                    if kl_perturb < kl_explore:
                        noise_scale *= 1.01
                    else:
                        noise_scale /= 1.01

                    qnet.load_state_dict(q_dict)
                    if random.random() < epsilon:
                        a = int(random.random() * action_dim)
                    else:
                        a = q_perturb.argmax(1).cpu().numpy()[0]

        # take action in env
        o_, r, done, info = env.step(a)
        o_ = torch.from_numpy(o_).to(device)
        # print(r)
        

        if info.get('episode'):
            infos = {
                'eplenmean': info['episode']['l'],
                'eprewmean': info['episode']['r'],
            }
        # return data and update observation

        yield (o, a, r, o_, int(done), imitation, infos)
        infos = dict()

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

            # if imitation:
            #     print(env.map)
            #     print(env.agents_pos)
            #     print(env.goals_pos)
            #     print(imitation_actions)
            


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
    learn(env, 20000000)