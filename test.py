import numpy as np
import torch
from environment import Environment
from model import Network
import config
from search import find_path
import pickle
import os
import matplotlib.pyplot as plt
import random
import argparse
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
test_case = 200

def create_test(num_agents):

    tests = {'maps': [], 'agents': [], 'goals': [], 'opt_steps': []}

    for _ in range(test_case):
        env = Environment(num_agents=num_agents)
        tests['maps'].append(np.copy(env.map))
        tests['agents'].append(np.copy(env.agents_pos))
        tests['goals'].append(np.copy(env.goals_pos))

        actions = find_path(env)
        while actions is None:
            env.reset()
            tests['maps'][-1] = np.copy(env.map)
            tests['agents'][-1] = np.copy(env.agents_pos)
            tests['goals'][-1] = np.copy(env.goals_pos)
            actions = find_path(env)

        tests['opt_steps'].append(len(actions))

    with open('./test{}.pkl'.format(env.num_agents), 'wb') as f:
        pickle.dump(tests, f)


def test_model(num_agents):


    network = Network(config.dueling)


    with open('./test{}.pkl'.format(num_agents), 'rb') as f:
        tests = pickle.load(f)

    checkpoint = config.save_interval
    
    x = []
    y1 = []
    y2 = []

    while os.path.exists('./models/'+str(checkpoint)+'.pth'):

        # print('true')
        state_dict = torch.load('./models/'+str(checkpoint)+'.pth')
        network.load_state_dict(state_dict)
        network.eval()


        env = Environment()
        case = 0
        show = True
        show_steps = 20
        sum_reward = 0
        fail = 0
        optimal = 0

        for i in range(200):
            env.load(tests['maps'][i], tests['agents'][i], tests['goals'][i])
            
            done = [False for _ in range(env.num_agents)]

            round_reward = 0
            while False in done and env.steps<config.max_steps:
                if i == case and show and env.steps < show_steps:
                    env.render()

                obs = env.observe()
                # obs = np.expand_dims(obs, axis=0)
                obs = torch.from_numpy(obs).float()

                with torch.no_grad():

                    q_vals = network(obs)


                if i == case and show and env.steps < show_steps:
                    print(q_vals)

                action = torch.argmax(q_vals, 1).tolist()

                if i == case and show and env.steps < show_steps:
                    print(action)

                for j in range(env.num_agents):
                    if done[j]:
                        action[j] = 0

                obs, reward, done, _ = env.step(action)
                # print(done)

                round_reward += sum(reward) / env.num_agents
            
            sum_reward += round_reward

            if not np.array_equal(env.agents_pos, env.goals_pos):
                fail += 1
                if show:
                    print(i)

            if env.steps == tests['opt_steps'][i]:
                optimal += 1

            if i == case and show:
                env.close()

        sum_reward /= test_case

        print('---------checkpoint '+str(checkpoint)+'---------------')
        print('test score: %.3f' %sum_reward)
        print('fail: %d' %fail)
        print('optimal: %d' %optimal)
        # print('best score: %.3f' %(sum(tests['rewards'])/test_case))

        # x.append(checkpoint)
        # y1.append(sum_reward)
        # y2.append(sum(tests['rewards'])/test_case)
        checkpoint += config.save_interval
    
    # plt.plot(x, y1, 'b-')
    # plt.plot(x, y2, 'g-')
    # plt.show()


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test MAPF model')

    parser.add_argument('--mode', type=str, choices=['test', 'create'], default='test', help='create test set or run test set')
    parser.add_argument('--number', type=int, default=config.num_agents, help='number of agents in environment')

    args = parser.parse_args()

    if args.mode == 'test':
        test_model(args.number)
    elif args.mode == 'create':
        create_test(args.number)
    