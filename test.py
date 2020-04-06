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
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
test_case = 200

def create_test():

    tests = {'maps': [], 'agents': [], 'goals': [], 'rewards': []}

    for _ in range(test_case):
        env = Environment()
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

        sum_reward = 0
        for action in actions:
            _, reward, _, _ = env.step(action)
            sum_reward += sum(reward) / env.num_agents

        tests['rewards'].append(sum_reward)

    with open('./test.pkl', 'wb') as f:
        pickle.dump(tests, f)


def test_model():


    network = Network(config.atom_num, config.dueling)


    with open('./test.pkl', 'rb') as f:
        tests = pickle.load(f)

    checkpoint = config.save_interval * 20
    
    x = []
    y1 = []
    y2 = []

    while os.path.exists('./models/'+str(checkpoint)+'.pth'):

        # print('true')
        state_dict = torch.load('./models/'+str(checkpoint)+'.pth')
        network.load_state_dict(state_dict)
        network.eval()
        if config.atom_num > 1:
            vrange = torch.linspace(config.min_value, config.max_value, config.atom_num)

        env = Environment()
        case = 56
        show = False
        show_steps = 15
        sum_reward = 0
        fail = 0
        optimal = 0

        for i in range(200):
            env.load(tests['maps'][i], tests['agents'][i], tests['goals'][i])
            
            done = False
            round_reward = 0
            while not done and env.steps<config.max_steps:
                if i == case and show and env.steps < show_steps:
                    env.render()

                obs = env.observe()
                obs = np.expand_dims(obs, axis=0)
                obs = torch.from_numpy(obs).float()

                with torch.no_grad():
                    q_vals = network(obs)

                if config.atom_num > 1:
                    q_vals = (q_vals.exp() * vrange).sum(2)

                if i == case and show and env.steps < show_steps:
                    print(q_vals)

                action = torch.argmax(q_vals, 2).tolist()[0]
                # print(action)
                obs, reward, done, _ = env.step(action)

                round_reward += sum(reward) / env.num_agents
            
            sum_reward += round_reward

            if not np.array_equal(env.agents_pos, env.goals_pos):
                fail += 1
                if show:
                    print(i)

            if round_reward == tests['rewards'][i]:
                optimal += 1

            if i == case and show:
                env.close()

        sum_reward /= test_case

        print('---------checkpoint '+str(checkpoint)+'---------------')
        print('test score: %.3f' %sum_reward)
        print('fail: %d' %fail)
        print('optimal: %d' %optimal)
        print('best score: %.3f' %(sum(tests['rewards'])/test_case))

        x.append(checkpoint)
        y1.append(sum_reward)
        y2.append(sum(tests['rewards'])/test_case)
        checkpoint += config.save_interval
    
    plt.plot(x, y1, 'b-')
    plt.plot(x, y2, 'g-')
    plt.show()

def test_init_model():


    network = Network(config.atom_num, config.dueling)
    
    with open('./test.pkl', 'rb') as f:
        tests = pickle.load(f)

    network.encoder.load_state_dict(torch.load('./encoder.pth'))
    network.q.load_state_dict(torch.load('./q.pth'))
    network.state.load_state_dict(torch.load('./state.pth'))
    network.eval()
    if config.atom_num > 1:
        vrange = torch.linspace(config.min_value, config.max_value, config.atom_num)

    env = Environment()
    case = 3
    show = True
    show_steps = 15
    sum_reward = 0
    fail = 0
    optimal = 0

    for i in range(200):
        env.load(tests['maps'][i], tests['agents'][i], tests['goals'][i])
        
        done = False
        round_reward = 0
        while not done and env.steps<config.max_steps:
            if i == case and show and env.steps < show_steps:
                env.render()

            obs = env.observe()
            obs = np.expand_dims(obs, axis=0)
            obs = torch.from_numpy(obs).float()

            with torch.no_grad():
                q_vals = network(obs)

            if config.atom_num > 1:
                q_vals = (q_vals.exp() * vrange).sum(2)

            if i == case and show and env.steps < show_steps:
                print(q_vals)

            action = torch.argmax(q_vals, 2).tolist()[0]
            # print(action)
            obs, reward, done, _ = env.step(action)

            round_reward += sum(reward) / env.num_agents
        
        sum_reward += round_reward

        if not np.array_equal(env.agents_pos, env.goals_pos):
            fail += 1
            if show:
                print(i)

        if round_reward == tests['rewards'][i]:
            optimal += 1

        if i == case and show:
            env.close()

    sum_reward /= test_case


    print('test score: %.3f' %sum_reward)
    print('fail: %d' %fail)
    print('optimal: %d' %optimal)
    print('best score: %.3f' %(sum(tests['rewards'])/test_case))

    

if __name__ == '__main__':
    # create_test()
    # test_model()
    test_init_model()