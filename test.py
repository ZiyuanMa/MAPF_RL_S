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
np.random.seed(0)
random.seed(0)

def create_test():

    tests = {'maps': [], 'agents': [], 'goals': [], 'rewards': []}

    for _ in range(100):
        env = Environment()
        tests['maps'].append(np.copy(env.map))
        tests['agents'].append(np.copy(env.agents_pos))
        tests['goals'].append(np.copy(env.goals_pos))

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
    # if checkpoint:
    #     network.load_state_dict(torch.load(checkpoint+'.checkpoint')[0])
    # else:

    with open('./test.pkl', 'rb') as f:
        tests = pickle.load(f)

    checkpoint = config.save_interval

    x = []
    y1 = []
    y2 = []

    while os.path.exists('./models/'+str(checkpoint)+'.checkpoint'):


        state_dict, atom_num = torch.load('./models/'+str(checkpoint)+'.checkpoint')
        network.load_state_dict(state_dict)
        if atom_num > 1:
            vrange = torch.linspace(config.min_value, config.max_value, atom_num)

        env = Environment()
        sum_reward = 0

        for i in range(100):
            env.load(tests['maps'][i], tests['agents'][i], tests['goals'][i])
            
            done = False

            while not done:
                if i == 50:
                    env.render()
                obs = env.observe()
                obs = np.expand_dims(obs, axis=0)
                obs = torch.from_numpy(obs).float()

                with torch.no_grad():
                    q_vals = network(obs)

                if atom_num > 1:
                    q_vals = (q_vals.exp() * vrange).sum(3)

                action = torch.argmax(q_vals, 2).tolist()[0]
                # print(action)
                obs, reward, done, _ = env.step(action)

                sum_reward += sum(reward) / env.num_agents

            if i == 50:
                env.close()

        sum_reward /= 100

        print('---------checkpoint '+str(checkpoint)+'---------------')
        print('test score: %.3f' %sum_reward)
        print('best score: %.3f' %(sum(tests['rewards'])/100))

        x.append(checkpoint)
        y1.append(sum_reward)
        y2.append(sum(tests['rewards'])/100)
        checkpoint += config.save_interval
    
    plt.plot(x, y1, 'b-')
    plt.plot(x, y2, 'g-')
    plt.show()


if __name__ == '__main__':
    # create_test()
    test_model()