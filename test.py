import numpy as np
import torch
from environment import Environment
from model import Network
import config
import random
from search import find_path
import pickle

environment = [
    [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],

    [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1],
    ],

    [
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1],
    ],

    [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],

    [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 0, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
]

agents_position = [
    [   
        [2, 0],
        [3, 0],
        [4, 0],
    ],

    [   
        [5, 0],
        [6, 0],
        [7, 0],
    ],

    [   
        [5, 0],
        [6, 0],
        [7, 0],
    ],

    [   
        [1, 1],
        [1, 2],
        [2, 1],
    ],

    [
        [4, 1],
        [1, 4],
        [4, 7],
    ],
]

goals_position = [
    [   
        [2, 7],
        [3, 7],
        [4, 7],
    ],

    [
        [2, 3],
        [2, 4],
        [2, 5],
    ],

    [
        [2, 7],
        [1, 7],
        [0, 7],
    ],

    [
        [6, 6],
        [6, 5],
        [5, 6],
    ],

    [
        [1, 4],
        [4, 7],
        [4, 1],
    ],

]
    



def create_env(obstacle_density):

    env = Environment(obstacle_density)

    return np.copy(env.map), np.copy(env.agents_pos), np.copy(env.goals_pos)

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


    network = CNN(config.atom_num, config.dueling)
    # if checkpoint:
    #     network.load_state_dict(torch.load(checkpoint+'.checkpoint')[0])
    # else:
    last_checkpoint = config.save_interval
    while os.path.exists('./'+str(last_checkpoint)+'.checkpoint'):
        last_checkpoint += config.save_interval

    last_checkpoint -= config.save_interval

    network.load_state_dict(torch.load(str(last_checkpoint)+'.checkpoint')[0])


    env = CustomEnv()
    sum_reward = 0

    with open('./test.pkl', 'rb') as f:
        tests = pickle.load(f)

    
    for i, test in enumerate(tests):
        env.load(test[0], test[1], test[2])
        
        done = False

        while not done:
            if (i==0 or i==10 or i == 20) and env.steps <= 15:
                env.render()

            obs = env.observe()
            obs = np.expand_dims(obs, axis=0)
            obs = torch.from_numpy(obs).float()

            with torch.no_grad():
                q_vals = network(obs)

            if (i==0 or i==10 or i == 20) and env.steps <= 15:
                print(q_vals)

            action = torch.argmax(q_vals, 1).item()
            # print(action)
            obs, reward, done, _ = env.step(action)

            # print(reward[0])
            sum_reward += reward
        
        env.close()

    print('test score: %.3f' %sum_reward)


    sum_reward = 0
    for i, test in enumerate(tests):
        env.load(test[0], test[1], test[2])

        map = (np.copy(env.map)==1).tolist()
                
        temp_agents_pos = np.copy(env.agent_pos)
        agents_pos= []
        agents_pos.append(tuple(temp_agents_pos))

        temp_goals_pos = np.copy(env.goal_pos)
        goals_pos = []
        goals_pos.append(tuple(temp_goals_pos))

        solver = CBSSolver(map, agents_pos, goals_pos)
        paths = solver.find_solution()

        if len(paths[0]) == 1:
            raise RuntimeError('agent_pos == goal_pos')


        for step in range(1, len(paths[0])):
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

            obs, reward, done, _  = env.step(actions[0])
            sum_reward += reward



    print('best score: %.3f' %sum_reward)


if __name__ == '__main__':

    create_test()
