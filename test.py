import numpy as np
import torch
from environment import Environment
from model import Network
import config
import random

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

if __name__ == '__main__':
    net = Network(config.atom_num, config.dueling)
    net.load_state_dict(torch.load('./500000.checkpoint')[0])
    net.eval()
    print('load')
    env = Environment(3)
    test_case = 0
    env.load(environment[test_case], 3, agents_position[test_case], goals_position[test_case])

    done = False
    # start one eposide
    while not done:
        # self.train_lock.wait()
        env.render()
        # observe
        obs = env.observe()
        obs = torch.from_numpy(obs).unsqueeze(0)

        with torch.no_grad():
            q_vals = net(obs)

        print(q_vals)


        actions = torch.argmax(q_vals, 2).numpy()[0].tolist()


        print(actions)
        _, _, done, _ = env.step(actions)
    print('done')
