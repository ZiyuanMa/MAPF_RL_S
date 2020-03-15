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
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
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
        [2, 2],
        [3, 2],
        [4, 2],
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
        [2, 5],
        [3, 5],
        [4, 5],
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
    net = Network()
    net.load_state_dict(torch.load('./model5.pth'))
    net.eval()
    print('load')
    env = Environment(3)
    env.load(environment[0], 3, agents_position[0], goals_position[0])

    done = False
    # start one eposide
    while not done:
        # self.train_lock.wait()
        env.render()
        # observe
        obs = env.joint_observe()
        obs = torch.from_numpy(obs)

        with torch.no_grad():
            q_vals = net(obs)

        print(q_vals)
        if random.random() < 0.5:
            # random action
            actions = np.random.randint(config.action_space, size=q_vals.size()[0])

        else:
            # greedy action

            actions = torch.argmax(q_vals, 1).numpy()


        print(actions)
        done = env.step(actions)
