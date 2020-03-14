import os
os.environ["OMP_NUM_THREADS"] = "1"
from worker import Play, Train
from model import Network
import config
import torch
import torch.multiprocessing as mp
import numpy as np
import random
import time
from environment import Environment
torch.manual_seed(0x5A31)
np.random.seed(0x5A31)
random.seed(0x5A31)


def drl():


    p_list = []

    network = Network()
    # network = network.float()
    network.eval()
    network.share_memory()

    training_q = mp.Queue(500)


    for _ in range(6):

        master_p = Play(3, network, training_q)
        master_p.start()
        p_list.append(master_p)


    training_p = Train(training_q, network)
    training_p.start()

    training_p.join()
    # p_list.append(training_p)

    for p in p_list:
        p.terminate()
        p.join()


def test():
    env = Environment(3)
    network = Network()
    network.eval()
    done = False
    while not done:
        env.render()
        obs = env.joint_observe()
        obs = torch.from_numpy(obs)

        with torch.no_grad():
            q_vals = network(obs)

        # get results
        actions = np.random.randint(config.action_space, size=q_vals.size()[0])
        
        done = env.step(actions)



if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    drl()
    # test()





