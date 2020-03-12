import os
os.environ["OMP_NUM_THREADS"] = "1"
from worker import Play, Train
from model import Network
import config
import torch
import torch.multiprocessing as mp
import numpy as np
import time
from environment import Environment


def drl():
    l = np.random.randint(1, 5, size=os.cpu_count())
    l = np.ones(os.cpu_count(), dtype=np.int)
    p_list = []

    network = Network()
    # network = network.float()
    network.eval()
    network.share_memory()

    training_l = mp.Event()
    training_q = mp.Queue()


    for agents_num in l:

        master_p = Play(agents_num, network, training_q, training_l)
        master_p.start()
        p_list.append(master_p)


    training_p = Train(training_q, training_l, network)
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





