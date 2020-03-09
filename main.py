import os
os.environ["OMP_NUM_THREADS"] = "1"
from worker import Play, Train
from model import Network
import torch.multiprocessing as mp
import numpy as np
from environment import Environment


def drl():
    l = np.random.randint(1, 4, size=12)
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
    p_list.append(training_p)

    for p in p_list:
        p.join()


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    drl()





