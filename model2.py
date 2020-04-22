import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import log_softmax

import math
import numpy as np

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)

class ResBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        identity = x

        out = F.relu(self.conv1(x))
        out = self.conv2(out)

        out += identity

        out = F.relu(out)

        return out



class Network(nn.Module):
    def __init__(self, dueling=True, map_size=config.map_size):
        super().__init__()

        self.encoder = nn.Sequential(
            
            nn.Conv2d(4, config.num_kernels, 3, 1, 1),
            nn.ReLU(),
            
            ResBlock(config.num_kernels),
            ResBlock(config.num_kernels),
            ResBlock(config.num_kernels),
            ResBlock(config.num_kernels),

            nn.Conv2d(config.num_kernels, 16, 1, 1),
            nn.ReLU(True),

            Flatten(),

        )

        self.linear = nn.Sequential(
            nn.Linear(16*config.map_size[0]*config.map_size[1], config.latent_dim),
            nn.ReLU(True),
        )

        # self.gru = nn.GRUCell(16*config.map_size[0]*config.map_size[1], config.latent_dim)

        self.adv = nn.Linear(config.latent_dim, config.action_space)

        self.state = nn.Linear(config.latent_dim, 1)

        for _, m in self.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
    def forward(self, x, hidden=None):

        latent = self.encoder(x)

        latent = self.linear(latent)

        # if hidden is not None:
        #     hidden = self.gru(latent, hidden)
        # else:
        #     hidden = self.gru(latent)
        
        adv_val = self.adv(latent)

        s_val = self.state(latent)

        q_val = s_val + adv_val - adv_val.mean(1, keepdim=True)

        return q_val, hidden
    
    # def bootstrap(self, x, steps=None, hidden=None):
    #     # batch_size x steps x obs
    #     step = x.size(1)

    #     x = x.view(-1, 4, *config.map_size)

    #     latent = self.encoder(x)

    #     latent = latent.view(config.batch_size, step, 16*config.map_size[0]*config.map_size[1]).permute(1, 0, 2)

    #     if hidden is not None:
    #         for i in range(step): 
    #             hidden = self.gru(latent[i], hidden)
    #     else:
            
    #         hidden = self.gru(latent[0])
    #         for i in range(1, step): 
    #             hidden = self.gru(latent[i], hidden)

    #     return hidden