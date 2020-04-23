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

        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)

        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity

        out = F.relu(out)

        return out



class Network(nn.Module):
    def __init__(self, dueling=True, map_size=config.map_size):
        super().__init__()

        self.encoder = nn.Sequential(
            
            nn.Conv2d(3+config.history_steps, config.num_kernels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(config.num_kernels),
            nn.ReLU(True),
            
            ResBlock(config.num_kernels),
            ResBlock(config.num_kernels),
            ResBlock(config.num_kernels),
            ResBlock(config.num_kernels),
            ResBlock(config.num_kernels),

            nn.Conv2d(2*config.num_kernels, 16, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            Flatten(),

        )

        self.linear = nn.Sequential(
            nn.Linear(16*config.map_size[0]*config.map_size[1], config.latent_dim),
            nn.ReLU(True),
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.ReLU(True),
        )

        self.adv = nn.Linear(config.latent_dim, config.action_space)

        self.state = nn.Linear(config.latent_dim, 1)

        for _, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

    
    def forward(self, x):

        latent = self.encoder(x)

        latent = self.linear(latent)
        
        adv_val = self.adv(latent)

        s_val = self.state(latent)

        q_val = s_val + adv_val - adv_val.mean(1, keepdim=True)

        return q_val
    