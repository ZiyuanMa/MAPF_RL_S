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
    def __init__(self, atom_num, dueling=True, map_size=config.map_size):
        super().__init__()

        self.encoder = nn.Sequential(
            
            nn.Conv2d(4, config.num_kernels, 3, 1, 1),
            nn.ReLU(),
            
            ResBlock(config.num_kernels),
            ResBlock(config.num_kernels),
            ResBlock(config.num_kernels),
            ResBlock(config.num_kernels),

            nn.Conv2d(config.num_kernels, 8, 1, 1),
            nn.ReLU(),

            Flatten(),

        )

        self.linear = nn.Sequential(
            nn.Linear(8*config.map_size[0]*config.map_size[1], 8*config.map_size[0]*config.map_size[1]),
            nn.ReLU(True),
            nn.Linear(8*config.map_size[0]*config.map_size[1], config.latent_dim),
            nn.ReLU(True),
        )

        self.gru = nn.GRU(config.latent_dim, config.latent_dim, batch_first=True)

        self.adv = nn.Linear(config.latent_dim, config.action_space)

        self.state = nn.Linear(config.latent_dim, 1)

        for _, m in self.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
    def forward(self, x, hidden=None):

        latent = self.encoder(x)

        latent = self.linear(latent)

        latent = latent.unsqueeze(1)
        if hidden is not None:
            latent, hidden = self.gru(latent, hidden)
        else:
            latent, hidden = self.gru(latent)
        latent = latent.squeeze(1)
        
        adv_val = self.adv(latent)

        s_val = self.state(latent)

        q_val = s_val + adv_val - adv_val.mean(1, keepdim=True)

        return q_val, hidden
    
    def bootstrap(self, x, steps):
        assert x.size(1) == config.bootstrap_steps

        x = x.view(-1, 4, *config.map_size)

        latent = self.encoder(x)

        latent = self.linear(latent)

        latent = latent.view(config.batch_size, config.bootstrap_steps, -1)

        latent = nn.utils.rnn.pack_padded_sequence(latent, steps, batch_first=True, enforce_sorted=False)

        _, hidden = self.gru(latent)

        hidden = nn.utils.rnn.pad_packed_sequence(hidden, batch_first=False, padding_value=0, total_length=None)

        return hidden