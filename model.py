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

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_sa_layers=config.num_sa_layers, num_sa_heads=config.num_sa_heads):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(4*config.map_size[0]*config.map_size[1], d_model),
            nn.ReLU(True),
        )
        
        self.self_attns = nn.ModuleList([nn.MultiheadAttention(d_model, config.num_sa_heads) 
                                                    for _ in range(config.num_sa_layers)])

        self.linears = nn.ModuleList([nn.Sequential(
                                                    nn.ReLU(True),
                                                    nn.Linear(d_model, d_model),
                                                    nn.ReLU(True),
                                    )

                                                    for _ in range(config.num_sa_layers)])


    def forward(self, src, src_mask=None, src_key_padding_mask=None):

        src = self.linear(src)

        for self_attn, linear in zip(self.self_attns,self.linears) :
        
            src = self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
            src = linear(src)

        return src


class Network(nn.Module):
    def __init__(self, atom_num, dueling=True, map_size=config.map_size):
        super().__init__()

        self.encoder = nn.Sequential(
            
            nn.Conv2d(3, config.num_kernels, 3, 1, 1),
            nn.ReLU(),
            
            ResBlock(config.num_kernels),
            ResBlock(config.num_kernels),
            ResBlock(config.num_kernels),
            ResBlock(config.num_kernels),

            nn.Conv2d(config.num_kernels, 4, 1, 1),
            nn.ReLU(),

            Flatten(),

        )

        self.self_attn = SelfAttention(config.latent_dim)
        self.linear = nn.Sequential(
            nn.Linear(4*config.map_size[0]*config.map_size[1], config.latent_dim),
            nn.ReLU(True),
        )

        
        self.adv = nn.Linear(2*config.latent_dim, config.action_space)

        if dueling:
            self.state = nn.Linear(2*config.latent_dim, 1)

        for _, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
    def forward(self, x):

        assert x.dim() == 5, str(x.shape)


        batch_size = x.size(0)

        x = x.view(-1, 3, config.map_size[0], config.map_size[1])

        latent = self.encoder(x)

        comm_latent = latent
        comm_latent = comm_latent.view(config.num_agents, batch_size, 4*config.map_size[0]*config.map_size[1])
        comm_latent = self.self_attn(comm_latent)
        comm_latent = comm_latent.view(config.num_agents*batch_size, config.latent_dim)

        latent = self.linear(latent)

        latent = torch.cat((latent, comm_latent), 1)
        
        adv_val = self.adv(latent)

        s_val = self.state(latent)

        q_val = s_val + adv_val - adv_val.mean(1, keepdim=True)

        return q_val.view(batch_size, config.num_agents, config.action_space)
    

# if __name__ == '__main__':
#     t = torch.rand(2, 4)
#     # values, indices = torch.argmax(t, 0)
#     print(t)
#     m = torch.LongTensor([[1],[2]])
#     print(t.gather(1,m))