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
    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super(ResBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, stride, padding),
            # nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size, stride, padding),
            # nn.BatchNorm2d(dim),
        )

    def forward(self, x):

        return F.relu(self.block(x) + x)

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_sa_layers=config.num_sa_layers, num_sa_heads=config.num_sa_heads):
        super(SelfAttention, self).__init__()

        self.self_attns = nn.ModuleList([nn.MultiheadAttention(d_model, config.num_sa_heads) for _ in range(config.num_sa_layers)])
        self.linears = nn.ModuleList([nn.Sequential(
                                                    nn.ReLU(True),
                                                    nn.Linear(d_model, d_model),
                                                    nn.ReLU(True),
                                                )

                                                for _ in range(config.num_sa_layers)])
        # self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(config.num_sa_layers)])

    def forward(self, src, src_mask=None, src_key_padding_mask=None):

        
        for self_attn, linear in zip(self.self_attns, self.linears):
        
            src = self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
            src = linear(src)

        return src


class Network(nn.Module):
    def __init__(self, atom_num, dueling):
        super(Network, self).__init__()


        self.atom_num = atom_num
        
        # self.conv_net = nn.Sequential(
        #     nn.Conv2d(3, config.num_kernels, 3, 1),
        #     nn.ReLU(True),
        #     nn.Conv2d(config.num_kernels, config.num_kernels, 3, 1),
        #     nn.ReLU(True),
        #     nn.Conv2d(config.num_kernels, config.num_kernels, 3, 1),
        #     nn.ReLU(True),
        #     Flatten(),
        #     nn.Linear(2*2*config.num_kernels, 2*2*config.num_kernels),
        #     nn.ReLU(True)

        # )

        self.conv_net = nn.Sequential(
            
            nn.Conv2d(3, config.num_kernels, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(config.num_kernels, config.num_kernels, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(config.num_kernels, config.num_kernels, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(config.num_kernels, config.num_kernels, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(config.num_kernels, config.num_kernels, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(config.num_kernels, config.num_kernels, 3, 1, 1),
            nn.ReLU(True),

            nn.Conv2d(config.num_kernels, 4, 1, 1),
            nn.ReLU(True),

            Flatten(),

            nn.Linear(4*8*8, config.latent_dim),
            nn.ReLU(True),

        )

        self.self_attn = SelfAttention(config.latent_dim)
        
        self.q = nn.Sequential(
            # nn.Linear(2*2*config.num_kernels, 2*2*config.num_kernels),
            # nn.ReLU(True),
            nn.Linear(config.latent_dim, config.action_space * atom_num)
        )

        if dueling:
            self.state = nn.Sequential(
                # nn.Linear(2*2*config.num_kernels, 2*2*config.num_kernels),
                # nn.ReLU(True),
                nn.Linear(config.latent_dim, atom_num)
            )

        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
    def forward(self, x):

        assert x.dim() == 5, str(x.shape)


        batch_size = x.size(0)

        x = x.view(-1, 3, 8, 8)

        latent = self.conv_net(x)

        latent = latent.view(config.num_agents, batch_size, 2*2*config.num_kernels)
        latent = self.self_attn(latent)
        latent = latent.view(config.num_agents*batch_size, 2*2*config.num_kernels)

        # latent = torch.cat((latent, latent_), 1)
        
        q_val = self.q(latent)


        if self.atom_num == 1:
            if hasattr(self, 'state'):
                s_val = self.state(latent)
                qvalue = s_val + q_val - q_val.mean(1, keepdim=True)
            return qvalue.view(batch_size, config.num_agents, config.action_space)
        else:

            q_val = q_val.view(batch_size*config.num_agents, config.action_space, self.atom_num)
            if hasattr(self, 'state'):
                s_val = self.state(latent).unsqueeze(1)
                q_val = s_val + q_val - q_val.mean(1, keepdim=True)
            logprobs = log_softmax(q_val, -1)
            return logprobs.view(batch_size, config.num_agents, config.action_space, self.atom_num)


# if __name__ == '__main__':
#     t = torch.rand(2, 4)
#     # values, indices = torch.argmax(t, 0)
#     print(t)
#     m = torch.LongTensor([[1],[2]])
#     print(t.gather(1,m))