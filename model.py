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


class SelfAttention(nn.Module):
    def __init__(self, d_model, num_sa_layers=config.num_sa_layers, num_sa_heads=config.num_sa_heads):
        super(SelfAttention, self).__init__()

        self.self_attns = nn.ModuleList([nn.MultiheadAttention(d_model, config.num_sa_heads) for _ in range(num_sa_layers)])
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_model),
                                                    nn.ReLU(True),
                                                    nn.Linear(d_model, d_model),
                                                )

                                                for _ in range(num_sa_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_sa_layers)])

    def forward(self, src, src_mask=None, src_key_padding_mask=None):

        
        for self_attn, linear, norm in zip(self.self_attns, self.linears, self.norms):
        
            src_out = self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
            src_out = linear(src_out)
            src = src + src_out
            src = norm(src)

        return src


class Network(nn.Module):
    def __init__(self, atom_num, dueling):
        super(Network, self).__init__()


        self.atom_num = atom_num
        
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 2, 1),
            nn.ReLU(True),
            Flatten(),

        )

        self.self_attn = SelfAttention(256)
        
        self.q = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, config.action_space * atom_num)
        )

        if dueling:
            self.state = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(True),
                nn.Linear(256, atom_num)
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