import config
import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(config.obs_dimension, config.num_kernels, 3, 1),
            nn.LeakyReLU(),
            nn.Conv2d(config.num_kernels, config.num_kernels, 3, 1),
            nn.LeakyReLU(),
            nn.Conv2d(config.num_kernels, config.num_kernels, 3, 1),
            nn.LeakyReLU(),
        )

        self.flatten = nn.Flatten()

        self_attn_layers = nn.TransformerEncoderLayer(d_model=2*2*config.num_kernels, nhead=config.num_sa_heads,dim_feedforward=2*2*config.num_kernels)
        self.self_attn = nn.TransformerEncoder(self_attn_layers, config.num_sa_layers)


        self.fc_net = nn.Sequential(
            nn.Linear(2*2*config.num_kernels, 2*2*config.num_kernels),
            nn.LeakyReLU(),
            nn.Linear(2*2*config.num_kernels, 2*2*config.num_kernels),
            nn.LeakyReLU(),
            nn.Linear(2*2*config.num_kernels, config.action_space),
        )
    
    def forward(self, x, mask=None):
        x = self.conv_net(x)
        x = self.flatten(x)
        if mask:
            pass
        else:
            x = torch.unsqueeze(x, 1)

        x = self.self_attn(x) + x
        if mask:
            pass
        else:
            x = torch.squeeze(x, 1)
        x = self.fc_net(x)
        return x

if __name__ == '__main__':
    t = torch.rand((4,3, 8, 8))
    n = Network()
    

    print(n(t))