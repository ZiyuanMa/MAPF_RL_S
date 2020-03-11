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
            nn.Dropout(0.1),
            nn.Linear(2*2*config.num_kernels, 2*2*config.num_kernels),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(2*2*config.num_kernels, config.action_space),
        )
    
    def forward(self, x, seq_mask=None):
        if len(x.size()) == 5:
            x = x.view(-1, 3, 8, 8)
        x = self.conv_net(x)
        x = self.flatten(x)
        if seq_mask is not None:
            # assert x.size()[0] == seq_mask.size()[0], 'batch mismatch 1'
            x = x.view(seq_mask.size()[1], seq_mask.size()[0], 2*2*config.num_kernels)
            x = self.self_attn(x, src_key_padding_mask=seq_mask) + x
            x = x.view(seq_mask.size()[0]*seq_mask.size()[1], 2*2*config.num_kernels)
            x = self.fc_net(x)
            x = x.view(seq_mask.size()[0], seq_mask.size()[1], config.action_space)
            # print(x.shape)
        else:

            x = torch.unsqueeze(x, 1)
            x = self.self_attn(x) + x
            x = torch.squeeze(x, 1)
            x = self.fc_net(x)

        return x

# if __name__ == '__main__':
#     t = torch.rand(2, 4)
#     # values, indices = torch.argmax(t, 0)
#     print(t)
#     m = torch.LongTensor([[1],[2]])
#     print(t.gather(1,m))