import config
import torch
import torch.nn as nn
import torch.nn.functional as F




class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, config.num_sa_heads)
        self.norm = nn.LayerNorm(d_model)
        

    def forward(self, src, src_mask=None, src_key_padding_mask=None):

        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + src2
        src = self.norm(src)

        return src


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(config.obs_dimension, config.num_kernels, 3, 1),
            nn.LeakyReLU(),
            nn.Conv2d(config.num_kernels, config.num_kernels, 3, 1),
            nn.LeakyReLU(),
            nn.Conv2d(config.num_kernels, config.num_kernels, 3, 1),
            nn.LeakyReLU(),

        )

        self.flatten = nn.Flatten()

        self.self_attn1 = SelfAttention(2*2*config.num_kernels)
        self.self_attn2 = SelfAttention(2*2*config.num_kernels)

        self.value_net = nn.Sequential(
            nn.Linear(2*2*config.num_kernels, 2*2*config.num_kernels),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(2*2*config.num_kernels, 2*2*config.num_kernels),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(2*2*config.num_kernels, 1),
        )

        self.advantage_net = nn.Sequential(
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

            x = x.view(seq_mask.size()[1], seq_mask.size()[0], 2*2*config.num_kernels)
            x = self.self_attn1(x)
            x = self.self_attn2(x)
            x = x.view(seq_mask.size()[0]*seq_mask.size()[1], 2*2*config.num_kernels)

            value = self.value_net(x)
            adv = self.advantage_net(x)
            q = value + adv - adv.mean(dim=-1, keepdim=True)

            q = q.view(seq_mask.size()[0], seq_mask.size()[1], config.action_space)

        else:

            x = torch.unsqueeze(x, 1)
            x = self.self_attn1(x)
            x = self.self_attn2(x)
            x = torch.squeeze(x, 1)

            value = self.value_net(x)
            adv = self.advantage_net(x)
            q = value + adv - adv.mean(dim=-1, keepdim=True)


        return q

# if __name__ == '__main__':
#     t = torch.rand(2, 4)
#     # values, indices = torch.argmax(t, 0)
#     print(t)
#     m = torch.LongTensor([[1],[2]])
#     print(t.gather(1,m))