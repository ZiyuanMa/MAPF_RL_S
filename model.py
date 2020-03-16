import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
        
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=size))

        return x.sign().mul(x.abs().sqrt())


class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, config.num_sa_heads)
        self.norm = nn.LayerNorm(d_model)
        

    def forward(self, src, src_mask=None, src_key_padding_mask=None):

        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + src2
        src = self.norm(src)

        return src


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
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

        self.self_attn = nn.Sequential(
            SelfAttention(2*2*config.num_kernels),
            SelfAttention(2*2*config.num_kernels),
            SelfAttention(2*2*config.num_kernels),
            SelfAttention(2*2*config.num_kernels),
        )

        self.value_net = nn.Sequential(
            nn.Linear(2*2*config.num_kernels, 2*2*config.num_kernels),
            nn.LeakyReLU(),

            NoisyLinear(2*2*config.num_kernels, 2*2*config.num_kernels),
            nn.LeakyReLU(),

            NoisyLinear(2*2*config.num_kernels, 1),
        )

        self.advantage_net = nn.Sequential(
            nn.Linear(2*2*config.num_kernels, 2*2*config.num_kernels),
            nn.LeakyReLU(),

            NoisyLinear(2*2*config.num_kernels, 2*2*config.num_kernels),
            nn.LeakyReLU(),

            NoisyLinear(2*2*config.num_kernels, config.action_space),
        )
    
    def forward(self, x, seq_mask=None):
        if len(x.size()) == 5:
            x = x.view(-1, 3, 8, 8)

        x = self.conv_net(x)
        x = self.flatten(x)
        
        if seq_mask is not None:

            x = x.view(seq_mask.size()[1], seq_mask.size()[0], 2*2*config.num_kernels)
            x = self.self_attn(x)
            x = x.view(seq_mask.size()[0]*seq_mask.size()[1], 2*2*config.num_kernels)

            value = self.value_net(x)
            adv = self.advantage_net(x)
            q = value + adv - adv.mean(dim=-1, keepdim=True)

            q = q.view(seq_mask.size()[0], seq_mask.size()[1], config.action_space)

        else:

            x = torch.unsqueeze(x, 1)
            x = self.self_attn(x)
            x = torch.squeeze(x, 1)

            value = self.value_net(x)
            adv = self.advantage_net(x)
            q = value + adv - adv.mean(dim=-1, keepdim=True)


        return q

    def reset_noise(self):
        """Reset all noisy layers."""
        self.value_net[2].reset_noise()
        self.value_net[4].reset_noise()
        self.advantage_net[2].reset_noise()
        self.advantage_net[4].reset_noise()

# if __name__ == '__main__':
#     t = torch.rand(2, 4)
#     # values, indices = torch.argmax(t, 0)
#     print(t)
#     m = torch.LongTensor([[1],[2]])
#     print(t.gather(1,m))