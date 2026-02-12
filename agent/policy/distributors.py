import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal

class CategoricalDist(nn.Module):
    def forward(self, logits):
        return Categorical(logits=logits)

    def post_process(self, action):
        return action.detach().cpu().item()

class DiagGaussianDist(nn.Module):
    def __init__(self, action_dim, low=-1.0, high=1.0):
        super().__init__()
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.register_buffer("low", torch.tensor(low))
        self.register_buffer("high", torch.tensor(high))

    def forward(self, mu):
        std = self.log_std.exp()
        return Normal(mu, std)

    def post_process(self, action):
        clamped = torch.clamp(action, self.low, self.high)
        out = clamped.detach().cpu().numpy()
        return np.atleast_1d(out).flatten()
