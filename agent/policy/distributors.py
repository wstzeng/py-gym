import torch
from torch import nn
from torch.distributions import Categorical, Normal, Beta

class BaseDistributor(nn.Module):
    """Base class to ensure consistent interface."""
    def forward(self, logits):
        raise NotImplementedError

    def post_process(self, action):
        """Standardize output to (N, A) torch tensor."""
        raise NotImplementedError

class CategoricalDist(BaseDistributor):
    def __init__(self):
        super().__init__()

    def forward(self, logits):
        # logits shape: (N, num_classes)
        return Categorical(logits=logits)

    def post_process(self, action):
        # action from sample() is (N,)
        # Force to (N, 1) to match (N, A) convention
        return action.view(-1, 1).detach()

class NormalDist(BaseDistributor):
    def __init__(self, action_dim):
        super().__init__()
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, mu):
        std = self.log_std.exp().expand_as(mu)
        return Normal(mu, std)

    def post_process(self, action):
        return action.detach()

class BetaDist(BaseDistributor):
    def __init__(
            self,
            action_low: float | list = -1.0,
            action_high: float | list = 1.0
    ):
        super().__init__()
        self.register_buffer("low", torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer("high", torch.tensor(action_high, dtype=torch.float32))

    def forward(self, logits):
        alpha_logits, beta_logits = torch.chunk(logits, 2, dim=-1)
        alpha = torch.nn.functional.softplus(alpha_logits) + 1.0
        beta = torch.nn.functional.softplus(beta_logits) + 1.0
        return Beta(alpha, beta)

    def post_process(self, action):
        return (action * (self.high - self.low) + self.low).detach()
