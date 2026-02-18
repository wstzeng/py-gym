import torch
from torch import nn
from torch.distributions import Categorical, Normal, Beta

__all__ = [
    "CategoricalHandler",
    "NormalHandler",
    "BetaHandler",
]

class BaseActionHandler(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim

    @property
    def required_input_dim(self):
        """The output dimension required from the actor network."""
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def get_log_prob(self, dist, action):
        raise NotImplementedError

    def post_process(self, action):
        raise NotImplementedError

    def apply_correction(self, log_prob, raw_action):
        raise NotImplementedError

    def to_env_format(self, action: torch.Tensor):
        raise NotImplementedError

class CategoricalHandler(BaseActionHandler):
    @property
    def required_input_dim(self):
        return self.action_dim

    def forward(self, logits):
        return Categorical(logits=logits)

    def get_log_prob(self, dist, action):
        return dist.log_prob(action.squeeze(-1))

    def post_process(self, action):
        return action.view(-1, 1).detach()

    def apply_correction(self, log_prob, raw_action):
        if log_prob.ndim > 1:
            return torch.diagonal(log_prob)
        return log_prob

    def to_env_format(self, action):
        return action.item()

class ContinuousActionHandler(BaseActionHandler):
    def get_log_prob(self, dist, action):
        return dist.log_prob(action)

    def to_env_format(self, action):
        act_np = action.detach().cpu().numpy()
        return act_np[0] if act_np.shape[0] == 1 else act_np

class NormalHandler(ContinuousActionHandler):
    def __init__(
            self,
            action_dim,
            action_low=-1.0,
            action_high=1.0,
            eps=1e-6
    ):
        super().__init__(action_dim)
        self.register_buffer("low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("high", torch.as_tensor(action_high, dtype=torch.float32))
        self.log_std = nn.Parameter(torch.full((1, action_dim), -0.5))
        self.eps = eps

    @property
    def required_input_dim(self):
        return self.action_dim

    def forward(self, mu):
        std = self.log_std.exp().expand_as(mu)
        return Normal(mu, std)

    def post_process(self, action):
        squashed = torch.tanh(action)
        return ((squashed + 1) / 2 * (self.high - self.low) + self.low).detach()

    def apply_correction(self, log_prob, raw_action):
        correction = torch.log(1 - torch.tanh(raw_action).pow(2) + self.eps)
        return (log_prob - correction).sum(dim=-1)

class BetaHandler(ContinuousActionHandler):
    def __init__(
            self, 
            action_dim, 
            action_low=-1.0, 
            action_high=1.0, 
            base=1.0
    ):
        super().__init__(action_dim)
        self.register_buffer("low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("high", torch.as_tensor(action_high, dtype=torch.float32))
        self.base = base

    @property
    def required_input_dim(self):
        return self.action_dim * 2

    def forward(self, logits):
        alpha_logits, beta_logits = torch.chunk(logits, 2, dim=-1)
        alpha = torch.nn.functional.softplus(alpha_logits) + self.base
        beta = torch.nn.functional.softplus(beta_logits) + self.base
        return Beta(alpha, beta)

    def post_process(self, action):
        return (action * (self.high - self.low) + self.low).detach()

    def apply_correction(self, log_prob, raw_action):
        return log_prob.sum(dim=-1)
