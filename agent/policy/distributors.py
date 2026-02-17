import torch
from torch import nn
from torch.distributions import Categorical, Normal, Beta

class BaseDistributor(nn.Module):
    """Base class to ensure consistent interface across discrete and continuous actions."""
    def forward(self, x):
        raise NotImplementedError

    def get_log_prob(self, dist, action):
        """Extract log probabilities from distribution given an action."""
        raise NotImplementedError

    def post_process(self, action):
        """Convert raw samples to environment-ready actions."""
        raise NotImplementedError

    def apply_correction(self, log_prob, raw_action):
        """Finalize log_prob calculation (summation, Jacobians, etc.)."""
        raise NotImplementedError

class CategoricalDist(BaseDistributor):
    def forward(self, logits):
        return Categorical(logits=logits)

    def get_log_prob(self, dist, action):
        # Squeeze the last dimension to prevent (N, N) broadcasting
        # Input action shape: (N, 1) -> Output log_prob: (N,)
        return dist.log_prob(action.squeeze(-1))

    def post_process(self, action):
        # Store as (N, 1) to keep buffer dimensions consistent
        return action.view(-1, 1).detach()

    def apply_correction(self, log_prob, raw_action):
        # For discrete, no summation or Jacobian needed
        # Ensure result is (N,)
        if log_prob.ndim > 1:
            return torch.diagonal(log_prob)
        return log_prob

class ContinuousDistributor(BaseDistributor):
    """Abstract class for continuous distributions that require dimension summation."""
    def get_log_prob(self, dist, action):
        # Continuous distributions usually return (N, A)
        return dist.log_prob(action)

    def apply_correction(self, log_prob, raw_action):
        # Subclasses handle specific Jacobian, then sum across action dimensions
        raise NotImplementedError

class NormalDist(ContinuousDistributor):
    def __init__(
            self,
            action_dim,
            action_low=-1.0,
            action_high=1.0,
            eps=1e-6
    ):
        super().__init__()
        self.register_buffer("low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("high", torch.as_tensor(action_high, dtype=torch.float32))
        self.log_std = nn.Parameter(torch.full((1, action_dim), -0.5))
        self.eps = eps

    def forward(self, mu):
        # Gaussian in unbounded space
        std = self.log_std.exp().expand_as(mu)
        return Normal(mu, std)

    def post_process(self, action):
        # Squash to [-1, 1] then scale to [low, high]
        squashed = torch.tanh(action)
        return ((squashed + 1) / 2 * (self.high - self.low) + self.low).detach()

    def apply_correction(self, log_prob, raw_action):
        # Squash correction: log(1 - tanh^2(x))
        # (N, A) - (N, A) -> (N,)
        correction = torch.log(1 - torch.tanh(raw_action).pow(2) + self.eps)
        return (log_prob - correction).sum(dim=-1)

class BetaDist(ContinuousDistributor):
    def __init__(
            self,
            action_low=-1.0,
            action_high=1.0,
            base=1.0
    ):
        super().__init__()
        self.register_buffer("low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("high", torch.as_tensor(action_high, dtype=torch.float32))
        self.base = base

    def forward(self, logits):
        # Expected input shape: (N, action_dim * 2)
        alpha_logits, beta_logits = torch.chunk(logits, 2, dim=-1)
        alpha = torch.nn.functional.softplus(alpha_logits) + self.base
        beta = torch.nn.functional.softplus(beta_logits) + self.base
        return Beta(alpha, beta)

    def post_process(self, action):
        # Map [0, 1] to [low, high]
        return (action * (self.high - self.low) + self.low).detach()

    def apply_correction(self, log_prob, raw_action):
        # Beta is naturally bounded, no Jacobian needed
        # (N, A) -> (N,)
        return log_prob.sum(dim=-1)
