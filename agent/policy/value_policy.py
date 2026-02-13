from .base_policy import BasePolicy

class ValuePolicy(BasePolicy):
    def __init__(self, critic):
        super().__init__(None)
        self.critic = critic

    def forward(self, x):
        return self.net(x)

    def get_distribution(self, x):
        raise NotImplementedError("ValuePolicy does not support get_distribution.")
