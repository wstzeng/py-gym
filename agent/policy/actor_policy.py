from .base_policy import BasePolicy

class ActorPolicy(BasePolicy):
    def __init__(self, actor_net, distributor):
        super().__init__(distributor)
        self.actor = actor_net

    def forward(self, x):
        return self.actor(x)

    def get_distribution(self, x):
        params = self.actor(x)
        return self.distributor(params)
