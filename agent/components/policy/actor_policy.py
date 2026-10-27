from .base_policy import StochasticPolicy


class ActorPolicy(StochasticPolicy):
    def __init__(self, actor, **kwargs):
        super().__init__(**kwargs)
        self.actor = actor

    @classmethod
    def sub_components(cls):
        return super().sub_components() + ["actor"]

    def forward(self, x):
        return self.get_distribution(x)

    def get_distribution(self, features):
        return self.handler(self.actor(features))
