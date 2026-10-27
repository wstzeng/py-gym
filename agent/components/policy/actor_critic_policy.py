from .base_policy import StochasticPolicy


class ActorCriticPolicy(StochasticPolicy):
    def __init__(self, actor, critic, **kwargs):
        super().__init__(**kwargs)
        # Composition allows different encoders or heads per policy.
        self.actor = actor
        self.critic = critic

    @classmethod
    def sub_components(cls):
        return super().sub_components() + ["actor", "critic"]

    def forward(self, x):
        return self.get_distribution(x)

    def get_distribution(self, features):
        return self.handler(self.actor(features))

    def get_value(self, x):
        return self.critic(x)
