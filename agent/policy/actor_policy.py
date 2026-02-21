from .base_policy import BasePolicy

class ActorPolicy(BasePolicy):
    def __init__(self, actor, **kwargs):
        super().__init__(**kwargs)
        self.actor = actor

    @classmethod
    def sub_components(cls):
        return super().sub_components() + ["actor"]

    def forward(self, x):
        return self.actor(x)

    def get_distribution(self, x):
        params = self.actor(x)
        return self.handler(params)
