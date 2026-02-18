from .base_policy import BasePolicy

class ActorCriticPolicy(BasePolicy):
    """
    Implementation of Actor-Critic architecture inheriting from BasePolicy.
    """
    def __init__(self, actor, critic, **kwargs):
        super().__init__(**kwargs)
        # Use composition to allow different encoders or heads
        self.actor = actor
        self.critic = critic

    def forward(self, x):
        """
        Forward pass returning both action distribution and state value.
        """
        return self.actor(x), self.critic(x)
    
    def get_distribution(self, x):
        """
        Convenience method if only action is needed.
        """
        logits = self.actor(x)
        return self.handler(logits)

    def get_value(self, x):
        """
        Convenience method if only state value is needed.
        """
        return self.critic(x)
