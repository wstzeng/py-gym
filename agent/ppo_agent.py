import torch
from dataclasses import dataclass
from .actor_critic_agent import ActorCriticAgent, ACConfig

@dataclass
class PPOConfig(ACConfig):
    eps_clip: float = 0.2
    k_epochs: int = 10
    gae_lambda: float = 0.95

class PPOAgent(ActorCriticAgent):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.cfg = PPOConfig(
            **{k: v for k, v in kwargs.items() if k in PPOConfig.__annotations__}
        )

    def update(self):
        """PPO Update logic using self.cfg parameters."""
        data = self.buffer.get_data(self.device)
        if not data or data["states"].size(0) == 0:
            return 0.0

        old_states = data["states"]
        old_actions = data["actions"]
        old_log_probs = data["log_probs"]
        old_values = data["values"]
        rewards = data["rewards"]
        dones = data["dones"]

        # Generalized Advantage Estimation (GAE)
        advantages = []
        last_gae, next_value = 0, 0
        for r, d, v in zip(reversed(rewards), reversed(dones), reversed(old_values)):
            delta = r + self.cfg.gamma * next_value * (1 - d) - v.item()
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * (1 - d) * last_gae
            advantages.insert(0, gae)
            last_gae, next_value = gae, v.item()

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        for _ in range(self.cfg.k_epochs):
            # The following components are inherited from ActorCriticAgent
            features = self.encoder(old_states)
            dist = self.policy.get_distribution(features)
            curr_values = self.policy.get_value(features).view(-1)

            # Support both Discrete and Continuous log_prob calculation
            target_actions = old_actions.squeeze(-1) if not self.continuous else old_actions
            curr_log_probs = dist.log_prob(target_actions)
            if self.continuous or len(curr_log_probs.shape) > 1:
                curr_log_probs = curr_log_probs.sum(dim=-1)

            # PPO Clipped Objective
            ratio = torch.exp(curr_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.cfg.eps_clip, 1 + self.cfg.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.critic_loss_fn(curr_values, returns)
            entropy = dist.entropy().mean()

            loss = actor_loss + self.cfg.critic_weight * critic_loss - self.cfg.entropy_weight * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        self.buffer.clear()
        return total_loss / self.cfg.k_epochs
