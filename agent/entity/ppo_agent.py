import torch
from dataclasses import dataclass
from .actor_critic_agent import ActorCriticAgent, ACConfig

@dataclass
class PPOConfig(ACConfig):
    eps_clip: float = 0.2
    k_epochs: int = 10
    gae_lambda: float = 0.95

class PPOAgent(ActorCriticAgent):
    _config_class = PPOConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self):
        data = self.buffer.get_data(self.device)
        if not data or data["states"].size(0) == 0:
            return {}

        old_states = data["states"]
        old_actions = data["actions"]
        old_log_probs = data["log_probs"]
        old_values = data["values"].view(-1)
        rewards = data["rewards"]
        dones = data["dones"]

        self._tracker.reset()

        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        next_value = 0.0

        for t in reversed(range(len(rewards))):
            if t < len(rewards) - 1:
                next_value = old_values[t + 1]
            else:
                next_value = 0.0

            mask = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * next_value * mask - old_values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * last_gae
            advantages[t] = gae
            last_gae = gae

        # Generalized Advantage Estimation (GAE)
        returns = advantages + old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.cfg.k_epochs):
            features = self.encoder(old_states)
            dist = self.policy.get_distribution(features)
            curr_values = self.policy.get_value(features).view(-1)

            raw_log_probs = self.policy.handler.get_log_prob(dist, old_actions)
            curr_log_probs = self.policy.handler.apply_correction(
                raw_log_probs,
                old_actions
            )

            ratio = torch.exp(curr_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.cfg.eps_clip, 1 + self.cfg.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.critic_loss_fn(curr_values, returns)

            entropy = dist.entropy()
            if len(entropy.shape) > 1:
                entropy = entropy.sum(dim=-1)
            entropy = entropy.mean()

            loss = actor_loss + self.cfg.critic_weight * critic_loss - self.cfg.entropy_weight * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self._tracker.store(
                loss=loss,
                actor=actor_loss,
                critic=critic_loss,
                entropy=entropy
            )

        self.buffer.clear()
        return self._tracker.result()
