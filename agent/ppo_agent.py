import torch
from dataclasses import dataclass
from .actor_critic_agent import ActorCriticAgent, ACConfig

@dataclass
class PPOConfig(ACConfig):
    eps_clip: float = 0.2
    k_epochs: int = 10
    gae_lambda: float = 0.95

class PPOAgent(ActorCriticAgent):
    config_class = PPOConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self):
        data = self.buffer.get_data(self.device)
        if not data or data["states"].size(0) == 0:
            return {}

        old_states = data["states"]
        old_actions = data["actions"]
        old_log_probs = data["log_probs"]
        old_values = data["values"]
        rewards = data["rewards"]
        dones = data["dones"]

        self.tracker.reset()

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

        for _ in range(self.cfg.k_epochs):
            features = self.encoder(old_states)
            dist = self.policy.get_distribution(features)
            curr_values = self.policy.get_value(features).view(-1)

            raw_log_probs = self.policy.handler.get_log_prob(dist, old_actions)

            # Handler handles summation
            curr_log_probs = self.policy.handler.apply_correction(
                raw_log_probs,
                old_actions
            )

            # PPO Clipped Objective
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

            self.tracker.store(
                loss=loss,
                actor=actor_loss,
                critic=critic_loss,
                entropy=entropy
            )

        self.buffer.clear()
        return self.tracker.result()
