import torch
from dataclasses import dataclass
from .base_agent import BaseAgent

@dataclass
class ACConfig:
    gamma: float = 0.99
    critic_weight: float = 0.5
    entropy_weight: float = 0.01

class ActorCriticAgent(BaseAgent):
    _config_class = ACConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.critic_loss_fn = kwargs['critic_loss']

        self._metric_weights['critic'] = self.cfg.critic_weight
        self._metric_weights['entropy'] = -self.cfg.entropy_weight

    def _acting(self, state, deterministic):
        features = self.encoder(state)
        dist = self.policy.get_distribution(features)

        raw_action = dist.mode if deterministic else dist.sample()
        raw_log_prob = dist.log_prob(raw_action)

        corrected_log_prob = self.policy.handler.apply_correction(
            raw_log_prob,
            raw_action
        )

        info = {
            "action": raw_action.detach().cpu().squeeze(0),
            "log_prob": corrected_log_prob.detach().item(),
            "value": self.policy.get_value(features).detach().item()
        }
        return raw_action, info

    def update(self):
        data = self.buffer.get_data(self.device)
        if not data: return {}

        self._tracker.reset()
        rewards, dones, old_values = data["rewards"], data["dones"], data["values"]

        # Bootstrapped returns
        returns = []
        g = old_values[-1].item() if not dones[-1] else 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            g = r + self.cfg.gamma * g * (1 - d)
            returns.insert(0, g)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Forward pass
        features = self.encoder(data["states"])
        dist = self.policy.get_distribution(features)
        curr_values = self.policy.get_value(features).view(-1)

        raw_log_probs = self.policy.handler.get_log_prob(dist, data["actions"])
        curr_log_probs = self.policy.handler.apply_correction(
            raw_log_probs,
            data["actions"]
        )

        advantages = (returns - curr_values.detach())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = -(curr_log_probs * advantages).mean()
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
