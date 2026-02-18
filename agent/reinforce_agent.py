import torch
from dataclasses import dataclass
from .base_agent import BaseAgent

@dataclass
class ReinforceConfig:
    gamma: float = 0.99

class ReinforceAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cfg = ReinforceConfig(
            **{k: v for k, v in kwargs.items() if k in ReinforceConfig.__annotations__}
        )

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
        }
        return raw_action, info

    def update(self):
        data = self.buffer.get_data(self.device)
        if not data:
            return {}

        self.tracker.reset()

        rewards = data["rewards"]
        # Calculate discounted returns
        returns = []
        g = 0
        for r in reversed(rewards):
            g = r + self.cfg.gamma * g
            returns.insert(0, g)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Re-calculate log probs for the batch
        features = self.encoder(data["states"])
        dist = self.policy.get_distribution(features)

        raw_log_probs = self.policy.handler.get_log_prob(dist, data["actions"])
        curr_log_probs = self.policy.handler.apply_correction(
            raw_log_probs,
            data["actions"]
        )

        # REINFORCE loss: -1/T * sum(log_prob * Gt)
        loss = -(curr_log_probs * returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.tracker.store(loss=loss)

        self.buffer.clear()
        return self.tracker.result()
