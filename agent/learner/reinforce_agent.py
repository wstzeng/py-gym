import torch

from .policy_gradient_agent import PolicyGradientAgent


class ReinforceAgent(PolicyGradientAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _act(self, state, deterministic: bool = False):
        features = self.encoder(state)
        dist = self.policy.get_distribution(features)

        raw_action = dist.mode if deterministic else dist.sample()
        raw_log_prob = dist.log_prob(raw_action)
        corrected_log_prob = self.policy.handler.apply_correction(
            raw_log_prob,
            raw_action,
        )

        info = {
            "action": raw_action.detach(),
            "log_prob": corrected_log_prob.detach(),
        }
        return raw_action, info

    def update(self):
        data = self.buffer.get_data(self.device)
        if not data:
            return {}

        self._tracker.reset()

        rewards = data["rewards"]

        returns = torch.zeros_like(rewards)
        g = 0.0
        for t in reversed(range(len(rewards))):
            g = rewards[t] + self.gamma * g
            returns[t] = g

        returns = (
            returns - returns.mean()
        ) / (returns.std(unbiased=False) + 1e-8)

        features = self.encoder(data["states"])
        dist = self.policy.get_distribution(features)

        raw_log_probs = self.policy.handler.get_log_prob(dist, data["actions"])
        curr_log_probs = self.policy.handler.apply_correction(
            raw_log_probs,
            data["actions"],
        )

        loss = -(curr_log_probs * returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._tracker.store(loss=loss)

        self.buffer.clear()
        return self._tracker.result()
