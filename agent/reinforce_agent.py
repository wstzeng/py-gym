import torch
from .base_agent import BaseAgent
from utils.torch_utils import state_to_tensor

class ReinforceAgent(BaseAgent):
    def __init__(
            self,
            encoder,
            policy,
            buffer,
            optimizer,
            device="auto",
            gamma=0.99,
            **kwargs
    ):
        super().__init__(encoder, device=device)
        self.policy = policy.to(self.device)
        self.buffer = buffer
        self.gamma = gamma
        self.optimizer = optimizer

    @state_to_tensor
    def _select_action_impl(self, state, deterministic):
        features = self.encoder(state)
        dist = self.policy.get_distribution(features)
        raw_action = dist.mode if deterministic else dist.sample()

        # Consistent with your PPO style: Use torch to handle dimensions
        action_for_buffer = torch.atleast_1d(raw_action.detach().cpu())

        info = {
            "state": state,
            "action": action_for_buffer,
            "log_prob": dist.log_prob(raw_action).sum().detach().item(),
        }

        return self.policy.distributor.post_process(raw_action), info

    def record(self, info, reward, done):
        self.buffer.store(info, reward, done)

    def update(self):
        data = self.buffer.get_data(self.device)
        if not data: return 0.0

        states = data["states"]
        actions = data["actions"]
        rewards = data["rewards"]
        returns = []
        g = 0
        for r in reversed(rewards):
            g = r + self.gamma * g
            returns.insert(0, g)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        features = self.encoder(states)
        dist = self.policy.get_distribution(features)

        if isinstance(dist, torch.distributions.Categorical):
            curr_log_probs = dist.log_prob(actions.view(-1))
        else:
            curr_log_probs = dist.log_prob(actions).sum(dim=-1)

        loss = -(curr_log_probs * returns).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.buffer.clear()
        return loss.item()
