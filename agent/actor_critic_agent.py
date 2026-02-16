import torch
from .base_agent import BaseAgent
from utils.torch_utils import state_to_tensor

class ActorCriticAgent(BaseAgent):
    def __init__(
            self,
            encoder,
            policy,
            buffer,
            optimizer,
            device="auto",
            gamma=0.99,
            critic_weight=0.5,
            entropy_weight=0.01,
            critic_loss=None,
            **kwargs
    ):
        super().__init__(encoder, device=device)
        self.policy = policy.to(self.device)
        self.buffer = buffer
        self.optimizer = optimizer
        self.gamma = gamma
        self.critic_weight = critic_weight
        self.entropy_weight = entropy_weight
        self.critic_loss_fn = critic_loss if critic_loss else torch.nn.SmoothL1Loss()

    @state_to_tensor
    def _select_action_impl(self, state, deterministic):
        features = self.encoder(state)
        dist = self.policy.get_distribution(features)
        raw_action = dist.mode if deterministic else dist.sample()

        action_for_buffer = torch.atleast_1d(raw_action.detach().cpu())

        info = {
            "state": state,
            "action": action_for_buffer,
            "log_prob": dist.log_prob(raw_action).sum().detach().item(),
            "value": self.policy.get_value(features).detach().item()
        }

        return self.policy.distributor.post_process(raw_action), info

    def record(self, info, reward, done):
        self.buffer.store(info, reward, done)

    def update(self):
        data = self.buffer.get_data(self.device)
        if not data: return 0.0

        old_states = data["states"]
        old_actions = data["actions"]
        rewards = data["rewards"]
        dones = data["dones"]

        returns = []
        g = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            g = r + self.gamma * g * (1 - d)
            returns.insert(0, g)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        features = self.encoder(old_states)
        dist = self.policy.get_distribution(features)
        curr_values = self.policy.get_value(features).view(-1)

        if isinstance(dist, torch.distributions.Categorical):
            curr_log_probs = dist.log_prob(old_actions.view(-1))
        else:
            curr_log_probs = dist.log_prob(old_actions).sum(dim=-1)

        advantages = returns - curr_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = -(curr_log_probs.view(-1) * advantages).mean()
        critic_loss = self.critic_loss_fn(curr_values, returns)
        entropy = dist.entropy().mean()

        loss = actor_loss + self.critic_weight * critic_loss - self.entropy_weight * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.buffer.clear()
        return loss.item()
