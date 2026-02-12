import torch
import numpy as np
from .base_agent import BaseAgent
from utils.torch_utils import state_to_tensor

class PPOAgent(BaseAgent):
    def __init__(
            self, encoder, policy, buffer, optimizer, device="auto",
            eps_clip=0.2, gamma=0.99, k_epochs=10, critic_weight=0.5,
            entropy_weight=0.01, gae_lambda=0.95,
            **kwargs
    ):
        super().__init__(encoder, device=device)
        self.policy = policy.to(self.device)
        self.buffer = buffer
        self.optimizer = optimizer
        self.eps_clip = eps_clip
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.k_epochs = k_epochs
        self.critic_weight = critic_weight
        self.entropy_weight = entropy_weight

    @state_to_tensor
    def _select_action_impl(self, state, deterministic):
        features = self.encoder(state)
        dist = self.policy.get_distribution(features)
        raw_action = dist.mode if deterministic else dist.sample()
        
        action_for_buffer = np.atleast_1d(raw_action.detach().cpu().numpy())

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
        old_states, old_actions, old_log_probs, old_values, rewards, dones = data
        if old_states.size(0) == 0: return 0.0

        # GAE
        advantages = []
        last_gae, next_value = 0, 0 
        for r, d, v in zip(reversed(rewards), reversed(dones), reversed(old_values)):
            delta = r + self.gamma * next_value * (1 - d) - v.item()
            gae = delta + self.gamma * self.gae_lambda * (1 - d) * last_gae
            advantages.insert(0, gae)
            last_gae, next_value = gae, v.item()

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        for _ in range(self.k_epochs):
            features = self.encoder(old_states)
            dist = self.policy.get_distribution(features)
            curr_values = self.policy.get_value(features).view(-1)
            
            if isinstance(dist, torch.distributions.Categorical):
                curr_log_probs = dist.log_prob(old_actions.view(-1))
            else:
                curr_log_probs = dist.log_prob(old_actions).sum(dim=-1)
            
            curr_log_probs = curr_log_probs.view(-1)
            entropy = dist.entropy().mean()

            ratio = torch.exp(curr_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = torch.nn.functional.smooth_l1_loss(curr_values, returns)
            
            loss = actor_loss + self.critic_weight * critic_loss - self.entropy_weight * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        self.buffer.clear()
        return total_loss / self.k_epochs
