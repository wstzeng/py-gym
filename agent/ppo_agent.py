import torch
import torch.nn as nn
import numpy as np
from .base_agent import BaseAgent

class PPOAgent(BaseAgent):
    def __init__(
        self, encoder, policy, buffer, optimizer,
        device="auto", eps_clip=0.2, gamma=0.99, k_epochs=10,
        critic_weight=0.5, entropy_weight=0.01, gae_lambda=0.95,
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

    def _select_action_impl(self, state, deterministic):
        # Policy inference and metadata collection
        state_tensor = torch.FloatTensor(state).to(self.device)
        features = self.encoder(state_tensor)
        
        # Consistent with ActorCritic: needs state for PPO re-evaluation
        logits, value = self.policy(features)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        
        info = {
            "state": state,
            "action": action.item(),
            "log_prob": dist.log_prob(action).detach(),
            "value": value.detach()
        }
        return action.item(), info

    def record(self, info, reward, done):
        # Matches PPOBuffer signature
        self.buffer.store(info, reward, done)

    def update(self):
        states, actions, old_log_probs, values, rewards, dones = self.buffer.get_data()
        if not states: return 0.0

        # Pre-process trajectory data
        old_states = torch.FloatTensor(np.array(states)).to(self.device)
        old_actions = torch.LongTensor(np.array(actions)).to(self.device)
        old_log_probs = torch.stack(old_log_probs).to(self.device).detach()
        old_values = torch.stack(values).squeeze().to(self.device).detach()

        # Compute GAE (Generalized Advantage Estimation)
        advantages = []
        last_gae = 0
        
        # We need the next value to compute TD-error: V(s_{t+1})
        # For the last step, if not done, we'd need a bootstrap, 
        # but here we assume the rollout ends at an episode boundary or T.
        next_value = 0 
        
        for r, d, v in zip(reversed(rewards), reversed(dones), reversed(old_values)):
            # TD-error delta = r + gamma * V(s_next) - V(s_now)
            delta = r + self.gamma * next_value * (1 - d) - v
            # GAE recursive formula
            gae = delta + self.gamma * self.gae_lambda * (1 - d) * last_gae
            advantages.insert(0, gae)
            
            last_gae = gae
            next_value = v

        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        # Returns for Critic are Advantage + Value_estimate
        returns = advantages + old_values
        
        # Standardize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Optimize policy for K epochs (Logic remains same)
        total_loss = 0
        for _ in range(self.k_epochs):
            features = self.encoder(old_states)
            logits, curr_values = self.policy(features)
            dist = torch.distributions.Categorical(logits=logits)
            
            curr_log_probs = dist.log_prob(old_actions)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(curr_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = torch.nn.functional.smooth_l1_loss(curr_values.squeeze(), returns)
            
            loss = actor_loss + self.critic_weight * critic_loss - self.entropy_weight * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        self.buffer.clear()
        return total_loss / self.k_epochs
