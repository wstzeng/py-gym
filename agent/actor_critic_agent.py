import torch
from .base_agent import BaseAgent

class ActorCriticAgent(BaseAgent):
    def __init__(
            self, encoder, policy, buffer, optimizer, 
            gamma=0.99, device="auto"):
        super().__init__(encoder, device=device)
        self.policy = policy.to(self.device)
        self.buffer = buffer
        self.gamma = gamma
        self.optimizer = optimizer

    def _select_action_impl(self, state, deterministic):
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        features = self.encoder(state_tensor)
        
        logits, value = self.policy(features)
        
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        info = {
            "log_prob": dist.log_prob(action),
            "value": value
        }
        
        return action.item(), info

    def record(self, info, reward, done):
        self.buffer.store(
            log_prob=info["log_prob"],
            value=info["value"],
            reward=reward,
            done=done
        )

    def update(self):
        log_probs, values, rewards, dones = self.buffer.get_data()
        if not log_probs:
            return 0.0

        returns = []
        g = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            g = r + self.gamma * g * (1 - d)
            returns.insert(0, g)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values).squeeze()  # From [T, 1] to [T]
        
        advantages = (returns - values.detach())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = torch.nn.functional.smooth_l1_loss(values, returns)
        
        loss = actor_loss + 0.1 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.buffer.clear()
        return loss.item()
