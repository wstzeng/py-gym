import torch
from .base_agent import BaseAgent
from utils.torch_utils import state_to_tensor

class ReinforceAgent(BaseAgent):
    def __init__(
            self, encoder, policy, buffer, optimizer,
            device="auto", gamma=0.99,
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
        
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        
        info = {
            "log_prob": dist.log_prob(action)
        }
        
        return action.item(), info
    
    def record(self, info, reward, done):
        self.buffer.store(
            log_prob=info["log_prob"], 
            reward=reward
        )
    
    def update(self):
        log_probs, rewards = self.buffer.get_data()
        if not log_probs:
            return 0.0
            
        returns = []
        g = 0
        for r in reversed(rewards):
            g = r + self.gamma * g
            returns.insert(0, g)
        
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs_t = torch.stack(log_probs) 
        loss = -(log_probs_t * returns).sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.buffer.clear()
        return loss.item()
