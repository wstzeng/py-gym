import torch
from .base_buffer import BaseBuffer

class TrajectoryBuffer(BaseBuffer):
    def __init__(self):
        super().__init__()
        self.clear()

    def store(self, info: dict, reward: float, done: bool):
        self.states.append(info.get('state'))
        self.actions.append(info.get('action'))
        self.log_probs.append(self._to_tensor(info.get('log_prob')))
        self.values.append(self._to_tensor(info.get('value', 0.0)))
        self.rewards.append(torch.tensor(reward, dtype=torch.float32))
        self.dones.append(torch.tensor(done, dtype=torch.bool))

    def _to_tensor(self, val):
        if val is None:
            return torch.tensor(0.0)
        if isinstance(val, torch.Tensor):
            return val.detach()
        return torch.tensor(val, dtype=torch.float32)

    def get_data(self, device="cpu") -> dict:
        if not self.states:
            return {}

        return {
            "states": torch.stack(self.states).to(device).squeeze(),
            "actions": torch.stack(self.actions).to(device).squeeze(),
            "log_probs": torch.stack(self.log_probs).to(device).view(-1),
            "values": torch.stack(self.values).to(device).view(-1),
            "rewards": torch.stack(self.rewards).to(device).view(-1),
            "dones": torch.stack(self.dones).to(device).view(-1)
        }

    def clear(self):
        self.states, self.actions, self.log_probs = [], [], []
        self.values, self.rewards, self.dones = [], [], []
