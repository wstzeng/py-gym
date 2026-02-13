import torch
import numpy as np
from .base_buffer import BaseBuffer

class TrajectoryBuffer(BaseBuffer):
    def __init__(self):
        self.clear()

    def store(self, info: dict, reward: float, done: bool):
        self.states.append(info.get('state'))
        self.actions.append(info.get('action'))
        self.log_probs.append(self._to_scalar(info.get('log_prob')))
        self.values.append(self._to_scalar(info.get('value', 0.0)))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def _to_scalar(self, val):
        if val is None: return 0.0
        return val.detach().cpu().item() if isinstance(val, torch.Tensor) else val

    def get_data(self, device="cpu") -> dict:
        if not self.states: return {}

        states_tensor = torch.as_tensor(np.array(self.states), dtype=torch.float32, device=device).squeeze(1)

        actions_arr = np.array(self.actions)
        if actions_arr.ndim == 1:
            actions_arr = actions_arr[:, np.newaxis]
        elif actions_arr.ndim > 2:
            actions_arr = actions_arr.squeeze()
            if actions_arr.ndim == 1:
                actions_arr = actions_arr[:, np.newaxis]

        return {
            "states": states_tensor,
            "actions": torch.as_tensor(actions_arr, device=device),
            "log_probs": torch.as_tensor(self.log_probs, dtype=torch.float32, device=device).view(-1),
            "values": torch.as_tensor(self.values, dtype=torch.float32, device=device).view(-1),
            "rewards": self.rewards,
            "dones": self.dones
        }

    def clear(self):
        self.states, self.actions, self.log_probs = [], [], []
        self.values, self.rewards, self.dones = [], [], []
