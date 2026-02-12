import torch
import numpy as np
from .base_buffer import BaseBuffer

class PPOBuffer(BaseBuffer):
    def __init__(self):
        self.clear()

    def store(self, info, reward, done):
        """
        Stores data. Handles both discrete (scalars) and continuous (tensors).
        """
        # Ensure states are numpy arrays for easier batch conversion
        self.states.append(info['state'])

        # Store actions as is; conversion happens in get_data
        self.actions.append(info['action'])

        # Metadata conversion to flat float for consistency
        self.log_probs.append(self._to_scalar(info['log_prob']))
        self.values.append(self._to_scalar(info['value']))
        
        self.rewards.append(reward)
        self.dones.append(done)

    def _to_scalar(self, val):
        if isinstance(val, torch.Tensor):
            return val.detach().cpu().item()
        return val

    def get_data(self, device="cpu"):
        states_arr = np.array(self.states)
        states_tensor = torch.as_tensor(states_arr, dtype=torch.float32, device=device).squeeze(1)

        actions_arr = np.array(self.actions)
        if actions_arr.ndim == 1:
            actions_arr = actions_arr[:, np.newaxis]

        if actions_arr.ndim > 2:
            actions_arr = actions_arr.squeeze()
            if actions_arr.ndim == 1:
                actions_arr = actions_arr[:, np.newaxis]

        if np.issubdtype(actions_arr.dtype, np.integer):
            actions_tensor = torch.as_tensor(actions_arr, dtype=torch.long, device=device)
        else:
            actions_tensor = torch.as_tensor(actions_arr, dtype=torch.float32, device=device)

        log_probs_tensor = torch.as_tensor(self.log_probs, dtype=torch.float32, device=device).view(-1)
        values_tensor = torch.as_tensor(self.values, dtype=torch.float32, device=device).view(-1)

        return (
            states_tensor, 
            actions_tensor, 
            log_probs_tensor, 
            values_tensor, 
            self.rewards, 
            self.dones
        )

    def clear(self):
        self.states, self.actions, self.log_probs = [], [], []
        self.values, self.rewards, self.dones = [], [], []
