import functools
import numpy as np
import torch

def state_to_tensor(func):
    """
    Decorator that automatically converts input state to a torch.Tensor.
    - Handles Batch Dimension: (D,) -> (1, D)
    - Handles Visual Shape (if needed): (H, W, C) -> (1, C, H, W)
    - Moves to the correct device based on 'self.device'
    """
    @functools.wraps(func)
    def wrapper(self, state, *args, **kwargs):
        if not isinstance(state, torch.Tensor):
            state_np = np.array(state)
            
            # Ensure 2D for Vector or 4D for Visual
            if state_np.ndim == 1:
                state_np = state_np[np.newaxis, :]
            elif state_np.ndim == 3:
                state_np = np.transpose(state_np, (2, 0, 1))[np.newaxis, :]
                
            state = torch.from_numpy(state_np).float().to(self.device)
            
        return func(self, state, *args, **kwargs)
    return wrapper
