# agents/base_agent.py
from abc import ABC, abstractmethod
import torch
from torch import nn
import numpy as np
import os
from .buffer import BaseBuffer
from utils.logger import logger

class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self._data = {}

    def store(self, **metrics):
        """Standardize metric storage: detach and move to CPU."""
        for k, v in metrics.items():
            if k not in self._data: self._data[k] = []
            val = v.item() if torch.is_tensor(v) else v
            self._data[k].append(val)

    def result(self) -> dict:
        """Calculate mean for all tracked metrics."""
        return {k: sum(v) / len(v) for k, v in self._data.items() if v}

class BaseAgent(nn.Module, ABC):
    def __init__(
            self,
            encoder: nn.Module,
            policy: nn.Module,
            buffer: BaseBuffer = None,
            optimizer: torch.optim.Optimizer = None,
            continuous: bool = False,
            **kwargs
    ):
        super().__init__()
        # Core components
        self.encoder = encoder
        self.policy = policy
        self.buffer = buffer
        self.optimizer = optimizer
        self.continuous = continuous
        self.tracker = MetricTracker()

    def __repr__(self):
        lines = [f"[{self.__class__.__name__}]"]
        lines.append(f"  - Continuous: {self.continuous}")
        lines.append(f"  - Device: {self.device}")

        lines.append("  - Components:")
        for name, module in self.named_children():
            mod_str = str(module).replace('\n', '\n\t')
            lines.append(f"\t{name}: {mod_str}")

        if self.buffer is not None:
            lines.append(f"  - Buffer: {self.buffer.__class__.__name__}")
        if self.optimizer is not None:
            lines.append(f"  - Optimizer: {self.optimizer.__class__.__name__}")

        if hasattr(self, 'cfg'):
            lines.append(f"  - Config: {self.cfg}")

        return "\n".join(lines)

    def summary(self):
        logger.info(f"[bold cyan]Agent Summary[/bold cyan]\n{self.__repr__()}")

    @property
    def device(self):
        """Dynamic detection of the model's device."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def record(self, info, reward, done):
        self.buffer.store(info, reward, done)

    def select_action(self, state, deterministic: bool = False):
        """Unified state preprocessing and action selection flow."""
        if not isinstance(state, torch.Tensor):
            state_np = np.array(state)
            # Standardize dimensions for Vector (N, D) or Visual (N, C, H, W)
            if state_np.ndim == 1:
                state_np = state_np[np.newaxis, :]
            elif state_np.ndim == 3:
                state_np = np.transpose(state_np, (2, 0, 1))[np.newaxis, :]
            state = torch.from_numpy(state_np).float().to(self.device)

        is_inference = not self.training or deterministic
        with torch.set_grad_enabled(not is_inference):
            action, info = self._select_action_impl(state, deterministic)
            info["state"] = state
            return self._to_env_action(action), info

    def _to_env_action(self, action: torch.Tensor):
        """Standardizes torch (N, A) output to environment-ready format."""
        action = action.detach().cpu()
        if self.continuous:
            return action.squeeze(0).numpy()
        else:
            try:
                return action.item()
            except RuntimeError:
                return action.squeeze(0).numpy()

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def _select_action_impl(self, state, deterministic):
        pass

    def save_checkpoints(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            name: module.state_dict()
            for name, module in self.named_children()
        }
        if self.optimizer is not None:
            checkpoint['optimizer'] = self.optimizer.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoints(self, path: str):
        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found at {path}")
            return

        checkpoint = torch.load(path, map_location=self.device)
        for name, module in self.named_children():
            if name in checkpoint:
                module.load_state_dict(checkpoint[name])

        if 'optimizer' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.eval()
