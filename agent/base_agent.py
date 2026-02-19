# agents/base_agent.py
from abc import ABC, abstractmethod
from dataclasses import fields
import numpy as np
import torch
from torch import nn
import os
from utils.logger import logger
from .buffer import BaseBuffer
from .utils import MetricTracker, get_agent_summary


class BaseAgent(nn.Module, ABC):
    _config_class = None

    def __init__(
            self,
            encoder: nn.Module,
            policy: nn.Module,
            buffer: BaseBuffer = None,
            optimizer: torch.optim.Optimizer = None,
            **kwargs
    ):
        super().__init__()
        # Core components
        self.encoder = encoder
        self.policy = policy
        self.buffer = buffer
        self.optimizer = optimizer

        self._metric_weights = {}
        self._tracker = MetricTracker(self._metric_weights)

        if self._config_class:
            valid_fields = {f.name for f in fields(self._config_class)}
            self.cfg = self._config_class(
                **{k: v for k, v in kwargs.items() if k in valid_fields}
            )

    def summary(self):
        summary_text = get_agent_summary(self)
        logger.info(summary_text)

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
            action, info = self._acting(state, deterministic)
            info["state"] = state
            return self._to_env_action(action), info

    def _to_env_action(self, action: torch.Tensor):
        return self.policy.handler.to_env_format(action)

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def _acting(self, state, deterministic):
        pass

    def save_checkpoints(self, path: str):
        if not os.path.exists(os.path.dirname(path)):
            logger.warning(f"{os.path.dirname(path)} not found for saving checkpoint")
            return

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
