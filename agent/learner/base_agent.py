import os
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn

from agent.utils import MetricTracker, get_agent_summary
from utils.logger import logger


class BaseAgent(nn.Module, ABC):
    def __init__(
            self,
            encoder: nn.Module,
            policy: nn.Module,
            **kwargs
    ):
        super().__init__()
        self.encoder = encoder
        self.policy = policy

        self._metric_weights = {}
        self._tracker = MetricTracker(self._metric_weights)

    @classmethod
    def sub_components(cls):
        return ["encoder", "policy"]

    def summary(self):
        return get_agent_summary(self)

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def record(self, info, reward, done):
        if getattr(self, "buffer", None) is None:
            raise RuntimeError(f"{self.__class__.__name__} has no buffer.")
        self.buffer.store(info, reward, done)

    def select_action(self, state, deterministic: bool = False):
        """Preprocess state, select an action, and convert it for the env."""
        if not isinstance(state, torch.Tensor):
            state_np = np.array(state)
            if state_np.ndim == 1:
                state_np = state_np[np.newaxis, :]
            else:
                raise NotImplementedError("state_shape >= 2 not implemented")
            state = torch.from_numpy(state_np).float().to(self.device)

        is_inference = not self.training or deterministic
        with torch.set_grad_enabled(not is_inference):
            action, info = self._act(state, deterministic)
            info["state"] = state
            return self.policy.handler.to_env_format(action), info

    @abstractmethod
    def _act(self, state, deterministic: bool = False):
        pass

    @abstractmethod
    def update(self):
        pass

    def save_checkpoints(self, path: str):
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            logger.warning(f"{parent} not found for saving checkpoint")
            return

        checkpoint = {
            name: module.state_dict()
            for name, module in self.named_children()
        }
        if getattr(self, "optimizer", None) is not None:
            checkpoint["optimizer"] = self.optimizer.state_dict()

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

        if (
                "optimizer" in checkpoint
                and getattr(self, "optimizer", None) is not None
        ):
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.eval()
