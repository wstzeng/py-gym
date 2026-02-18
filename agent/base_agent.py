# agents/base_agent.py
from abc import ABC, abstractmethod
import torch
from torch import nn
from dataclasses import fields, is_dataclass, asdict
import numpy as np
import tomli_w
import os
from .buffer import BaseBuffer
from utils.logger import logger

class MetricTracker:
    def __init__(self, weights: dict):
        self.metric_weights = weights
        self.reset()

    def reset(self):
        self._data = {}

    def store(self, **metrics):
        for k, v in metrics.items():
            if k not in self._data:
                self._data[k] = []
            val = v.detach().cpu().item() if torch.is_tensor(v) else v

            if k in self.metric_weights:
                val *= self.metric_weights[k]

            self._data[k].append(val)

    def result(self) -> dict:
        return {
            k: sum(v) / len(v) for k, v in self._data.items() if len(v) > 0
        }

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
        lines = [f"[bold cyan][{self.__class__.__name__} Summary][/bold cyan]"]
        
        # 2. Metadata (Basic Info)
        lines.append(f"Device: {self.device}")
        lines.append(f"Optimizer: {self.optimizer.__class__.__name__ if self.optimizer else 'None'}")

        # 3. Agent Structure (Full PyTorch Repr)
        lines.append("\n[bold yellow][Model Architecture][/bold yellow]")
        lines.append(str(self))

        # 4. Config (Parameters)
        if hasattr(self, 'cfg') and self.cfg is not None:
            lines.append("\n[bold yellow][Configuration][/bold yellow]")
            cfg_dict = asdict(self.cfg) if is_dataclass(self.cfg) else getattr(self.cfg, '__dict__', {})
            for k, v in cfg_dict.items():
                lines.append(f"  {k}: {v}")

        # 5. Join and Log
        logger.info("\n".join(lines))

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
