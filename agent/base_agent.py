# agents/base_agent.py
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import os

from utils.logger import logger

class BaseAgent(nn.Module, ABC):
    def __init__(self, encoder: nn.Module, device: str = "auto"):
        super().__init__()
        self.encoder = encoder
        # Device management
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.encoder.to(self.device)
        self.component_map = {
            'policy': 'policy',
            'critic': 'critic',
            'optimizer': 'optimizer'
        }

    def select_action(
            self,
            state: torch.FloatTensor,
            deterministic: bool = False
    ):
        """
        Inference entry point. 
        Args:
            state: current environment observation.
            deterministic: if True, use max-likelihood action (testing). 
                        if False, sample from distribution (training).
        """
        # We ensure the correct gradient mode here to prevent memory leaks during testing
        with torch.set_grad_enabled(self.training and not deterministic):
            return self._select_action_impl(state, deterministic)

    @abstractmethod
    def _select_action_impl(self, state, deterministic):
        """Hidden implementation to be overridden by child classes."""
        pass

    @abstractmethod
    def record(self, info: dict, reward: float, done: bool, **kwargs):
        pass

    @abstractmethod
    def update(self):
        pass
    
    def save_checkpoints(self, path: str):
        """
        Generic save method that captures the state of all core components.
        """
        import os
        folder = os.path.dirname(path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)

        checkpoint = {
            'encoder': self.encoder.state_dict(),
        }
        
        if hasattr(self, 'policy'):
            checkpoint['policy'] = self.policy.state_dict()
        if hasattr(self, 'critic'):
            checkpoint['critic'] = self.critic.state_dict()
        if hasattr(self, 'optimizer'):
            checkpoint['optimizer'] = self.optimizer.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"[bold green]Checkpoint saved[/bold green] to [cyan]{path}[/cyan]")

    def load_checkpoints(self, path: str):
        """
        Generic load method that restores state for all present components.
        """
        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found at {path}")
            return

        checkpoint = torch.load(path, map_location=self.device)
        
        if 'encoder' in checkpoint:
            self.encoder.load_state_dict(checkpoint['encoder'])
        
        for key, attr in self.component_map.items():
            if hasattr(self, attr) and key in checkpoint:
                getattr(self, attr).load_state_dict(checkpoint[key])
        
        self.eval()
        logger.info(f"[bold blue]Checkpoint loaded[/bold blue] from [cyan]{path}[/cyan]")
