from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BasePolicy(nn.Module, ABC):
    def __init__(self, handler = None):
        super().__init__()
        self.handler = handler

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_distribution(self, x) -> torch.distributions.Distribution:
        """Returns a torch.distributions object."""
        pass
