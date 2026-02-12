from abc import ABC, abstractmethod
import torch.nn as nn

class BasePolicy(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass
