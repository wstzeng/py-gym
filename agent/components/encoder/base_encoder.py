from abc import ABC, abstractmethod
import torch.nn as nn

class BaseEncoder(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        pass
