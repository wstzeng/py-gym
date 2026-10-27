from torch import nn
from abc import ABC, abstractmethod

class BaseBuffer(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def store(self, **kwargs):
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def get_data(self):
        pass
