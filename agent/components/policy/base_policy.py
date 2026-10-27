from abc import ABC, abstractmethod

from torch import nn


class BasePolicy(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @classmethod
    def sub_components(cls):
        return []

    @abstractmethod
    def forward(self, x):
        pass


class StochasticPolicy(BasePolicy, ABC):
    def __init__(self, handler, **kwargs):
        super().__init__()
        self.handler = handler

    @classmethod
    def sub_components(cls):
        return super().sub_components() + ["handler"]

    @abstractmethod
    def get_distribution(self, features):
        pass
