from abc import ABC

from torch import optim

from .base_agent import BaseAgent
from agent.components.buffer import BaseBuffer
from utils.logger import logger

class PolicyGradientAgent(BaseAgent, ABC):
    _default_hyper_params = {
        "gamma": 0.99,
        "entropy_weight": 0.0
    }

    def __init__(
            self,
            optimizer: optim.Optimizer,
            buffer: BaseBuffer | None = None,
            hyper_params: dict[str, any] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.optimizer = optimizer
        self.buffer = buffer
        self.hyper_params = hyper_params or {}

        self.set_hyper_params()

    def set_hyper_params(self):
        for name, default_val in self._default_hyper_params.items():
            if name not in self.hyper_params:
                self.hyper_params[name] = default_val
                logger.warning(
                    f"[{self.__class__.__name__}] hyper_param '{name}' not in config. "
                    f"Using default: {default_val}"
                )
            setattr(self, name, self.hyper_params[name])

    @classmethod
    def sub_components(cls):
        return super().sub_components() + ["buffer"]
