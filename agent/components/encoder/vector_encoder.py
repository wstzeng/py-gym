import torch
from torch import nn
from .base_encoder import BaseEncoder

class VectorEncoder(BaseEncoder):
    def __init__(
            self,
            network: nn.Module,
            input_dim: int = None,
            feature_dim: int = None
    ):
        super().__init__()
        self.net = network

        if feature_dim is not None:
            self._feature_dim = feature_dim
        elif input_dim is not None:
            self._feature_dim = self._detect_feature_dim(input_dim)
        else:
            raise ValueError("Must provide either feature_dim or input_dim for auto-detection.")

    def _detect_feature_dim(self, input_dim: int) -> int:
        # Create a dummy input to trace the network's output shape
        self.net.eval()
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_dim)
            output = self.net(dummy_input)
        return output.shape[-1]

    def forward(self, x):
        return self.net(x)

    @property
    def feature_dim(self):
        return self._feature_dim
