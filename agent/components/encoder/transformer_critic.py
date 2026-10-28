import torch
from torch import nn


class TransformerCritic(nn.Module):
    """Value network that attends over tokens of the encoder feature vector.

    A single observation vector carries no temporal axis, so the features are
    split into ``n_tokens`` chunks before self-attention. Attention is therefore
    over feature groups, not over time.
    """

    requires_input_dim = True

    def __init__(
            self,
            input_dim: int,
            d_model: int = 64,
            nhead: int = 4,
            depth: int = 2,
            n_tokens: int = 8,
            dim_feedforward: int = 128,
            dropout: float = 0.0
    ):
        super().__init__()

        if input_dim % n_tokens != 0:
            raise ValueError(
                f"input_dim={input_dim} is not divisible by n_tokens={n_tokens}"
            )

        self.n_tokens = n_tokens
        self.token_dim = input_dim // n_tokens

        self.project = nn.Linear(self.token_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.view(x.size(0), self.n_tokens, self.token_dim)
        hidden = self.encoder(self.project(tokens))
        return self.head(hidden.mean(dim=1))
