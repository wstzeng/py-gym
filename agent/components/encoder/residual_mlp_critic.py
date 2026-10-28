import torch
from torch import nn


class _ResidualFFBlock(nn.Module):
    """Dense mixing with a post-norm residual, no attention."""

    def __init__(
            self,
            d_model: int,
            hidden: int,
            dropout: float = 0.0
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden + self.ff(hidden))


class ResidualMLPCritic(nn.Module):
    """Value network with the transformer's residual stream but no attention.

    Control arm for :class:`TransformerCritic`: same residual width, same
    post-norm placement, same GELU feed-forward, same depth, but every token
    sees every other one through a dense projection instead of attention.

    If this arm matches the transformer, the gain came from normalisation and
    residual connections rather than from self-attention.
    """

    requires_input_dim = True

    def __init__(
            self,
            input_dim: int,
            d_model: int = None,
            hidden: int = 256,
            depth: int = 2,
            dropout: float = 0.0
    ):
        super().__init__()

        d_model = d_model or input_dim

        self.project = (
            nn.Identity() if d_model == input_dim
            else nn.Linear(input_dim, d_model)
        )
        self.blocks = nn.ModuleList(
            _ResidualFFBlock(d_model, hidden, dropout) for _ in range(depth)
        )
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.project(x)
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(self.norm_out(hidden))
