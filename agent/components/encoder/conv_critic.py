import torch
from torch import nn


class _ConvBlock(nn.Module):
    """Local (kernel-window) mixing with a post-norm residual."""

    def __init__(self, d_model: int, kernel_size: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: (batch, tokens, channels); conv wants (batch, channels, tokens).
        mixed = self.conv(hidden.transpose(1, 2)).transpose(1, 2)
        return self.norm(hidden + self.dropout(mixed))


class ConvCritic(nn.Module):
    """Value network that mixes adjacent feature tokens with 1D convolutions.

    Ablation sibling of :class:`TransformerCritic`. The token split, the
    residual stream, the post-norm placement and the activation are identical;
    only the mixing operator differs (kernel-sized window instead of dense
    attention). The dense counterpart without attention is
    :class:`ResidualMLPCritic`.

    Convolution shares its weights across every token position, so matching the
    parameter count of a transformer forces a wider channel dimension. That is
    why the default ``d_model`` is larger than the transformer's, and why the
    per-sample MAC count is several times higher at equal parameter count.

    The kernel must be odd so the token axis keeps its length; the encoder is
    deliberately depth-limited and shallow, it is a critic, not a feature body.
    """

    requires_input_dim = True

    def __init__(
            self,
            input_dim: int,
            d_model: int = 104,
            depth: int = 2,
            n_tokens: int = 8,
            kernel_size: int = 3,
            dropout: float = 0.0
    ):
        super().__init__()

        if input_dim % n_tokens != 0:
            raise ValueError(
                f"input_dim={input_dim} is not divisible by n_tokens={n_tokens}"
            )
        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size={kernel_size} must be odd to preserve the token axis"
            )

        self.n_tokens = n_tokens
        self.token_dim = input_dim // n_tokens

        self.project = nn.Linear(self.token_dim, d_model)
        self.blocks = nn.ModuleList(
            _ConvBlock(d_model, kernel_size, dropout) for _ in range(depth)
        )
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.view(x.size(0), self.n_tokens, self.token_dim)
        hidden = self.project(tokens)
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(self.norm_out(hidden.mean(dim=1)))
