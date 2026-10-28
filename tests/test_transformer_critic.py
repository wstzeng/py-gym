import pytest
import torch

from agent.components.encoder import TransformerCritic
from utils.builder import build_component


def test_critic_returns_one_value_per_sample():
    critic = TransformerCritic(
        input_dim=64,
        d_model=32,
        nhead=4,
        depth=2,
        n_tokens=8,
    )

    output = critic(torch.randn(5, 64))

    assert output.shape == (5, 1)


def test_critic_rejects_non_divisible_token_split():
    with pytest.raises(ValueError, match="not divisible"):
        TransformerCritic(input_dim=64, n_tokens=5)


def test_builder_injects_input_dim():
    critic, output_dim = build_component(
        {
            "type": "TransformerCritic",
            "params": {
                "d_model": 32,
                "nhead": 4,
                "depth": 1,
                "n_tokens": 8,
            },
        },
        input_dim=64,
    )

    assert output_dim is None
    assert critic(torch.randn(3, 64)).shape == (3, 1)


def test_builder_fails_without_input_dim():
    with pytest.raises(ValueError, match="requires an input dimension"):
        build_component({"type": "TransformerCritic", "params": {"n_tokens": 8}})
