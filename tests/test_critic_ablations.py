"""Shape and parameter-parity tests for the critic ablation variants.

The parameter counts encode the experimental design: each tier is a set of
critics with (nearly) identical capacity but a different mixing operator, so a
performance difference can be attributed to that operator instead of capacity.
If a refactor breaks parity, these tests are supposed to go red.
"""

import pytest
import torch

from agent.components.encoder import (
    ConvCritic,
    ResidualMLPCritic,
    TransformerCritic,
)
from utils.builder import build_component

INPUT_DIM = 64

SMALL_TARGET = 2_113
LARGE_TARGET = 67_585


def count(module):
    return sum(p.numel() for p in module.parameters())


def mlp_small():
    return torch.nn.Sequential(
        torch.nn.Linear(INPUT_DIM, 32),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(32, 1),
    )


def mlp_large():
    return torch.nn.Sequential(
        torch.nn.Linear(INPUT_DIM, 1024),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(1024, 1),
    )


# arm name -> (factory, tier)
ARMS = {
    "mlp_small": (mlp_small, "small"),
    "conv_small": (lambda: ConvCritic(INPUT_DIM, d_model=18), "small"),
    "attn_small": (
        lambda: TransformerCritic(
            INPUT_DIM, d_model=16, nhead=2, depth=1, dim_feedforward=32
        ),
        "small",
    ),
    "mlp_large": (mlp_large, "large"),
    "conv_large": (lambda: ConvCritic(INPUT_DIM, d_model=104), "large"),
    "attn_large": (lambda: TransformerCritic(INPUT_DIM), "large"),
    "resmlp": (lambda: ResidualMLPCritic(INPUT_DIM), "large"),
}


@pytest.mark.parametrize("name", ARMS)
def test_critic_returns_one_value_per_sample(name):
    factory, _ = ARMS[name]
    critic = factory()

    assert critic(torch.randn(5, INPUT_DIM)).shape == (5, 1)


@pytest.mark.parametrize("name", ARMS)
def test_parameter_parity_within_tier(name):
    factory, tier = ARMS[name]
    target = SMALL_TARGET if tier == "small" else LARGE_TARGET
    tolerance = 0.15 if tier == "small" else 0.02

    actual = count(factory())

    assert abs(actual - target) / target <= tolerance, (
        f"{name} has {actual} parameters, expected {target} "
        f"(+/- {tolerance:.0%})"
    )


def test_mlp_large_matches_transformer_exactly():
    assert count(mlp_large()) == count(TransformerCritic(INPUT_DIM))


def test_builder_injects_input_dim_for_new_critics():
    for cls_type, params in (
        ("ConvCritic", {"d_model": 18}),
        ("ResidualMLPCritic", {"hidden": 256}),
    ):
        critic, output_dim = build_component(
            {"type": cls_type, "params": params},
            input_dim=INPUT_DIM,
        )

        assert output_dim is None
        assert critic(torch.randn(3, INPUT_DIM)).shape == (3, 1)


def test_builder_fails_without_input_dim():
    with pytest.raises(ValueError, match="requires an input dimension"):
        build_component({"type": "ConvCritic", "params": {"d_model": 18}})
    with pytest.raises(ValueError, match="requires an input dimension"):
        build_component({"type": "ResidualMLPCritic"})


def test_conv_critic_rejects_even_kernel():
    with pytest.raises(ValueError, match="must be odd"):
        ConvCritic(INPUT_DIM, kernel_size=2)
