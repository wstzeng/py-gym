"""Shape and parameter-parity tests for the critic ablation variants.

The parameter counts encode the experimental design: within a tier every critic
has (nearly) the same capacity but a different mixing operator, so a performance
difference can be attributed to that operator instead of capacity. If a refactor
breaks parity, these tests are supposed to go red.

Tiers:
    small   ~2.1k   the critic the shipped stable recipe actually uses
    mid    ~17k     the matrix proper: all four mixing families at equal capacity
    anchor ~67.6k   the capacity of the original TransformerCritic result, kept
                    as a cheap pure-MLP reference point
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

# tier -> (target parameter count, relative tolerance)
TIERS = {
    "small": (2_113, 0.15),
    "mid": (17_150, 0.02),
    "anchor": (67_585, 0.02),
}


def count(module):
    return sum(p.numel() for p in module.parameters())


def mlp(hidden):
    return torch.nn.Sequential(
        torch.nn.Linear(INPUT_DIM, hidden),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(hidden, 1),
    )


# arm name -> (factory, tier)
ARMS = {
    "mlp_small": (lambda: mlp(32), "small"),
    "conv_small": (lambda: ConvCritic(INPUT_DIM, d_model=18), "small"),
    "attn_small": (
        lambda: TransformerCritic(
            INPUT_DIM, d_model=16, nhead=2, depth=1, dim_feedforward=32
        ),
        "small",
    ),
    "mlp_mid": (lambda: mlp(256), "mid"),
    "conv_mid": (lambda: ConvCritic(INPUT_DIM, d_model=52), "mid"),
    "attn_mid": (
        lambda: TransformerCritic(
            INPUT_DIM, d_model=32, nhead=4, depth=2, dim_feedforward=64
        ),
        "mid",
    ),
    "resmlp_mid": (lambda: ResidualMLPCritic(INPUT_DIM, hidden=64), "mid"),
    "mlp_anchor": (lambda: mlp(1024), "anchor"),
}


@pytest.mark.parametrize("name", ARMS)
def test_critic_returns_one_value_per_sample(name):
    factory, _ = ARMS[name]
    critic = factory()

    assert critic(torch.randn(5, INPUT_DIM)).shape == (5, 1)


@pytest.mark.parametrize("name", ARMS)
def test_parameter_parity_within_tier(name):
    factory, tier = ARMS[name]
    target, tolerance = TIERS[tier]

    actual = count(factory())

    assert abs(actual - target) / target <= tolerance, (
        f"{name} has {actual} parameters, expected {target} "
        f"(+/- {tolerance:.0%})"
    )


def test_mlp_anchor_matches_transformer_exactly():
    factory, _ = ARMS["mlp_anchor"]

    assert count(factory()) == count(TransformerCritic(INPUT_DIM))


def test_builder_injects_input_dim_for_new_critics():
    for cls_type, params in (
        ("ConvCritic", {"d_model": 52}),
        ("ResidualMLPCritic", {"hidden": 64}),
    ):
        critic, output_dim = build_component(
            {"type": cls_type, "params": params},
            input_dim=INPUT_DIM,
        )

        assert output_dim is None
        assert critic(torch.randn(3, INPUT_DIM)).shape == (3, 1)


def test_builder_fails_without_input_dim():
    with pytest.raises(ValueError, match="requires an input dimension"):
        build_component({"type": "ConvCritic", "params": {"d_model": 52}})
    with pytest.raises(ValueError, match="requires an input dimension"):
        build_component({"type": "ResidualMLPCritic"})


def test_conv_critic_rejects_even_kernel():
    with pytest.raises(ValueError, match="must be odd"):
        ConvCritic(INPUT_DIM, kernel_size=2)
