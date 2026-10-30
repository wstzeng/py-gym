"""Guards for the component taxonomy described in AGENTS.md.

The component registry is one flat namespace, so a value network dropped into
``encoder/`` still builds and still trains — it just returns a scalar where the
policy expects a feature vector. No other test can see that mistake, because
nothing about it is an error at runtime. These assertions encode the rule:

    an encoder produces features and exposes ``feature_dim``;
    a critic produces one value per sample and lives on the policy side.
"""

import torch

from agent.components import encoder as encoder_pkg
from agent.components import policy as policy_pkg
from agent.components.encoder import BaseEncoder

CRITICS = ("ConvCritic", "ResidualMLPCritic", "TransformerCritic")


def test_every_encoder_export_is_a_real_encoder():
    for name in encoder_pkg.__all__:
        cls = getattr(encoder_pkg, name)

        assert issubclass(cls, BaseEncoder), f"{name} is not a BaseEncoder"
        assert hasattr(cls, "feature_dim"), f"{name} exposes no feature_dim"


def test_critics_are_exported_from_the_policy_side():
    for name in CRITICS:
        assert hasattr(policy_pkg, name), f"{name} missing from policy/__all__"
        assert name in policy_pkg.__all__
        assert not hasattr(encoder_pkg, name), f"{name} belongs in policy/"


def test_critics_return_one_value_per_sample():
    for name in CRITICS:
        critic = getattr(policy_pkg, name)(input_dim=64)
        values = critic(torch.zeros(3, 64))

        assert values.shape == (3, 1), f"{name} returned {tuple(values.shape)}"
        assert not isinstance(critic, BaseEncoder)


def test_critics_declare_that_they_need_the_parent_dimension():
    for name in CRITICS:
        cls = getattr(policy_pkg, name)

        assert getattr(cls, "requires_input_dim", False) is True
