import numpy as np
import torch

from agent.components.policy.action_handler import (
    BetaHandler,
    CategoricalHandler,
    NormalHandler,
)


def test_normal_handler_maps_actions_to_bounds():
    handler = NormalHandler(
        action_dim=2,
        action_low=-2.0,
        action_high=3.0,
    )

    action = handler.to_env_format(torch.zeros(1, 2))

    assert action.shape == (2,)
    assert np.allclose(action, [0.5, 0.5])


def test_beta_handler_maps_actions_to_bounds():
    handler = BetaHandler(
        action_dim=2,
        action_low=-1.0,
        action_high=1.0,
    )

    action = handler.to_env_format(torch.full((1, 2), 0.5))

    assert action.shape == (2,)
    assert np.allclose(action, [0.0, 0.0])


def test_categorical_handler_maps_to_scalar():
    handler = CategoricalHandler(action_dim=3)

    action = handler.to_env_format(torch.tensor([2]))

    assert action == 2
