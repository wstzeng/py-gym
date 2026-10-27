import copy

import pytest
import torch
from torch import nn

from utils.builder import build_agent


def make_ppo_config() -> dict:
    return {
        "env": {
            "id": "CartPole-v1",
            "default": {},
            "training": {},
            "testing": {},
        },
        "agent": {
            "type": "PPOAgent",
            "encoder": {
                "type": "VectorEncoder",
                "layers": [
                    {"type": "Linear", "out_features": 32},
                    {"type": "LeakyReLU"},
                ],
            },
            "policy": {
                "type": "ActorCriticPolicy",
                "handler": {
                    "type": "CategoricalHandler",
                    "params": {"action_dim": 2},
                },
                "actor": {
                    "type": "Sequential",
                    "layers": [{"type": "Linear", "out_features": 2}],
                },
                "critic": {
                    "type": "Sequential",
                    "layers": [{"type": "Linear", "out_features": 1}],
                },
            },
            "buffer": {"type": "TrajectoryBuffer"},
        },
        "hyper_params": {
            "gamma": 0.9,
            "eps_clip": 0.1,
            "k_epochs": 2,
            "critic_weight": 0.5,
            "entropy_weight": 0.01,
            "gae_lambda": 0.9,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lrs": {"default": 1e-3, "actor": 1e-4},
                },
            },
            "criterion": {"critic_loss": "MSELoss"},
        },
    }


def make_reinforce_config() -> dict:
    config = make_ppo_config()
    config["agent"]["type"] = "ReinforceAgent"
    config["agent"]["policy"] = {
        "type": "ActorPolicy",
        "handler": {
            "type": "CategoricalHandler",
            "params": {"action_dim": 2},
        },
        "actor": {
            "type": "Sequential",
            "layers": [{"type": "Linear", "out_features": 2}],
        },
    }
    config["hyper_params"].pop("criterion")
    return config


def test_builder_threads_dimensions_and_hyper_params():
    agent = build_agent(make_ppo_config())

    assert agent.encoder.feature_dim == 32
    assert isinstance(agent.policy.actor, nn.Sequential)
    assert agent.policy.actor[0].in_features == agent.encoder.feature_dim
    assert agent.policy.critic[0].in_features == agent.encoder.feature_dim

    assert agent.hyper_params["gamma"] == 0.9
    assert agent.hyper_params["critic_weight"] == 0.5
    assert agent.gamma == 0.9
    assert agent.eps_clip == 0.1
    assert agent.k_epochs == 2
    assert agent.gae_lambda == 0.9

    learning_rates = {group["lr"] for group in agent.optimizer.param_groups}
    assert learning_rates == {1e-4, 1e-3}


def test_select_action_and_one_update():
    agent = build_agent(make_ppo_config())
    state = torch.zeros(1, 4)

    action, info = agent.select_action(state)
    assert isinstance(action, int)
    assert "log_prob" in info
    assert "value" in info

    agent.buffer.store(info, reward=1.0, done=True)
    metrics = agent.update()

    assert "loss" in metrics
    assert torch.isfinite(torch.as_tensor(metrics["loss"]))


def test_reinforce_agent_builds_and_updates():
    agent = build_agent(make_reinforce_config())

    action, info = agent.select_action(torch.zeros(1, 4))
    assert isinstance(action, int)
    assert "value" not in info

    agent.buffer.store(info, reward=1.0, done=True)
    metrics = agent.update()

    assert "loss" in metrics


def test_unknown_agent_key_is_rejected():
    config = copy.deepcopy(make_ppo_config())
    config["agent"]["unexpected"] = {"type": "TrajectoryBuffer"}

    with pytest.raises(ValueError, match="Unknown agent config keys"):
        build_agent(config)


def test_missing_optimizer_is_rejected():
    config = copy.deepcopy(make_ppo_config())
    config["hyper_params"].pop("optimizer")

    with pytest.raises(ValueError, match="hyper_params.optimizer"):
        build_agent(config)
