import torch

from agent.learner.reinforce_agent import ReinforceAgent


def test_returns_reset_at_episode_boundaries():
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
    dones = torch.tensor([0.0, 1.0, 0.0, 1.0])

    returns = ReinforceAgent.compute_returns(rewards, dones, gamma=0.5)

    assert torch.allclose(returns, torch.tensor([1.5, 1.0, 1.5, 1.0]))


def test_single_episode_matches_discounted_sum():
    rewards = torch.tensor([1.0, 2.0, 3.0])
    dones = torch.tensor([0.0, 0.0, 1.0])

    returns = ReinforceAgent.compute_returns(rewards, dones, gamma=0.5)

    assert torch.allclose(returns, torch.tensor([2.75, 3.5, 3.0]))


def test_truncated_episode_does_not_borrow_future_rewards():
    rewards = torch.tensor([1.0, 2.0])
    dones = torch.tensor([0.0, 0.0])

    returns = ReinforceAgent.compute_returns(rewards, dones, gamma=0.9)

    assert torch.allclose(returns, torch.tensor([2.8, 2.0]))
