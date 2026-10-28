from pathlib import Path

import numpy as np
import pytest
import torch

from config import AgentConfig
from utils.builder import build_agent
from utils.seed import episode_seed, resolve_seed, set_global_seed
from utils.train import train_loop

TEMPLATE = Path(__file__).resolve().parents[1] / "config" / "templates" / "cartpole_ppo.toml"


def load_config() -> AgentConfig:
    return AgentConfig.load(str(TEMPLATE))


def test_global_seed_repeats_torch_stream():
    set_global_seed(1234)
    first = torch.randn(8)
    set_global_seed(1234)
    second = torch.randn(8)

    assert torch.equal(first, second)


def test_global_seed_repeats_numpy_stream():
    set_global_seed(1234)
    first = np.random.rand(4)
    set_global_seed(1234)
    second = np.random.rand(4)

    assert np.array_equal(first, second)


def test_global_seed_changes_stream():
    set_global_seed(1)
    first = torch.randn(8)
    set_global_seed(2)
    second = torch.randn(8)

    assert not torch.equal(first, second)


def test_episode_seed_is_deterministic():
    assert episode_seed(7, 3) == episode_seed(7, 3)


def test_episode_seed_varies_with_episode_and_run():
    seeds = {episode_seed(7, i) for i in range(20)}

    assert len(seeds) == 20
    assert episode_seed(7, 0) != episode_seed(8, 0)


def test_episode_seed_is_independent_of_reset_order():
    """Consuming the global RNG must not shift per-episode seeds."""
    first = episode_seed(11, 5)

    set_global_seed(999)
    torch.randn(100)

    assert episode_seed(11, 5) == first


def test_resolve_seed_passes_through_and_draws_fresh():
    assert resolve_seed(42) == 42

    drawn = {resolve_seed() for _ in range(5)}

    assert len(drawn) == 5
    assert all(isinstance(seed, int) and 0 <= seed < 2**31 for seed in drawn)


def test_weight_init_reproducible_for_same_seed():
    config = load_config().to_dict()

    set_global_seed(2026)
    first = build_agent(config)
    set_global_seed(2026)
    second = build_agent(config)

    for (name, a), (_, b) in zip(first.named_parameters(), second.named_parameters()):
        assert torch.equal(a, b), f"{name} differs between identical seeds"


def test_weight_init_differs_for_different_seed():
    config = load_config().to_dict()

    set_global_seed(1)
    first = build_agent(config)
    set_global_seed(2)
    second = build_agent(config)

    assert any(
        not torch.equal(a, b)
        for a, b in zip(first.parameters(), second.parameters())
    )


class Recorder:
    """Minimal stand-in for the dashboard, capturing reward rows."""

    def __init__(self):
        self.rows = []

    def update(self, iteration, reward, **metrics):
        self.rows.append((iteration, reward))


def run_one_iteration(seed: int):
    config = load_config()
    set_global_seed(seed)
    agent = build_agent(config.to_dict())
    recorder = Recorder()

    train_loop(
        env_name=config.env.id,
        agent=agent,
        iterations=1,
        episodes=2,
        dashboard=recorder,
        env_kwargs={**config.env.default, **config.env.training},
        seed=seed,
    )

    return recorder.rows


@pytest.mark.integration
@pytest.mark.parametrize("seed", [3])
def test_training_is_reproducible_for_same_seed(seed):
    assert run_one_iteration(seed) == run_one_iteration(seed)


@pytest.mark.integration
def test_training_without_seed_still_runs():
    config = load_config()
    agent = build_agent(config.to_dict())
    recorder = Recorder()

    train_loop(
        env_name=config.env.id,
        agent=agent,
        iterations=1,
        episodes=1,
        dashboard=recorder,
    )

    assert len(recorder.rows) == 1
