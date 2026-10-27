from pathlib import Path

import gymnasium as gym
import pytest
import torch

from config import AgentConfig
from utils.builder import build_agent

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "config" / "templates"

TEMPLATES = [
    "cartpole_ppo.toml",
    "cartpole_reinforce.toml",
    "lunarlander_ppo.toml",
]


def box2d_available() -> bool:
    try:
        import Box2D  # noqa: F401
    except ImportError:
        return False
    return True


def load_template(filename: str) -> AgentConfig:
    return AgentConfig.load(str(TEMPLATES_DIR / filename))


def skip_without_box2d(config: AgentConfig):
    """LunarLander is the project's reference Box2D environment."""
    if "LunarLander" in config.env.id and not box2d_available():
        pytest.skip(
            "Box2D is not installed; see README (system swig + box2d extra)"
        )


@pytest.mark.integration
@pytest.mark.parametrize("filename", TEMPLATES)
def test_template_builds_and_runs_one_update(filename):
    config = load_template(filename)
    skip_without_box2d(config)

    agent = build_agent(config.to_dict())
    state_dim = agent.encoder.net[0].in_features

    action, info = agent.select_action(torch.zeros(1, state_dim))
    assert action is not None
    assert "log_prob" in info

    agent.record(info, reward=1.0, done=True)
    metrics = agent.update()

    assert "loss" in metrics
    assert torch.isfinite(torch.as_tensor(metrics["loss"]))


@pytest.mark.integration
@pytest.mark.parametrize("filename", TEMPLATES)
def test_template_environment_steps(filename):
    config = load_template(filename)
    skip_without_box2d(config)

    env_kwargs = {**config.env.default, **config.env.training}
    env = gym.make(config.env.id, **env_kwargs)
    try:
        observation, _ = env.reset(seed=0)
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    finally:
        env.close()

    assert observation.shape == env.observation_space.shape
    assert reward is not None
    assert isinstance(terminated or truncated, bool)
