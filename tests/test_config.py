from pathlib import Path

import pytest

from config import AgentConfig

TEMPLATES = Path(__file__).resolve().parents[1] / "config" / "templates"


@pytest.mark.parametrize(
    "filename",
    [
        "cartpole_ppo.toml",
        "cartpole_reinforce.toml",
        "lunarlander_ppo.toml",
    ],
)
def test_templates_load(filename):
    config = AgentConfig.load(str(TEMPLATES / filename))

    assert config.env.id
    assert "type" in config.agent
    assert "encoder" in config.agent
    assert "policy" in config.agent


def test_state_dim_is_rejected():
    with pytest.raises(ValueError, match="Invalid env config"):
        AgentConfig._from_dict({
            "env": {
                "id": "CartPole-v1",
                "state_dim": 4,
            },
        })
