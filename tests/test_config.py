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


def test_unknown_top_level_section_is_rejected():
    with pytest.raises(ValueError, match="Unknown config section"):
        AgentConfig._from_dict({"env": {"id": "CartPole-v1"}, "agents": {}})


def test_save_writes_toml_atomically(tmp_path):
    import tomllib

    config = AgentConfig._from_dict({
        "env": {"id": "CartPole-v1", "default": {"max_episode_steps": 500}},
        "agent": {"type": "PPOAgent", "encoder": {"type": "VectorEncoder"}},
        "hyper_params": {"lr": 0.001},
        "metadata": {"seed": 7, "wall_seconds": 12.5},
    })
    path = tmp_path / "config.toml"

    config.save(str(path))

    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    assert raw["metadata"]["seed"] == 7
    assert not (tmp_path / "config.toml.tmp").exists()

    loaded = AgentConfig.load(str(path))
    assert loaded.to_dict() == config.to_dict()


def test_save_refuses_other_formats(tmp_path):
    config = AgentConfig._from_dict({"env": {"id": "CartPole-v1"}})

    with pytest.raises(ValueError, match=".toml"):
        config.save(str(tmp_path / "config.json"))


def test_legacy_json_still_loads(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        '{"env": {"id": "CartPole-v1"}, "metadata": {"seed": 3}}'
    )

    config = AgentConfig.load(str(path))

    assert config.env.id == "CartPole-v1"
    assert config.metadata["seed"] == 3


def test_unknown_extension_is_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("env: {}")

    with pytest.raises(ValueError, match="Unsupported config format"):
        AgentConfig.load(str(path))
