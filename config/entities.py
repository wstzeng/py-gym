import os
import json
import tomllib
import tomli_w
from dataclasses import dataclass, asdict, field
from pathlib import Path

@dataclass
class EnvConfig:
    id: str = None
    state_dim: int = None
    default: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    testing: dict = field(default_factory=dict)

@dataclass
class AgentDetailConfig:
    type: str = None
    components: dict = field(default_factory=dict)

@dataclass
class AgentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentDetailConfig = field(default_factory=AgentDetailConfig)
    hyper_params: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def save(self, path: str):
        """Dispatches to the appropriate saver based on file extension."""
        ext = Path(path).suffix.lower()
        if ext == '.toml':
            self.save_toml(path)
        else:
            self.save_json(path)

    def save_json(self, path: str):
        """Saves configuration to a JSON file."""
        os.makedirs(
            os.path.dirname(path),
            exist_ok=True
        )
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4)

    def save_toml(self, path: str):
        """Saves configuration to a TOML file using tomli-w."""
        os.makedirs(
            os.path.dirname(path),
            exist_ok=True
        )
        # tomli-w uses binary mode
        with open(path, 'wb') as f:
            tomli_w.dump(asdict(self), f)

    @classmethod
    def load(cls, path: str):
        """Dispatches to the appropriate loader based on file extension."""
        ext = Path(path).suffix.lower()
        if ext == '.toml':
            return cls.load_toml(path)
        elif ext == '.json':
            return cls.load_json(path)
        else:
            raise ValueError(f"Unsupported config format: {ext}")

    @classmethod
    def load_json(cls, path: str):
        """Loads configuration from a JSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSON config not found: {path}")

        with open(path, 'r') as f:
            data = json.load(f)

        return cls._from_dict(data)

    @classmethod
    def load_toml(cls, path: str):
        """Loads configuration from a TOML file using tomllib."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"TOML config not found: {path}")

        with open(path, 'rb') as f:
            data = tomllib.load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict):
        """Helper to create an instance from a dictionary."""
        # Using separate params for each field to match coding style
        return cls(
            env=EnvConfig(
                **data.get('env', {})
            ),
            agent=AgentDetailConfig(
                **data.get('agent', {})
            ),
            hyper_params=data.get('hyper_params', {}),
            metadata=data.get('metadata', {})
        )

    def to_dict(self):
        return asdict(self)
