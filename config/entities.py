import os
import json
import tomllib
import tomli_w
from dataclasses import dataclass, asdict, field
from pathlib import Path

# Every config file has exactly these top-level sections. Anything else is a
# typo, and silently ignoring it would drop settings without a warning.
SECTIONS = ("env", "agent", "hyper_params", "metadata")

@dataclass
class EnvConfig:
    id: str = None
    default: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    testing: dict = field(default_factory=dict)

@dataclass
class AgentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: dict = field(default_factory=dict)
    hyper_params: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def save(self, path: str):
        """Writes the config snapshot as TOML, atomically.

        TOML is the only format the framework writes: it reads back with the
        standard library, keeps comments in hand-written files, and is what
        every recipe in ``config/`` already uses. The write goes to a sibling
        ``.tmp`` and is renamed into place, because the presence of this file is
        what marks a run as completed (see ``manage.py``): a half-written file
        must never be mistaken for a finished run.
        """
        if Path(path).suffix.lower() != '.toml':
            raise ValueError(f"Config snapshots are written as .toml, got: {path}")
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        tmp = f"{path}.tmp"
        try:
            with open(tmp, 'wb') as f:
                tomli_w.dump(asdict(self), f)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

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
        """Loads a legacy JSON config snapshot (read-only fallback)."""
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
        unknown = set(data) - set(SECTIONS)
        if unknown:
            raise ValueError(
                f"Unknown config section(s): {', '.join(sorted(unknown))}. "
                f"Expected one of: {', '.join(SECTIONS)}"
            )

        try:
            env = EnvConfig(**data.get('env', {}))
        except TypeError as exc:
            raise ValueError(f"Invalid env config: {exc}") from exc

        return cls(
            env=env,
            agent=data.get('agent', {}),
            hyper_params=data.get('hyper_params', {}),
            metadata=data.get('metadata', {})
        )

    def to_dict(self):
        return asdict(self)
