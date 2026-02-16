import os
import json
from dataclasses import dataclass, asdict, field, fields

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
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load(cls, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r') as f:
            data = json.load(f)

        return cls(
            env=EnvConfig(**data.get('env', {})),
            agent=AgentDetailConfig(**data.get('agent', {})),
            hyper_params=data.get('hyper_params', {}),
            metadata=data.get('metadata', {})
        )

    def to_dict(self):
        return asdict(self)
