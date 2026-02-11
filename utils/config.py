import os
import json
from dataclasses import dataclass, asdict

@dataclass
class AgentConfig:
    env_id: str
    is_continuous: bool
    encoder_type: str
    hidden_dims: list
    hyper_params: dict
    k_epochs: int
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
            return cls(**data)

