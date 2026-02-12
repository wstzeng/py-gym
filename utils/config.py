import os
import json
from dataclasses import dataclass, asdict, field, fields

@dataclass
class AgentConfig:
    env_id: str = None
    is_continuous: bool = False
    encoder_type: str = None
    hidden_dims: list = field(default_factory=list)
    hyper_params: dict = field(default_factory=dict)
    k_epochs: int = 1
    state_dim: int = None
    action_dim: int = None
    components: dict = field(default_factory=dict)
    optimizer: dict = field(default_factory=dict)
    agent: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def save(self, path: str):
        """
        Saves config to JSON, including runtime metadata.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load(cls, path: str):
        """
        Loads JSON and filters out keys not defined in the dataclass.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
            
        with open(path, 'r') as f:
            data = json.load(f)

        # Filter: Only keep keys that exist in AgentConfig fields
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        return cls(**filtered_data)
