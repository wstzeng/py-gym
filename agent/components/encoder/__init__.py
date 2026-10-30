"""Encoders: raw observation -> latent feature vector.

An encoder is the agent-level feature body. Every class exported here must
subclass :class:`BaseEncoder` and expose ``feature_dim``, because
``build_agent()`` threads that number into the policy. Value networks do not
belong here: they consume features and return a scalar, so they live in
``agent/components/policy/`` next to the critic slot that uses them.
"""

from .base_encoder import BaseEncoder
from .vector_encoder import VectorEncoder

__all__ = [
    "VectorEncoder",
]
