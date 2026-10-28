"""Deterministic seeding helpers.

Seeding happens in two separate places and both matter:

* :func:`set_global_seed` must run *before* the agent is built, because weight
  initialisation consumes torch's global RNG. Seeding only inside the training
  loop gives every run a different initial policy.
* :func:`episode_seed` derives a per-episode environment seed from the run seed
  and the episode index. Because it is a pure function of ``(seed, index)``,
  every configuration run with the same seed sees the same sequence of initial
  states regardless of how many episodes it happened to reset. That is what
  makes cross-run and cross-architecture comparisons meaningful.

Reproducibility caveat: torch CPU kernels may change reduction order with the
thread count, so bit-identical reruns also require the same ``OMP_NUM_THREADS``.
"""

import random
import secrets

import numpy as np
import torch


def set_global_seed(seed: int) -> int:
    """Seed python, numpy and torch. Returns the seed actually used."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    return seed


def episode_seed(seed: int, episode_index: int) -> int:
    """Environment seed for one episode, independent of reset order."""
    state = np.random.SeedSequence([int(seed), int(episode_index)]).generate_state(1)
    return int(state[0])


def resolve_seed(seed: int | None = None) -> int:
    """Return the caller's seed, or draw a fresh one so runs stay unlinked."""
    if seed is not None:
        return int(seed)
    return secrets.randbits(31)
