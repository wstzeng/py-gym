"""Provenance recorded alongside every run.

A result is only reusable if the run can be traced back to the code that
produced it, the threads it ran with, and the library versions it ran against.
None of that survives in ``training_log.csv``, so it goes into the run metadata.

Two deliberate choices:

* ``git status --porcelain --untracked-files=no`` decides ``git_dirty``. Untracked
  files are ignored on purpose: a sweep writes untracked result directories while
  it runs, and flagging every run as dirty would make the flag useless.
* The thread count comes from torch, not from ``OMP_NUM_THREADS``. The
  environment variable can be set to something torch clamps or ignores; what
  matters for bit-identical reruns is the count torch actually used.
"""

import os
import platform
import subprocess
import sys

import numpy as np
import torch

GIT_TIMEOUT = 5


def _run_git(*args) -> str | None:
    """Run ``git <args>`` and return stripped stdout, or ``None`` on any failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_revision() -> tuple[str | None, bool | None]:
    """``(sha, dirty)`` for the current checkout, or ``(None, None)`` outside a repo."""
    sha = _run_git("rev-parse", "HEAD")
    if not sha:
        return None, None
    status = _run_git("status", "--porcelain", "--untracked-files=no")
    return sha, bool(status)


def collect() -> dict:
    """Everything worth recording about the machine a run happened on.

    Keys whose value cannot be determined are omitted, not set to ``None``,
    because the metadata is also written to TOML and TOML has no null.
    """
    sha, dirty = git_revision()
    provenance = {
        "git_sha": sha,
        "git_dirty": dirty,
        "torch_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python_executable": sys.executable,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }
    return {key: value for key, value in provenance.items() if value is not None}
