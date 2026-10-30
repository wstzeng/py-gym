"""Where experiment records and their artifacts live.

Records (``experiments/``) are tracked and branch-scoped; artifacts
(``artifacts/``) are an ignored cache. Both trees share one relative layout, so
a record can name its artifacts without absolute paths.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORD_ROOT = REPO_ROOT / "experiments"
ARTIFACT_ROOT = REPO_ROOT / "artifacts"


def sweep_dir(name, root=None):
    """``<root>/sweeps/<name>``."""
    return Path(root or RECORD_ROOT) / "sweeps" / name


def run_dir(sweep_root, arm, seed):
    """``<sweep>/<arm>/seed<N>`` — the layout both roots share."""
    return Path(sweep_root) / arm / f"seed{int(seed)}"


def record_dir(name, arm, seed, root=None):
    """Record directory for one (sweep, arm, seed)."""
    return run_dir(sweep_dir(name, root), arm, seed)


def artifact_dir(name, arm, seed, root=None):
    """Artifact directory for one (sweep, arm, seed)."""
    return run_dir(sweep_dir(name, root or ARTIFACT_ROOT), arm, seed)


def artifact_for(record):
    """Cache directory mirroring a record directory.

    Records under ``experiments/`` are tracked; their heavy siblings (the
    checkpoint, the per-run plot) live at the same relative path under
    ``artifacts/``. Anything outside the record tree — a ``--dry`` temp dir or a
    ``--save-dir`` pointing elsewhere — keeps its artifacts next to itself, so
    the mapping is always "same layout, one root apart".
    """
    record = Path(record).expanduser().resolve()
    try:
        relative = record.relative_to(RECORD_ROOT)
    except ValueError:
        return record
    return ARTIFACT_ROOT / relative
