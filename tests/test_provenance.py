import os
import subprocess

import pytest

from utils import provenance
from utils.provenance import collect, git_revision


def test_git_revision_matches_git(tmp_path):
    sha, dirty = git_revision()

    assert sha == subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    assert isinstance(dirty, bool)


def test_git_revision_survives_a_missing_git(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("git not found")

    monkeypatch.setattr(provenance.subprocess, "run", boom)

    assert git_revision() == (None, None)


def test_git_revision_ignores_untracked_files(tmp_path, monkeypatch):
    """A sweep writes untracked result directories; that must not read as dirty."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "tracked.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    monkeypatch.chdir(repo)

    _, clean = git_revision()
    (repo / "experiments").mkdir()
    (repo / "experiments" / "run.csv").write_text("a\n")
    _, with_untracked = git_revision()
    (repo / "tracked.py").write_text("x = 2\n")
    _, modified = git_revision()

    assert clean is False
    assert with_untracked is False
    assert modified is True


def test_collect_reports_threads_and_versions():
    data = collect()

    assert data["torch_threads"] >= 1
    assert data["python"].count(".") >= 1
    assert data["torch"]
    assert data["numpy"]
    assert data["python_executable"]


def test_collect_omits_unknown_values(monkeypatch):
    monkeypatch.setattr(provenance, "git_revision", lambda: (None, None))
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

    data = collect()

    # TOML has no null, so absent values must be absent keys, not None.
    assert "git_sha" not in data
    assert "git_dirty" not in data
    assert all(value is not None for value in data.values())


def test_collect_records_omp_override(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")

    assert collect()["omp_num_threads"] == "7"


def test_collect_is_toml_serialisable(tmp_path):
    import tomli_w

    path = tmp_path / "meta.toml"

    tomli_w.dump({"metadata": collect()}, path.open("wb"))

    assert path.read_text().strip()
