import csv
from pathlib import Path

import pytest

from scripts.sweep import (
    core_slices,
    expand_configs,
    plan_runs,
    run_dir_for,
    solved_iteration,
    summarize,
)


def write_csv(path, rewards):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Iteration", "AverageReward"])
        for i, reward in enumerate(rewards):
            writer.writerow([i, reward])


def test_plan_runs_covers_full_matrix():
    runs = plan_runs(["a.toml", "b.toml"], [0, 1])

    assert [(r["stem"], r["seed"]) for r in runs] == [
        ("a", 0), ("a", 1), ("b", 0), ("b", 1)
    ]


def test_plan_runs_rejects_duplicate_seed():
    with pytest.raises(SystemExit, match="duplicate"):
        plan_runs(["a.toml"], [0, 0])


def test_expand_configs_keeps_order_and_deduplicates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "b.toml").write_text("")
    (tmp_path / "a.toml").write_text("")

    assert expand_configs(["*.toml", "a.toml"]) == ["a.toml", "b.toml"]


def test_expand_configs_fails_on_empty_glob(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit, match="no config matched"):
        expand_configs(["missing/*.toml"])


def test_core_slices_are_disjoint_when_fitting():
    assert core_slices(2, 4, nproc=16) == ["0-3", "4-7"]


def test_core_slices_skip_pinning_when_oversubscribed():
    assert core_slices(6, 4, nproc=16) == [None] * 6


def test_solved_iteration_slides_over_the_plateau():
    rewards = [0.0] * 90 + [300.0] * 10

    # A 100-wide window never accumulates enough of the plateau.
    assert solved_iteration(rewards, window=100, threshold=200) is None
    # A 10-wide window crosses at i=96: three zeros and seven 300s average 210.
    assert solved_iteration(rewards, window=10, threshold=200) == 96


def test_solved_iteration_returns_none_for_a_ramp():
    assert solved_iteration([float(i) for i in range(120)]) is None


def test_summarize_reports_tail_statistics(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_csv(run_dir / "training_log.csv", [float(i) for i in range(120)])

    stats = summarize(run_dir)

    assert stats["iterations"] == 120
    assert stats["last50"] == 94.5
    assert stats["last100"] == 69.5
    assert stats["best"] == 119.0
    assert stats["best_iter"] == 119
    assert stats["solved_iter"] is None


def test_summarize_tolerates_missing_csv(tmp_path):
    assert summarize(tmp_path) == {}


def test_run_dir_layout():
    run = {"stem": "04_conv_large", "seed": 2}

    assert run_dir_for("experiments/sweeps/x", run) == Path(
        "experiments/sweeps/x/04_conv_large/seed2"
    )
