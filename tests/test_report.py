import csv
import json
from pathlib import Path

import numpy as np
import pytest
import tomli_w

from utils.experiment import paths
from utils.experiment.report import (
    SUMMARY_FIELDS,
    aggregate,
    collect_rows,
    load_curves,
    load_metadata,
    parse_panels,
    rolling_mean,
    solved_iteration,
    stacked,
    summarize,
    summary_table,
    write_report,
    write_summary,
)


def write_csv(path, rewards):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Iteration", "AverageReward"])
        for i, reward in enumerate(rewards):
            writer.writerow([i, reward])


def make_run(root, arm, seed, rewards=None, **metadata):
    """One run directory: a config snapshot plus an optional curve."""
    run_dir = root / arm / f"seed{seed}"
    run_dir.mkdir(parents=True)
    with (run_dir / "config.toml").open("wb") as handle:
        tomli_w.dump({"metadata": {"seed": seed, **metadata}}, handle)
    if rewards is not None:
        write_csv(run_dir / "training_log.csv", rewards)
    return run_dir


PROVENANCE = {
    "torch_threads": 4,
    "python": "3.12.4",
    "torch": "2.9.0",
    "numpy": "2.3.1",
    "wall_seconds": 3000,
}


def test_run_dir_layout():
    assert paths.run_dir("experiments/sweeps/x", "04_conv_large", 2) == Path(
        "experiments/sweeps/x/04_conv_large/seed2"
    )


def test_artifact_for_mirrors_a_record_under_experiments():
    record = paths.RECORD_ROOT / "sweeps" / "x" / "arm" / "seed0"

    assert paths.artifact_for(record) == paths.ARTIFACT_ROOT / "sweeps/x/arm/seed0"


def test_artifact_for_keeps_external_records_next_to_themselves(tmp_path):
    # A dry run or an explicit --save-dir outside experiments/ has no mirror.
    assert paths.artifact_for(tmp_path) == tmp_path.resolve()


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


def test_collect_rows_reads_snapshot_metadata(tmp_path):
    make_run(tmp_path, "arm_a", 0, [240.0] * 100, iterations=100, episodes=20,
             **PROVENANCE)

    rows, incomplete = collect_rows(tmp_path)

    assert incomplete == []
    row = rows[0]
    assert row["config"] == "arm_a"
    assert row["seed"] == 0
    assert row["threads"] == 4
    assert row["wall_min"] == 50.0
    assert row["iterations"] == 100  # counted from the curve, not the metadata


def test_collect_rows_flags_a_directory_without_a_snapshot(tmp_path):
    run_dir = tmp_path / "arm_b" / "seed0"
    write_csv(run_dir / "training_log.csv", [1.0])

    rows, incomplete = collect_rows(tmp_path)

    assert rows == []
    assert incomplete == [str(run_dir)]


def test_collect_rows_falls_back_to_a_json_snapshot(tmp_path):
    run_dir = tmp_path / "arm_c" / "seed0"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps({"metadata": {"seed": 0, "iterations": 5}})
    )

    rows, _ = collect_rows(tmp_path)

    assert rows[0]["iterations"] == 5


def test_load_metadata_returns_none_without_a_snapshot(tmp_path):
    assert load_metadata(tmp_path) is None


def test_collect_rows_reads_legacy_run_json_for_timings(tmp_path):
    run_dir = make_run(tmp_path, "arm_a", 0, [240.0] * 100)
    (run_dir / "run.json").write_text(
        json.dumps({"wall_seconds": 600, "threads": 4})
    )

    rows, _ = collect_rows(tmp_path)

    assert rows[0]["wall_min"] == 10.0
    assert rows[0]["threads"] == 4


def test_write_summary_is_derived_and_has_no_rc(tmp_path):
    make_run(tmp_path, "arm_a", 0, [240.0] * 100, iterations=100, **PROVENANCE)
    rows, _ = collect_rows(tmp_path)

    path = write_summary(tmp_path, rows)

    with path.open() as handle:
        written = list(csv.DictReader(handle))
    assert list(written[0]) == SUMMARY_FIELDS
    assert "rc" not in written[0]
    assert written[0]["status"] == "ok"


def row(config, seed, last100, solved_iter="", wall_min="50.0"):
    return {
        "config": config,
        "seed": str(seed),
        "iterations": "600",
        "last50": "200.0",
        "last50_sd": "10.0",
        "last100": str(last100),
        "last100_sd": "5.0",
        "best": "250.0",
        "best_iter": "500",
        "solved_iter": solved_iter,
        "wall_min": wall_min,
    }


def test_aggregate_censors_unsolved_runs():
    rows = [
        row("a", 0, 240, solved_iter="280"),
        row("a", 1, 230, solved_iter="300"),
        row("a", 2, 120, solved_iter=""),
    ]

    arm = aggregate(rows)[0]

    assert arm["seeds"] == 3
    assert arm["solved"] == "2/3"
    assert arm["solved_at"] == "290 +/- 14"
    assert arm["last100"] == "197 +/- 67"


def test_aggregate_single_seed_has_no_spread():
    arm = aggregate([row("a", 0, 240, solved_iter="280")])[0]

    assert arm["solved_at"] == "280"
    assert arm["last100"] == "240"


def test_aggregate_tolerates_missing_metrics():
    arm = aggregate([{"config": "a", "seed": "0", "wall_min": "1.0"}])[0]

    assert arm["last100"] == "—"
    assert arm["solved"] == "0/1"


def test_report_records_provenance_and_incomplete_runs(tmp_path):
    make_run(tmp_path, "arm_a", 0, [240.0] * 100, iterations=100, episodes=20,
             git_sha="deadbeefdeadbeef", git_dirty=False, **PROVENANCE)
    make_run(tmp_path, "arm_a", 1, [230.0] * 100, iterations=100, episodes=20,
             git_sha="deadbeefdeadbeef", git_dirty=False, **PROVENANCE)
    write_csv(tmp_path / "arm_b" / "seed0" / "training_log.csv", [1.0, 2.0])

    text = write_report(tmp_path).read_text()

    assert "Sweep report" in text
    assert "`deadbeefdead` (clean)" in text
    assert "| runs | 3 (1 configs, seeds 0, 1) |" in text
    assert "| complete | 2 ok, 1 incomplete |" in text
    assert "| `arm_a` | 2 | 235 +/- 7 | 99 +/- 0 | 2/2 | 50 +/- 0 min |" in text
    assert "| `arm_b/seed0` |" in text
    assert "| rc |" not in text


def test_report_flags_a_dirty_tree(tmp_path):
    make_run(tmp_path, "arm_a", 0, [240.0] * 100,
             git_sha="abcdef1234567890", git_dirty=True)

    assert "dirty working tree" in write_report(tmp_path).read_text()


def test_report_says_so_when_provenance_is_absent(tmp_path):
    make_run(tmp_path, "arm_a", 0, [0.0] * 100)

    text = write_report(tmp_path).read_text()

    assert "run predates provenance" in text
    assert "never" in text  # unsolved run still rendered


def test_write_report_needs_a_run(tmp_path):
    assert write_report(tmp_path) is None


@pytest.fixture
def sweep(tmp_path):
    write_csv(tmp_path / "arm_a" / "seed0" / "training_log.csv", [0.0, 10.0, 20.0])
    write_csv(tmp_path / "arm_a" / "seed1" / "training_log.csv", [10.0, 10.0, 10.0])
    write_csv(tmp_path / "arm_b" / "seed0" / "training_log.csv", [-5.0, -5.0, -5.0])
    return tmp_path


def test_load_curves_groups_by_arm_in_seed_order(sweep):
    curves = load_curves(sweep)

    assert sorted(curves) == ["arm_a", "arm_b"]
    assert curves["arm_a"] == [[0.0, 10.0, 20.0], [10.0, 10.0, 10.0]]


def test_load_curves_can_skip_incomplete_runs(sweep):
    curves = load_curves(sweep, skip=[sweep / "arm_a" / "seed0"])

    assert curves["arm_a"] == [[10.0, 10.0, 10.0]]


def test_load_curves_ignores_incomplete_runs(sweep):
    # A run that died before writing its log simply has no curve.
    (sweep / "arm_c" / "seed0").mkdir(parents=True)

    assert "arm_c" not in load_curves(sweep)


def test_rolling_mean_keeps_the_curve_centred():
    out = rolling_mean([0.0, 0.0, 3.0, 3.0, 3.0], window=3)

    assert out.tolist() == [1.0, 2.0, 3.0]


def test_rolling_mean_is_a_no_op_for_short_curves():
    assert rolling_mean([1.0, 2.0], window=5).tolist() == [1.0, 2.0]


def test_stacked_trims_to_the_shortest_seed(sweep):
    curves = load_curves(sweep)
    curves["arm_a"][1] = [10.0, 10.0]  # one seed stopped early

    data = stacked(curves["arm_a"], window=1)

    assert data.shape == (2, 2)


def test_stacked_returns_an_empty_matrix_without_seeds():
    assert stacked([], window=1).shape == (0, 0)


def test_parse_panels_defaults_to_one_panel_of_everything(sweep):
    curves = load_curves(sweep)

    assert parse_panels(None, curves) == [("", ["arm_a", "arm_b"])]


def test_parse_panels_splits_titles_and_arms(sweep):
    curves = load_curves(sweep)

    panels = parse_panels(["big:arm_b", "small: arm_a , arm_c"], curves)

    # Unknown arms are dropped, known ones keep the requested order.
    assert panels == [("big", ["arm_b"]), ("small", ["arm_a"])]


def test_parse_panels_rejects_a_missing_arm_list(sweep):
    with pytest.raises(SystemExit, match="must look like"):
        parse_panels(["only a title"], load_curves(sweep))


def test_summary_table_reports_tail_mean_and_spread(sweep):
    rows = summary_table(load_curves(sweep))

    assert rows[0][0] == "arm_a"
    assert rows[0][1] == 2
    assert rows[0][2] == pytest.approx(np.mean([0.0, 10.0, 20.0, 10.0, 10.0, 10.0]))
    assert rows[1][2] == pytest.approx(-5.0)
