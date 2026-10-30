"""Experiment reports and plots: derived files only.

Everything is rendered from files a run already wrote — its ``config.toml``
snapshot and its ``training_log.csv`` — so redrawing never needs a retrain::

    python manage.py report --sweep critic_ablation_s0-2
    python manage.py report --run experiments/sweeps/<name>/<arm>/seed0

A run counts as complete when its config snapshot is present; directories
without one are listed as incomplete and left out of the plot. Panels default to
one, containing every arm. Pass ``--panel`` (repeatable) to split them, which
keeps a 3-parameter arm from being drowned out by a 70-parameter one::

    --panel 'small tier:01_mlp_small,03_conv_small,05_attn_small' \
    --panel 'mid tier:02_mlp_mid,04_conv_mid,06_attn_mid,07_resmlp_mid'

Per-run plots live in the artifact cache and are rebuilt by the same command,
which is what makes ``artifacts/`` disposable.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from utils.experiment import paths

REWARD_COLUMN = "AverageReward"
THRESHOLD = 200.0
SOLVED_THRESHOLD = 200.0
WINDOW = 100

SUMMARY_FIELDS = [
    "config", "seed", "status", "wall_min", "iterations",
    "last50", "last50_sd", "last100", "last100_sd",
    "best", "best_iter", "solved_iter",
]


def solved_iteration(rewards, window=WINDOW, threshold=SOLVED_THRESHOLD):
    """First iteration whose trailing-window mean reaches the threshold."""
    for i in range(window - 1, len(rewards)):
        if sum(rewards[i - window + 1:i + 1]) / window >= threshold:
            return i
    return None


def summarize(run_dir):
    """Reward summary for one run, empty when the CSV is missing."""
    path = Path(run_dir) / "training_log.csv"
    if not path.exists():
        return {}

    with path.open() as handle:
        rewards = [float(row[REWARD_COLUMN]) for row in csv.DictReader(handle)]
    if not rewards:
        return {}

    result = {"iterations": len(rewards)}
    for window in (50, 100):
        tail = rewards[-window:]
        result[f"last{window}"] = round(statistics.mean(tail), 1)
        result[f"last{window}_sd"] = (
            round(statistics.stdev(tail), 1) if len(tail) > 1 else 0.0
        )

    best = max(rewards)
    result["best"] = round(best, 1)
    result["best_iter"] = rewards.index(best)
    result["solved_iter"] = solved_iteration(rewards)
    return result


def load_snapshot(record):
    """The config snapshot: TOML, or a legacy JSON one.

    Returns ``None`` when the directory holds no snapshot, which is exactly how
    an interrupted run is recognised.
    """
    record = Path(record)
    toml_path = record / "config.toml"
    json_path = record / "config.json"

    try:
        if toml_path.exists():
            with toml_path.open("rb") as handle:
                return tomllib.load(handle)
        if json_path.exists():
            return json.loads(json_path.read_text())
    except (OSError, ValueError):
        return {}
    return None


def load_metadata(run_dir):
    """Metadata of the snapshot, or ``None`` when there is no snapshot."""
    snapshot = load_snapshot(run_dir)
    if snapshot is None:
        return None
    return snapshot.get("metadata", {})


def load_legacy_run(run_dir):
    """Timings and thread counts recorded before the completion contract.

    Records written by the retired sweep runner keep their cost in ``run.json``;
    new runs record ``wall_seconds`` in the config snapshot instead. This is a
    read-only fallback, the same way ``config.json`` is.
    """
    try:
        return json.loads((Path(run_dir) / "run.json").read_text())
    except (OSError, ValueError):
        return {}


def collect_rows(root):
    """One row per completed run, plus the directories that never finished."""
    rows, incomplete = [], []
    for run_dir in sorted(Path(root).glob("*/seed*")):
        metadata = load_metadata(run_dir)
        if metadata is None:
            incomplete.append(str(run_dir))
            continue

        legacy = load_legacy_run(run_dir)
        wall = metadata.get("wall_seconds")
        if wall is None:
            wall = legacy.get("wall_seconds")
        threads = metadata.get("torch_threads")
        if threads is None:
            threads = legacy.get("threads")

        row = {
            "config": run_dir.parent.name,
            "seed": int(run_dir.name.removeprefix("seed")),
            "status": "ok",
            "iterations": metadata.get("iterations", ""),
            "episodes": metadata.get("episodes", ""),
            "threads": threads if threads is not None else "",
            "git_sha": metadata.get("git_sha"),
            "git_dirty": metadata.get("git_dirty", False),
            "python": metadata.get("python"),
            "torch": metadata.get("torch"),
            "numpy": metadata.get("numpy"),
            "wall_min": round(float(wall) / 60, 1) if wall is not None else "",
        }
        row.update(summarize(run_dir))
        rows.append(row)

    rows.sort(key=lambda row: (row["config"], row["seed"]))
    return rows, incomplete


def write_summary(root, rows):
    """Rewrite ``summary.csv`` from the rows; it is derived, never typed."""
    path = Path(root) / "summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _numbers(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        values.append(float(value))
    return values


def _mean_sd(values):
    """``"mean +/- sd"``, or ``"—"`` when there is nothing to report."""
    if not values:
        return "—"
    mean = statistics.mean(values)
    if len(values) < 2:
        return f"{mean:.0f}"
    return f"{mean:.0f} +/- {statistics.stdev(values):.0f}"


def aggregate(rows):
    """Per-config summary across seeds. Unsolved runs are censored, not zero."""
    groups = {}
    for row in rows:
        groups.setdefault(row["config"], []).append(row)

    summary = []
    for config in sorted(groups):
        group = groups[config]
        solved = _numbers(group, "solved_iter")
        summary.append(
            {
                "config": config,
                "seeds": len(group),
                "last100": _mean_sd(_numbers(group, "last100")),
                "solved_at": _mean_sd(solved),
                "solved": f"{len(solved)}/{len(group)}",
                "wall": _mean_sd(_numbers(group, "wall_min")),
            }
        )
    return summary


def _distinct(values):
    seen = [str(value) for value in values if value not in (None, "")]
    return sorted(set(seen))


def _describe_code(rows):
    shas = _distinct([row.get("git_sha") for row in rows])
    if not shas:
        return "not recorded (run predates provenance)"
    dirty = any(row.get("git_dirty") for row in rows)
    label = ", ".join(f"`{sha[:12]}`" for sha in shas)
    return f"{label} (dirty working tree)" if dirty else f"{label} (clean)"


def _describe_versions(rows, key):
    return ", ".join(_distinct([row.get(key) for row in rows])) or "—"


def _wall(value):
    return f"{value} min" if value not in (None, "") else "—"


def render_report(root, rows, incomplete=()):
    """Markdown report built only from recorded data, never from memory."""
    failed = len(incomplete)
    iterations = _distinct([row["iterations"] for row in rows])
    episodes = _distinct([row["episodes"] for row in rows])
    arms = _distinct([row["config"] for row in rows])
    seeds = _distinct([row["seed"] for row in rows])
    total_wall = sum(_numbers(rows, "wall_min"))

    lines = [
        f"# Sweep report: {Path(root).name}",
        "",
        "Generated from each run's `config.toml` and `training_log.csv`.",
        "Regenerate with",
        "",
        "```bash",
        "python manage.py report \\",
        f"  --sweep {Path(root).name}",
        "```",
        "",
        "Do not edit by hand: numbers here and in `summary.csv` must not drift.",
        "",
        "## Setup",
        "",
        "| Setting | Value |",
        "| :--- | :--- |",
        f"| runs | {len(rows) + failed} ({len(arms)} configs, seeds {', '.join(seeds) or '—'}) |",
        f"| complete | {len(rows)} ok, {failed} incomplete |",
        f"| iterations x episodes | {' / '.join(iterations) or '—'} x "
        f"{' / '.join(episodes) or '—'} |",
        f"| torch threads | {_describe_versions(rows, 'threads')} |",
        f"| code | {_describe_code(rows)} |",
        f"| python / torch / numpy | {_describe_versions(rows, 'python')} / "
        f"{_describe_versions(rows, 'torch')} / "
        f"{_describe_versions(rows, 'numpy')} |",
        f"| sum of run wall | {total_wall:.0f} min |",
        "",
        "## Per arm",
        "",
        "| arm | seeds | last-100 | solved at | solved | wall |",
        "| :--- | ---: | :--- | :--- | ---: | ---: |",
    ]
    for arm in aggregate(rows):
        wall = "—" if arm["wall"] == "—" else f"{arm['wall']} min"
        lines.append(
            f"| `{arm['config']}` | {arm['seeds']} | {arm['last100']} | "
            f"{arm['solved_at']} | {arm['solved']} | {wall} |"
        )

    lines += [
        "",
        "`solved at` is the first iteration whose trailing-100 mean reaches 200;",
        "runs that never get there are excluded from that column and counted in",
        "`solved` as unsolved. Values are mean +/- sd across seeds.",
        "",
        "## Per run",
        "",
        "| arm | seed | last-50 | last-100 | solved at | best | wall | git |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for row in rows:
        sha = (row.get("git_sha") or "—")[:12]
        if row.get("git_dirty"):
            sha += "*"
        solved = row.get("solved_iter") or "never"
        lines.append(
            f"| `{row['config']}` | {row['seed']} | "
            f"{row.get('last50', '—')} | {row.get('last100', '—')} | {solved} | "
            f"{row.get('best', '—')} | {_wall(row.get('wall_min'))} | `{sha}` |"
        )

    lines += ["", "## Incomplete", ""]
    if incomplete:
        lines += ["| directory |", "| :--- |"]
        for run_dir in incomplete:
            try:
                shown = Path(run_dir).relative_to(root)
            except ValueError:
                shown = run_dir
            lines.append(f"| `{shown}` |")
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def _write_derived(root, rows, incomplete):
    if not rows and not incomplete:
        return None

    if rows:
        write_summary(root, rows)
    path = Path(root) / "REPORT.md"
    path.write_text(render_report(root, rows, incomplete))
    return path


def write_report(root):
    """Rewrite summary.csv and REPORT.md. Returns the report path, or None."""
    rows, incomplete = collect_rows(root)
    return _write_derived(root, rows, incomplete)


RUN_PLOT = "training_log.png"


def load_run(record):
    """``(iterations, rewards, metrics)`` from one run's CSV, or ``None``.

    Metrics are whatever columns the run recorded, so the plot keeps showing
    every series without this module hard-coding PPO's names.
    """
    path = Path(record) / "training_log.csv"
    try:
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return None
    if not rows:
        return None

    iterations = [int(row["Iteration"]) for row in rows]
    rewards = [float(row[REWARD_COLUMN]) for row in rows]

    metrics = {}
    for name in rows[0]:
        if name in ("Iteration", REWARD_COLUMN):
            continue
        try:
            metrics[name] = [float(row[name]) for row in rows]
        except (TypeError, ValueError):
            continue  # a ragged column is not worth failing the plot for
    return iterations, rewards, metrics


def centered_sma(values, window):
    """Centred moving average, the same smoothing the in-memory plotter used."""
    values = np.asarray(values, dtype=float)
    half = max(0, int(window) // 2)
    return np.array([
        values[max(0, i - half):min(len(values), i + half + 1)].mean()
        for i in range(len(values))
    ])


def run_label(record):
    """``env (agent)`` for the title, falling back to the directory name."""
    snapshot = load_snapshot(record) or {}
    env = snapshot.get("env", {}).get("id")
    agent = snapshot.get("agent", {}).get("type")
    if env and agent:
        return f"{env} ({agent})"
    return env or Path(record).name


def plot_run(record, out=None):
    """Redraw one run's training curve from its CSV.

    The PNG is derived, so it belongs in the artifact cache next to the
    checkpoint: deleting ``artifacts/`` loses a picture, never a number. The
    default destination is the mirrored path, which keeps a dry run or an
    external ``--save-dir`` self-contained. Returns the path, or ``None`` when
    the run has no log.
    """
    data = load_run(record)
    if data is None:
        return None

    iterations, rewards, metrics = data
    path = Path(out) if out else paths.artifact_for(record) / RUN_PLOT

    # Imported here so report stays light for callers that only read rows.
    from utils.logger.plot_config import set_plot_style

    set_plot_style()
    fig, ax1 = plt.subplots(figsize=(12, 7))
    try:
        ax2 = ax1.twinx()

        colors = ["tab:orange", "tab:red", "tab:green", "tab:brown"]
        for i, (name, values) in enumerate(metrics.items()):
            ax2.plot(
                iterations, values, color=colors[i % 4], linestyle="--",
                alpha=0.4, label=name.capitalize(),
            )

        trend = centered_sma(rewards, max(2, len(rewards) // 50))
        ax1.plot(
            iterations, rewards, color="tab:purple", alpha=0.15, zorder=2,
            label="Raw Reward",
        )
        ax1.plot(
            iterations, trend, color="tab:blue", linewidth=2.5, zorder=3,
            label="Trend (SMA)",
        )

        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)
        ax1.set_xlabel("Iteration", fontweight="bold")
        ax1.set_ylabel("Reward", color="tab:blue", fontweight="bold")
        ax2.set_ylabel("Metrics", color="tab:gray", fontweight="bold")
        ax1.set_title(f"Final Training Summary: {run_label(record)}", pad=20)

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=300)
    finally:
        plt.close(fig)
        plt.rcdefaults()  # the run plot is xkcd-styled; do not leak it
    return path


def refresh_run_plots(root, incomplete=()):
    """Redraw every completed run's PNG; returns how many were written."""
    skipped = {Path(item).resolve() for item in incomplete}
    written = 0
    for run_dir in sorted(Path(root).glob("*/seed*")):
        if run_dir.resolve() in skipped:
            continue
        if plot_run(run_dir) is not None:
            written += 1
    return written


def load_curves(root, skip=()):
    """Map arm name -> list of per-seed reward sequences, sorted by seed."""
    skip = {Path(path) for path in skip}
    curves = {}
    for log in sorted(Path(root).glob("*/seed*/training_log.csv")):
        if log.parent in skip:
            continue
        arm = log.parent.parent.name
        with log.open() as handle:
            rewards = [float(row[REWARD_COLUMN]) for row in csv.DictReader(handle)]
        curves.setdefault(arm, []).append(rewards)
    return curves


def rolling_mean(values, window):
    """Centred moving average, so the x axis still lines up with iterations."""
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def stacked(curves, window):
    """Seeds x iterations matrix, trimmed to the shortest run in the arm."""
    runs = [rolling_mean(run, window) for run in curves]
    if not runs:
        return np.empty((0, 0))
    length = min(len(run) for run in runs)
    return np.vstack([run[:length] for run in runs])


def parse_panels(specs, curves):
    """``['Title:a,b', ...]`` -> ``[(title, [existing arms])]``.

    Arms that do not appear in the sweep are dropped with a warning rather than
    silently producing an empty panel.
    """
    if not specs:
        return [("", sorted(curves))]

    panels = []
    for spec in specs:
        title, _, arm_list = spec.partition(":")
        if not arm_list:
            raise SystemExit(f"error: panel {spec!r} must look like 'Title:arm1,arm2'")
        wanted = [arm.strip() for arm in arm_list.split(",") if arm.strip()]
        missing = [arm for arm in wanted if arm not in curves]
        if missing:
            print(f"warning: {title or 'panel'} has no data for {missing}", file=sys.stderr)
        present = [arm for arm in wanted if arm in curves]
        if present:
            panels.append((title.strip(), present))
    return panels


def plot_sweep(root, curves, panels, window=20, threshold=THRESHOLD, title=None):
    """Render one subplot per panel and return the figure."""
    count = len(panels)
    fig, axes = plt.subplots(
        1, count, figsize=(6.2 * count, 5.2), sharey=True, squeeze=False
    )
    axes = axes[0]

    for ax, (panel_title, arms) in zip(axes, panels):
        for arm in arms:
            data = stacked(curves[arm], window)
            mean, sd = data.mean(axis=0), data.std(axis=0)
            x = np.arange(len(mean)) + window - 1
            line, = ax.plot(x, mean, label=f"{arm} (n={len(data)})")
            if len(data) > 1:
                ax.fill_between(x, mean - sd, mean + sd, alpha=0.15, color=line.get_color())
        if threshold is not None:
            ax.axhline(threshold, ls="--", c="gray", lw=1, label="solved")
        if panel_title:
            ax.set_title(panel_title, fontsize=11)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")

    axes[0].set_ylabel(f"episode reward (smoothed w={window})")
    fig.suptitle(title or f"{Path(root).name}", fontsize=12)
    fig.tight_layout()
    return fig


def summary_table(curves):
    """Final rolling-100 per arm, for printing next to the figure."""
    rows = []
    for arm, seeds in sorted(curves.items()):
        tail = [float(np.mean(run[-100:])) for run in seeds]
        rows.append((arm, len(seeds), float(np.mean(tail)), float(np.std(tail))))
    return rows


def resolve_sweep(value):
    """Accept either a sweep directory or a name under ``experiments/sweeps``."""
    path = Path(value)
    return path if path.is_dir() else paths.sweep_dir(value)


def render(root, args):
    """Rewrite the sweep's derived artifacts; returns an exit code."""
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"error: no sweep at {root}")

    rows, incomplete = collect_rows(root)
    if not rows and not incomplete:
        raise SystemExit(f"error: no completed runs under {root}")

    if rows:
        print(f"summary: {write_summary(root, rows)}")
    report = _write_derived(root, rows, incomplete)
    if report:
        print(f"report: {report}")

    if args.no_plot:
        return 0

    curves = load_curves(root, skip=incomplete)
    if not curves:
        raise SystemExit(f"error: no completed training_log.csv under {root}")

    panels = parse_panels(args.panel, curves)
    fig = plot_sweep(root, curves, panels, args.smooth, args.threshold, args.title)
    out = Path(args.out) if args.out else root / "comparison.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")

    if not args.no_run_plots:
        print(f"run plots: {refresh_run_plots(root, incomplete)} refreshed")

    for arm, seeds, mean, sd in summary_table(curves):
        print(f"  {arm:20s} n={seeds} last-100 {mean:6.1f} +/- {sd:5.1f}")
    return 0


def run(args):
    """Entry point for ``manage.py report``; returns an exit code."""
    if args.run:
        path = plot_run(args.run)
        if path is None:
            raise SystemExit(f"error: no training_log.csv under {args.run}")
        print(f"wrote {path}")
        return 0
    return render(resolve_sweep(args.sweep), args)
