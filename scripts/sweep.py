"""Bounded-parallel sweeps over (config, seed) pairs.

Run from the repository root:

    poetry run python scripts/sweep.py \
        --configs 'config/ablation/lunarlander/*.toml' \
        --seeds 0 1 2 -T 600 -N 20 --jobs 4 --threads 4

Every pair gets its own directory, ``experiments/sweeps/<name>/<stem>/seed<N>/``,
passed to ``manage.py`` via ``--save-dir``. That sidesteps the timestamped
default path, which two runs starting in the same second would otherwise share.

Finished runs are skipped when the sweep is re-invoked, so a long sweep is
resumable; pass ``--force`` to redo them.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import queue
import shutil
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_ROOT = Path("experiments") / "sweeps"
SOLVED_THRESHOLD = 200.0
WINDOW = 100

_ACTIVE = set()
_ACTIVE_LOCK = threading.Lock()


def expand_configs(patterns):
    """Glob every pattern into a de-duplicated, order-preserving config list."""
    configs = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise SystemExit(f"error: no config matched {pattern!r}")
        configs.extend(matches)
    return list(dict.fromkeys(configs))


def plan_runs(configs, seeds):
    """Build the (config, seed) matrix, rejecting accidental duplicates."""
    runs = []
    seen = set()
    for config in configs:
        for seed in seeds:
            key = (config, int(seed))
            if key in seen:
                raise SystemExit(f"error: duplicate run: {config} seed={seed}")
            seen.add(key)
            runs.append({"config": config, "seed": int(seed), "stem": Path(config).stem})
    return runs


def core_slices(jobs, threads, nproc=None):
    """Disjoint ``taskset`` CPU lists, or ``None`` when oversubscribed."""
    nproc = nproc or os.cpu_count() or 1
    if threads <= 0 or jobs * threads > nproc:
        return [None] * jobs
    return [f"{i * threads}-{i * threads + threads - 1}" for i in range(jobs)]


def run_dir_for(root, run):
    return Path(root) / run["stem"] / f"seed{run['seed']}"


def is_complete(run_dir):
    marker = Path(run_dir) / "run.json"
    if not marker.exists():
        return False
    try:
        return json.loads(marker.read_text()).get("returncode") == 0
    except json.JSONDecodeError:
        return False


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
        rewards = [float(row["AverageReward"]) for row in csv.DictReader(handle)]
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


def build_command(run, args, out_dir, cpus):
    cmd = [
        args.python, "-B", "manage.py", "train", run["config"],
        "-T", str(args.iterations),
        "-N", str(args.episodes),
        "--record", *args.record,
        "--seed", str(run["seed"]),
        "--save-dir", str(out_dir),
    ]
    return ["taskset", "-c", cpus] + cmd if cpus else cmd


def _terminate_active():
    with _ACTIVE_LOCK:
        active = list(_ACTIVE)
    for proc in active:
        if proc.poll() is None:
            proc.terminate()


def execute(run, args, root, cpus, progress):
    out_dir = run_dir_for(root, run)
    record = {
        "config": run["config"],
        "seed": run["seed"],
        "directory": str(out_dir),
        "cpus": cpus,
    }

    if is_complete(out_dir) and not args.force:
        record.update({"returncode": 0, "wall_seconds": 0.0, "skipped": True})
        progress(record)
        return record

    if out_dir.exists() and args.force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_command(run, args, out_dir, cpus)
    env = dict(os.environ)
    if args.threads > 0:
        env["OMP_NUM_THREADS"] = str(args.threads)
        env["MKL_NUM_THREADS"] = str(args.threads)

    started_at = datetime.now().isoformat(timespec="seconds")
    started = time.monotonic()
    with (out_dir / "train.log").open("w") as log:
        proc = subprocess.Popen(
            cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        with _ACTIVE_LOCK:
            _ACTIVE.add(proc)
        try:
            returncode = proc.wait()
        finally:
            with _ACTIVE_LOCK:
                _ACTIVE.discard(proc)

    record.update({
        "returncode": returncode,
        "wall_seconds": round(time.monotonic() - started, 1),
        "started": started_at,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "command": cmd,
        "threads": args.threads,
        "iterations": args.iterations,
        "episodes": args.episodes,
        "skipped": False,
    })
    (out_dir / "run.json").write_text(json.dumps(record, indent=2))

    progress(record)
    return record


def write_summary(root, records):
    rows = []
    for record in records:
        row = {
            "config": Path(record["config"]).stem,
            "seed": record["seed"],
            "rc": record["returncode"],
            "wall_min": round(record["wall_seconds"] / 60, 1),
        }
        row.update(summarize(record["directory"]))
        rows.append(row)

    # Completed order is scheduler-dependent; sort so reruns diff cleanly.
    rows.sort(key=lambda row: (row["config"], row["seed"]))

    path = Path(root) / "summary.csv"
    if rows:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return path, rows


def print_summary(rows):
    header = (f"{'config':22s} {'seed':>4s} {'rc':>3s} {'wall':>7s} "
              f"{'last50':>7s} {'last100':>8s} {'solved@':>8s} {'best':>7s}")
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        solved = row.get("solved_iter")
        print(
            f"{row['config']:22s} {row['seed']:4d} {row['rc']:3d} "
            f"{row['wall_min']:6.1f}m {row.get('last50', float('nan')):7.1f} "
            f"{row.get('last100', float('nan')):8.1f} "
            f"{('never' if solved is None else solved):>8} "
            f"{row.get('best', float('nan')):7.1f}"
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a config x seed sweep")
    parser.add_argument("--configs", nargs="+", required=True, help="Glob patterns")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("-T", "--iterations", type=int, default=600)
    parser.add_argument("-N", "--episodes", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=4, help="Concurrent runs")
    parser.add_argument("--threads", type=int, default=4, help="Torch threads per run")
    parser.add_argument("--record", nargs="+", default=["cli"])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--name", default=None, help="Sweep directory name")
    parser.add_argument("--force", action="store_true", help="Redo completed runs")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.chdir(REPO_ROOT)

    configs = expand_configs(args.configs)
    runs = plan_runs(configs, args.seeds)
    root = SWEEP_ROOT / (args.name or datetime.now().strftime("sweep_%Y%m%d_%H%M%S"))

    slices = core_slices(args.jobs, args.threads)
    print(f"sweep: {len(runs)} runs ({len(configs)} configs x {len(args.seeds)} seeds)")
    print(f"root:  {root}")
    print(
        f"jobs:  {args.jobs} x {args.threads} threads | "
        f"cpus: {', '.join(s for s in slices if s) or 'unpinned'}"
    )

    if args.plan_only:
        for run in runs:
            print(f"  {run['stem']:24s} seed={run['seed']} -> {run_dir_for(root, run)}")
        return 0

    root.mkdir(parents=True, exist_ok=True)
    slot_pool = queue.Queue()
    for cpus in slices:
        slot_pool.put(cpus)

    records = []
    lock = threading.Lock()
    finished = 0

    def progress(record):
        nonlocal finished
        with lock:
            finished += 1
            parts = summarize(record["directory"])
            status = "skip" if record.get("skipped") else f"rc={record['returncode']}"
            tail = (
                f"last100={parts['last100']:.1f}" if "last100" in parts else "no csv"
            )
            print(
                f"[{finished}/{len(runs)}] {Path(record['config']).stem:22s} "
                f"seed={record['seed']} {status:6s} "
                f"{record['wall_seconds'] / 60:6.1f}m {tail}",
                flush=True,
            )

    def task(run):
        cpus = slot_pool.get()
        try:
            return execute(run, args, root, cpus, progress)
        finally:
            slot_pool.put(cpus)

    started = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(task, run) for run in runs]
            for future in as_completed(futures):
                records.append(future.result())
    except KeyboardInterrupt:
        _terminate_active()
        print("\ninterrupted: terminated running jobs", file=sys.stderr)
        return 130

    path, rows = write_summary(root, records)
    print_summary(rows)
    print(f"\ntotal wall: {(time.monotonic() - started) / 60:.1f} min")
    print(f"summary:    {path}")
    failed = [row for row in rows if row["rc"] != 0]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
