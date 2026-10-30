import os
import time
import argparse
import tempfile
from datetime import datetime

from config import AgentConfig
from utils.builder import build_agent
from utils.train import train_loop
from utils.test import test_loop
from utils.logger import logger, setup_logger, Dashboard
from utils.provenance import collect as collect_provenance
from utils.seed import resolve_seed, set_global_seed
from utils.experiment import paths

# Files a crashed run may have left behind. The config snapshot is never one of
# them: it is written last, atomically, and marks the run as completed.
RECORD_FILES = ("training_log.csv",)
ARTIFACT_FILES = ("model.ckpt", "training_log.png", "train.log")


def prepare_run_dirs(record_dir, artifact_dir):
    """Clear a partial run, or refuse to overwrite a completed one.

    The record directory holds the config snapshot, so its presence is what makes
    a run complete. Only the known per-run files are removed: ``--save-dir`` may
    share a directory with unrelated files. Both directories are kept, since
    ``artifact_for`` may point at a directory that already holds other runs.
    """
    if os.path.exists(os.path.join(record_dir, "config.toml")):
        raise SystemExit(
            f"error: {record_dir} already holds a completed run (config.toml). "
            "Delete it or choose another --save-dir."
        )

    os.makedirs(record_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)
    for directory, names in ((record_dir, RECORD_FILES), (artifact_dir, ARTIFACT_FILES)):
        for name in names:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                logger.warning(f"Removing leftover from an incomplete run: {path}")
                os.remove(path)


def get_base_parser():
    """Shared arguments for both train and test modes."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dry', action='store_true', help='Save to system temp directory')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed. Omitted: a fresh seed is drawn and recorded in config.',
    )
    return parser

def run_train(args):
    # 1. Load config and setup path
    config = AgentConfig.load(args.config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed = resolve_seed(args.seed)

    if args.save_dir:
        save_dir = args.save_dir
    elif args.dry:
        save_dir = tempfile.mkdtemp(prefix="pygym_")
        logger.warning(f"[Dry-run] Results will be stored in: {save_dir}")
    else:
        save_dir = os.path.join("experiments", f"{config.env.id}_{timestamp}")

    # Records are tracked; checkpoints and plots are not, so they go to the
    # mirrored cache tree. Outside experiments/ the two paths coincide.
    artifact_dir = str(paths.artifact_for(save_dir))
    prepare_run_dirs(save_dir, artifact_dir)
    logger.info(f"Record: {save_dir} | Artifacts: {artifact_dir}")

    # 2. Inject Metadata
    config.metadata.update({
        "iterations": args.iterations,
        "episodes": args.episodes,
        "device": args.device,
        "timestamp": timestamp,
        "dry_run": args.dry,
        "seed": seed
    })
    # Provenance: which code, how many threads, which library versions. This is
    # part of the snapshot written when the run completes.
    config.metadata.update(collect_provenance())

    # 3. Build Components
    # Seed before building: weight initialisation consumes torch's global RNG.
    set_global_seed(seed)
    logger.info(f"Seed: {seed}")
    agent = build_agent(config_dict=config.to_dict(), device=args.device)
    dashboard = Dashboard(
        env_name=config.env.id,
        agent_name=agent.__class__.__name__,
        total_iterations=args.iterations,
        save_dir=save_dir,
        artifact_dir=artifact_dir,
        modes=args.record
    )

    # 4. Training Loop
    train_env_kwargs = {**config.env.default, **config.env.training}
    started = time.monotonic()
    try:
        train_loop(
            env_name=config.env.id,
            agent=agent,
            iterations=args.iterations,
            episodes=args.episodes,
            dashboard=dashboard,
            env_kwargs=train_env_kwargs,
            seed=seed,
        )
    except KeyboardInterrupt:
        dashboard.close()
        logger.warning(
            "[bold red]Interrupted. [/bold red]"
            f"Training log available in {save_dir}"
        )
        return

    # Final artifacts packing. The snapshot goes last and atomically: its
    # presence is what tells every other tool that this run completed.
    agent.save_checkpoints(os.path.join(artifact_dir, "model.ckpt"))
    dashboard.close()
    config.metadata["wall_seconds"] = round(time.monotonic() - started, 1)
    config.save(os.path.join(save_dir, "config.toml"))
    logger.info(f"Results archived in: {save_dir} (artifacts: {artifact_dir})")

    # 5. Optional Evaluation
    if args.eval > 0:
        logger.info(f"Starting post-training evaluation ({args.eval} episodes)")
        test_env_kwargs = {**config.env.default, **config.env.testing}
        test_loop(
            env_name=config.env.id,
            agent=agent,
            episodes=args.eval,
            env_kwargs=test_env_kwargs,
            seed=seed + 1,
        )

def run_test(args):
    """Run evaluation from an existing experiment directory."""
    toml_config_path = os.path.join(args.exp_dir, "config.toml")
    json_config_path = os.path.join(args.exp_dir, "config.json")
    ckpt_path = os.path.join(str(paths.artifact_for(args.exp_dir)), "model.ckpt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint for {args.exp_dir} not found (expected {ckpt_path}). "
            "The record is tracked but its artifacts live in the cache tree."
        )

    if os.path.exists(toml_config_path):
        config_path = toml_config_path
    elif os.path.exists(json_config_path):
        config_path = json_config_path
    else:
        raise FileNotFoundError(
            f"Config not found in {args.exp_dir} (expected config.toml, legacy config.json)"
        )

    config = AgentConfig.load(config_path)
    # Older checkpoints predate seed tracking: fall back to a fixed seed so
    # evaluation stays reproducible instead of drifting per run.
    seed = resolve_seed(args.seed) if args.seed is not None else int(
        config.metadata.get("seed", 0)
    )
    set_global_seed(seed)
    logger.info(f"Seed: {seed}")

    agent = build_agent(config_dict=config.to_dict(), device=args.device)
    agent.load_checkpoints(ckpt_path)

    dashboard = Dashboard(
        env_name=config.env.id,
        agent_name=agent.__class__.__name__,
        total_iterations=args.episodes,
        modes=['cli']
    )

    test_env_kwargs = {**config.env.default, **config.env.testing}
    logger.info(f"Testing model from {args.exp_dir} on {config.env.id}")

    try:
        test_loop(
            env_name=config.env.id,
            agent=agent,
            episodes=args.episodes,
            dashboard=dashboard,
            env_kwargs=test_env_kwargs,
            seed=seed,
        )
    finally:
        dashboard.close()

def run_exp(args):
    # TODO: Implement experiment list/archive logic
    raise NotImplementedError("Experiment management mode (TBD)")

def run_report(args):
    from utils.experiment import report
    return report.run(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Py-Gym Framework Manager")
    subparsers = parser.add_subparsers(dest='mode', required=True)
    base_p = get_base_parser()

    # Train Command
    train_p = subparsers.add_parser('train', parents=[base_p])
    train_p.add_argument('config', type=str)
    train_p.add_argument('-T', '--iterations', type=int, default=500)
    train_p.add_argument('-N', '--episodes', type=int, default=10)
    train_p.add_argument('--record', nargs='+', default=['cli', 'live'])
    train_p.add_argument('--eval', type=int, default=0, help='Number of eval episodes after training')
    train_p.add_argument(
        '--save-dir',
        type=str,
        default=None,
        help='Explicit artifact directory, overriding --dry and the timestamped default.',
    )

    # Test Command
    test_p = subparsers.add_parser('test', parents=[base_p])
    test_p.add_argument('exp_dir', type=str)
    test_p.add_argument('-N', '--episodes', type=int, default=5)

    # Exp Command
    exp_p = subparsers.add_parser('exp', help='Manage saved experiments')
    exp_p.add_argument('action', choices=['archive', 'list'])

    # Report Command
    report_p = subparsers.add_parser('report', help='Regenerate a sweep report')
    report_p.add_argument('--sweep', required=True, help='Sweep name or directory')
    report_p.add_argument('--out', default=None, help='Output PNG (default: <sweep>/comparison.png)')
    report_p.add_argument('--panel', action='append', default=None, help="'Title:arm1,arm2'")
    report_p.add_argument('--smooth', type=int, default=20)
    report_p.add_argument('--threshold', type=float, default=200.0)
    report_p.add_argument('--title', default=None)
    report_p.add_argument('--no-plot', action='store_true', help='Only rewrite REPORT.md')

    args = parser.parse_args()

    level = 'INFO'
    if getattr(args, 'debug', False):
        level = 'DEBUG'
        args.dry = True
    logger = setup_logger(level=level)

    modes = {
        'train': run_train,
        'test': run_test,
        'exp': run_exp,
        'report': run_report,
    }
    raise SystemExit(modes[args.mode](args) or 0)
