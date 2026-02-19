import os
import argparse
import tempfile
from datetime import datetime

from config import AgentConfig
from utils.builder import build_agent
from utils.train import train_loop
from utils.test import test_loop
from utils.logger import logger, Dashboard

def get_base_parser():
    """Shared arguments for both train and test modes."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dry', action='store_true', help='Save to system temp directory')
    return parser

def run_train(args):
    # 1. Load config and setup path
    config = AgentConfig.load(args.config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.dry:
        save_dir = tempfile.mkdtemp(prefix="pygym_")
        logger.warning(f"[Dry-run] Artifacts will be stored in: {save_dir}")
    else:
        save_dir = os.path.join("experiments", f"{config.env.id}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

    # 2. Inject Metadata
    config.metadata.update({
        "iterations": args.iterations,
        "episodes": args.episodes,
        "device": args.device,
        "timestamp": timestamp,
        "dry_run": args.dry
    })

    # 3. Build Components
    agent = build_agent(config_dict=config.to_dict(), device=args.device)
    dashboard = Dashboard(
        env_name=config.env.id,
        agent_name=agent.__class__.__name__,
        total_iterations=args.iterations,
        save_dir=save_dir,
        modes=args.record
    )

    # 4. Training Loop
    train_env_kwargs = {**config.env.default, **config.env.training}
    try:
        train_loop(
            env_name=config.env.id,
            agent=agent,
            iterations=args.iterations,
            episodes=args.episodes,
            dashboard=dashboard,
            env_kwargs=train_env_kwargs,
        )
    except KeyboardInterrupt:
        logger.warning(f"Interuptted. Training log available in {save_dir}")
        dashboard.close()
        return

    # Final artifacts packing
    agent.save_checkpoints(os.path.join(save_dir, "model.ckpt"))
    config.save(os.path.join(save_dir, "config.json"))
    config.save(os.path.join(save_dir, "config.toml"))
    dashboard.close()
    logger.info(f"Results archived in: {save_dir}")

    # 5. Optional Evaluation
    if args.eval > 0:
        logger.info(f"Starting post-training evaluation ({args.eval} episodes)")
        test_env_kwargs = {**config.env.default, **config.env.testing}
        test_loop(
            env_name=config.env.id,
            agent=agent,
            episodes=args.eval,
            env_kwargs=test_env_kwargs,
        )

def run_test(args):
    """Run evaluation from an existing experiment directory."""
    toml_config_path = os.path.join(args.exp_dir, "config.toml")
    json_config_path = os.path.join(args.exp_dir, "config.json")
    ckpt_path = os.path.join(args.exp_dir, "model.ckpt")

    if os.path.exists(toml_config_path):
        config_path = toml_config_path
    elif os.path.exists(json_config_path):
        config_path = json_config_path
    else:
        raise FileNotFoundError(f"Config not found in {args.exp_dir}")

    config = AgentConfig.load(config_path)
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
        )
    finally:
        dashboard.close()

def run_exp(args):
    # TODO: Implement experiment list/archive logic
    raise NotImplementedError("Experiment management mode (TBD)")

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

    # Test Command
    test_p = subparsers.add_parser('test', parents=[base_p])
    test_p.add_argument('exp_dir', type=str)
    test_p.add_argument('-N', '--episodes', type=int, default=5)

    # Exp Command
    exp_p = subparsers.add_parser('exp', help='Manage saved experiments')
    exp_p.add_argument('action', choices=['archive', 'list'])

    args = parser.parse_args()

    modes = {
        'train': run_train,
        'test': run_test,
        'exp': run_exp
    }
    modes[args.mode](args)
