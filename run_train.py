import os
import argparse
from datetime import datetime
from dataclasses import asdict

from utils.builder import build_agent
from utils.train import train_loop
from utils.test import test_loop
from utils.config import AgentConfig
from utils.logger import logger, Dashboard

def main(
        config_path: str,
        iterations: int,
        episodes: int,
        device: str = "cpu",
        record_modes: list = ['cli'],
) -> None:
    # 1. Load config and inject runtime parameters
    config = AgentConfig.load(config_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config.metadata = {
        "iterations": iterations,
        "episodes": episodes,
        "device": device,
        "timestamp": timestamp
    }

    # 2. Prepare experiment directory
    exp_name = f"{config.env.id}_{timestamp}"
    save_dir = os.path.join("experiments", exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # 3. Build Agent (Pass config dict to builder)
    agent = build_agent(
        config_dict=config.to_dict(),
        device=device
    )

    # 4. Setup Monitor
    monitor = Dashboard(
        env_name=config.env.id,
        agent_name=agent.__class__.__name__,
        total_iterations=iterations,
        save_dir=save_dir,
        modes=record_modes,
    )

    # 5. Environment Kwargs Merging
    # Combine default env settings with specific training/testing settings
    train_env_kwargs = {**config.env.default, **config.env.training}
    test_env_kwargs = {**config.env.default, **config.env.testing}

    # 6. Execution
    logger.info(f"Starting experiment: [cyan bold]{exp_name}[/cyan bold]")
    try:
        train_loop(
            env_name=config.env.id,
            agent=agent,
            iterations=iterations,
            episodes=episodes,
            monitor=monitor,
            env_kwargs=train_env_kwargs,
        )
    except KeyboardInterrupt:
        logger.error("\nTraining interrupted by user. Packing current artifacts...")
    finally:
        # 7. Save Checkpoints & Config
        agent.save_checkpoints(os.path.join(save_dir, "model.ckpt"))
        config.save(os.path.join(save_dir, "config.json"))
        monitor.close()
        logger.info(f"Experiment artifacts packed into: {save_dir}")

    # 8. Immediate test run
    test_loop(
        env_name=config.env.id,
        agent=agent,
        env_kwargs=test_env_kwargs,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and pack RL experiments.")
    parser.add_argument('config', type=str, help='Path to experiment json')
    parser.add_argument('-T', '--iterations', type=int, default=500)
    parser.add_argument('-N', '--episodes', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--record', nargs='+', default=['cli', 'live', 'file'])

    args = parser.parse_args()
    main(
        config_path=args.config,
        iterations=args.iterations,
        episodes=args.episodes,
        device=args.device,
        record_modes=args.record
    )
