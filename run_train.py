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
        monitor_modes: list = ['cli', 'live', 'file'],
) -> None:
    # 1. Load config and inject runtime parameters
    config = AgentConfig.load(config_path)
    config_dict = asdict(config)
    env_configs = config_dict.get('env_configs', {})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config.metadata = {
        "iterations": iterations,
        "episodes": episodes,
        "device": device,
        "timestamp": timestamp
    }

    # 2. Prepare experiment directory
    exp_name = f"{config.env_id}_{timestamp}"
    save_dir = os.path.join("experiments", exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # 3. Build Agent (Pass config dict to builder)
    agent = build_agent(
        config_dict=asdict(config),
        device=device
    )

    # 4. Setup Monitor with the specific save_dir
    monitor = Dashboard(
        env_name=config.env_id,
        agent_name=agent.__class__.__name__,
        total_iterations=iterations,
        save_dir=save_dir,
        modes=monitor_modes,
    )

    # 5. Execution
    logger.info(f"Starting experiment: [cyan bold]{exp_name}[/cyan bold]")
    try:
        train_loop(
            env_name=config.env_id,
            agent=agent,
            iterations=iterations,
            episodes=episodes,
            monitor=monitor,
            env_kwargs=env_configs.get('train'),
        )
    except KeyboardInterrupt:
        logger.error("\nTraining interrupted by user. Packing current artifacts...")
    finally:
        # 6. Automatic Packing (Always run even if interrupted)
        agent.save_checkpoints(os.path.join(save_dir, "model.ckpt"))
        config.save(os.path.join(save_dir, "config.json"))
        monitor.close()

        logger.info(f"Experiment artifacts packed into: {save_dir}")

    # 7. Immediate test run
    test_loop(
        env_name=config.env_id,
        agent=agent,
        env_kwargs=env_configs.get('test'),
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and pack RL experiments.")
    parser.add_argument('config', type=str, help='Path to experiment json')
    parser.add_argument('-T', '--iterations', type=int, default=500)
    parser.add_argument('-N', '--episodes', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--monitor', nargs='+', default=['cli', 'live', 'file'])

    args = parser.parse_args()
    main(
        config_path=args.config,
        iterations=args.iterations,
        episodes=args.episodes,
        device=args.device,
        monitor_modes=args.monitor
    )
