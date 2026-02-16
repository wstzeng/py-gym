import os
import json
import argparse
from utils.builder import build_agent
from utils.test import test_loop
from utils.logger import logger

def main(
        exp_dir: str,
        episodes: int = 5,
        device: str = "cpu"
) -> None:
    """
    Run testing from a packed experiment directory.
    """
    # 1. Path resolution
    config_path = os.path.join(exp_dir, "config.json")
    ckpt_path = os.path.join(exp_dir, "model.ckpt")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found in {exp_dir}")

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # 2. Re-construct agent and load weights
    agent = build_agent(config_dict=config_dict, device=device)
    agent.load_checkpoints(ckpt_path)

    # 3. Extract environment settings from new structure
    env_cfg = config_dict.get('env', {})
    env_id = env_cfg.get('id')

    # Merge default settings with testing-specific settings
    default_kwargs = env_cfg.get('default', {})
    testing_kwargs = env_cfg.get('testing', {})
    test_env_kwargs = {**default_kwargs, **testing_kwargs}

    # 4. Execute test
    logger.info(f"Testing model from {exp_dir} on {env_id}...")
    test_loop(
        env_name=env_id,
        agent=agent,
        episodes=episodes,
        env_kwargs=test_env_kwargs,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a trained agent from an experiment directory.")
    parser.add_argument('exp_dir', type=str, help='Path to the experiment folder')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()
    main(
        exp_dir=args.exp_dir,
        episodes=args.episodes,
        device=args.device
    )
