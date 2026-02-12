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
    
    # 1. Load the blueprint and weights from the specific experiment
    config_path = f"{exp_dir}/config.json"
    ckpt_path = f"{exp_dir}/model.ckpt"
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # 2. Re-construct the exact same agent
    agent = build_agent(config_dict=config_dict, device=device)
    
    # 3. Load trained weights
    agent.load_checkpoints(ckpt_path)
    
    # 4. Execute test
    logger.info(f"Testing model from {exp_dir} on {config_dict['env_id']}...")
    test_loop(
        env_name=config_dict['env_id'], 
        agent=agent, 
        episodes=episodes
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str, help='Path to the experiment folder')
    parser.add_argument('--episodes', type=int, default=5)
    
    args = parser.parse_args()
    main(exp_dir=args.exp_dir, episodes=args.episodes)
