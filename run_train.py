import os
import json
import argparse
from datetime import datetime
from utils.builder import build_agent
from utils.train import train_loop
from utils.test import test_loop

def main(
        config_path: str,
        iterations: int,
        episodes: int,
        device: str = "cpu"
) -> None:
    """
    Unified entry point for running experiments with automatic artifact packing.
    """
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    env_name = config_dict.get('env_id')
    if not env_name:
        raise ValueError("The config file must specify an 'env_id'.")

    # 1. Prepare unique experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{env_name}_{timestamp}"
    save_dir = os.path.join("experiments", exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # 2. Build Agent
    agent = build_agent(
        config_dict=config_dict,
        device=device
    )

    # 3. Execution
    print(f"Starting experiment: {exp_name}")
    train_loop(
        env_name=env_name,
        agent=agent,
        iterations=iterations,
        episodes=episodes,
        monitor_mode=['cli', 'live', 'file']
    )

    # 4. Automatic Packing
    # Save model weights
    agent.save_checkpoints(os.path.join(save_dir, "model.ckpt"))
    
    # Save a copy of the config for reproducibility
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(config_dict, f, indent=4)
        
    print(f"Experiment artifacts packed into: {save_dir}")

    # 5. Immediate test run
    test_loop(env_name=env_name, agent=agent)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and pack RL experiments.")
    parser.add_argument('config', type=str, help='Path to experiment json')
    parser.add_argument(
        '-T', '--iterations',
        type=int, default=500, help='Training iterations'
    )
    parser.add_argument(
        '-N', '--episodes',
        type=int, default=10, help='Evaluation window size'
    )
    parser.add_argument(
        '--device',
        type=str, default='cpu', help='Device (cpu/cuda)'
    )
    parser.add_argument(
        '--monitor', 
        nargs='+', 
        default=['cli', 'live', 'file'],
        help='Monitoring modes'
    )
    
    args = parser.parse_args()
    main(
        config_path=args.config, 
        iterations=args.iterations, 
        episodes=args.episodes, 
        device=args.device,
    )
