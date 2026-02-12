import gymnasium as gym
from agent import BaseAgent

def test_loop(
        env_name: str,
        agent: BaseAgent,
        episodes: int = 5,
        max_steps: int = 1000,
):
    env = gym.make(env_name, render_mode='human')
    
    for _ in range(episodes):
        state, _ = env.reset()
        for _ in range(max_steps):
            env.render()
            
            # Unpack (action, info) based on the new standardized agent interface
            # We only need 'action' for the environment step
            action, info = agent.select_action(state, deterministic=True)

            state, _, terminated, truncated, _ = env.step(action)
            
            if terminated or truncated:
                break
    
    env.close()
