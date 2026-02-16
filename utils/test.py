import gymnasium as gym
from agent import BaseAgent

def test_loop(
        env_name: str,
        agent: BaseAgent,
        episodes: int = 5,
        max_steps: int = 1000,
        env_kwargs: dict = None
):
    kwargs = env_kwargs or {}
    env = gym.make(env_name, **kwargs)

    for _ in range(episodes):
        state, _ = env.reset()
        for _ in range(max_steps):
            action, _ = agent.select_action(state, deterministic=True)
            state, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                break

    env.close()
