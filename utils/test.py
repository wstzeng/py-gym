import gymnasium as gym
from agent import BaseAgent
from utils.logger import Dashboard

def test_loop(
        env_name: str,
        agent: BaseAgent,
        episodes: int = 5,
        dashboard: Dashboard = None,
        max_steps: int = 1000,
        env_kwargs: dict = None
):
    kwargs = env_kwargs or {}
    env = gym.make(env_name, **kwargs)
    agent.eval()

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(max_steps):
            action, _ = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        if dashboard:
            dashboard.update(ep, total_reward)

    env.close()
