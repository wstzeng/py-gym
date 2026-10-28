import gymnasium as gym
from agent import BaseAgent
from utils.logger import Dashboard
from utils.seed import episode_seed

def test_loop(
        env_name: str,
        agent: BaseAgent,
        episodes: int = 5,
        dashboard: Dashboard = None,
        max_steps: int = 1000,
        env_kwargs: dict = None,
        seed: int = None
):
    kwargs = env_kwargs or {}
    env = gym.make(env_name, **kwargs)
    agent.eval()

    for ep in range(episodes):
        reset_kwargs = {"seed": episode_seed(seed, ep)} if seed is not None else {}
        state, _ = env.reset(**reset_kwargs)
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
