import gymnasium as gym
from agent import BaseAgent
from utils.logger import Dashboard
from utils.seed import episode_seed

def train_loop(
        env_name: str,
        agent: BaseAgent,
        iterations: int,
        episodes: int,
        dashboard: Dashboard = None,
        env_kwargs: dict = None,
        seed: int = None
):
    kwargs = env_kwargs or {}
    env = gym.make(env_name, **kwargs)
    agent.train()

    for t in range(iterations):
        total_rewards = []

        for i in range(episodes):
            episode = t * episodes + i
            reset_kwargs = (
                {"seed": episode_seed(seed, episode)} if seed is not None else {}
            )
            state, _ = env.reset(**reset_kwargs)
            done = False
            total_reward = 0

            while not done:
                action, info = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.record(info, reward, done)

                state = next_state
                total_reward += reward

            total_rewards.append(total_reward)

        loss_metrics = agent.update()

        avg_reward = sum(total_rewards) / episodes

        if dashboard:
            dashboard.update(t, avg_reward, **loss_metrics)

    env.close()
