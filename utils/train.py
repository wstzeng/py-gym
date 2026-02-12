import gymnasium as gym
from agent import BaseAgent
from utils.logger import Dashboard

def train_loop(
        env_name: str,
        agent: BaseAgent,
        iterations: int,
        episodes: int,
        monitor: Dashboard,
):
    env = gym.make(env_name)
    agent.train()

    for t in range(1, iterations + 1):
        total_rewards = []

        for _ in range(episodes):
            state, _ = env.reset()
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

        # Update Agent
        loss = agent.update()
        
        # Update Monitor
        avg_reward = sum(total_rewards) / episodes
        monitor.update(t, avg_reward, loss)

    env.close()
