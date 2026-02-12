# utils/train.py
import gymnasium as gym
import torch
from agent import BaseAgent
from .monitor.training_monitor import TrainingMonitor as Monitor

def train_loop(
    env_name: str,
    agent: BaseAgent,
    iterations: int,
    episodes: int,
    monitor_mode = 'cli',
):
    env = gym.make(env_name)
    monitor = Monitor(
        env_name, agent.__class__.__name__, iterations,
        mode=monitor_mode, window_size=50
    )

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

        loss = agent.update()
        
        avg_reward = sum(total_rewards) / episodes
        monitor.update(t, avg_reward, loss)

    monitor.close()
    env.close()
