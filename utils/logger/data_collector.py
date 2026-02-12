import os
import numpy as np
from collections import deque

class DataCollector:
    def __init__(
            self,
            window_size: int = 100,
            save_dir: str = None
    ):
        self.window_size = window_size
        self.save_dir = save_dir
        self.iterations = []
        self.full_rewards = []
        self.full_losses = []
        
        # Windows for live feedback
        self.reward_window = deque(maxlen=window_size)
        self.loss_window = deque(maxlen=window_size)
        
        self._log_file = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.log_path = os.path.join(save_dir, 'training_log.csv')
            self._log_file = open(self.log_path, 'w')
            self._log_file.write("Iteration,AverageReward,Loss\n")

    def add(
            self,
            iteration: int,
            reward: float,
            loss: float
    ):
        self.iterations.append(iteration)
        self.full_rewards.append(reward)
        self.full_losses.append(loss)
        self.reward_window.append(reward)
        self.loss_window.append(loss)
        
        if self._log_file:
            self._log_file.write(f"{iteration},{reward:.4f},{loss:.6f}\n")
            self._log_file.flush()

    def get_live_stats(
            self,
            window: int = 20
    ) -> dict:
        """
        Calculates stats for the current window (live plotting).
        """
        y = np.array(self.reward_window)
        if len(y) == 0:
            return {}
        
        half_w = window // 2
        smoothed, std_up, std_lo = [], [], []
        
        for i in range(len(y)):
            start = max(0, i - half_w)
            end = min(len(y), i + half_w + 1)
            view = y[start:end]
            mu, std = np.mean(view), np.std(view)
            smoothed.append(mu)
            std_up.append(mu + 1.28 * std)
            std_lo.append(mu - 1.28 * std)
            
        return {
            'x': list(self.iterations)[-len(y):],
            'raw': y,
            'smoothed': smoothed,
            'std_up': std_up,
            'std_lo': std_lo,
            'loss': list(self.loss_window)
        }

    def close(self):
        if self._log_file:
            self._log_file.close()
