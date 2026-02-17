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

        self.metrics_history = {}
        self.metrics_windows = {}

        self.reward_window = deque(maxlen=window_size)

        self._log_file = None
        self._csv_header_written = False
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.log_path = os.path.join(save_dir, 'training_log.csv')

    def add(
            self,
            iteration: int,
            reward: float,
            **metrics
    ):
        self.iterations.append(iteration)
        self.full_rewards.append(reward)
        self.reward_window.append(reward)

        for name, value in metrics.items():
            if name not in self.metrics_history:
                self.metrics_history[name] = []
                self.metrics_windows[name] = deque(maxlen=self.window_size)
            self.metrics_history[name].append(value)
            self.metrics_windows[name].append(value)

        if self.save_dir:
            if not self._log_file:
                self._log_file = open(self.log_path, 'w')

            if not self._csv_header_written:
                header = ["Iteration", "AverageReward"] + list(metrics.keys())
                self._log_file.write(",".join(header) + "\n")
                self._csv_header_written = True

            values = [str(iteration), f"{reward:.4f}"]
            values += [f"{metrics.get(k, 0.0):.6f}" for k in metrics.keys()]
            self._log_file.write(",".join(values) + "\n")
            self._log_file.flush()

    def get_live_stats(self, window: int = 20) -> dict:
        y = np.array(self.reward_window)
        if len(y) == 0: return {}

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

        stats = {
            'x': list(self.iterations)[-len(y):],
            'raw': y,
            'smoothed': smoothed,
            'std_up': std_up,
            'std_lo': std_lo,
        }
        for name, win in self.metrics_windows.items():
            stats[name] = list(win)

        return stats

    def close(self):
        if self._log_file:
            self._log_file.close()
