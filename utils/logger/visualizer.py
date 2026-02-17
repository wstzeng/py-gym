import os
import numpy as np
import matplotlib.pyplot as plt
from .plot_config import set_plot_style

class Visualizer:
    def __init__(
            self,
            agent_name: str,
            env_name: str
    ):
        self.agent_name = agent_name
        self.env_name = env_name
        self.styles = set_plot_style()
        self.fig = None
        self.fill_area = None

    def setup_live(self):
        plt.ion()
        self.fig, self.ax1 = plt.subplots(figsize=(10, 6))
        self.fig.subplots_adjust(top=0.88)
        self.ax2 = self.ax1.twinx()

        # Setup Labels
        self.ax1.set_xlabel('Iteration', fontweight='bold')
        self.ax1.set_ylabel('Reward', fontweight='bold', color='tab:blue')
        self.ax2.set_ylabel('Metrics (Loss/Entropy)', fontweight='bold', color='tab:gray')

        self.ax1.tick_params(axis='y', labelcolor='tab:blue')
        self.ax2.tick_params(axis='y', labelcolor='tab:gray')

        self.ax1.set_zorder(self.ax2.get_zorder() + 1)
        self.ax1.patch.set_visible(False)

        # Fix Plots: Reward Related
        self.line_raw, = self.ax1.plot([], [], color='tab:purple', alpha=0.3, zorder=2, label='Raw')
        self.line_trend, = self.ax1.plot([], [], color='tab:blue', linewidth=2, zorder=3, label='Trend')

        # Dynamic Plots: Multiple Metrics
        self.metric_lines = {}

        self.fig.suptitle(f'{self.agent_name}: {self.env_name}', **self.styles['suptitle'])

    def update_live(self, stats: dict):
        if not self.fig or not stats:
            return

        # 1. Update Reward Plots
        self.line_raw.set_data(stats['x'], stats['raw'])
        self.line_trend.set_data(stats['x'], stats['smoothed'])

        # 2. Update Dynamic Metrics (ax2)
        # Excludes known keys related to Reward in Stats
        known_keys = {'x', 'raw', 'smoothed', 'std_up', 'std_lo'}
        for name in stats.keys():
            if name in known_keys:
                continue

            # New Line for new metrics
            if name not in self.metric_lines:
                color = ["tab:orange", "tab:red", "tab:green", "tab:brown"][len(self.metric_lines) % 4]
                line, = self.ax2.plot([], [], color=color, linestyle='--', alpha=0.6, label=name.capitalize())
                self.metric_lines[name] = line
                self.ax2.legend(loc='upper right', fontsize='small')

            self.metric_lines[name].set_data(stats['x'], stats[name])

        # 3. Standard Deviation Trending
        if len(stats['x']) > 1:
            if self.fill_area: self.fill_area.remove()
            self.fill_area = self.ax1.fill_between(
                stats['x'], stats['std_lo'], stats['std_up'],
                color='tab:blue', alpha=0.12, zorder=2, linewidth=0
            )

        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def save_final(self, iterations, rewards, metrics_history, save_path):
        """
        metrics_history: dict { 'loss': [...], 'entropy': [...] }
        """
        plt.ioff()
        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax2 = ax1.twinx()

        # Smoothing logic
        y_r = np.array(rewards)
        window = max(2, len(iterations) // 50)
        half_w = window // 2
        smoothed = [np.mean(y_r[max(0, i-half_w):min(len(y_r), i+half_w+1)]) for i in range(len(y_r))]

        # Layer 1: Metrics (Bottom)
        colors = ["tab:orange", "tab:red", "tab:green", "tab:brown"]
        for i, (name, values) in enumerate(metrics_history.items()):
            if len(values) == len(iterations):
                ax2.plot(iterations, values, color=colors[i % 4], linestyle='--', alpha=0.4, label=name.capitalize())

        # Layer 2 & 3: Reward (Top)
        ax1.plot(iterations, y_r, color='tab:purple', alpha=0.15, zorder=2, label='Raw Reward')
        ax1.plot(iterations, smoothed, color='tab:blue', linewidth=2.5, zorder=3, label='Trend (SMA)')

        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)

        ax1.set_xlabel('Iteration', fontweight='bold')
        ax1.set_ylabel('Reward', color='tab:blue', fontweight='bold')
        ax2.set_ylabel('Metrics', color='tab:gray', fontweight='bold')

        plt.title(f'Final Training Summary: {self.env_name} ({self.agent_name})', pad=20)

        # Combine legends
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='upper left')

        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
