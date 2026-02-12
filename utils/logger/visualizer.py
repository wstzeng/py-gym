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
        """Initializes the live plotting window with specific layering."""
        plt.ion()
        self.fig, self.ax1 = plt.subplots(figsize=(10, 6))
        self.fig.subplots_adjust(top=0.88)
        self.ax2 = self.ax1.twinx()

        self.ax1.set_xlabel('Iteration', fontweight='bold')
        self.ax1.set_ylabel('Reward', fontweight='bold', color='tab:blue')
        self.ax2.set_ylabel('Loss', fontweight='bold', color='tab:orange')
        
        self.ax1.tick_params(axis='y', labelcolor='tab:blue')
        self.ax2.tick_params(axis='y', labelcolor='tab:orange')

        # Layering: ax1 (Reward) on top of ax2 (Loss)
        self.ax1.set_zorder(self.ax2.get_zorder() + 1)
        self.ax1.patch.set_visible(False)

        # Initialize lines with explicit zorders
        self.line_raw, = self.ax1.plot([], [], color='tab:purple', alpha=0.3, zorder=2, label='Raw')
        self.line_trend, = self.ax1.plot([], [], color='tab:blue', linewidth=2, zorder=3, label='Trend')
        self.line_loss, = self.ax2.plot([], [], color='tab:orange', linestyle='--', alpha=0.6, zorder=1)
        
        self.fig.suptitle(f'{self.agent_name}: {self.env_name}', **self.styles['suptitle'])

    def update_live(
            self,
            stats: dict
    ):
        """Updates live lines and confidence intervals."""
        if not self.fig or not stats:
            return
        
        self.line_raw.set_data(stats['x'], stats['raw'])
        self.line_trend.set_data(stats['x'], stats['smoothed'])
        self.line_loss.set_data(stats['x'], stats['loss'])

        if len(stats['x']) > 1:
            if self.fill_area:
                self.fill_area.remove()
            # Standard Deviation area also set to zorder 2 (below trend)
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

    def save_final(
            self,
            iterations: list,
            rewards: list,
            losses: list,
            save_path: str
    ):
        """Generates the high-res summary plot with Reward Trend as the top layer."""
        plt.ioff()
        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax2 = ax1.twinx()
        
        # Calculate smoothing
        y_r = np.array(rewards)
        window = max(20, len(iterations) // 50)
        half_w = window // 2
        smoothed = [
            np.mean(y_r[max(0, i-half_w):min(len(y_r), i+half_w+1)]) 
            for i in range(len(y_r))
        ]

        # Layer 1: Loss (Bottom)
        ax2.plot(iterations, losses, color='tab:orange', linestyle='--', alpha=0.4, zorder=1, label='Loss')
        
        # Layer 2: Raw Reward (Middle)
        ax1.plot(iterations, y_r, color='tab:purple', alpha=0.15, zorder=2, label='Raw Reward')
        
        # Layer 3: Trend (Top)
        ax1.plot(iterations, smoothed, color='tab:blue', linewidth=2.5, zorder=3, label='Trend (SMA)')

        # Ensure ax1 is visibly on top of ax2
        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)

        ax1.set_xlabel('Iteration', fontweight='bold')
        ax1.set_ylabel('Reward', color='tab:blue', fontweight='bold')
        ax2.set_ylabel('Loss', color='tab:orange', fontweight='bold')
        
        plt.title(f'Final Training Summary: {self.env_name} ({self.agent_name})', pad=20)
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
