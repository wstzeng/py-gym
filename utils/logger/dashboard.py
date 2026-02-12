import os
from .data_collector import DataCollector
from .visualizer import Visualizer
from utils.logger import logger, console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

class Dashboard:
    def __init__(
            self,
            env_name: str,
            agent_name: str,
            total_iterations: int,
            window_size: int = 100,
            modes: list = ['cli'],
            save_dir: str = 'log',
    ):
        self.modes = modes
        self.save_dir = save_dir
        
        # Initialize components
        self.data = DataCollector(
            window_size=window_size, 
            save_dir=save_dir if 'file' in modes else None
        )
        self.viz = Visualizer(agent_name, env_name)
        
        if 'live' in modes:
            self.viz.setup_live()

        self._setup_progress(env_name, total_iterations)

    def _setup_progress(
            self,
            env_name: str,
            total: int
    ):
        if 'cli' not in self.modes:
            self.progress = None
            return

        self.progress = Progress(
            TextColumn("[bold cyan]{task.fields[env]:>16.16}", justify="left"),
            BarColumn(bar_width=20),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TextColumn("[bold green]Reward: {task.fields[avg_r]:>8.2f}"),
            "•",
            TextColumn("[bold magenta]Loss: {task.fields[loss]:>7.4f}"),
            "•",
            TimeElapsedColumn(),
            "/",
            TimeRemainingColumn(),
            console=console,
            transient=True
        )
        self.progress.start()
        self.task = self.progress.add_task(
            "Training", 
            total=total, 
            env=env_name, 
            avg_r=0.0, 
            loss=0.0
        )

    def update(
            self,
            iteration: int,
            avg_reward: float,
            loss: float
    ):
        self.data.add(iteration, avg_reward, loss)
        
        if self.progress:
            self.progress.update(
                self.task, 
                completed=iteration, 
                avg_r=avg_reward, 
                loss=loss
            )
        
        if 'live' in self.modes:
            stats = self.data.get_live_stats()
            self.viz.update_live(stats)

    def close(self):
        if self.progress:
            self.progress.stop()
        
        logger.info(f"[bold green]Training on {self.viz.env_name} completed![/bold green]")

        if 'file' in self.modes:
            self.viz.save_final(
                self.data.iterations, 
                self.data.full_rewards, 
                self.data.full_losses,
                os.path.join(self.save_dir, "training_log.png")
            )
            logger.info(f"Full history plot saved to {self.save_dir}")

        self.data.close()
