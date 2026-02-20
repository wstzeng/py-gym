import logging
from rich.logging import RichHandler
from rich.console import Console

console = Console()
logger = logging.getLogger("Py-Gym")

def setup_logger(
        level: str = "INFO",
        name: str = "Py-Gym"
):
    target_logger = logging.getLogger(name)
    target_logger.setLevel(level)

    if not target_logger.handlers:
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            markup=True,
        )
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        target_logger.addHandler(handler)

    target_logger.propagate = False

    return target_logger
