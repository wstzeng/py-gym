import logging
from rich.logging import RichHandler
from rich.console import Console

console = Console()

def setup_logger(
        name: str = "Py-Gym"
):
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(
            console=console, 
            rich_tracebacks=True, 
            markup=True
        )]
    )
    return logging.getLogger(name)

logger = setup_logger()
