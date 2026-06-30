"""
src/utils/logger.py — Structured logging utility declaration.
Owner: Tech Lead (TL-2, Sprint 1)
"""
import logging

from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True)



def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger with a consistent structured format.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger
    
    logger.setLevel(logging.INFO)
    
    handler = RichHandler(
        console = _console,
        show_time=True,
        show_path=True,
        show_level=True,
    )
    formatter = logging.Formatter(
        "%(message)s",
        style="%",
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger
    
