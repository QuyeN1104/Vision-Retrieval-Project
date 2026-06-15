"""
src/utils/logger.py — Structured logging utility declaration.
Owner: Tech Lead (TL-2, Sprint 1)
"""
import logging


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger with a consistent structured format.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured Logger instance.
    """
    raise NotImplementedError("To be implemented by Tech Lead (TL-2) in Sprint 1")
