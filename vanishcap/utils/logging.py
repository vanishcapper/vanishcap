"""Logging utilities for vanishcap."""

import logging
from typing import Optional


class WorkerNameFilter(logging.Filter):
    """Filter that extracts just the worker name from the full logger name."""

    def filter(self, record):
        """Extract worker name from the full logger name and pad it."""
        # Get the last part of the logger name (after the last dot)
        worker_name = record.name.split(".")[-1]
        # Pad to match length of "controller" (10 chars)
        record.worker_name = worker_name.ljust(10)
        # Get first letter of level name twice (e.g., "WW" for WARNING)
        record.level_short = record.levelname[0] * 2
        return True


def get_worker_logger(worker_name: str, log_level: Optional[str] = None) -> logging.Logger:
    """Get a logger for a worker with the specified log level.

    Args:
        worker_name: Name of the worker
        log_level: Optional log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                  If not specified, defaults to INFO

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(f"vanishcap.workers.{worker_name}")

    # Set log level from config or default to INFO
    if log_level:
        try:
            level = getattr(logging, log_level.upper())
        except AttributeError:
            level = logging.INFO
    else:
        level = logging.INFO

    logger.setLevel(level)

    # Only add handler if logger doesn't already have handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        # Add filter to extract worker name
        name_filter = WorkerNameFilter()
        handler.addFilter(name_filter)
        formatter = logging.Formatter("%(asctime)s - %(level_short)s - %(worker_name)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
