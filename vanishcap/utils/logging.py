"""Logging utilities for vanishcap."""

import logging
import os
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


# Global file handler for all loggers
_global_file_handler = None


def get_worker_logger(worker_name: str, log_level: Optional[str] = None, log_file: Optional[str] = None) -> logging.Logger:
    """Get a logger for a worker with the specified log level.

    Args:
        worker_name: Name of the worker
        log_level: Optional log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                  If not specified, defaults to INFO
        log_file: Optional path to log file. If provided and not None, logs will be written to both
                 console and file. This is a global setting - all loggers will use the
                 same file handler.

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

    # Only add handlers if logger doesn't already have handlers
    if not logger.handlers:
        # Add console handler
        console_handler = logging.StreamHandler()
        # Add filter to extract worker name
        name_filter = WorkerNameFilter()
        console_handler.addFilter(name_filter)
        formatter = logging.Formatter("%(asctime)s - %(level_short)s - %(worker_name)s %(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Add global file handler if log file path is provided and not None
        global _global_file_handler
        if log_file and _global_file_handler is None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            _global_file_handler = logging.FileHandler(log_file, mode='w')  # Use 'w' mode to overwrite
            _global_file_handler.addFilter(name_filter)
            _global_file_handler.setFormatter(formatter)

        if _global_file_handler is not None:
            logger.addHandler(_global_file_handler)

    return logger
