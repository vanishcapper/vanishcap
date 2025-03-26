"""Logging utilities for veil."""

import logging
from typing import Optional


def get_worker_logger(worker_name: str, log_level: Optional[str] = None) -> logging.Logger:
    """Get a logger for a worker with the specified log level.

    Args:
        worker_name: Name of the worker
        log_level: Optional log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                  If not specified, defaults to INFO

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(f"veil.workers.{worker_name}")
    
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
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger 