"""Base worker class for event handling."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from veil.event import Event


class Worker(ABC):
    """Abstract base class for workers that handle named events with data."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the worker with a name and optional configuration.

        Args:
            name: The name of this worker instance
            config: Optional dictionary of configuration values
        """
        self.name = name
        self.config: Dict[str, Any] = config or {}

    @abstractmethod
    def __call__(self, event: Event) -> Event:
        """Handle an event and return a response event.

        Args:
            event: The event to handle

        Returns:
            Event: A new event object containing this worker's response
        """
        pass 