"""Event data structure for the veil system."""

from dataclasses import dataclass
from typing import Any


@dataclass
class Event:
    """Represents an event in the system with its source and data.

    Attributes:
        worker_name: The name of the worker that generated the event
        event_name: The name of the event
        data: Optional data associated with the event
    """

    worker_name: str
    event_name: str
    data: Any = None 