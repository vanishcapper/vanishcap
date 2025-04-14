"""Event data structure for the vanishcap system."""

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Event:
    """Represents an event in the system with its source and data.

    Attributes:
        worker_name: The name of the worker that generated the event
        event_name: The name of the event
        data: Optional data associated with the event
        timestamp: Time when the event was created (seconds since epoch)
        frame_number: Optional frame number associated with the event
    """

    worker_name: str
    event_name: str
    data: Any = None
    timestamp: float = field(default_factory=time.time)
    frame_number: Optional[int] = None
