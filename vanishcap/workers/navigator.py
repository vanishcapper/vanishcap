"""Worker for processing detections and emitting navigation commands."""

import queue
from typing import Any, Dict

from vanishcap.event import Event
from vanishcap.worker import Worker


class Navigator(Worker):
    """Worker that processes detections and emits navigation commands."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the navigator worker.

        Args:
            config: Configuration dictionary containing:
                - target_class: Class ID to track
                - log_level: Optional log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        super().__init__("navigator", config)
        self.target_class = config["target_class"]
        self.logger.info("Initialized navigator worker with target class: %s", self.target_class)

        # Initialize state
        self.detection_queue = queue.Queue()

    def _task(self) -> None:
        """Run one iteration of the navigator loop."""
        # Get detections from queue
        try:
            detections = self.detection_queue.get_nowait()
        except queue.Empty:
            return

        # Process detections
        target_detections = [d for d in detections if d["class_name"] == self.target_class]

        if target_detections:
            # Get the largest target (closest to camera)
            target = max(target_detections, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))

            # Calculate target position
            bbox = target["bbox"]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            # Emit navigation event
            self._emit(
                Event(self.name, "navigation", {"x": center_x, "y": center_y, "confidence": target["confidence"]})
            )

    def __call__(self, event: Event) -> None:
        """Handle incoming events.

        Args:
            event: Event to handle
        """
        if event.event_name == "detection":
            self.detection_queue.put(event.data)
        else:
            self.logger.debug("Received unknown event: %s", event.event_name)

    def _finish(self) -> None:
        """Clean up navigator resources."""
        # Nothing to clean up for navigator
