"""Worker for processing detections and emitting navigation commands."""

import queue
from typing import Any, Dict

from vanishcap.event import Event
from vanishcap.worker import Worker


class Navigator(Worker):
    """Worker that processes detections and emitting navigation commands."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the navigator worker.

        Args:
            config: Configuration dictionary containing:
                - target_class: Name of the class to track (e.g. "person", "car")
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
            self.logger.debug("Processing %d detections", len(detections))
        except queue.Empty:
            return

        # Process detections
        target_detections = [d for d in detections if d["class_name"] == self.target_class]
        self.logger.debug("Found %d detections of target class %s", len(target_detections), self.target_class)

        if target_detections:
            # Get the largest target (closest to camera)
            target = max(target_detections, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
            self.logger.debug("Selected largest target with confidence: %.2f", target["confidence"])

            # Get normalized coordinates from detector
            target_x = target["x"]
            target_y = target["y"]
            self.logger.debug("Target position: (%.2f, %.2f)", target_x, target_y)

            # Emit target event with normalized coordinates
            self._emit(
                Event(
                    self.name,
                    "target",
                    {
                        "x": target_x,
                        "y": target_y,
                        "confidence": target["confidence"],
                    },
                )
            )
            self.logger.debug("Emitted target event with position (%.2f, %.2f)", target_x, target_y)
        else:
            self.logger.debug("No targets of class %s found in frame", self.target_class)

    def __call__(self, event: Event) -> None:
        """Handle incoming events.

        Args:
            event: Event to handle
        """
        if event.event_name == "detection":
            self.logger.debug("Received detection event with %d detections", len(event.data))
            self.detection_queue.put(event.data)
        else:
            self.logger.debug("Received unknown event: %s", event.event_name)

    def _finish(self) -> None:
        """Clean up navigator resources."""
        # Nothing to clean up for navigator
