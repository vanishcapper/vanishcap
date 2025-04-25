"""Worker for processing detections and emitting navigation commands."""

from typing import Any, Dict

from vanishcap.event import Event
from vanishcap.worker import Worker


class Navigator(Worker):
    """Worker that processes detections and emitting navigation commands."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the navigator worker.

        Args:
            config: Configuration dictionary containing:
                - name: Name of the worker
                - target_class: Class name to track
                - log_level: Optional log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        super().__init__(config)
        self.target_class = config["target_class"]
        self.logger.warning("Initialized navigator worker with target class: %s", self.target_class)

        # Initialize state

    def _task(self) -> None:
        """Run one iteration of the navigator loop."""
        # Get latest events
        latest_detection_event = self._get_latest_events_and_clear().get("detection", None)

        # Process the latest detection event if found
        if latest_detection_event is None:
            return

        detections = latest_detection_event.data
        frame_number = latest_detection_event.frame_number
        self.logger.info("Processing latest detections (frame %d) with %d detections", frame_number, len(detections))

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

            # Get normalized bounding box coordinates
            target_bbox = target["bbox"]
            self.logger.debug("Target bounding box: %s", target_bbox)

            # Emit target event with normalized coordinates and frame number
            self._emit(
                Event(
                    self.name,
                    "target",
                    {
                        "x": target_x,
                        "y": target_y,
                        "confidence": target["confidence"],
                        "bbox": target_bbox,
                    },
                    frame_number=frame_number,
                )
            )
            self.logger.debug("Emitted target event with position (%.2f, %.2f)", target_x, target_y)
        else:
            self.logger.debug("No targets of class %s found in frame - emitting empty target event", self.target_class)
            # Emit empty target event
            self._emit(
                Event(
                    self.name,
                    "target",
                    None,
                    frame_number=frame_number,
                )
            )

    def _finish(self) -> None:
        """Clean up navigator resources."""
        # Nothing to clean up for navigator
