"""Worker for capturing video frames from a source."""

from typing import Any, Dict

import cv2
from vanishcap.event import Event
from vanishcap.worker import Worker


class Video(Worker):
    """Worker that captures frames from a video source."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the video worker.

        Args:
            config: Configuration dictionary containing:
                - source: Path to video file or camera index
                - log_level: Optional log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        super().__init__("video", config)
        self.source = config["source"]
        self.logger.warning("Initialized video worker with source: %s", self.source)

        # Initialize video capture
        self.cap = cv2.VideoCapture(self.source)  # pylint: disable=no-member
        if not self.cap.isOpened():
            self.logger.error("Failed to open video source: %s", self.source)
            raise RuntimeError(f"Failed to open video source: {self.source}")

        self.logger.warning("Successfully opened video source")

        # Initialize state
        self.current_frame = None
        self.frame_number = 0  # Track frame number

    def _task(self) -> None:
        """Run one iteration of the video loop."""
        # Read frame
        ret, frame = self.cap.read()
        if not ret:
            self.logger.error("Failed to read frame")
            self._stop_event.set()
            return

        # Increment frame number
        self.frame_number += 1

        # Emit frame event with frame number
        self.logger.info("Acquired frame %d", self.frame_number)
        self._emit(Event(self.name, "frame", {"frame": frame, "frame_number": self.frame_number}))

    def __call__(self, event: Event) -> None:
        """Handle incoming events.

        Args:
            event: Event to handle
        """
        if event.event_name == "command":
            self.logger.info("Received command: %s", event.data)
        else:
            self.logger.debug("Received unknown event: %s", event.event_name)

    def _finish(self) -> None:
        """Clean up video resources."""
        if self.cap is not None:
            self.cap.release()
            self.logger.warning("Video capture released")
