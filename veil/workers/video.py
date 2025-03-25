"""Video worker for reading frames from a video source."""

import cv2
from typing import Any, Dict, Optional

from veil.event import Event
from veil.worker import Worker


class Video(Worker):
    """Worker for reading frames from a video source."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the video worker.

        Args:
            config: Configuration dictionary containing:
                - source: Path to video file or camera index
                - frame_skip: Number of frames to skip between reads (optional)
        """
        super().__init__("video")
        self.source = config["source"]
        self.frame_skip = config.get("frame_skip", 1)
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False

    def __call__(self, event: Event) -> Event:
        """Handle events for the video worker.

        Args:
            event: The event to handle

        Returns:
            Event: Response event

        Raises:
            ValueError: If the event name is not recognized
        """
        if event.event_name == "run":
            return self._handle_run()
        elif event.event_name == "stop":
            return self._handle_stop()
        else:
            raise ValueError(f"Unknown event name: {event.event_name}")

    def _handle_run(self) -> Event:
        """Handle the run event by starting the video capture loop.

        Returns:
            Event: Response event
        """
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                return Event("video", "error", "Failed to open video source")

        self.running = True
        frame_count = 0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                return Event("video", "end", None)

            # Skip frames if configured
            frame_count += 1
            if frame_count % self.frame_skip != 0:
                continue

            # Send frame event
            return Event("video", "frame", frame)

        return Event("video", "stop", None)

    def _handle_stop(self) -> Event:
        """Handle the stop event by stopping the video capture.

        Returns:
            Event: Response event
        """
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        return Event("video", "stop", None) 