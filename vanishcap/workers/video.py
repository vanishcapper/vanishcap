"""Worker for capturing video frames from a source."""

from queue import Empty
from typing import Any, Dict, Optional

from vidgear.gears import CamGear, WriteGear
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
                - save_path: Optional path to save the video (if not provided, video won't be saved)
        """
        super().__init__("video", config)
        self.source = config["source"]
        self.logger.warning("Initialized video worker with source: %s", self.source)

        # Initialize video capture with CamGear
        options = {
            "THREADED_QUEUE_MODE": False,  # Disable threaded queue mode
            "THREAD_TIMEOUT": 1.0,  # Set thread timeout to 1 second
        }
        self.cap = CamGear(source=self.source, **options).start()
        if self.cap is None:
            self.logger.error("Failed to open video source: %s", self.source)
            raise RuntimeError(f"Failed to open video source: {self.source}")

        # Initialize video writer if save_path is provided
        self.writer: Optional[WriteGear] = None
        if "save_path" in config:
            output_params = {
                "-input_framerate": 30,
                "-vcodec": "libx264",
                "-crf": 23,
                "-preset": "fast",
            }
            self.writer = WriteGear(output=config["save_path"], **output_params)
            self.logger.warning("Video will be saved to: %s", config["save_path"])

        self.logger.warning("Successfully opened video source")

        # Initialize state
        self.current_frame = None
        self.frame_number = 0  # Track frame number

    def _task(self) -> None:
        """Run one iteration of the video loop."""
        try:
            # Read frame
            frame = self.cap.read()
            if frame is None:
                self.logger.error("Failed to read frame")
                self._stop_event.set()
                return

            # Increment frame number
            self.frame_number += 1

            # Save frame if writer is enabled
            if self.writer is not None:
                self.writer.write(frame)

            # Emit frame event with frame number
            self.logger.info("Acquired frame %d", self.frame_number)
            self._emit(Event(self.name, "frame", {"frame": frame, "frame_number": self.frame_number}))
        except Empty:
            self.logger.warning("Video stream queue empty - stopping worker")
            self._stop_event.set()

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
        if self.writer is not None:
            self.writer.close()
            self.logger.warning("Video writer closed")
        self.cap = None
        self.logger.warning("Video capture released")
