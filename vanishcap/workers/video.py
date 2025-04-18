"""Worker for capturing video frames from a source."""

import time
from queue import Empty
from typing import Any, Dict, Optional

from vidgear.gears import CamGear, WriteGear
from vanishcap.event import Event
from vanishcap.worker import Worker


# pylint: disable=too-many-instance-attributes
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

        # Determine if source is a stream (URL), camera (int), or file
        if isinstance(self.source, int):
            self.is_stream = True
            source_type = "camera"
        else:
            stream_prefixes = ["udp://", "rtsp://", "http://", "https://"]
            self.is_stream = any(self.source.startswith(prefix) for prefix in stream_prefixes)
            source_type = "stream" if self.is_stream else "file"
        self.logger.warning("Source type: %s", source_type)

        # Initialize video capture with CamGear
        options = {
            "THREADED_QUEUE_MODE": not self.is_stream,  # Only enable threaded queue mode for files
            "THREAD_TIMEOUT": 5.0,  # Increase thread timeout to 5 seconds
            "QUEUE_TIMEOUT": 5.0,  # Add queue timeout
            "BUFFER_QUEUE_SIZE": 100,  # Increase buffer size
        }
        self.cap = CamGear(source=self.source, **options).start()
        if self.cap is None:
            self.logger.error("Failed to open video source: %s", self.source)
            raise RuntimeError(f"Failed to open video source: {self.source}")

        # Get video properties
        self.fps = self.cap.framerate
        self.logger.warning("Video framerate: %.2f FPS", self.fps)
        self.frame_time = 1.0 / self.fps if self.fps > 0 and not self.is_stream else 0.0
        self.last_frame_time = 0.0

        # Initialize video writer if save_path is provided
        self.writer: Optional[WriteGear] = None
        if "save_path" in config:
            output_params = {
                "-input_framerate": self.fps,
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
            # Check if enough time has passed for the next frame (only for files)
            current_time = time.time()
            if not self.is_stream and self.frame_time > 0:
                time_since_last_frame = current_time - self.last_frame_time
                if time_since_last_frame < self.frame_time:
                    sleep_time = self.frame_time - time_since_last_frame
                    self.logger.debug("Sleeping for %.3f seconds to maintain framerate", sleep_time)
                    time.sleep(sleep_time)
                    current_time = time.time()  # Update current time after sleep

            # Read frame
            frame = self.cap.read()
            if frame is None:
                self.logger.error("Failed to read frame")
                self._stop_event.set()
                return

            # Update last frame time
            self.last_frame_time = current_time

            # Increment frame number
            self.frame_number += 1

            # Save frame if writer is enabled
            if self.writer is not None:
                self.writer.write(frame)

            # Emit frame event with frame number
            self.logger.info("Acquired frame %d (time since last frame: %.3f seconds)",
                           self.frame_number, current_time - self.last_frame_time)
            self._emit(Event(self.name, "frame", frame, frame_number=self.frame_number))
        except Empty:
            self.logger.warning("Video stream queue empty - stopping worker")
            self._stop_event.set()

    def _finish(self) -> None:
        """Clean up video resources."""
        if self.writer is not None:
            self.writer.close()
            self.logger.warning("Video writer closed")
        self.cap = None
        self.logger.warning("Video capture released")
