"""Worker for displaying video frames and detection results."""

# pylint: disable=wrong-import-position,too-many-instance-attributes,no-member

import time
import traceback
from typing import Any, Dict, Optional

import cv2

from vanishcap.event import Event
from vanishcap.worker import Worker


class Ui(Worker):
    """Worker that displays video frames and detection results."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the UI worker.

        Args:
            config: Configuration dictionary containing:
                - name: Name of the worker
                - window_size: Optional tuple of (width, height) for the display window
                - log_level: Optional log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                - fps: Optional target frame rate (default: 30)
        """
        super().__init__(config)
        # Display settings
        self.display_config = {
            "window_size": config.get("window_size", (800, 600)),
            "target_fps": config.get("fps", 30),
            "frame_time": 1.0 / config.get("fps", 30),
            "last_frame_time": 0,
        }
        self.logger.warning(
            "Initialized UI worker with window size: %s and target FPS: %s",
            self.display_config["window_size"],
            self.display_config["target_fps"],
        )

        # Initialize OpenCV window
        try:
            cv2.resizeWindow("vanishcap", *self.display_config["window_size"])
            self.logger.warning("OpenCV window initialized")
        except Exception as e:
            self.logger.error("Failed to initialize OpenCV window: %s", e)
            traceback.print_exc()
            raise

        # Initialize state
        self.current_frame_event: Optional[Event] = None
        self.current_detections = []
        self.worker_profiles = {}  # Maps worker name to last task time

        # Font settings for OpenCV
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1
        self.font_color = (255, 255, 255)  # White
        self.box_color = (0, 255, 0)  # Green

    def _denormalize_coordinates(self, x: float, y: float, width: float, height: float) -> tuple[int, int]:
        """Convert normalized coordinates [-1, 1] back to pixel coordinates.

        Args:
            x: Normalized x coordinate [-1, 1]
            y: Normalized y coordinate [-1, 1]
            width: Frame width in pixels
            height: Frame height in pixels

        Returns:
            tuple[int, int]: Pixel coordinates
        """
        # Convert from [-1, 1] to pixel coordinates
        pixel_x = int((x + 1) * width / 2)
        pixel_y = int((y + 1) * height / 2)
        return pixel_x, pixel_y

    def _task(self) -> None:  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        """Run one iteration of the UI loop."""
        # Check for quit key (ESC)
        if (cv2.waitKey(1) & 0xFF == 27) or cv2.getWindowProperty("vanishcap", cv2.WND_PROP_VISIBLE) < 1:
            self.logger.warning("Received quit event from OpenCV")
            self._emit(Event(self.name, "stop", None))
            self._stop_event.set()
            return  # Exit task early if quitting

        # Get latest events from BaseWorker
        latest_events = self._get_latest_events_and_clear()

        for _, event in latest_events.items():
            if event.event_name == "frame":
                self.current_frame_event = event
            elif event.event_name == "detection":
                self.current_detections = event.data
            elif event.event_name == "worker_profile":
                worker_name = event.worker_name
                task_time = event.data["task_time"]
                self.worker_profiles[worker_name] = task_time

        # Update display if we have a frame and enough time has passed
        if (
            self.current_frame_event is None
            or time.time() - self.display_config["last_frame_time"] < self.display_config["frame_time"]
        ):
            # No new frame to render or too soon, skip render
            return

        # Extract frame data from the instance variable
        frame = self.current_frame_event.data.copy()  # Make a copy to draw on
        frame_number = self.current_frame_event.frame_number

        # Log which frame we're displaying
        current_time = time.time()
        self.logger.info(
            "Displaying frame %d %dms after it was acquired",
            frame_number,
            1000 * (current_time - self.current_frame_event.timestamp),
        )

        # Get original frame dimensions
        orig_height, orig_width = frame.shape[:2]

        # Scale frame to window size
        frame = cv2.resize(frame, self.display_config["window_size"])

        # Draw detection boxes using self.current_detections
        for detection in self.current_detections:
            # Get normalized bbox coordinates
            norm_x1, norm_y1, norm_x2, norm_y2 = detection["bbox"]

            # Convert normalized coordinates to pixel coordinates
            x1, y1 = self._denormalize_coordinates(norm_x1, norm_y1, orig_width, orig_height)
            x2, y2 = self._denormalize_coordinates(norm_x2, norm_y2, orig_width, orig_height)

            # Scale bbox coordinates to match scaled frame size
            x1 = int(x1 * self.display_config["window_size"][0] / orig_width)
            y1 = int(y1 * self.display_config["window_size"][1] / orig_height)
            x2 = int(x2 * self.display_config["window_size"][0] / orig_width)
            y2 = int(y2 * self.display_config["window_size"][1] / orig_height)

            # Invert y-coordinates for OpenCV's coordinate system
            window_height = self.display_config["window_size"][1]
            y1 = window_height - y1
            y2 = window_height - y2

            # Draw rectangle (OpenCV uses BGR color order)
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.box_color, 2)

            # Draw label (adjust y position for inverted coordinates)
            label = f"{detection['class_name']} ({detection['confidence']:.2f})"
            cv2.putText(frame, label, (x1, y1 + 20), self.font, self.font_scale, self.font_color, self.font_thickness)

            # Log detection drawing
            self.logger.debug(
                "Drawing %s (%.2f) at normalized (%.2f, %.2f) - (%.2f, %.2f)",
                detection["class_name"],
                detection["confidence"],
                norm_x1,
                norm_y1,
                norm_x2,
                norm_y2,
            )

        # Log summary of all boxes drawn, sorted by class_id
        if self.current_detections:
            summary = ", ".join(
                f"{d['class_name']}({d['confidence']:.2f})"
                for d in sorted(self.current_detections, key=lambda x: x["class_id"])
            )
            self.logger.info("Boxes drawn: %s", summary)

        # Draw frame number in top right
        frame_text = f"Frame: {frame_number}"
        text_size = cv2.getTextSize(frame_text, self.font, self.font_scale, self.font_thickness)[0]
        text_x = self.display_config["window_size"][0] - text_size[0] - 10
        text_y = text_size[1] + 10
        cv2.putText(
            frame, frame_text, (text_x, text_y), self.font, self.font_scale, self.font_color, self.font_thickness
        )

        # Draw profiling data using self.worker_profiles
        y = 30  # Start below frame number
        for worker_name, task_time in self.worker_profiles.items():
            text = f"{worker_name}: {task_time*1000:.1f}ms"
            cv2.putText(frame, text, (10, y), self.font, self.font_scale, self.font_color, self.font_thickness)
            y += 20

        # Update display
        cv2.imshow("vanishcap", frame)
        self.display_config["last_frame_time"] = time.time()

    def _finish(self) -> None:
        """Clean up OpenCV resources."""
        cv2.destroyAllWindows()
        self.logger.warning("OpenCV window closed")
