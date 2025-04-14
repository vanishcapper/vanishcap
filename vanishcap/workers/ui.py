"""Worker for displaying video frames and detection results."""

# pylint: disable=wrong-import-position

import os
import queue
import time
import traceback
from typing import Any, Dict

# Suppress pygame startup message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import cv2  # noqa: C0413
import pygame  # noqa: C0413
import pygame.display  # noqa: C0413
import pygame.event  # noqa: C0413
import pygame.image  # noqa: C0413

from vanishcap.event import Event  # noqa: C0413
from vanishcap.worker import Worker  # noqa: C0413


class Ui(Worker):
    """Worker that displays video frames and detection results."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the UI worker.

        Args:
            config: Configuration dictionary containing:
                - window_size: Optional tuple of (width, height) for the display window
                - log_level: Optional log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                - fps: Optional target frame rate (default: 30)
        """
        super().__init__("ui", config)
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

        # Initialize pygame modules
        try:
            pygame.display.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode(self.display_config["window_size"])
            pygame.display.set_caption("vanishcap")
            self.logger.warning("Pygame modules initialized")
        except Exception as e:
            self.logger.error("Failed to initialize pygame modules: %s", e)
            traceback.print_exc()
            raise

        # Initialize state
        self.current_frame_event = None
        self.current_detections = []

        # Profiling data
        self.worker_profiles = {}  # Maps worker name to last task time
        self.profile_font = pygame.font.Font(None, 20)  # Small font for profiling text

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

    def _task(self) -> None:  # pylint: disable=too-many-locals
        """Run one iteration of the UI loop."""
        # Handle pygame events
        for event in pygame.event.get():
            if not (
                event.type == pygame.QUIT  # pylint: disable=no-member
                or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)  # pylint: disable=no-member
            ):
                continue
            self.logger.warning("Received quit event")
            self._emit(Event(self.name, "stop", None))
            self._stop_event.set()
            pygame.display.quit()
            if self._event_queue is not None:
                try:
                    event = self._event_queue.get_nowait()
                    self(event)
                except queue.Empty:
                    pass
            return

        # Update display if we have a frame and enough time has passed
        if (
            self.current_frame_event is None
            or time.time() - self.display_config["last_frame_time"] < self.display_config["frame_time"]
        ):
            return

        # Extract frame data
        frame = self.current_frame_event.data
        frame_number = self.current_frame_event.frame_number

        # Log which frame we're displaying
        self.logger.info("Displaying frame %d", frame_number)

        # Convert frame to pygame surface
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member
        frame_surface = pygame.image.frombuffer(frame_rgb.tobytes(), frame_rgb.shape[1::-1], "RGB")

        # Get original frame dimensions
        orig_height, orig_width = frame.shape[:2]

        # Scale frame to window size
        frame_surface = pygame.transform.scale(frame_surface, self.display_config["window_size"])

        # Draw frame
        self.screen.blit(frame_surface, (0, 0))

        # Draw detection boxes
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

            # Draw rectangle
            pygame.draw.rect(self.screen, (0, 255, 0), (x1, y1, x2 - x1, y2 - y1), 2)

            # Draw label
            label = f"{detection['class_name']} ({detection['confidence']:.2f})"
            text = self.profile_font.render(label, True, (0, 255, 0))
            self.screen.blit(text, (x1, y1 - 20))

            # Log detection drawing
            self.logger.info(
                "Drawing %s (%.2f) at normalized (%.2f, %.2f) - (%.2f, %.2f)",
                detection["class_name"],
                detection["confidence"],
                norm_x1,
                norm_y1,
                norm_x2,
                norm_y2,
            )

        # Draw frame number in top right
        frame_text = self.profile_font.render(f"Frame: {frame_number}", True, (255, 255, 255))
        text_rect = frame_text.get_rect()
        text_rect.topright = (self.display_config["window_size"][0] - 10, 10)
        self.screen.blit(frame_text, text_rect)

        # Draw profiling data
        y = 10
        for worker_name, task_time in self.worker_profiles.items():
            text = self.profile_font.render(f"{worker_name}: {task_time*1000:.1f}ms", True, (255, 255, 255))
            self.screen.blit(text, (10, y))
            y += 20

        # Update display
        pygame.display.flip()
        self.display_config["last_frame_time"] = time.time()

    def __call__(self, event: Event) -> None:
        """Handle incoming events.

        Args:
            event: Event to handle
        """
        if event.event_name == "frame":
            # Store the full event
            self.current_frame_event = event
        elif event.event_name == "detection":
            self.current_detections = event.data
        elif event.event_name == "worker_profile":
            # Update profiling data for the worker
            worker_name = event.worker_name
            task_time = event.data["task_time"]  # Extract task_time from profile data
            self.worker_profiles[worker_name] = task_time
        elif event.event_name == "command":
            # Handle command events
            if event.data == "quit":
                self.logger.warning("Received quit event")
                self._stop_event.set()
                return
        else:
            self.logger.debug("Received unknown event: %s", event.event_name)

    def _finish(self) -> None:
        """Clean up pygame resources."""
        pygame.font.quit()
        pygame.display.quit()
        self.logger.warning("Pygame modules stopped")
