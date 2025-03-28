"""Worker for displaying video frames and detection results."""

import traceback
from typing import Any, Dict, Optional
import queue
import time
import os
import cv2

# Suppress pygame startup message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame
import pygame.display
import pygame.event
import pygame.image

from veil.event import Event
from veil.worker import Worker


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
        self.window_size = config.get("window_size", (800, 600))
        self.target_fps = config.get("fps", 30)
        self.frame_time = 1.0 / self.target_fps
        self.last_frame_time = 0
        self.logger.warning(
            f"Initialized UI worker with window size: {self.window_size} and target FPS: {self.target_fps}"
        )

        # Initialize pygame modules
        try:
            pygame.display.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Veil UI")
            self.logger.warning("Pygame modules initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize pygame modules: {e}")
            traceback.print_exc()
            raise

        # Initialize state
        self.current_frame_event = None
        self.current_detections = []

        # Profiling data
        self.worker_profiles: Dict[str, float] = {}  # Maps worker name to last task time
        self.profile_font = pygame.font.Font(None, 20)  # Small font for profiling text

    def _task(self) -> None:
        """Run one iteration of the UI loop."""
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
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
        if self.current_frame_event is None:
            return

        current_time = time.time()
        if current_time - self.last_frame_time >= self.frame_time:
            # Extract frame data
            frame = self.current_frame_event.data["frame"]
            frame_number = self.current_frame_event.data["frame_number"]

            # Log which frame we're displaying
            self.logger.info(f"Displaying frame {frame_number}")

            # Convert frame to pygame surface
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.image.frombuffer(frame_rgb.tobytes(), frame_rgb.shape[1::-1], "RGB")

            # Get original frame dimensions
            orig_height, orig_width = frame.shape[:2]

            # Scale frame to window size
            frame_surface = pygame.transform.scale(frame_surface, self.window_size)

            # Draw frame
            self.screen.blit(frame_surface, (0, 0))

            # Draw detection boxes
            for detection in self.current_detections:
                bbox = detection["bbox"]
                # Scale bbox coordinates to match scaled frame size
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                x1 = int(x1 * self.window_size[0] / orig_width)
                y1 = int(y1 * self.window_size[1] / orig_height)
                x2 = int(x2 * self.window_size[0] / orig_width)
                y2 = int(y2 * self.window_size[1] / orig_height)
                pygame.draw.rect(self.screen, (0, 255, 0), (x1, y1, x2 - x1, y2 - y1), 2)

                # Draw label
                label = f"{detection['class_name']} ({detection['confidence']:.2f})"
                text = self.profile_font.render(label, True, (0, 255, 0))
                self.screen.blit(text, (x1, y1 - 20))

            # Draw frame number in top right
            frame_text = self.profile_font.render(f"Frame: {frame_number}", True, (255, 255, 255))
            text_rect = frame_text.get_rect()
            text_rect.topright = (self.window_size[0] - 10, 10)
            self.screen.blit(frame_text, text_rect)

            # Draw profiling data
            y = 10
            for worker_name, task_time in self.worker_profiles.items():
                text = self.profile_font.render(f"{worker_name}: {task_time*1000:.1f}ms", True, (255, 255, 255))
                self.screen.blit(text, (10, y))
                y += 20

            # Update display
            pygame.display.flip()
            self.last_frame_time = current_time

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
            # Log detection drawing
            for det in self.current_detections:
                x1, y1, x2, y2 = det["bbox"]
                self.logger.info(
                    f"Drawing {det['class_name']} ({det['confidence']:.2f}) at pixels ({x1:.1f}, {y1:.1f}) - ({x2:.1f}, {y2:.1f})"
                )
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
            self.logger.debug(f"Received unknown event: {event.event_name}")

    def _finish(self) -> None:
        """Clean up pygame resources."""
        pygame.font.quit()
        pygame.display.quit()
        self.logger.warning("Pygame modules stopped")
