"""Worker for displaying video frames and detection results."""

import traceback
from typing import Any, Dict, List, Optional, Tuple
import queue
import time
import cv2
import pygame
import pygame.display
import pygame.event
import pygame.image
import pygame.time

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
        self.logger.info(f"Initialized UI worker with window size: {self.window_size} and target FPS: {self.target_fps}")

        # Initialize pygame modules
        try:
            pygame.display.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Veil UI")
            self.logger.info("Pygame modules initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize pygame modules: {e}")
            traceback.print_exc()
            raise

        # Initialize state
        self.current_frame = None
        self.current_detections = []

        # Profiling data
        self.worker_profiles: Dict[str, float] = {}  # Maps worker name to last task time
        self.profile_font = pygame.font.Font(None, 20)  # Small font for profiling text

    def _task(self) -> None:
        """Run one iteration of the UI loop."""
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                print(f"DEBUG: UI worker received ESC/QUIT event")
                self.logger.info("Received quit event")
                # Emit stop event to notify controller
                print(f"DEBUG: UI worker emitting stop event")
                self._emit(Event(self.name, "stop", None))
                # Set stop flag to stop UI loop
                print(f"DEBUG: UI worker setting stop flag")
                self._stop_event.set()
                # Close pygame window
                print(f"DEBUG: UI worker closing pygame window")
                pygame.display.quit()
                # Process any pending events before returning
                if self._event_queue is not None:
                    try:
                        event = self._event_queue.get_nowait()
                        self(event)
                    except queue.Empty:
                        pass
                return

        # Update display if we have a frame and enough time has passed
        if self.current_frame is not None:
            current_time = time.time()
            if (current_time - self.last_frame_time) >= self.frame_time:
                # Flip frame horizontally
                frame_flipped = cv2.flip(self.current_frame, 1)  # 1 means flip horizontally

                # Convert frame to RGB for pygame
                frame_rgb = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)

                # Convert to pygame surface
                frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))

                # Scale frame to window size
                frame_surface = pygame.transform.scale(frame_surface, self.window_size)

                # Draw frame
                self.screen.blit(frame_surface, (0, 0))

                # Draw detections if any
                if self.current_detections:
                    for detection in self.current_detections:
                        bbox = detection["bbox"]

                        # Get frame dimensions for scaling
                        frame_height, frame_width = self.current_frame.shape[:2]

                        # Scale bbox to window size and flip x coordinates
                        x1, y1, x2, y2 = [
                            int(bbox[0] * self.window_size[0] / frame_width),
                            int(bbox[1] * self.window_size[1] / frame_height),
                            int(bbox[2] * self.window_size[0] / frame_width),
                            int(bbox[3] * self.window_size[1] / frame_height)
                        ]

                        # Flip x coordinates
                        x1_flipped = self.window_size[0] - x2
                        x2_flipped = self.window_size[0] - x1

                        # Draw bounding box
                        pygame.draw.rect(self.screen, (0, 255, 0), (x1_flipped, y1, x2_flipped-x1_flipped, y2-y1), 2)

                        # Draw label
                        label = f"{detection['class_name']} ({detection['confidence']:.2f})"
                        font = pygame.font.Font(None, 24)
                        text = font.render(label, True, (0, 255, 0))
                        self.screen.blit(text, (x1_flipped, y1-20))

                # Draw profiling information in bottom left
                y_offset = self.window_size[1] - 25  # Start 25 pixels from bottom
                for worker_name, task_time in self.worker_profiles.items():
                    # Format time in milliseconds with 2 decimal places
                    time_str = f"{task_time * 1000:.2f}ms"
                    profile_text = f"{worker_name}: {time_str}"
                    text = self.profile_font.render(profile_text, True, (255, 255, 255))
                    self.screen.blit(text, (10, y_offset))
                    y_offset -= 20  # Move up 20 pixels for next worker

                # Update display
                pygame.display.flip()
                self.last_frame_time = current_time

    def __call__(self, event: Event) -> None:
        """Handle incoming events.

        Args:
            event: Event to handle
        """
        if event.event_name == "frame":
            self.current_frame = event.data
            self.logger.debug(f"Received frame event")
        elif event.event_name == "detection":
            self.current_detections = event.data
            self.logger.debug(f"Received detection event with {len(event.data)} detections")
        elif event.event_name == "stop":
            self.logger.info("Received stop event")
            self._stop_event.set()
            pygame.display.quit()
        elif event.event_name == "worker_profile":
            # Update profiling data for the worker
            self.worker_profiles[event.worker_name] = event.data["task_time"]
            self.logger.debug(f"Updated profile for {event.worker_name}: {event.data['task_time']:.3f}s")
        else:
            self.logger.debug(f"Received unknown event: {event.event_name}")

    def _finish(self) -> None:
        """Clean up pygame resources."""
        pygame.font.quit()
        pygame.display.quit()
        self.logger.info("Pygame modules stopped")