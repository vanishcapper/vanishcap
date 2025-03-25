"""UI worker for displaying frames and detections using pygame."""

import pygame
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from veil.event import Event
from veil.worker import Worker


class UI(Worker):
    """Worker that displays frames and detections using pygame."""

    def __init__(self, name: str, config: Dict[str, Any] = None) -> None:
        """Initialize the UI worker.

        Args:
            name: The name of this worker instance
            config: Optional dictionary of configuration values
        """
        super().__init__(name, config)
        self.window_size = self.config.get("window_size", (800, 600))
        self.display = None
        self.clock = None
        self.font = None
        self.running = False

    def __call__(self, event: Event) -> Event:
        """Handle events and update the display.

        Args:
            event: The event to handle

        Returns:
            Event: A response event indicating display status
        """
        if event.event_name == "start":
            if not self.running:
                pygame.init()
                self.display = pygame.display.set_mode(self.window_size)
                pygame.display.set_caption("Veil Detection Display")
                self.clock = pygame.time.Clock()
                self.font = pygame.font.Font(None, 36)
                self.running = True
            return Event(self.name, "ready", None)

        elif event.event_name == "frame":
            if not self.running:
                return Event(self.name, "error", "Display not initialized")

            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return Event(self.name, "quit", None)

            # Convert frame to pygame surface
            frame = event.data["frame"]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frame = cv2.resize(frame, self.window_size)
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))

            # Draw frame
            self.display.blit(surface, (0, 0))

            # Draw frame number
            frame_text = self.font.render(f"Frame: {event.data['frame_number']}", True, (255, 255, 255))
            self.display.blit(frame_text, (10, 10))

            pygame.display.flip()
            self.clock.tick(30)  # Limit to 30 FPS

            return Event(self.name, "displayed", None)

        elif event.event_name == "detection":
            if not self.running:
                return Event(self.name, "error", "Display not initialized")

            # Get the frame from the event data
            frame = event.data.get("frame")
            if frame is None:
                return Event(self.name, "error", "No frame data in detection event")

            # Convert frame to pygame surface
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.window_size)
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))

            # Draw frame
            self.display.blit(surface, (0, 0))

            # Draw detections
            for detection in event.data["detections"]:
                bbox = detection["bbox"]
                # Scale bbox to window size
                bbox = self._scale_bbox(bbox, frame.shape[:2], self.window_size)
                # Draw rectangle
                pygame.draw.rect(
                    self.display,
                    (0, 255, 0),  # Green color for boxes
                    bbox,
                    2,
                )
                # Draw class and confidence
                text = f"{detection['class']}: {detection['confidence']:.2f}"
                text_surface = self.font.render(text, True, (255, 255, 255))
                self.display.blit(text_surface, (bbox[0], bbox[1] - 25))

            # Draw frame number
            frame_text = self.font.render(f"Frame: {event.data['frame_number']}", True, (255, 255, 255))
            self.display.blit(frame_text, (10, 10))

            pygame.display.flip()
            self.clock.tick(30)

            return Event(self.name, "displayed", None)

        elif event.event_name == "stop":
            if self.running:
                pygame.quit()
                self.running = False
            return Event(self.name, "stopped", None)

        return Event(self.name, "error", f"Unknown event: {event.event_name}")

    def _scale_bbox(self, bbox: List[float], src_size: Tuple[int, int], dst_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Scale bounding box coordinates to match display size.

        Args:
            bbox: Original bounding box [x1, y1, x2, y2]
            src_size: Original image size (height, width)
            dst_size: Target display size (width, height)

        Returns:
            Tuple[int, int, int, int]: Scaled bounding box coordinates
        """
        scale_x = dst_size[0] / src_size[1]
        scale_y = dst_size[1] / src_size[0]
        return (
            int(bbox[0] * scale_x),
            int(bbox[1] * scale_y),
            int(bbox[2] * scale_x),
            int(bbox[3] * scale_y),
        ) 