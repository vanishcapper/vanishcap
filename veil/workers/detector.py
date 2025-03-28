"""Worker for detecting objects in frames using YOLOv5."""

from typing import Any, Dict, List, Optional
import numpy as np
import queue

from ultralytics import YOLO
from veil.event import Event
from veil.worker import Worker


class Detector(Worker):
    """Worker that processes frames and emits detection events."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the detector worker.

        Args:
            config: Configuration dictionary containing:
                - model: Path to YOLO model file
                - frame_skip: Number of frames to skip between detections
                - log_level: Optional log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        super().__init__("detector", config)
        self.model_path = config["model"]
        self.frame_skip = config.get("frame_skip", 1)  # Default to 1 frame
        self.logger.warning(f"Initialized detector worker with model: {self.model_path}")

        # Initialize YOLO model
        self.model = YOLO(self.model_path)
        self.logger.warning("Successfully loaded YOLO model")

        # Warm up model with blank image
        self.logger.warning("Warming up model with blank image...")
        blank_image = np.zeros((640, 640, 3), dtype=np.uint8)  # Standard YOLO input size
        _ = self.model(blank_image, verbose=False)
        self.logger.warning("Model warmup complete")

        # Initialize state
        self.frame_count = 0
        self.latest_frame_event = None

    def __call__(self, event: Event) -> None:
        """Handle incoming events.

        Args:
            event: Event to handle
        """
        if event.event_name == "frame":
            self.frame_count += 1
            if self.frame_count % self.frame_skip == 0:
                # Store the full event
                self.latest_frame_event = event
                self.logger.info(f"Received frame {event.data['frame_number']}")
        else:
            self.logger.debug(f"Received unknown event: {event.event_name}")

    def _task(self) -> None:
        """Run one iteration of the detector loop."""
        # Skip if no frame available
        if self.latest_frame_event is None:
            return

        # Extract frame data
        frame = self.latest_frame_event.data["frame"]
        frame_number = self.latest_frame_event.data["frame_number"]

        # Log which frame we're processing
        self.logger.info(f"Processing frame {frame_number}")

        # Run detection with verbose=False to suppress YOLO's default logging
        results = self.model(frame, verbose=False)

        # Process results
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                # Log detection details
                self.logger.info(f"Detected {class_name} ({confidence:.2f}) at ({x1:.1f}, {y1:.1f}) - ({x2:.1f}, {y2:.1f})")

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name
                })

        # Emit detection event
        self._emit(Event(self.name, "detection", detections))

        # Clear the frame event after processing
        self.latest_frame_event = None

    def _dispatch(self, event: Event) -> None:
        """Handle an incoming event from the Controller.

        For frame events, we only keep the latest one in the queue.
        For other events, we use the default queue behavior.

        Args:
            event: Event to handle
        """
        if event.event_name == "frame":
            # Clear any existing frame events from the queue
            try:
                while True:
                    old_event = self._event_queue.get_nowait()
                    if old_event.event_name == "frame":
                        self.logger.debug(f"Discarding old frame event {old_event.data['frame_number']}")
            except queue.Empty:
                pass

            # Add the new frame event
            self.logger.debug(f"Worker {self.name} dispatching frame event {event.data['frame_number']}")
            self._event_queue.put(event)
            self.logger.debug(f"Worker {self.name} frame event {event.data['frame_number']} queued")
        else:
            # Use default queue behavior for non-frame events
            self.logger.debug(f"Worker {self.name} dispatching event {event.event_name}")
            self._event_queue.put(event)
            self.logger.debug(f"Worker {self.name} event {event.event_name} queued")

    def _finish(self) -> None:
        """Clean up detector resources."""
        self.logger.warning("Detector worker cleanup complete")