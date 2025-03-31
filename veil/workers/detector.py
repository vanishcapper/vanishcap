"""Worker for detecting objects in frames using YOLOv5."""

import os
import queue
from pathlib import Path
from typing import Any, Dict

import numpy as np
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
                - use_tensorrt: Optional bool to use TensorRT (default: False)
                - tensorrt_engine: Optional path to save/load TensorRT engine
        """
        super().__init__("detector", config)
        self.model_path = config["model"]
        self.frame_skip = config.get("frame_skip", 1)  # Default to 1 frame
        self.use_tensorrt = config.get("use_tensorrt", False)
        self.tensorrt_engine = config.get("tensorrt_engine", None)
        self.logger.warning("Initialized detector worker with model: %s", self.model_path)

        # Initialize YOLO model
        if self.use_tensorrt:
            if self.tensorrt_engine and os.path.exists(self.tensorrt_engine):
                # Load existing TensorRT engine
                self.logger.warning("Loading TensorRT engine from: %s", self.tensorrt_engine)
                self.model = YOLO(self.tensorrt_engine)
                self.logger.warning("Successfully loaded TensorRT engine")
            else:
                # Convert PyTorch model to TensorRT
                self.logger.warning("Converting PyTorch model to TensorRT...")
                self.model = YOLO(self.model_path)
                if self.tensorrt_engine:
                    # Save engine to specified path
                    self.model.export(format="engine", device=0, half=True, workspace=4, verbose=False)
                    self.logger.warning("Saved TensorRT engine to: %s", self.tensorrt_engine)
                else:
                    # Use default engine path
                    engine_path = str(Path(self.model_path).with_suffix(".engine"))
                    self.logger.warning("Saved TensorRT engine to: %s", engine_path)
        else:
            # Use regular PyTorch model
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
                self.logger.info("Received frame %d", event.data["frame_number"])
        else:
            self.logger.debug("Received unknown event: %s", event.event_name)

    def _task(self) -> None:
        """Run one iteration of the detector loop."""
        # Skip if no frame available
        if self.latest_frame_event is None:
            return

        # Extract frame data
        frame = self.latest_frame_event.data["frame"]
        frame_number = self.latest_frame_event.data["frame_number"]

        # Log which frame we're processing
        self.logger.info("Processing frame %d", frame_number)

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
                self.logger.info(
                    "Detected %s (%.2f) at (%.1f, %.1f) - (%.1f, %.1f)",
                    class_name,
                    confidence,
                    x1,
                    y1,
                    x2,
                    y2,
                )

                detections.append(
                    {"bbox": [x1, y1, x2, y2], "confidence": confidence, "class_id": class_id, "class_name": class_name}
                )

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
                        self.logger.debug("Discarding old frame event %d", old_event.data["frame_number"])
            except queue.Empty:
                pass

            # Add the new frame event
            self.logger.debug("Worker %s dispatching frame event %d", self.name, event.data["frame_number"])
            self._event_queue.put(event)
            self.logger.debug("Worker %s frame event %d queued", self.name, event.data["frame_number"])
        else:
            # Use default queue behavior for non-frame events
            self.logger.debug("Worker %s dispatching event %s", self.name, event.event_name)
            self._event_queue.put(event)
            self.logger.debug("Worker %s event %s queued", self.name, event.event_name)

    def _finish(self) -> None:
        """Clean up detector resources."""
        self.logger.warning("Detector worker cleanup complete")
