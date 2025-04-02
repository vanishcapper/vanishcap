"""Worker for detecting objects in frames using YOLOv5."""

import os
import queue
from typing import Any, Dict

import numpy as np
from ultralytics import YOLO

from vanishcap.event import Event
from vanishcap.worker import Worker


class Detector(Worker):
    """Worker that processes frames and emits detection events."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the detector worker.

        Args:
            config: Configuration dictionary containing:
                - model: Base path to YOLO model (without extension)
                - frame_skip: Number of frames to skip between detections
                - log_level: Optional log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                - backend: Optional backend to use (pytorch, tensorrt, onnx) (default: pytorch)
        """
        super().__init__("detector", config)
        self.model_base = config["model"]
        self.frame_skip = config.get("frame_skip", 1)  # Default to 1 frame
        self.backend = config.get("backend", "pytorch")
        self.logger.warning("Initialized detector worker with model base: %s", self.model_base)

        # Create assets directory if it doesn't exist
        self.assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assets")
        os.makedirs(self.assets_dir, exist_ok=True)
        self.logger.warning("Using assets directory: %s", self.assets_dir)

        # Determine model path based on backend
        if self.backend == "tensorrt":
            model_path = os.path.join(self.assets_dir, f"{self.model_base}.engine")
            if not os.path.exists(model_path):
                # Convert PyTorch model to TensorRT
                self.logger.warning("Converting PyTorch model to TensorRT...")
                pytorch_path = os.path.join(self.assets_dir, f"{self.model_base}.pt")
                model = YOLO(pytorch_path, task="detect")
                # Save engine to assets directory
                model.export(format="engine", device=0, half=True, workspace=4, verbose=False, save_dir=self.assets_dir)
                self.logger.warning("Saved TensorRT engine to: %s", model_path)
        elif self.backend == "onnx":
            model_path = os.path.join(self.assets_dir, f"{self.model_base}.onnx")
            if not os.path.exists(model_path):
                # Convert PyTorch model to ONNX
                self.logger.warning("Converting PyTorch model to ONNX...")
                pytorch_path = os.path.join(self.assets_dir, f"{self.model_base}.pt")
                model = YOLO(pytorch_path, task="detect")
                # Save ONNX to assets directory
                model.export(format="onnx", verbose=False, save_dir=self.assets_dir)
                self.logger.warning("Saved ONNX model to: %s", model_path)
        elif self.backend == "pytorch":
            # Use regular PyTorch model
            model_path = os.path.join(self.assets_dir, f"{self.model_base}.pt")
        else:
            # Raise exception for unknown backend
            raise ValueError(f"Unknown backend: {self.backend}. Supported backends are: pytorch, tensorrt, onnx")

        # Load the model (same syntax for all backends)
        self.logger.warning("Loading %s model from: %s", self.backend, model_path)
        self.model = YOLO(model_path, task="detect")
        self.logger.warning("Successfully loaded %s model", self.backend)

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
