"""Worker for detecting objects in frames using YOLOv5."""

import os
import time
from typing import Any, Dict

import numpy as np
import torch
from ultralytics import YOLO

from vanishcap.event import Event
from vanishcap.worker import Worker


class Detector(Worker):  # pylint: disable=too-many-instance-attributes
    """Worker that processes frames and emits detection events."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the detector worker.

        Args:
            config: Configuration dictionary containing:
                - name: Name of the worker
                - model: Base path to YOLO model (without extension)
                - frame_skip: Number of frames to skip between detections
                - log_level: Optional log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                - model_verbose: Optional verbose level for model (False, True) (default: False)
                - backend: Optional backend to use (pytorch, tensorrt, onnx) (default: pytorch)
        """
        super().__init__(config)
        self.model_base = config["model"]
        self.frame_skip = config.get("frame_skip", 1)  # Default to 1 frame
        self.backend = config.get("backend", "pytorch")
        self.model_verbose = config.get("model_verbose", False)
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
                model.export(
                    format="engine",
                    device=0,
                    half=True,
                    workspace=4,
                    verbose=self.model_verbose,
                    save_dir=self.assets_dir,
                )
                self.logger.warning("Saved TensorRT engine to: %s", model_path)
        elif self.backend == "onnx":
            model_path = os.path.join(self.assets_dir, f"{self.model_base}.onnx")
            if not os.path.exists(model_path):
                # Convert PyTorch model to ONNX
                self.logger.warning("Converting PyTorch model to ONNX...")
                pytorch_path = os.path.join(self.assets_dir, f"{self.model_base}.pt")
                model = YOLO(pytorch_path, task="detect")
                # Save ONNX to assets directory
                model.export(format="onnx", verbose=self.model_verbose, save_dir=self.assets_dir)
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

        if torch.cuda.is_available():
            self.logger.warning("CUDA is available on device %s", torch.cuda.get_device_name())
            self.device = torch.device("cuda")
        else:
            self.logger.warning("CUDA is not available, using CPU")
            self.device = torch.device("cpu")

        # Warm up model with blank image
        self.logger.warning("Warming up model with blank image...")
        blank_image = np.zeros((640, 640, 3), dtype=np.uint8)  # Standard YOLO input size
        _ = self.model(blank_image, verbose=self.model_verbose, device=self.device)
        self.logger.warning("Model warmup complete")

        # Initialize state
        self.frame_count = 0

    def _normalize_coordinates(
        self, coords: tuple[float, float, float, float], width: float, height: float
    ) -> tuple[float, float, float, float]:
        """Normalize coordinates to [-1, 1] range and swap tl,br -> bl,tr.

        Args:
            coords: Tuple of (x1, y1, x2, y2) coordinates in pixels
            width: Frame width in pixels
            height: Frame height in pixels

        Returns:
            tuple[float, float, float, float]: Normalized (x1, y1, x2, y2) coordinates in [-1, 1] range
        """
        x1, y1, x2, y2 = coords
        # Convert to [-1, 1] range
        norm_x1 = (x1 / width) * 2 - 1
        norm_y1 = -((y2 / height) * 2 - 1)
        norm_x2 = (x2 / width) * 2 - 1
        norm_y2 = -((y1 / height) * 2 - 1)
        return norm_x1, norm_y1, norm_x2, norm_y2

    def _task(self) -> None:  # pylint: disable=too-many-locals
        """Run one iteration of the detector loop: get latest events and process frame."""

        latest_frame_event = self._get_latest_events_and_clear().get("frame", None)
        if latest_frame_event is None:
            self.logger.debug("No frame event to process in this task iteration.")
            return

        # Code below is now executed only if latest_frame_event is not None
        self.frame_count += 1  # Increment frame count only when a frame is considered

        # Check if frame should be skipped
        if self.frame_count % self.frame_skip != 0:
            self.logger.debug(
                "Skipping frame %d due to frame_skip (%d %% %d != 0)",
                latest_frame_event.frame_number,
                self.frame_count,
                self.frame_skip,
            )
            return  # Skip processing this frame

        # Process the frame (logic moved back from _process_frame)
        frame_event = latest_frame_event
        # Extract frame data
        frame = frame_event.data
        frame_number = frame_event.frame_number

        # Log which frame we're processing
        self.logger.info("Processing frame %d", frame_number)

        start_time = time.perf_counter()
        results = self.model(frame, verbose=self.model_verbose, device=self.device)
        processing_time = time.perf_counter() - start_time
        self.logger.info(
            "Frame %d detection took %d ms (%d ms after frame acquisition)",
            frame_number,
            processing_time * 1000,
            1000 * (time.time() - frame_event.timestamp),
        )

        # Get frame dimensions
        height, width = frame.shape[:2]

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

                # Normalize bounding box coordinates to [-1, 1] range and swap to bl -> tr
                norm_x1, norm_y1, norm_x2, norm_y2 = self._normalize_coordinates((x1, y1, x2, y2), width, height)

                # Calculate center point in normalized coordinates
                center_x = (norm_x1 + norm_x2) / 2
                center_y = (norm_y1 + norm_y2) / 2

                # Log detection details
                self.logger.debug(
                    "Detected %s (%.2f) at (%.2f, %.2f) - (%.2f, %.2f): orig coords: (%f, %f, %f, %f)",
                    class_name,
                    confidence,
                    norm_x1,
                    norm_y1,
                    norm_x2,
                    norm_y2,
                    x1,
                    y1,
                    x2,
                    y2,
                )

                detections.append(
                    {
                        "bbox": [norm_x1, norm_y1, norm_x2, norm_y2],
                        "confidence": confidence,
                        "class_id": class_id,
                        "class_name": class_name,
                        "x": center_x,
                        "y": center_y,
                    }
                )

        # Log summary of all detections, sorted by class_id
        if detections:
            summary = ", ".join(
                f"{d['class_name']}({d['confidence']:.2f})" for d in sorted(detections, key=lambda x: x["class_id"])
            )
            self.logger.info("Detections in frame %d: %s", frame_number, summary)
        else:
            self.logger.info("No detections in frame %d", frame_number)

        # Emit detection event with frame number
        self._emit(Event(self.name, "detection", detections, frame_number=frame_number))

    def _finish(self) -> None:
        """Clean up detector resources."""
        self.logger.warning("Detector worker cleanup complete")
