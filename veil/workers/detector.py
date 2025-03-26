"""Worker for detecting objects in frames using YOLOv5."""

from typing import Any, Dict, List, Optional
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
        self.frame_skip = config.get("frame_skip", 1)
        self.logger.info(f"Initialized detector worker with model: {self.model_path}")
        
        # Initialize YOLO model
        self.model = YOLO(self.model_path)
        self.logger.info("Successfully loaded YOLO model")
        
        # Initialize state
        self.frame_count = 0
        self.frame_queue = queue.Queue()

    def _task(self) -> None:
        """Run one iteration of the detector loop."""
        # Get frame from queue
        try:
            frame = self.frame_queue.get_nowait()
        except queue.Empty:
            return
            
        # Run detection
        results = self.model(frame)
        
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
                
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name
                })
        
        # Emit detection event
        self._emit(Event(self.name, "detection", detections))

    def __call__(self, event: Event) -> None:
        """Handle incoming events.

        Args:
            event: Event to handle
        """
        if event.event_name == "frame":
            self.frame_count += 1
            if self.frame_count % self.frame_skip == 0:
                self.frame_queue.put(event.data)
        else:
            self.logger.debug(f"Received unknown event: {event.event_name}")

    def _finish(self) -> None:
        """Clean up detector resources."""
        # Nothing to clean up for detector
        pass 