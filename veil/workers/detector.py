"""Detector worker for object detection using YOLO."""

from typing import Any, Dict, List

import cv2
from ultralytics import YOLO

from veil.event import Event
from veil.worker import Worker


class Detector(Worker):
    """Worker that performs object detection on frames."""

    def __init__(self, config: Dict[str, Any] = None) -> None:
        """Initialize the detector worker.

        Args:
            config: Optional dictionary of configuration values
        """
        super().__init__("detector", config)
        self.model = YOLO(self.config.get("model", "yolov5s.pt"))
        self.frame_skip = self.config.get("frame_skip", 1)
        self.frame_count = 0

    def __call__(self, event: Event) -> Event:
        """Handle frame events and perform detection.

        Args:
            event: The event to handle

        Returns:
            Event: A response event containing detection results
        """
        if event.event_name == "frame":
            self.frame_count += 1
            if self.frame_count % self.frame_skip != 0:
                return Event(self.name, "skip", None)

            frame = event.data["frame"]
            results = self.model(frame)
            
            # Extract detections
            detections = []
            for r in results:
                for box in r.boxes:
                    detections.append({
                        "class": int(box.cls[0]),
                        "confidence": float(box.conf[0]),
                        "bbox": box.xyxy[0].tolist(),
                    })

            return Event(self.name, "detection", {
                "frame_number": event.data["frame_number"],
                "detections": detections,
            })
        
        return Event(self.name, "error", f"Unknown event: {event.event_name}") 