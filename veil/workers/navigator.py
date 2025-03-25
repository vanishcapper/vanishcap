"""Navigator worker for handling navigation based on detections."""

from typing import Any, Dict, List

from veil.event import Event
from veil.worker import Worker


class Navigator(Worker):
    """Worker that handles navigation based on object detections."""

    def __init__(self, config: Dict[str, Any] = None) -> None:
        """Initialize the navigator worker.

        Args:
            config: Optional dictionary of configuration values
        """
        super().__init__("navigator", config)
        self.target_class = 0  # 0 is person in COCO dataset

    def __call__(self, event: Event) -> Event:
        """Handle detection events and determine navigation actions.

        Args:
            event: The event to handle

        Returns:
            Event: A response event containing navigation commands
        """
        if event.event_name == "detection":
            detections = event.data["detections"]
            frame_number = event.data["frame_number"]
            
            # Find humans in the frame
            humans = [d for d in detections if d["class"] == self.target_class]
            
            if not humans:
                return Event(self.name, "no_target", {
                    "frame_number": frame_number,
                    "action": "search",
                })
            
            # Get the largest human detection
            largest_human = max(humans, key=lambda x: self._bbox_area(x["bbox"]))
            
            # Calculate center of detection
            bbox = largest_human["bbox"]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Determine action based on position
            action = self._determine_action(center_x, center_y)
            
            return Event(self.name, "command", {
                "frame_number": frame_number,
                "action": action,
                "target_position": {
                    "x": center_x,
                    "y": center_y,
                },
            })
        
        return Event(self.name, "error", f"Unknown event: {event.event_name}")

    def _bbox_area(self, bbox: List[float]) -> float:
        """Calculate the area of a bounding box.

        Args:
            bbox: List of [x1, y1, x2, y2] coordinates

        Returns:
            float: Area of the bounding box
        """
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    def _determine_action(self, center_x: float, center_y: float) -> str:
        """Determine navigation action based on target position.

        Args:
            center_x: X coordinate of target center
            center_y: Y coordinate of target center

        Returns:
            str: Navigation action to take
        """
        # Simple logic based on target position in frame
        if center_x < 0.3:
            return "move_left"
        elif center_x > 0.7:
            return "move_right"
        elif center_y < 0.3:
            return "move_forward"
        elif center_y > 0.7:
            return "move_backward"
        else:
            return "hover" 