"""Worker for displaying video frames and detection results."""

# pylint: disable=wrong-import-position,too-many-instance-attributes,no-member,too-many-locals
# pylint: disable=too-many-branches,too-many-statements,too-many-nested-blocks

import math  # Added for grid calculation
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np  # Added for canvas creation and manipulation

from vanishcap.event import Event
from vanishcap.worker import Worker


class Ui(Worker):
    """Worker that displays video frames and detection results from multiple sources in a tiled view."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the UI worker.

        Args:
            config: Configuration dictionary containing:
                - name: Name of the worker
                - window_size: Optional tuple of (width, height) for the display window (default: 800x600)
                - log_level: Optional log level
                - background_color: Optional BGR tuple for empty tiles (default: [100, 100, 100] gray)
                - events: List of event configurations for video sources
        """
        super().__init__(config)
        # Display settings
        self.window_width, self.window_height = config.get("window_size", (800, 600))
        self.background_color = np.array(config.get("background_color", [100, 100, 100]), dtype=np.uint8)

        # Count video feeds from events config
        self.expected_video_feeds = self._count_video_feeds(config.get("events", []))
        self.logger.warning(
            "Initialized UI worker with window size: %dx%d and %d expected video feeds",
            self.window_width,
            self.window_height,
            self.expected_video_feeds,
        )

        # Initialize state for multiple sources
        self.latest_frames: Dict[str, Event] = {}  # video_source_name -> latest frame Event
        self.latest_detections: Dict[str, List[Dict]] = {}  # detector_source_name -> latest list of detections
        self.worker_profiles: Dict[str, float] = {}  # worker_name -> last task time
        self.frame_sizes: Dict[str, Tuple[int, int]] = {}  # video_source_name -> (width, height)

        # Font settings for OpenCV
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1
        self.font_color = (255, 255, 255)  # White
        self.box_color = (0, 255, 0)
        self.profile_color = (200, 200, 0)  # Cyan/Yellow for profile text

    def _count_video_feeds(self, events: List[Dict[str, str]]) -> int:
        """Count the number of video feeds from the events configuration.

        Args:
            events: List of event configurations

        Returns:
            int: Number of expected video feeds
        """
        video_feeds = 0
        for event in events:
            if isinstance(event, dict) and len(event) == 1:
                _, event_type = next(iter(event.items()))
                if event_type == "frame":
                    video_feeds += 1
        return video_feeds

    def _denormalize_coordinates(self, x: float, y: float, width: float, height: float) -> tuple[int, int]:
        """Convert normalized coordinates [-1, 1] back to pixel coordinates.

        Args:
            x: Normalized x coordinate [-1, 1]
            y: Normalized y coordinate [-1, 1]
            width: Frame width in pixels
            height: Frame height in pixels

        Returns:
            tuple[int, int]: Pixel coordinates
        """
        # Convert from [-1, 1] to pixel coordinates
        pixel_x = int((x + 1) * width / 2)
        pixel_y = int((y + 1) * height / 2)
        return pixel_x, pixel_y

    def _get_associated_workers(self, video_source_name: str) -> Dict[str, str]:
        """Infer associated worker names based on a video source name (e.g., video1 -> detector1)."""
        base_name = video_source_name.replace("video", "")  # Assumes 'video' prefix
        return {
            "video": video_source_name,
            "detector": f"detector{base_name}",
            "navigator": f"navigator{base_name}",
            "drone": f"drone{base_name}",
            # Add other worker types if needed
        }

    def _draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> None:
        """Draw detection boxes and labels onto a frame."""
        if not detections:
            return

        orig_height, orig_width = frame.shape[:2]

        for detection in detections:
            norm_x1, norm_y1, norm_x2, norm_y2 = detection["bbox"]

            # Convert normalized coordinates to pixel coordinates relative to ORIGINAL frame size
            # (assuming detections were made on the original before potential resize)
            # We need a way to know the original size if the frame passed here is already resized...
            # Let's ASSUME for now the frame passed here is the *original* size from the video worker.
            # The tiling logic will handle resizing later.
            px1 = int((norm_x1 + 1) * orig_width / 2)
            py1 = int((norm_y1 + 1) * orig_height / 2)
            px2 = int((norm_x2 + 1) * orig_width / 2)
            py2 = int((norm_y2 + 1) * orig_height / 2)

            # OpenCV uses top-left origin, y increases downwards.
            # Our normalized coords might be bottom-left origin. Let's assume they are.
            # Convert bottom-left origin normalized [-1, 1] to top-left pixel coords [0, H] or [0, W]
            x1 = px1
            y1 = orig_height - py2  # y1 corresponds to higher normalized y (top of box)
            x2 = px2
            y2 = orig_height - py1  # y2 corresponds to lower normalized y (bottom of box)

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.box_color, 2)

            # Draw label
            label = f"{detection['class_name']} ({detection['confidence']:.2f})"
            # Put label slightly below the top-left corner of the box
            text_y = y1 + 15 if y1 > 15 else y1 - 5  # Adjust if box is near top edge
            cv2.putText(frame, label, (x1, text_y), self.font, self.font_scale, self.font_color, self.font_thickness)

    def _draw_profiling(self, frame: np.ndarray, associated_workers: Dict[str, str]) -> None:
        """Draw profiling information onto a frame for associated workers."""
        self.logger.debug("Worker profiles: %s", self.worker_profiles)
        y_offset = 20  # Start drawing below top edge
        for worker_type, worker_name in associated_workers.items():
            task_time = self.worker_profiles.get(worker_name)
            if task_time is not None:
                text = f"{worker_type}: {task_time*1000:.1f}ms"
                self.logger.debug("Drawing profiling text: %s", text)
                cv2.putText(
                    frame, text, (10, y_offset), self.font, self.font_scale, self.profile_color, self.font_thickness
                )
                y_offset += 15  # Move down for next line

    def _calculate_window_size(self) -> Tuple[int, int]:
        """Calculate the optimal window size based on frame dimensions and number of feeds.

        Returns:
            Tuple[int, int]: (width, height) of the window
        """
        if not self.frame_sizes or self.expected_video_feeds <= 0:
            return self.window_width, self.window_height

        # Get the maximum frame dimensions
        max_width = max(w for w, _ in self.frame_sizes.values())
        max_height = max(h for _, h in self.frame_sizes.values())

        # Calculate grid dimensions
        cols = math.ceil(math.sqrt(max(1, self.expected_video_feeds)))
        rows = math.ceil(self.expected_video_feeds / max(1, cols))

        # Calculate total window size needed
        total_width = max_width * cols
        total_height = max_height * rows

        # Add some padding
        padding = 20
        total_width += padding * (cols + 1)
        total_height += padding * (cols + 1)

        return total_width, total_height

    def _task(self) -> None:
        """Run one iteration of the UI loop: process events, create tiled view, display."""
        # --- Event Processing ---
        latest_events = self._get_latest_events_and_clear()

        for _, event in latest_events.items():
            worker_name = event.worker_name
            event_name = event.event_name

            if event_name == "frame":
                self.latest_frames[worker_name] = event
                # Store frame dimensions if not already known
                if worker_name not in self.frame_sizes and event.data is not None:
                    height, width = event.data.shape[:2]
                    self.frame_sizes[worker_name] = (width, height)
            elif event_name == "detection":
                # Assuming event.data is the list of detection dicts
                self.latest_detections[worker_name] = event.data
            elif event_name == "worker_profile":
                # Assuming event.data is {'task_time': float}
                if isinstance(event.data, dict) and "task_time" in event.data:
                    self.worker_profiles[worker_name] = event.data["task_time"]
                    self.logger.debug(
                        "Updated worker profile for %s: %s", worker_name, self.worker_profiles[worker_name]
                    )
                    self.logger.debug("Worker profiles: %s", self.worker_profiles)
                else:
                    self.logger.warning("Received malformed worker_profile event from %s: %s", worker_name, event.data)

        # --- Tiled Frame Rendering ---
        video_sources = sorted(self.latest_frames.keys())
        num_sources = len(video_sources)

        if num_sources == 0:
            # If no frames received yet, display a blank screen or placeholder
            canvas = np.full((self.window_height, self.window_width, 3), self.background_color, dtype=np.uint8)
            cv2.putText(
                canvas,
                "Waiting for video streams...",
                (50, self.window_height // 2),
                self.font,
                1.0,
                self.font_color,
                1,
            )
        else:
            # Calculate optimal window size based on frame dimensions
            window_width, window_height = self._calculate_window_size()

            # Update window size if needed
            if window_width != self.window_width or window_height != self.window_height:
                self.window_width, self.window_height = window_width, window_height
                cv2.resizeWindow("vanishcap", self.window_width, self.window_height)

            # Calculate grid dimensions
            cols = math.ceil(math.sqrt(self.expected_video_feeds))
            cols = max(1, cols)  # Ensure cols is at least 1 to avoid division by zero

            # Calculate tile dimensions based on maximum frame size
            max_width = max(w for w, _ in self.frame_sizes.values())
            max_height = max(h for _, h in self.frame_sizes.values())

            # Add padding between tiles
            padding = 20
            tile_width = max_width + padding
            tile_height = max_height + padding

            # Create the main canvas to draw all tiles onto
            canvas = np.full((self.window_height, self.window_width, 3), self.background_color, dtype=np.uint8)

            for i, video_source_name in enumerate(video_sources):
                frame_event = self.latest_frames.get(video_source_name)
                if not frame_event:
                    continue

                frame = frame_event.data  # Assume data is the numpy frame
                if frame is None:
                    # Handle case where frame data might be None
                    tile_content = np.full((tile_height, tile_width, 3), [0, 0, 50], dtype=np.uint8)  # Dark Red
                    cv2.putText(
                        tile_content,
                        f"{video_source_name}: No Frame",
                        (10, tile_height // 2),
                        self.font,
                        0.6,
                        self.font_color,
                        1,
                    )
                else:
                    frame = frame.copy()  # Work on a copy

                    # --- Annotate the frame ---
                    associated_workers = self._get_associated_workers(video_source_name)
                    detector_name = associated_workers.get("detector")

                    # Draw detections if available
                    if detector_name:
                        detections = self.latest_detections.get(detector_name, [])
                        self._draw_detections(frame, detections)

                    # Draw profiling info
                    self._draw_profiling(frame, associated_workers)

                    # Draw frame number / source name
                    frame_info_text = f"{video_source_name} | Frame: {frame_event.frame_number}"
                    # Position bottom-left for frame info text
                    text_x = 10
                    text_y = frame.shape[0] - 10  # 10 pixels from bottom
                    cv2.putText(
                        frame,
                        frame_info_text,
                        (text_x, text_y),
                        self.font,
                        self.font_scale,
                        self.font_color,
                        self.font_thickness,
                    )

                    # Use frame at native resolution
                    tile_content = frame

                # --- Place the tile onto the main canvas ---
                row_idx = i // cols
                col_idx = i % cols
                y_start = row_idx * tile_height + padding
                y_end = y_start + frame.shape[0] if frame is not None else tile_height
                x_start = col_idx * tile_width + padding
                x_end = x_start + frame.shape[1] if frame is not None else tile_width

                # Ensure dimensions match exactly before assignment
                canvas[y_start:y_end, x_start:x_end] = tile_content[: (y_end - y_start), : (x_end - x_start)]

        # --- Display and Handle Quit ---
        try:
            cv2.imshow("vanishcap", canvas)
            # Check for quit key (ESC) or window close
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or cv2.getWindowProperty("vanishcap", cv2.WND_PROP_VISIBLE) < 1:
                self.logger.warning("Received quit event from OpenCV")
                self._emit(Event(self.name, "stop", None))  # Ask controller to stop everything
                self._stop_event.set()  # Stop this worker's loop
                return
        except cv2.error as e:
            if "window was destroyed" in str(e).lower():
                self.logger.warning("OpenCV window was closed externally.")
                self._emit(Event(self.name, "stop", None))
                self._stop_event.set()
            else:
                self.logger.error("OpenCV error during imshow/waitKey: %s", e, exc_info=True)
                self._emit(Event(self.name, "stop", None))  # Stop on other CV errors too
                self._stop_event.set()

    def _finish(self) -> None:
        """Clean up UI resources (Window closing is now handled by Controller)."""
        # cv2.destroyAllWindows() # Removed - Handled by Controller.__exit__
        self.logger.warning("UI worker finished.")
