"""Unit tests for the UI worker."""

# pylint: disable=protected-access
# pylint: disable=unnecessary-pass
# pylint: disable=too-many-instance-attributes

import unittest
from unittest.mock import patch
import numpy as np

from vanishcap.workers.ui import Ui
from vanishcap.event import Event


class TestUI(unittest.TestCase):
    """Test cases for the UI worker."""

    def setUp(self):  # pylint: disable=arguments-differ
        """Set up test fixtures."""
        self.config = {"name": "ui", "window_size": (800, 600), "fps": 30, "log_level": "DEBUG"}
        self.mock_frame = np.zeros((600, 800, 3), dtype=np.uint8)

        # Create patches for OpenCV functions
        self.patcher_cv2 = patch("vanishcap.workers.ui.cv2")

        # Start patches
        self.mock_cv2 = self.patcher_cv2.start()

        # Configure mock cv2
        self.mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        self.mock_cv2.WINDOW_NORMAL = 0
        self.mock_cv2.WND_PROP_VISIBLE = 0
        self.mock_cv2.waitKey.return_value = 0
        self.mock_cv2.getWindowProperty.return_value = 1
        self.mock_cv2.getTextSize.return_value = ((100, 20), 0)

        # Create UI worker instance
        self.ui = Ui(self.config)

    def tearDown(self):
        """Clean up after each test."""
        self.patcher_cv2.stop()

    def test_initialization(self):
        """Test initialization of UI worker (attributes)."""
        self.assertEqual(self.ui.font, self.mock_cv2.FONT_HERSHEY_SIMPLEX)
        self.assertIsNone(self.ui.current_frame_event)
        self.assertEqual(self.ui.current_detections, [])
        self.assertEqual(self.ui.worker_profiles, {})

    def test_handle_frame_event(self):
        """Test handling of frame events (state update only)."""
        frame_event = Event("video", "frame", self.mock_frame, frame_number=1)
        # Check state before
        self.assertIsNone(self.ui.current_frame_event)
        self.ui._dispatch(frame_event)
        self.ui._task()  # Process the event
        # Check state after - current_frame is updated by _task
        self.assertEqual(self.ui.current_frame_event, frame_event)

    def test_handle_detection_event(self):
        """Test handling of detection events (state update only)."""
        detections = [{"bbox": (0.1, 0.1, 0.5, 0.5), "confidence": 0.9, "class_name": "person"}]
        detection_event = Event("detector", "detection", detections, frame_number=1)
        self.ui._dispatch(detection_event)
        self.ui._task()  # Process the event
        self.assertEqual(self.ui.current_detections, detections)

    def test_handle_profile_event(self):
        """Test handling of worker profile events (state update only)."""
        event_source = "profiler"
        profile_data = {"worker": "test_worker", "avg_loop_time": 0.1, "avg_task_time": 0.05, "task_time": 0.05}
        profile_event = Event(event_source, "worker_profile", profile_data, frame_number=0)
        self.ui._dispatch(profile_event)
        self.ui._task()  # Process the event
        # Check state updates from _task
        self.assertIn(event_source, self.ui.worker_profiles)
        self.assertEqual(self.ui.worker_profiles[event_source], profile_data["task_time"])

    def test_finish(self):
        """Test cleanup in finish method."""
        self.ui._finish()
        self.mock_cv2.destroyAllWindows.assert_called_once()


if __name__ == "__main__":
    unittest.main()
