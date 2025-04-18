"""Unit tests for the Navigator worker."""

# pylint: disable=protected-access

import unittest
from unittest.mock import patch

from vanishcap.workers.navigator import Navigator
from vanishcap.event import Event


class TestNavigator(unittest.TestCase):
    """Test cases for the Navigator worker."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test config
        self.config = {"target_class": "person", "log_level": "DEBUG"}

        # Create mock for _emit method
        self.patcher_emit = patch("vanishcap.worker.Worker._emit")
        self.mock_emit = self.patcher_emit.start()

        # Create navigator instance
        self.navigator = Navigator(self.config)

    def tearDown(self):
        """Clean up test fixtures."""
        self.patcher_emit.stop()

    def test_initialization(self):
        """Test initialization of navigator worker."""
        self.assertEqual(self.navigator.target_class, "person")
        self.assertEqual(self.navigator.name, "navigator")

    def test_handle_detection_event(self):
        """Test handling a detection event."""
        # Create test detections
        detections = [
            {"class_name": "person", "bbox": [0, 0, 100, 100], "confidence": 0.9, "x": 0.5, "y": 0.5},
            {"class_name": "car", "bbox": [0, 0, 200, 200], "confidence": 0.8, "x": 0.6, "y": 0.6},
        ]

        # Create and handle detection event
        detection_event = Event("detector", "detection", detections, frame_number=1)
        self.navigator._dispatch(detection_event)
        self.navigator._task()

        # Verify target event was emitted with correct data
        self.mock_emit.assert_called_once()
        emitted_event = self.mock_emit.call_args[0][0]
        self.assertEqual(emitted_event.worker_name, "navigator")
        self.assertEqual(emitted_event.event_name, "target")
        self.assertEqual(emitted_event.data["x"], 0.5)
        self.assertEqual(emitted_event.data["y"], 0.5)
        self.assertEqual(emitted_event.data["confidence"], 0.9)

    def test_select_largest_target(self):
        """Test selecting the largest target from multiple detections."""
        # Create test detections with different sizes
        detections = [
            {"class_name": "person", "bbox": [0, 0, 100, 100], "confidence": 0.9, "x": 0.5, "y": 0.5},  # Area: 10000
            {"class_name": "person", "bbox": [0, 0, 200, 200], "confidence": 0.8, "x": 0.6, "y": 0.6},  # Area: 40000
        ]

        # Create and handle detection event
        detection_event = Event("detector", "detection", detections, frame_number=1)
        self.navigator._dispatch(detection_event)
        self.navigator._task()

        # Verify largest target was selected
        self.mock_emit.assert_called_once()
        emitted_event = self.mock_emit.call_args[0][0]
        self.assertEqual(emitted_event.worker_name, "navigator")
        self.assertEqual(emitted_event.event_name, "target")
        self.assertEqual(emitted_event.data["x"], 0.6)  # From larger target
        self.assertEqual(emitted_event.data["y"], 0.6)

    def test_no_target_detections(self):
        """Test handling detection event with no targets of the specified class."""
        # Create test detections without target class
        detections = [{"class_name": "car", "bbox": [0, 0, 100, 100], "confidence": 0.9, "x": 0.5, "y": 0.5}]

        # Create and handle detection event
        detection_event = Event("detector", "detection", detections, frame_number=1)
        self.navigator._dispatch(detection_event)
        self.navigator._task()

        # Verify no target event was emitted
        self.mock_emit.assert_not_called()

    def test_empty_detection_queue(self):
        """Test handling of empty detection queue."""
        # Run task with empty queue
        self.navigator._task()

        # Verify no target event was emitted
        self.mock_emit.assert_not_called()

    def test_unknown_event(self):
        """Test handling of unknown events."""
        # Create unknown event
        event = Event("unknown", "unknown", {}, frame_number=1)
        self.navigator._dispatch(event)
        self.navigator._task()

        # Verify no target event was emitted
        self.mock_emit.assert_not_called()

    def test_finish(self):
        """Test cleanup in finish method."""
        # Call finish method
        self.navigator._finish()

        # Verify no errors occurred
        # (No cleanup needed for navigator, so nothing to verify)


if __name__ == "__main__":
    unittest.main()
