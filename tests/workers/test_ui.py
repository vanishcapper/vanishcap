"""Unit tests for the UI worker."""

# pylint: disable=protected-access
# pylint: disable=unnecessary-pass
# pylint: disable=too-many-instance-attributes

import unittest
from unittest.mock import MagicMock, patch
import time  # pylint: disable=unused-import
import numpy as np
import pygame  # pylint: disable=unused-import

from vanishcap.workers.ui import Ui
from vanishcap.event import Event


class TestUI(unittest.TestCase):
    """Test cases for the UI worker."""

    def setUp(self):  # pylint: disable=arguments-differ
        """Set up test fixtures."""
        self.config = {"window_size": (800, 600), "fps": 30, "log_level": "DEBUG"}
        self.mock_frame = np.zeros((600, 800, 3), dtype=np.uint8)

        # Create mock surface
        self.mock_surface = MagicMock()
        self.mock_surface.get_rect.return_value = MagicMock(topright=(790, 10))

        # Create mock font
        self.mock_font = MagicMock()
        self.mock_font.render.return_value = self.mock_surface

        # Create patches
        self.patcher_display = patch("pygame.display")
        self.patcher_font = patch("pygame.font")
        self.patcher_image = patch("pygame.image")
        self.patcher_draw = patch("pygame.draw")
        self.patcher_transform = patch("pygame.transform")
        self.patcher_event = patch("pygame.event")

        # Start patches
        self.mock_display = self.patcher_display.start()
        self.mock_font = self.patcher_font.start()
        self.mock_image = self.patcher_image.start()
        self.mock_draw = self.patcher_draw.start()
        self.mock_transform = self.patcher_transform.start()
        self.mock_event = self.patcher_event.start()

        # Configure mock display
        self.mock_screen = MagicMock()
        self.mock_display.set_mode.return_value = self.mock_screen
        self.mock_display.init.return_value = None
        self.mock_display.quit.return_value = None
        self.mock_display.flip.return_value = None
        self.mock_display.get_init.return_value = True

        # Configure mock font
        self.mock_font.init.return_value = None
        self.mock_font.Font.return_value = self.mock_font
        self.mock_font.quit.return_value = None

        # Configure mock image
        self.mock_image.frombuffer.return_value = self.mock_surface

        # Configure mock transform
        self.mock_transform.scale.return_value = self.mock_surface

        # Configure mock event
        self.mock_event.get.return_value = []

        # Create UI worker instance
        self.ui = Ui(self.config)

    def tearDown(self):
        """Clean up after each test."""
        self.patcher_display.stop()
        self.patcher_font.stop()
        self.patcher_image.stop()
        self.patcher_draw.stop()
        self.patcher_transform.stop()
        self.patcher_event.stop()

    def test_initialization(self):
        """Test initialization of UI worker (attributes)."""
        self.assertIsNotNone(self.ui.profile_font)
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
        self.mock_font.quit.assert_called_once()
        self.mock_display.quit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
