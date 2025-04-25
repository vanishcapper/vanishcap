"""Unit tests for the Video worker."""

# pylint: disable=protected-access

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from vanishcap.workers.video import Video
from vanishcap.event import Event


class TestVideo(unittest.TestCase):  # pylint: disable=too-many-instance-attributes
    """Test cases for the Video worker."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a configuration dictionary
        self.config = {"source": "test_video.mp4", "log_level": "DEBUG", "save_path": "/tmp/output.mp4"}

        # Create mock frame
        self.mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Create mock CamGear instance
        self.mock_cap = MagicMock()
        self.mock_cap.read.return_value = self.mock_frame

        # Set the framerate attribute directly on the mock
        self.mock_cap.framerate = 30.0

        # Create mock CamGear class that returns a started instance
        self.mock_camgear = MagicMock()
        self.mock_camgear.return_value.start.return_value = self.mock_cap

        # Create mock WriteGear
        self.mock_writer = MagicMock()

        # Create patches
        self.patcher_camgear = patch("vanishcap.workers.video.CamGear", self.mock_camgear)
        self.patcher_writegear = patch("vanishcap.workers.video.WriteGear", return_value=self.mock_writer)

        # Start patches
        self.mock_camgear = self.patcher_camgear.start()
        self.mock_writegear = self.patcher_writegear.start()

    def tearDown(self):
        """Clean up after each test."""
        self.patcher_camgear.stop()
        self.patcher_writegear.stop()

    def test_initialization(self):
        """Test initialization of video worker."""
        # Create video worker
        video = Video(self.config)

        # Verify initialization
        self.assertEqual(video.source, self.config["source"])
        self.assertEqual(video.frame_number, 0)
        self.assertIsNone(video.current_frame)
        self.mock_camgear.assert_called_once()
        self.mock_writegear.assert_called_once()

    def test_initialization_without_save_path(self):
        """Test initialization without save path."""
        config = self.config.copy()
        del config["save_path"]

        # Create video worker
        video = Video(config)

        # Verify initialization
        self.assertEqual(video.source, config["source"])
        self.assertIsNone(video.writer)
        self.mock_camgear.assert_called_once()
        self.mock_writegear.assert_not_called()

    def test_task_frame_capture(self):
        """Test frame capture in task loop."""
        # Create video worker
        video = Video(self.config)

        # Run task
        video._task()

        # Verify frame capture
        self.mock_cap.read.assert_called_once()
        self.assertEqual(video.frame_number, 1)

        # Verify that writer.write was called once with any frame
        self.mock_writer.write.assert_called_once()

    def test_task_empty_queue(self):
        """Test handling of empty queue in task loop."""
        # Configure mock to return None (empty queue)
        self.mock_cap.read.return_value = None

        # Create video worker
        video = Video(self.config)

        # Run task
        video._task()

        # Verify stop event was set
        self.assertTrue(video._stop_event.is_set())

    def test_event_handling(self):
        """Test event handling."""
        # Create video worker
        video = Video(self.config)

        # Test command event
        command_event = Event("video", "command", {"action": "test"})
        video._dispatch(command_event)

        # Process events in task
        video._task()

        # Test unknown event
        unknown_event = Event("video", "unknown", {})
        video._dispatch(unknown_event)

        # Process events in task
        video._task()

    def test_finish(self):
        """Test cleanup in finish method."""
        # Create video worker
        video = Video(self.config)

        # Call finish
        video._finish()

        # Verify cleanup
        self.mock_writer.close.assert_called_once()
        self.assertIsNone(video.cap)

    def test_frame_event_emission(self):
        """Test frame event emission."""
        # Create video worker
        video = Video(self.config)

        # Mock emit method
        video._emit = MagicMock()

        # Run task
        video._task()

        # Verify frame event was emitted
        video._emit.assert_called_once()
        event = video._emit.call_args[0][0]
        self.assertEqual(event.worker_name, "video")
        self.assertEqual(event.event_name, "frame")
        self.assertEqual(event.frame_number, 1)
        np.testing.assert_array_equal(event.data, self.mock_frame)


if __name__ == "__main__":
    unittest.main()
