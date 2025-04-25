"""Unit tests for the Video worker."""

# pylint: disable=protected-access

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from vanishcap.workers.video import Video
from vanishcap.event import Event


class TestVideo(unittest.TestCase):  # pylint: disable=too-many-instance-attributes
    """Test cases for the Video worker."""

    def setUp(self) -> None:
        """Set up test environment."""
        self.config = {"name": "video", "source": 0, "log_level": "DEBUG", "save_path": "test_output.mp4"}

        # Create mock frame
        self.mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Create mock CamGear instance
        self.mock_cap = MagicMock()
        self.mock_cap.read.return_value = self.mock_frame
        self.mock_cap.framerate = 30.0  # Set framerate as a regular attribute
        self.mock_cap.start.return_value = self.mock_cap  # Make start() return self

        # Create mock CamGear class that returns our mock_cap
        self.mock_camgear = MagicMock()
        self.mock_camgear.return_value = self.mock_cap

        # Create mock WriteGear
        self.mock_writer = MagicMock()

        # Create and start patches
        self.patcher_camgear = patch("vanishcap.workers.video.CamGear", self.mock_camgear)
        self.patcher_writegear = patch("vanishcap.workers.video.WriteGear", return_value=self.mock_writer)

        # Start patches before creating worker
        self.mock_camgear = self.patcher_camgear.start()
        self.mock_writegear = self.patcher_writegear.start()

        # Create worker after patches are in place
        self.worker = Video(self.config)

    def tearDown(self):
        """Clean up after each test."""
        self.patcher_camgear.stop()
        self.patcher_writegear.stop()

    def test_initialization(self):
        """Test initialization of video worker."""
        # Verify initialization
        self.assertEqual(self.worker.source, self.config["source"])
        self.assertEqual(self.worker.frame_number, 0)
        self.assertIsNone(self.worker.current_frame)
        self.mock_camgear.assert_called_once()
        self.mock_writegear.assert_called_once()

    def test_initialization_without_save_path(self):
        """Test initialization without save path."""
        config = self.config.copy()
        del config["save_path"]

        # Create video worker without save path
        worker_no_save = Video(config)

        # Verify initialization
        self.assertEqual(worker_no_save.source, config["source"])
        self.assertIsNone(worker_no_save.writer)
        self.mock_camgear.assert_called()  # Changed to assert_called since we have two instances
        self.mock_writegear.assert_called_once()

    def test_task_frame_capture(self):
        """Test frame capture in task loop."""
        # Run task
        self.worker._task()

        # Verify frame capture
        self.mock_cap.read.assert_called_once()
        self.assertEqual(self.worker.frame_number, 1)

        # Verify that writer.write was called once with any frame
        self.mock_writer.write.assert_called_once()

    def test_task_empty_queue(self):
        """Test handling of empty queue in task loop."""
        # Configure mock to return None (empty queue)
        self.mock_cap.read.return_value = None

        # Run task
        self.worker._task()

        # Verify stop event was set
        self.assertTrue(self.worker._stop_event.is_set())

    def test_event_handling(self):
        """Test event handling."""
        # Test command event
        command_event = Event("video", "command", {"action": "test"})
        self.worker._dispatch(command_event)

        # Process events in task
        self.worker._task()

        # Test unknown event
        unknown_event = Event("video", "unknown", {})
        self.worker._dispatch(unknown_event)

        # Process events in task
        self.worker._task()

    def test_finish(self):
        """Test cleanup in finish method."""
        # Call finish
        self.worker._finish()

        # Verify cleanup
        self.mock_writer.close.assert_called_once()
        self.assertIsNone(self.worker.cap)

    def test_frame_event_emission(self):
        """Test frame event emission."""
        # Mock emit method
        self.worker._emit = MagicMock()

        # Run task
        self.worker._task()

        # Verify frame event was emitted
        self.worker._emit.assert_called_once()
        event = self.worker._emit.call_args[0][0]
        self.assertEqual(event.worker_name, "video")
        self.assertEqual(event.event_name, "frame")
        self.assertEqual(event.frame_number, 1)
        np.testing.assert_array_equal(event.data, self.mock_frame)


if __name__ == "__main__":
    unittest.main()
