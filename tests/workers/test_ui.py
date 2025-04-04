"""Unit tests for the UI worker."""

# pylint: disable=protected-access

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pygame

from vanishcap.workers.ui import Ui
from vanishcap.event import Event


class TestUI(unittest.TestCase):  # pylint: disable=too-many-instance-attributes
    """Test cases for the UI worker."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test config
        self.config = {"window_size": (800, 600), "fps": 30, "log_level": "DEBUG"}

        # Create mock for pygame modules
        self.patcher_display = patch("pygame.display")
        self.mock_display = self.patcher_display.start()
        self.mock_screen = MagicMock()
        self.mock_display.set_mode.return_value = self.mock_screen

        self.patcher_font = patch("pygame.font")
        self.mock_font = self.patcher_font.start()
        self.mock_font_instance = MagicMock()
        self.mock_font.Font.return_value = self.mock_font_instance

        self.patcher_image = patch("pygame.image")
        self.mock_image = self.patcher_image.start()

        self.patcher_transform = patch("pygame.transform")
        self.mock_transform = self.patcher_transform.start()

        self.patcher_draw = patch("pygame.draw")
        self.mock_draw = self.patcher_draw.start()

        self.patcher_event = patch("pygame.event")
        self.mock_event = self.patcher_event.start()
        self.mock_event.get.return_value = []  # No events by default

        # Create mock for _emit method
        self.patcher_emit = patch("vanishcap.worker.Worker._emit")
        self.mock_emit = self.patcher_emit.start()

        # Create UI instance
        self.ui = Ui(self.config)

    def tearDown(self):
        """Clean up test fixtures."""
        self.patcher_display.stop()
        self.patcher_font.stop()
        self.patcher_image.stop()
        self.patcher_transform.stop()
        self.patcher_draw.stop()
        self.patcher_event.stop()
        self.patcher_emit.stop()

    def test_initialization(self):
        """Test initialization of UI worker."""
        self.assertEqual(self.ui.display_config["window_size"], (800, 600))
        self.assertEqual(self.ui.display_config["target_fps"], 30)
        self.assertEqual(self.ui.display_config["frame_time"], 1.0 / 30)
        self.assertEqual(self.ui.name, "ui")

        # Verify pygame initialization
        self.mock_display.init.assert_called_once()
        self.mock_font.init.assert_called_once()
        self.mock_display.set_mode.assert_called_once_with((800, 600))
        self.mock_display.set_caption.assert_called_once_with("vanishcap")

    def test_denormalize_coordinates(self):
        """Test coordinate denormalization."""
        # Test conversion from normalized [-1, 1] to pixel coordinates
        x, y = self.ui._denormalize_coordinates(0.0, 0.0, 100, 100)
        self.assertEqual(x, 50)  # Center x
        self.assertEqual(y, 50)  # Center y

        x, y = self.ui._denormalize_coordinates(1.0, 1.0, 100, 100)
        self.assertEqual(x, 100)  # Right edge
        self.assertEqual(y, 100)  # Bottom edge

        x, y = self.ui._denormalize_coordinates(-1.0, -1.0, 100, 100)
        self.assertEqual(x, 0)  # Left edge
        self.assertEqual(y, 0)  # Top edge

    def test_handle_frame_event(self):
        """Test handling of frame events."""
        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_event = Event("camera", "frame", {"frame": frame, "frame_number": 42})

        # Handle frame event
        self.ui(frame_event)

        # Verify frame was stored
        self.assertEqual(self.ui.current_frame_event, frame_event)

        # Run task to process frame
        self.ui._task()

        # Verify frame processing
        self.mock_image.frombuffer.assert_called_once()
        self.mock_transform.scale.assert_called_once()
        self.mock_screen.blit.assert_called()  # Multiple blits for frame and text

    def test_handle_detection_event(self):
        """Test handling of detection events."""
        # Create test detections
        detections = [
            {"class_name": "person", "bbox": [-0.5, -0.5, 0.5, 0.5], "confidence": 0.9}  # Normalized coordinates
        ]
        detection_event = Event("detector", "detection", detections)

        # Handle detection event
        self.ui(detection_event)

        # Verify detections were stored
        self.assertEqual(self.ui.current_detections, detections)

        # Create and handle a frame to trigger detection drawing
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_event = Event("camera", "frame", {"frame": frame, "frame_number": 42})
        self.ui(frame_event)
        self.ui._task()

        # Verify detection drawing
        self.mock_draw.rect.assert_called()  # Rectangle drawn for detection
        self.mock_font_instance.render.assert_called()  # Text rendered for label

    def test_handle_profile_event(self):
        """Test handling of worker profile events."""
        # Create test profile event
        profile_event = Event("detector", "worker_profile", {"task_time": 0.016})  # 16ms

        # Handle profile event
        self.ui(profile_event)

        # Verify profile was stored
        self.assertEqual(self.ui.worker_profiles["detector"], 0.016)

        # Create and handle a frame to trigger profile display
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_event = Event("camera", "frame", {"frame": frame, "frame_number": 42})
        self.ui(frame_event)
        self.ui._task()

        # Verify profile display
        render_calls = self.mock_font_instance.render.call_args_list
        profile_text_rendered = False
        for call_args in render_calls:
            if "detector: 16.0ms" in call_args[0][0]:
                profile_text_rendered = True
                break
        self.assertTrue(profile_text_rendered)

    def test_handle_quit_event(self):
        """Test handling of quit events."""
        # Mock pygame event types
        pygame.QUIT = 12  # Actual pygame.QUIT value
        pygame.KEYDOWN = 2  # Actual pygame.KEYDOWN value
        pygame.K_ESCAPE = 27  # Actual pygame.K_ESCAPE value

        # Create pygame quit event
        quit_event = MagicMock()
        quit_event.type = pygame.QUIT
        self.mock_event.get.return_value = [quit_event]

        # Create and handle a frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_event = Event("camera", "frame", {"frame": frame, "frame_number": 42})
        self.ui(frame_event)

        # Set last frame time to allow frame processing
        self.ui.display_config["last_frame_time"] = 0

        # Run task to process quit event
        self.ui._task()

        # Verify quit handling
        self.mock_emit.assert_called_once()
        emitted_event = self.mock_emit.call_args[0][0]
        self.assertEqual(emitted_event.event_name, "stop")
        self.mock_display.quit.assert_called_once()

        # Test escape key quit
        self.mock_emit.reset_mock()
        self.mock_display.quit.reset_mock()

        escape_event = MagicMock()
        escape_event.type = pygame.KEYDOWN
        escape_event.key = pygame.K_ESCAPE
        self.mock_event.get.return_value = [escape_event]

        # Run task to process escape key event
        self.ui._task()

        # Verify quit handling
        self.mock_emit.assert_called_once()
        emitted_event = self.mock_emit.call_args[0][0]
        self.assertEqual(emitted_event.event_name, "stop")
        self.mock_display.quit.assert_called_once()

    def test_finish(self):
        """Test cleanup in finish method."""
        # Call finish method
        self.ui._finish()

        # Verify pygame cleanup
        self.mock_display.quit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
