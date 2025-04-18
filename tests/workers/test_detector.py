"""Unit tests for the Detector worker."""

# pylint: disable=protected-access

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from vanishcap.workers.detector import Detector
from vanishcap.event import Event


class TestDetector(unittest.TestCase):  # pylint: disable=too-many-instance-attributes
    """Test cases for the Detector worker."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a configuration dictionary
        self.config = {"model": "yolov5s", "frame_skip": 2, "log_level": "DEBUG", "backend": "pytorch"}

        # Create mock frame
        self.mock_frame = np.zeros((640, 640, 3), dtype=np.uint8)

        # Create mock YOLO model
        self.mock_model = MagicMock()
        self.mock_model.names = {0: "person", 1: "car"}

        # Mock detection results with PyTorch tensors
        mock_box = MagicMock()
        mock_box.xyxy = [torch.tensor([100, 200, 300, 400], dtype=torch.float32)]  # x1, y1, x2, y2
        mock_box.conf = [torch.tensor([0.95], dtype=torch.float32)]  # confidence
        mock_box.cls = [torch.tensor([0], dtype=torch.long)]  # class id

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        self.mock_model.return_value = [mock_result]

        # Create patches
        self.patcher_yolo = patch("vanishcap.workers.detector.YOLO", return_value=self.mock_model)
        self.patcher_makedirs = patch("os.makedirs")
        self.patcher_path_exists = patch("os.path.exists", return_value=True)

        # Start patches
        self.mock_yolo = self.patcher_yolo.start()
        self.mock_makedirs = self.patcher_makedirs.start()
        self.mock_path_exists = self.patcher_path_exists.start()

    def tearDown(self):
        """Clean up after each test."""
        self.patcher_yolo.stop()
        self.patcher_makedirs.stop()
        self.patcher_path_exists.stop()

    def test_initialization_pytorch(self):
        """Test initialization with PyTorch backend."""
        detector = Detector(self.config)
        self.assertEqual(detector.backend, "pytorch")
        self.assertEqual(detector.frame_skip, 2)
        self.mock_yolo.assert_called_once()
        self.mock_makedirs.assert_called_once()

    def test_initialization_tensorrt(self):
        """Test initialization with TensorRT backend."""
        config = self.config.copy()
        config["backend"] = "tensorrt"

        # Mock path exists to False to test conversion
        with patch("os.path.exists", return_value=False):
            detector = Detector(config)

        self.assertEqual(detector.backend, "tensorrt")
        self.mock_yolo.assert_called()
        self.mock_model.export.assert_called_once()

    def test_initialization_onnx(self):
        """Test initialization with ONNX backend."""
        config = self.config.copy()
        config["backend"] = "onnx"

        # Mock path exists to False to test conversion
        with patch("os.path.exists", return_value=False):
            detector = Detector(config)

        self.assertEqual(detector.backend, "onnx")
        self.mock_yolo.assert_called()
        self.mock_model.export.assert_called_once()

    def test_initialization_invalid_backend(self):
        """Test initialization with invalid backend."""
        config = self.config.copy()
        config["backend"] = "invalid"

        with self.assertRaises(ValueError):
            Detector(config)

    def test_normalize_coordinates(self):
        """Test coordinate normalization."""
        detector = Detector(self.config)
        # Set image dimensions required by the method
        # These were implicitly assumed before but are needed for the method signature
        img_width = 640
        img_height = 640

        # Test normalization with different values, passing width and height
        norm_x, norm_y = detector._normalize_coordinates(320, 320, img_width, img_height)
        self.assertEqual(norm_x, 0.0)
        self.assertEqual(norm_y, 0.0)

        norm_x, norm_y = detector._normalize_coordinates(640, 640, img_width, img_height)
        self.assertEqual(norm_x, 1.0)
        self.assertEqual(norm_y, 1.0)

        norm_x, norm_y = detector._normalize_coordinates(0, 0, img_width, img_height)
        self.assertEqual(norm_x, -1.0)
        self.assertEqual(norm_y, -1.0)

    def test_event_handling(self):
        """Test event handling."""
        detector = Detector(self.config)

        # Test frame event
        frame_event1 = Event("video", "frame", self.mock_frame, frame_number=1)
        detector._dispatch(frame_event1)
        detector._task()
        self.assertEqual(detector.frame_count, 1)

        # Second frame
        frame_event2 = Event("video", "frame", self.mock_frame, frame_number=2)
        detector._dispatch(frame_event2)
        detector._task()
        self.assertEqual(detector.frame_count, 2)

        # Test unknown event
        unknown_event = Event("video", "unknown", {})
        detector._dispatch(unknown_event)
        detector._task()

    def test_task_execution(self):
        """Test task execution with detections."""
        detector = Detector(self.config)
        detector._emit = MagicMock()

        # Set up first frame event (will be skipped)
        frame_event1 = Event("video", "frame", self.mock_frame, frame_number=1)
        detector._dispatch(frame_event1)
        detector._task()

        # Set up second frame event (will be processed)
        frame_event2 = Event("video", "frame", self.mock_frame, frame_number=2)
        detector._dispatch(frame_event2)
        detector._task()

        # Verify detection event was emitted
        detector._emit.assert_called_once()
        event = detector._emit.call_args[0][0]
        self.assertEqual(event.worker_name, "detector")
        self.assertEqual(event.event_name, "detection")
        self.assertEqual(len(event.data), 1)  # One detection

        # Verify detection data
        detection = event.data[0]
        self.assertEqual(detection["class_name"], "person")
        self.assertAlmostEqual(detection["confidence"], 0.95)
        self.assertTrue("bbox" in detection)
        self.assertTrue("x" in detection)
        self.assertTrue("y" in detection)

        # Verify frame_number is passed through
        self.assertEqual(event.frame_number, 2)

    def test_task_no_frame(self):
        """Test task execution without frame."""
        detector = Detector(self.config)
        detector._emit = MagicMock()

        # Run task without frame
        detector._task()

        # Verify no event was emitted
        detector._emit.assert_not_called()

    def test_dispatch_frame_event(self):
        """Test dispatching frame events."""
        detector = Detector(self.config)

        # Create two frame events
        frame_event1 = Event("video", "frame", self.mock_frame, frame_number=1)
        frame_event2 = Event("video", "frame", self.mock_frame, frame_number=2)

        # Dispatch first frame event
        detector._dispatch(frame_event1)
        detector._task()
        self.assertEqual(detector.frame_count, 1)

        # Dispatch second frame event
        detector._dispatch(frame_event2)
        detector._task()
        self.assertEqual(detector.frame_count, 2)

    def test_dispatch_other_event(self):
        """Test dispatching non-frame events."""
        detector = Detector(self.config)

        # Create a non-frame event
        other_event = Event("video", "other", {"data": "test"})

        # Dispatch the event
        detector._dispatch(other_event)
        detector._task()

        # Verify the event was handled without errors
        self.assertEqual(detector.frame_count, 0)

    def test_finish(self):
        """Test cleanup in finish method."""
        detector = Detector(self.config)
        detector._finish()
        # Verify cleanup completed without errors


if __name__ == "__main__":
    unittest.main()
