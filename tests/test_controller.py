"""Unit tests for the Controller class."""

# pylint: disable=protected-access,duplicate-code

import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch
import yaml

from vanishcap.controller import Controller
from vanishcap.event import Event
from vanishcap.worker import Worker


class TestWorker(Worker):
    """Test worker implementation."""

    def __init__(self, name: str, config: dict):
        """Initialize the test worker."""
        super().__init__(name, config)
        self.processed_events = []

    def _task(self):
        """Mock task implementation."""
        # Process any pending events
        events = self._get_latest_events_and_clear()
        for event in events.values():
            self._process_event(event)
        time.sleep(0.001)  # Small sleep to prevent CPU spinning

    def _process_event(self, event):
        """Record received events."""
        self.processed_events.append(event)


class TestController(unittest.TestCase):
    """Test cases for the Controller class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a configuration dictionary
        self.config = {
            "controller": {"offline": True, "log_level": "DEBUG"},
            "test_worker": {"log_level": "DEBUG"},
            "other_worker": {"log_level": "DEBUG"},
        }

        # Create temp config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f)

        # Create invalid config file for testing
        self.invalid_config_path = os.path.join(self.temp_dir, "invalid_config.yaml")
        with open(self.invalid_config_path, "w", encoding="utf-8") as f:
            f.write("invalid: yaml: content:")

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temp config files
        os.unlink(self.config_path)
        os.unlink(self.invalid_config_path)
        os.rmdir(self.temp_dir)

    def _setup_mock_workers(self, controller):
        """Set up mock workers for testing."""
        controller.workers = {
            "test_worker": TestWorker("test_worker", self.config["test_worker"]),
            "other_worker": TestWorker("other_worker", self.config["other_worker"]),
        }

    @patch.object(Controller, "_init_workers")
    @patch.object(Controller, "_build_event_routes")
    def test_initialization(self, mock_build_routes, mock_init_workers):
        """Test controller initialization."""
        controller = Controller(self.config_path)
        self._setup_mock_workers(controller)
        self.assertIsNotNone(controller)
        self.assertEqual(controller.config["controller"]["log_level"], "DEBUG")
        self.assertTrue(controller.config["controller"]["offline"])
        mock_init_workers.assert_called_once()
        mock_build_routes.assert_called_once()

    def test_initialization_missing_config(self):
        """Test initialization with missing config file."""
        with self.assertRaises(FileNotFoundError):
            Controller("nonexistent.yaml")

    @patch.object(Controller, "_init_workers")
    @patch.object(Controller, "_build_event_routes")
    def test_event_routing(self, mock_build_routes, mock_init_workers):
        """Test event routing between workers."""
        controller = Controller(self.config_path)
        self._setup_mock_workers(controller)
        mock_init_workers.assert_called_once()
        mock_build_routes.assert_called_once()

        # Set up event routes manually for testing
        controller.event_routes = {"test_event": {"test_worker"}}

        # Start the workers
        for worker in controller.workers.values():
            worker.start(controller)

        # Create and route a test event
        event = Event("other_worker", "test_event", {"test": "data"})
        controller(event)

        # Give the worker thread time to process the event
        time.sleep(0.1)

        # Verify the event was routed to test_worker
        test_worker = controller.workers.get("test_worker")
        self.assertIsNotNone(test_worker)
        self.assertEqual(len(test_worker.processed_events), 1)
        received_event = test_worker.processed_events[0]
        self.assertEqual(received_event.event_name, "test_event")
        self.assertEqual(received_event.worker_name, "other_worker")
        self.assertEqual(received_event.data, {"test": "data"})

        # Stop the workers
        for worker in controller.workers.values():
            worker.stop()

    @patch.object(Controller, "_init_workers")
    @patch.object(Controller, "_build_event_routes")
    def test_stop_event(self, mock_build_routes, mock_init_workers):
        """Test handling of stop events."""
        controller = Controller(self.config_path)
        self._setup_mock_workers(controller)
        mock_init_workers.assert_called_once()
        mock_build_routes.assert_called_once()

        # Create mock workers
        controller.workers["test_worker"] = MagicMock()
        controller.workers["other_worker"] = MagicMock()

        # Send stop event
        event = Event("test_worker", "stop")
        controller(event)

        # Verify all workers were stopped
        for worker in controller.workers.values():
            worker.stop.assert_called_once()

    @patch.object(Controller, "_init_workers")
    @patch.object(Controller, "_build_event_routes")
    def test_context_manager(self, mock_build_routes, mock_init_workers):
        """Test controller as context manager."""
        with Controller(self.config_path) as controller:
            self._setup_mock_workers(controller)
            mock_init_workers.assert_called_once()
            mock_build_routes.assert_called_once()
            self.assertIsNotNone(controller)
            # Mock the workers
            controller.workers["test_worker"] = MagicMock()
            controller.workers["other_worker"] = MagicMock()

        # Verify all workers were stopped on exit
        for worker in controller.workers.values():
            worker.stop.assert_called_once()

    def test_init_normal(self):
        """Test normal initialization."""
        with patch.object(Controller, "_init_workers"), patch.object(Controller, "_build_event_routes"):
            controller = Controller(self.config_path)
            self.assertEqual(controller.config, self.config)

    def test_init_invalid_config(self):
        """Test initialization with invalid config."""
        with self.assertRaises(ValueError):
            Controller(self.invalid_config_path)


if __name__ == "__main__":
    unittest.main()
