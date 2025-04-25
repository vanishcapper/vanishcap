"""Unit tests for the Worker base class."""

# pylint: disable=protected-access,broad-exception-raised

import threading
import time
import unittest
from unittest.mock import MagicMock

from vanishcap.worker import Worker
from vanishcap.event import Event


class TestWorker(Worker):
    """Test implementation of the Worker class."""

    def __init__(self, config: dict):
        """Initialize test worker."""
        super().__init__(config)
        self.task_called = False
        self.event_handled = False
        self.last_event = None
        self.finish_called = False

    def _task(self):
        """Mock task implementation."""
        self.task_called = True
        # Process any pending events
        events = self._get_latest_events_and_clear()
        for event in events.values():
            self._process_event(event)
        time.sleep(0.01)  # Small sleep to simulate work

    def _process_event(self, event):
        """Handle test events."""
        self.event_handled = True
        self.last_event = event

    def _finish(self):
        """Record cleanup call."""
        self.finish_called = True


class TestBaseWorker(unittest.TestCase):
    """Test cases for the base Worker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {"name": "test_worker", "profile_window": 1.0, "log_level": "DEBUG"}
        self.worker = TestWorker(self.config)
        self.mock_controller = MagicMock()

    def test_initialization(self):
        """Test worker initialization."""
        self.assertEqual(self.worker.name, "test_worker")
        self.assertEqual(self.worker.config, self.config)
        self.assertIsNotNone(self.worker.logger)
        self.assertFalse(self.worker._stop_event.is_set())
        self.assertIsNotNone(self.worker._event_lock)
        self.assertEqual(self.worker._profile_window, 1.0)

    def test_start_stop(self):
        """Test starting and stopping the worker."""
        # Start worker
        self.worker.start(self.mock_controller)
        self.assertIsNotNone(self.worker._run_thread)
        self.assertTrue(self.worker._run_thread.is_alive())
        self.assertFalse(self.worker._stop_event.is_set())

        # Let it run briefly
        time.sleep(0.1)

        # Stop worker
        self.worker.stop()
        self.assertTrue(self.worker._stop_event.is_set())
        self.assertFalse(self.worker._run_thread.is_alive())
        self.assertTrue(self.worker.finish_called)

    def test_event_handling(self):
        """Test event handling through dispatch."""
        test_event = Event("other_worker", "test_event", {"test": "data"})

        # Start worker
        self.worker.start(self.mock_controller)

        # Dispatch event
        self.worker._dispatch(test_event)

        # Wait briefly for event processing
        time.sleep(0.1)

        # Check event was handled
        self.assertTrue(self.worker.event_handled)
        self.assertEqual(self.worker.last_event, test_event)

        self.worker.stop()

    def test_event_emission(self):
        """Test event emission to controller."""
        self.worker._controller = self.mock_controller
        test_event = Event("test_worker", "test_event", {"test": "data"})

        self.worker._emit(test_event)
        self.mock_controller.assert_called_once_with(test_event)

    def test_profiling(self):
        """Test task time profiling."""
        self.worker.start(self.mock_controller)
        time.sleep(0.1)  # Let it run a few iterations

        # Should have recorded some task time
        self.assertGreater(self.worker._last_task_time, 0)
        self.assertGreater(self.worker._max_task_time, 0)

        self.worker.stop()

    def test_main_thread_execution(self):
        """Test running worker in main thread."""
        stop_thread = threading.Thread(target=lambda: time.sleep(0.1) or self.worker.stop())
        stop_thread.start()

        self.worker.start(self.mock_controller, run_in_main_thread=True)
        self.assertTrue(self.worker.task_called)
        self.assertTrue(self.worker.finish_called)

    def test_event_handling_latest_only(self):
        """Test that only latest events are processed."""
        events = [
            Event("other_worker", "test_event", {"test": "data1"}),
            Event("other_worker", "test_event", {"test": "data2"}),  # Same event type
        ]

        self.worker.start(self.mock_controller)

        # Dispatch events in quick succession
        for event in events:
            self.worker._dispatch(event)

        # Wait briefly for event processing
        time.sleep(0.1)

        # Check only the latest event was processed
        self.assertTrue(self.worker.event_handled)
        self.assertEqual(self.worker.last_event, events[-1])

        self.worker.stop()

    def test_error_handling(self):
        """Test error handling in run loop."""

        def failing_task():
            raise Exception("Test error")

        self.worker._task = failing_task
        self.worker.start(self.mock_controller)
        time.sleep(0.1)  # Let it hit the error

        # Worker should have stopped and cleaned up
        self.assertFalse(self.worker._run_thread.is_alive())
        self.assertTrue(self.worker.finish_called)


if __name__ == "__main__":
    unittest.main()
