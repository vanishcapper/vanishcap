"""Unit tests for the Event class."""

import unittest
import time  # Add time import for sleep if needed later, or just for assertAlmostEqual context
from vanishcap.event import Event


class TestEvent(unittest.TestCase):
    """Test cases for the Event class."""

    def test_event_creation(self):
        """Test creating an event with all fields."""
        event = Event("test_worker", "test_event", {"test": "data"}, frame_number=None)
        self.assertEqual(event.worker_name, "test_worker")
        self.assertEqual(event.event_name, "test_event")
        self.assertEqual(event.data, {"test": "data"})
        self.assertIsNone(event.frame_number)

    def test_event_creation_no_data(self):
        """Test creating an event without data."""
        event = Event("test_worker", "test_event", frame_number=None)
        self.assertEqual(event.worker_name, "test_worker")
        self.assertEqual(event.event_name, "test_event")
        self.assertIsNone(event.data)
        self.assertIsNone(event.frame_number)

    def test_event_equality(self):
        """Test equality between two identical events."""
        event1 = Event("test_worker", "test_event", {"test": "data"}, frame_number=None)
        time.sleep(0.00001)
        event2 = Event("test_worker", "test_event", {"test": "data"}, frame_number=None)

        self.assertEqual(event1.worker_name, event2.worker_name)
        self.assertEqual(event1.event_name, event2.event_name)
        self.assertEqual(event1.data, event2.data)
        self.assertEqual(event1.frame_number, event2.frame_number)
        self.assertAlmostEqual(event1.timestamp, event2.timestamp, delta=0.01)

    def test_event_inequality(self):
        """Test inequality between different events."""
        event1 = Event("test_worker", "test_event", {"test": "data"}, frame_number=None)
        event2 = Event("other_worker", "test_event", {"test": "data"}, frame_number=None)
        event3 = Event("test_worker", "other_event", {"test": "data"}, frame_number=None)
        event4 = Event("test_worker", "test_event", {"other": "data"}, frame_number=None)
        event5 = Event("test_worker", "test_event", {"test": "data"}, frame_number=1)

        self.assertNotEqual(event1, event2)
        self.assertNotEqual(event1, event3)
        self.assertNotEqual(event1, event4)
        self.assertNotEqual(event1, event5)

    def test_event_str(self):
        """Test string representation of an event."""
        event = Event("test_worker", "test_event", {"test": "data"}, frame_number=None)
        str_repr = str(event)
        self.assertIn("test_worker", str_repr)
        self.assertIn("test_event", str_repr)
        self.assertIn("test", str_repr)
        self.assertIn("data", str_repr)
        self.assertIn("frame_number=None", str_repr)

    def test_event_repr(self):
        """Test repr representation of an event."""
        event = Event("test_worker", "test_event", {"test": "data"}, frame_number=None)
        repr_str = repr(event)
        self.assertIn("Event", repr_str)
        self.assertIn("test_worker", repr_str)
        self.assertIn("test_event", repr_str)
        self.assertIn("test", repr_str)
        self.assertIn("data", repr_str)
        self.assertIn("frame_number=None", repr_str)


if __name__ == "__main__":
    unittest.main()
