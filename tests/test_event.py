"""Unit tests for the Event class."""

import unittest
from vanishcap.event import Event


class TestEvent(unittest.TestCase):
    """Test cases for the Event class."""

    def test_event_creation(self):
        """Test creating an event with all fields."""
        event = Event("test_worker", "test_event", {"test": "data"})
        self.assertEqual(event.worker_name, "test_worker")
        self.assertEqual(event.event_name, "test_event")
        self.assertEqual(event.data, {"test": "data"})

    def test_event_creation_no_data(self):
        """Test creating an event without data."""
        event = Event("test_worker", "test_event")
        self.assertEqual(event.worker_name, "test_worker")
        self.assertEqual(event.event_name, "test_event")
        self.assertIsNone(event.data)

    def test_event_equality(self):
        """Test equality between two identical events."""
        event1 = Event("test_worker", "test_event", {"test": "data"})
        event2 = Event("test_worker", "test_event", {"test": "data"})
        self.assertEqual(event1, event2)

    def test_event_inequality(self):
        """Test inequality between different events."""
        event1 = Event("test_worker", "test_event", {"test": "data"})
        event2 = Event("other_worker", "test_event", {"test": "data"})
        event3 = Event("test_worker", "other_event", {"test": "data"})
        event4 = Event("test_worker", "test_event", {"other": "data"})

        self.assertNotEqual(event1, event2)  # Different worker
        self.assertNotEqual(event1, event3)  # Different event name
        self.assertNotEqual(event1, event4)  # Different data

    def test_event_str(self):
        """Test string representation of an event."""
        event = Event("test_worker", "test_event", {"test": "data"})
        str_repr = str(event)
        self.assertIn("test_worker", str_repr)
        self.assertIn("test_event", str_repr)
        self.assertIn("test", str_repr)
        self.assertIn("data", str_repr)

    def test_event_repr(self):
        """Test repr representation of an event."""
        event = Event("test_worker", "test_event", {"test": "data"})
        repr_str = repr(event)
        self.assertIn("Event", repr_str)
        self.assertIn("test_worker", repr_str)
        self.assertIn("test_event", repr_str)
        self.assertIn("test", repr_str)
        self.assertIn("data", repr_str)


if __name__ == "__main__":
    unittest.main()
