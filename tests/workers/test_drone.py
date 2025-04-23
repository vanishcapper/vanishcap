"""Tests for the drone worker."""

import time
import unittest
from unittest.mock import MagicMock, patch

from vanishcap.event import Event
from vanishcap.workers.drone import Drone


# pylint: disable=protected-access
class TestDrone(unittest.TestCase):
    """Test the drone worker."""

    def setUp(self) -> None:
        """Set up the test."""
        self.config = {
            "name": "test",
            "driver": {
                "name": "tello",
                "ip": "192.168.10.1",
                "max_linear_velocity": 50.0,
                "max_angular_velocity": 50.0,
                "max_vertical_velocity": 30.0,
                "field_of_view": 82.6,
                "disable_yaw": False,
                "disable_xy": False,
                "disable_z": False,
            },
            "auto_takeoff": True,
            "target_timeout": 1.0,
            "movement_threshold": 0.1,
        }
        self.patcher_driver = patch("vanishcap.drivers.tello.TelloDriver")
        self.mock_driver_class = self.patcher_driver.start()
        self.mock_driver = MagicMock()
        self.mock_driver.get_max_linear_velocity.return_value = 50.0
        self.mock_driver.get_max_angular_velocity.return_value = 50.0
        self.mock_driver.get_max_vertical_velocity.return_value = 30.0
        self.mock_driver.get_field_of_view.return_value = 82.6
        self.mock_driver_class.return_value = self.mock_driver

    def tearDown(self) -> None:
        """Clean up after the test."""
        self.patcher_driver.stop()

    def test_initialization(self) -> None:
        """Test initialization of drone worker."""
        drone = Drone(self.config)
        self.assertEqual(drone.logger.name, "vanishcap.workers.drone")
        self.assertEqual(drone.auto_takeoff, True)
        self.assertEqual(drone.target_timeout, 1.0)
        self.assertEqual(drone.movement_threshold, 0.1)
        self.assertEqual(drone.driver.get_max_linear_velocity(), 50.0)
        self.assertEqual(drone.driver.get_max_angular_velocity(), 50.0)
        self.assertEqual(drone.driver.get_max_vertical_velocity(), 30.0)
        self.assertEqual(drone.driver.get_field_of_view(), 82.6)

    def test_initialization_default_values(self) -> None:
        """Test initialization with default values."""
        drone = Drone({})
        self.assertEqual(drone.logger.name, "vanishcap.workers.drone")
        self.assertFalse(drone.auto_takeoff)
        self.assertEqual(drone.target_timeout, 1.0)
        self.assertEqual(drone.movement_threshold, 0.1)
        self.assertEqual(drone.driver.get_max_linear_velocity(), 50.0)
        self.assertEqual(drone.driver.get_max_angular_velocity(), 50.0)
        self.assertEqual(drone.driver.get_max_vertical_velocity(), 30.0)
        self.assertEqual(drone.driver.get_field_of_view(), 82.6)

    def test_dispatch_command(self) -> None:
        """Test command dispatching."""
        drone = Drone(self.config)
        drone._dispatch_command("takeoff")
        drone.driver.takeoff.assert_called_once()

    def test_dispatch_command_error_handling(self) -> None:
        """Test error handling in command dispatching."""
        drone = Drone(self.config)
        drone.driver.takeoff.side_effect = RuntimeError("Test error")
        with self.assertLogs(drone.logger, level="ERROR"):
            drone._dispatch_command("takeoff")

    def test_follow_target(self) -> None:
        """Test target following logic."""
        drone = Drone(self.config)
        drone.ready_to_process_targets = True
        drone.current_target = {
            "x": 0.5,
            "y": 0.5,
            "z": 0.5,
            "bbox": [0.3, 0.4, 0.6, 0.7],
            "processed": False,
        }
        drone._follow_target()
        drone.driver.send_rc_control.assert_called()

    def test_follow_target_threshold(self) -> None:
        """Test movement threshold in target following."""
        drone = Drone(self.config)
        drone.ready_to_process_targets = True
        drone.current_target = {
            "x": 0.01,
            "y": 0.01,
            "z": 0.01,
            "bbox": [0.3, 0.4, 0.6, 0.7],
            "processed": False,
        }
        drone._follow_target()
        drone.driver.send_rc_control.assert_called_with(0, 0, 0, 0)

    def test_target_timeout(self) -> None:
        """Test target timeout handling."""
        drone = Drone(self.config)
        target_event = Event(
            "test",
            "target",
            {
                "x": 0.5,
                "y": 0.5,
                "z": 0.5,
                "bbox": [0.3, 0.4, 0.6, 0.7],
            },
        )
        current_time = time.time()
        drone._process_target_event(target_event, current_time)
        self.assertIsNotNone(drone.last_target_time)

        # Simulate target timeout
        with patch("time.time") as mock_time:
            mock_time.return_value = current_time + drone.target_timeout + 1
            drone._process_current_target(mock_time())
            self.assertIsNone(drone.current_target)

    def test_event_handling(self) -> None:
        """Test event handling."""
        drone = Drone(self.config)
        target_event = Event(
            "test",
            "target",
            {
                "x": 0.5,
                "y": 0.5,
                "z": 0.5,
                "bbox": [0.3, 0.4, 0.6, 0.7],
            },
        )
        current_time = time.time()
        drone._process_target_event(target_event, current_time)
        self.assertEqual(drone.current_target["x"], target_event.data["x"])
        self.assertEqual(drone.current_target["y"], target_event.data["y"])
        self.assertEqual(drone.current_target["z"], target_event.data["z"])
        self.assertEqual(drone.current_target["bbox"], target_event.data["bbox"])

    def test_finish(self) -> None:
        """Test cleanup in finish method."""
        drone = Drone(self.config)
        drone._finish()
        drone.driver.disconnect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
