"""Tests for the base drone driver."""

import unittest
from unittest.mock import MagicMock, patch

from vanishcap.drivers.base import BaseDroneDriver


class TestDroneDriver(BaseDroneDriver):
    """Concrete implementation of BaseDroneDriver for testing."""

    def connect(self) -> None:
        """Connect to the drone."""
        pass

    def disconnect(self) -> None:
        """Disconnect from the drone."""
        pass

    def takeoff(self) -> None:
        """Take off."""
        pass

    def land(self) -> None:
        """Land."""
        pass

    def _send_rc_control(self, left_right: int, forward_backward: int, up_down: int, yaw: int) -> None:
        """Send RC control commands."""
        pass

    def get_current_state(self) -> dict:
        """Get the current state of the drone."""
        return {}

    def streamon(self) -> None:
        """Start video stream."""
        pass

    def streamoff(self) -> None:
        """Stop video stream."""
        pass


class TestBaseDroneDriver(unittest.TestCase):
    """Test the base drone driver."""

    def setUp(self) -> None:
        """Set up the test."""
        self.config = {
            "name": "test",
            "max_linear_velocity": 50.0,
            "max_angular_velocity": 50.0,
            "max_vertical_velocity": 30.0,
            "field_of_view": 82.6,
            "disable_yaw": True,
            "disable_xy": True,
            "disable_z": True,
        }

    def test_init(self) -> None:
        """Test driver initialization."""
        driver = TestDroneDriver(self.config)
        self.assertEqual(driver.max_linear_velocity, 50.0)
        self.assertEqual(driver.max_angular_velocity, 50.0)
        self.assertEqual(driver.max_vertical_velocity, 30.0)
        self.assertEqual(driver.field_of_view, 82.6)
        self.assertTrue(driver.disable_yaw)
        self.assertTrue(driver.disable_xy)
        self.assertTrue(driver.disable_z)
        self.assertEqual(driver.logger.name, "test")

    def test_init_defaults(self) -> None:
        """Test driver initialization with defaults."""
        driver = TestDroneDriver({})
        self.assertEqual(driver.max_linear_velocity, 50.0)
        self.assertEqual(driver.max_angular_velocity, 50.0)
        self.assertEqual(driver.max_vertical_velocity, 30.0)
        self.assertEqual(driver.field_of_view, 82.6)
        self.assertFalse(driver.disable_yaw)
        self.assertFalse(driver.disable_xy)
        self.assertFalse(driver.disable_z)
        self.assertEqual(driver.logger.name, "root")

    def test_send_rc_control(self) -> None:
        """Test RC control with disable flags."""
        driver = TestDroneDriver(self.config)
        with patch.object(driver, "_send_rc_control") as mock_send:
            driver.send_rc_control(100, 100, 100, 100)
            mock_send.assert_called_once_with(0, 0, 0, 0)

    def test_send_rc_control_no_disables(self) -> None:
        """Test RC control without disable flags."""
        driver = TestDroneDriver({})
        with patch.object(driver, "_send_rc_control") as mock_send:
            driver.send_rc_control(100, 100, 100, 100)
            mock_send.assert_called_once_with(100, 100, 100, 100)

    def test_get_max_velocities(self) -> None:
        """Test getting max velocities."""
        driver = TestDroneDriver(self.config)
        self.assertEqual(driver.get_max_linear_velocity(), 50.0)
        self.assertEqual(driver.get_max_angular_velocity(), 50.0)
        self.assertEqual(driver.get_max_vertical_velocity(), 30.0)
        self.assertEqual(driver.get_field_of_view(), 82.6)
