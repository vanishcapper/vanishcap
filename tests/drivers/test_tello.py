"""Tests for the Tello drone driver."""

import unittest
from unittest.mock import MagicMock, patch

from vanishcap.drivers.tello import TelloDriver


class TestTelloDriver(unittest.TestCase):
    """Test the Tello drone driver."""

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
            "ip": "192.168.10.2",
        }

    @patch("vanishcap.drivers.tello.Tello")
    def test_init(self, mock_tello) -> None:
        """Test driver initialization."""
        driver = TelloDriver(self.config)
        mock_tello.assert_called_once()
        self.assertEqual(driver.max_linear_velocity, 50.0)
        self.assertEqual(driver.max_angular_velocity, 50.0)
        self.assertEqual(driver.max_vertical_velocity, 30.0)
        self.assertEqual(driver.field_of_view, 82.6)
        self.assertTrue(driver.disable_yaw)
        self.assertTrue(driver.disable_xy)
        self.assertTrue(driver.disable_z)
        self.assertEqual(driver.logger.name, "test")

    @patch("vanishcap.drivers.tello.Tello")
    def test_init_defaults(self, mock_tello) -> None:
        """Test driver initialization with defaults."""
        driver = TelloDriver({})
        mock_tello.assert_called_once()
        self.assertEqual(driver.max_linear_velocity, 50.0)
        self.assertEqual(driver.max_angular_velocity, 50.0)
        self.assertEqual(driver.max_vertical_velocity, 30.0)
        self.assertEqual(driver.field_of_view, 82.6)
        self.assertFalse(driver.disable_yaw)
        self.assertFalse(driver.disable_xy)
        self.assertFalse(driver.disable_z)
        self.assertEqual(driver.logger.name, "root")

    @patch("vanishcap.drivers.tello.Tello")
    def test_connect(self, mock_tello) -> None:
        """Test connecting to the drone."""
        driver = TelloDriver(self.config)
        driver.connect()
        driver.tello.connect.assert_called_once()

    @patch("vanishcap.drivers.tello.Tello")
    def test_disconnect(self, mock_tello) -> None:
        """Test disconnecting from the drone."""
        driver = TelloDriver(self.config)
        driver.disconnect()
        driver.tello.end.assert_called_once()

    @patch("vanishcap.drivers.tello.Tello")
    def test_takeoff(self, mock_tello) -> None:
        """Test taking off."""
        driver = TelloDriver(self.config)
        driver.takeoff()
        driver.tello.takeoff.assert_called_once()

    @patch("vanishcap.drivers.tello.Tello")
    def test_land(self, mock_tello) -> None:
        """Test landing."""
        driver = TelloDriver(self.config)
        driver.land()
        driver.tello.land.assert_called_once()

    @patch("vanishcap.drivers.tello.Tello")
    def test_send_rc_control(self, mock_tello) -> None:
        """Test sending RC control commands."""
        driver = TelloDriver(self.config)
        driver._send_rc_control(100, 100, 100, 100)
        driver.tello.send_rc_control.assert_called_once_with(100, 100, 100, 100)

    @patch("vanishcap.drivers.tello.Tello")
    def test_get_current_state(self, mock_tello) -> None:
        """Test getting current state."""
        driver = TelloDriver(self.config)
        mock_state = {"bat": 80, "h": 100, "vgx": 10, "agx": 0.1, "agy": 0.2, "agz": 0.3}
        driver.tello.get_current_state.return_value = mock_state
        self.assertEqual(driver.get_current_state(), mock_state)
        driver.tello.get_current_state.assert_called_once()

    @patch("vanishcap.drivers.tello.Tello")
    def test_streamon(self, mock_tello) -> None:
        """Test starting video stream."""
        driver = TelloDriver(self.config)
        driver.streamon()
        driver.tello.streamon.assert_called_once()

    @patch("vanishcap.drivers.tello.Tello")
    def test_streamoff(self, mock_tello) -> None:
        """Test stopping video stream."""
        driver = TelloDriver(self.config)
        driver.streamoff()
        driver.tello.streamoff.assert_called_once()
