"""Unit tests for the WiFi utility module."""

import unittest
from unittest.mock import MagicMock, patch

from vanishcap.utils.wifi import WifiError, WifiManager


class TestWifiManager(unittest.TestCase):
    """Test cases for the WifiManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "connect": {"ssid": "test_network", "password": "test_password", "max_retries": 1, "retry_delay": 1.0},
            "reconnect": True,
            "log_level": "DEBUG",
        }

    @patch("subprocess.run")
    def test_initialization_with_connect(self, mock_run):
        """Test initialization with WiFi connection configuration."""
        # Mock successful WiFi operations
        mock_run.return_value = MagicMock(returncode=0)

        # Create WiFi manager
        wifi_manager = WifiManager(self.config)

        # Verify initialization
        self.assertEqual(wifi_manager.config, self.config)
        self.assertIsNotNone(wifi_manager.logger)
        self.assertIsNone(wifi_manager.previous_wifi)
        self.assertIsNone(wifi_manager.wifi_device)

    def test_initialization_without_connect(self):
        """Test initialization without WiFi connection configuration."""
        config = {"log_level": "DEBUG"}
        wifi_manager = WifiManager(config)

        # Verify initialization
        self.assertEqual(wifi_manager.config, config)
        self.assertIsNotNone(wifi_manager.logger)
        self.assertIsNone(wifi_manager.previous_wifi)
        self.assertIsNone(wifi_manager.wifi_device)

    @patch("subprocess.run")
    def test_scan_success(self, mock_run):
        """Test successful WiFi scanning."""
        # Mock successful scan
        mock_run.return_value = MagicMock(returncode=0)

        wifi_manager = WifiManager(self.config)
        result = wifi_manager.scan()

        self.assertTrue(result)
        mock_run.assert_called_once_with(["nmcli", "device", "wifi", "list", "--rescan", "yes"], check=True)

    @patch("subprocess.run")
    def test_scan_failure(self, mock_run):
        """Test failed WiFi scanning."""
        # Mock failed scan
        mock_run.side_effect = Exception("Scan failed")

        wifi_manager = WifiManager(self.config)
        result = wifi_manager.scan()

        self.assertFalse(result)

    @patch("subprocess.run")
    def test_find_wifi_device(self, mock_run):
        """Test finding a valid WiFi device."""
        # Mock device list with a valid WiFi device
        mock_run.return_value = MagicMock(returncode=0, stdout="wlan0:wifi:connected\neth0:ethernet:connected")

        wifi_manager = WifiManager(self.config)
        device = wifi_manager._find_wifi_device()

        self.assertEqual(device, "wlan0")

    @patch("subprocess.run")
    def test_get_current_wifi(self, mock_run):
        """Test getting current WiFi connection."""
        # Mock current WiFi connection
        mock_run.return_value = MagicMock(returncode=0, stdout="wlan0:wifi:connected:test_network")

        wifi_manager = WifiManager(self.config)
        wifi_info = wifi_manager.get_current_wifi()

        self.assertEqual(wifi_info, ("test_network", "wlan0"))

    @patch("subprocess.run")
    def test_connect_success(self, mock_run):
        """Test successful WiFi connection."""
        # Mock successful connection
        mock_run.return_value = MagicMock(returncode=0)

        wifi_manager = WifiManager(self.config)
        result = wifi_manager.connect("test_network", "test_password")

        self.assertTrue(result)
        mock_run.assert_called_with(
            ["nmcli", "device", "wifi", "connect", "test_network", "password", "test_password"],
            capture_output=True,
            text=True,
            check=False,
        )

    @patch("subprocess.run")
    def test_connect_failure(self, mock_run):
        """Test failed WiFi connection."""
        # Mock failed connection
        mock_run.return_value = MagicMock(returncode=1, stderr="Connection failed")

        wifi_manager = WifiManager(self.config)
        result = wifi_manager.connect("test_network", "test_password")

        self.assertFalse(result)

    @patch("subprocess.run")
    def test_store_current_wifi(self, mock_run):
        """Test storing current WiFi information."""
        # Mock current WiFi connection
        mock_run.return_value = MagicMock(returncode=0, stdout="wlan0:wifi:connected:test_network")

        wifi_manager = WifiManager(self.config)
        wifi_manager.store_current_wifi()

        self.assertEqual(wifi_manager.previous_wifi, ("test_network", "wlan0"))
        self.assertEqual(wifi_manager.wifi_device, "wlan0")

    @patch("subprocess.run")
    def test_reconnect_previous(self, mock_run):
        """Test reconnecting to previous WiFi network."""
        # Mock successful reconnection
        mock_run.return_value = MagicMock(returncode=0)

        wifi_manager = WifiManager(self.config)
        wifi_manager.previous_wifi = ("test_network", "wlan0")
        wifi_manager.reconnect_previous()

        mock_run.assert_called_with(
            ["nmcli", "device", "wifi", "connect", "test_network"], capture_output=True, text=True, check=False
        )

    @patch("subprocess.run")
    def test_context_manager(self, mock_run):
        """Test WiFi manager as context manager."""
        # Mock current WiFi connection
        mock_run.return_value = MagicMock(returncode=0, stdout="wlan0:wifi:connected:test_network")

        with WifiManager(self.config) as wifi_manager:
            self.assertIsInstance(wifi_manager, WifiManager)
            wifi_manager.store_current_wifi()

        # Verify reconnect was attempted
        mock_run.assert_called_with(
            ["nmcli", "device", "wifi", "connect", "test_network"], capture_output=True, text=True, check=False
        )


if __name__ == "__main__":
    unittest.main()
