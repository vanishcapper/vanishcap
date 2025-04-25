"""Utility module for WiFi management."""

import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from vanishcap.utils.logging import get_worker_logger
from vanishcap.event import Event


class WifiError(Exception):
    """Exception raised when WiFi operations fail."""

    pass


class WifiManager:
    """Manager for WiFi connections."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the WiFi manager.

        Args:
            config: Configuration dictionary containing:
                - connect: Optional dictionary with:
                    - ssid: SSID to connect to
                    - password: Password for the SSID
                - reconnect: Whether to reconnect to previous SSID after workers finish
                - log_level: Optional log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        # Set up logging
        self.logger = get_worker_logger("wifi", config.get("log_level", "WARNING"))
        self.previous_wifi: Optional[Tuple[str, str]] = None
        self.wifi_device: Optional[str] = None

        # Store configuration
        self.config = config
        self.previous_ssid: Optional[str] = None

        # Connect to WiFi if configured
        if "connect" in config:
            max_retries = 10
            retry_delay = 2.0  # seconds

            # Try initial connect first
            if self.connect(config["connect"]["ssid"], config["connect"].get("password", "")):
                return

            # If initial connect failed, scan and retry the sequence
            for attempt in range(max_retries):
                try:
                    self.logger.warning(
                        "Initial connect failed, scanning and retrying... (attempt %d/%d)", attempt + 1, max_retries
                    )
                    if not self.scan():
                        self.logger.warning("Failed to scan for WiFi networks")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                        continue

                    if self.connect(config["connect"]["ssid"], config["connect"].get("password", "")):
                        return

                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                except Exception as e:  # pylint: disable=broad-except
                    self.logger.warning("Connection attempt %d/%d failed: %s", attempt + 1, max_retries, e)
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)

            self.logger.error("Failed to connect to WiFi after %d attempts", max_retries)
            raise WifiError(f"Failed to connect to WiFi network: {config['connect']['ssid']}")
        else:
            self.logger.warning("Running in offline mode - skipping WiFi management")

    def scan(self) -> bool:
        """Scan for available WiFi networks.

        Returns:
            True if scan completed successfully, False otherwise
        """
        try:
            self.logger.warning("Scanning for WiFi networks...")
            subprocess.run(
                ["nmcli", "device", "wifi", "list", "--rescan", "yes"],
                check=True,
            )
            self.logger.info("WiFi scan completed successfully")
            return True
        except subprocess.CalledProcessError:
            self.logger.error("Failed to scan for WiFi networks")
            return False

    def _find_wifi_device(self) -> Optional[str]:
        """Find a valid WiFi device.

        Returns:
            Optional[str]: Name of a valid WiFi device, or None if none found
        """
        try:
            result = subprocess.run(
                ["nmcli", "--terse", "--fields", "DEVICE,TYPE,STATE", "device", "status"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                self.logger.error("Failed to get device status: %s", result.stderr)
                return None

            # Parse terse output (DEVICE:TYPE:STATE)
            for line in result.stdout.splitlines():
                device, dev_type, state = line.split(":")
                if dev_type.lower() == "wifi" and state.lower() != "unmanaged":
                    return device
            return None
        except subprocess.CalledProcessError as e:
            self.logger.error("Failed to get device status: %s", e)
            return None
        except Exception as e:  # pylint: disable=broad-except
            self.logger.error("Error getting device status: %s", e)
            return None

    def get_current_wifi(self) -> Optional[Tuple[str, str]]:
        """Get the currently connected WiFi SSID and interface.

        Returns:
            Optional[Tuple[str, str]]: Tuple of (ssid, interface) if connected, None otherwise
        """
        try:
            # Get list of WiFi interfaces with terse output
            result = subprocess.run(
                ["nmcli", "--terse", "--fields", "DEVICE,TYPE,STATE,CONNECTION", "device", "status"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                self.logger.error("Failed to get device status: %s", result.stderr)
                return None

            # Parse terse output (DEVICE:TYPE:STATE:CONNECTION)
            for line in result.stdout.splitlines():
                device, dev_type, state, connection = line.split(":")
                if dev_type.lower() == "wifi" and state.lower() == "connected":
                    if connection and connection != "off/any":
                        return (connection, device)
            return None
        except subprocess.CalledProcessError as e:
            self.logger.error("Failed to get WiFi status: %s", e)
            return None
        except Exception as e:  # pylint: disable=broad-except
            self.logger.error("Error getting WiFi status: %s", e)
            return None

    def connect(self, ssid: str, password: str = "") -> bool:
        """Connect to a WiFi network.

        Args:
            ssid: SSID to connect to
            password: Optional password for the network

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Check if we're already connected to the target SSID
            current_wifi = self.get_current_wifi()
            if current_wifi and current_wifi[0] == ssid:
                self.logger.info("Already connected to WiFi network: %s", ssid)
                return True

            # Get WiFi device to use
            device = self.wifi_device
            if not device:
                device = self._find_wifi_device()
                if not device:
                    self.logger.error("No valid WiFi device found")
                    return False
                self.wifi_device = device

            # Only try to disconnect if we're connected to something
            if current_wifi:
                result = subprocess.run(
                    ["nmcli", "device", "disconnect", device], capture_output=True, text=True, check=False
                )
                if result.returncode != 0:
                    self.logger.warning("Failed to disconnect from current network: %s", result.stderr)
                time.sleep(2)  # Wait for disconnect

            # Connect to new network
            if password:
                cmd = ["nmcli", "device", "wifi", "connect", ssid, "password", password]
            else:
                cmd = ["nmcli", "device", "wifi", "connect", ssid]

            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                self.logger.info("Successfully connected to WiFi network: %s", ssid)
                return True

            self.logger.error("Failed to connect to WiFi: %s", result.stderr)
            return False
        except subprocess.CalledProcessError as e:
            self.logger.error("Failed to connect to WiFi: %s", e)
            return False
        except Exception as e:  # pylint: disable=broad-except
            self.logger.error("Error connecting to WiFi: %s", e)
            return False

    def store_current_wifi(self) -> None:
        """Store the current WiFi connection information."""
        wifi_info = self.get_current_wifi()
        if wifi_info:
            self.previous_wifi = wifi_info
            self.wifi_device = wifi_info[1]  # Store the device name

    def reconnect_previous(self) -> None:
        """Reconnect to the previously connected WiFi network."""
        if self.previous_wifi:
            ssid = self.previous_wifi[0]  # Only use the SSID
            self.logger.info("Reconnecting to previous WiFi network: %s", ssid)
            self.connect(ssid)

    def __enter__(self) -> "WifiManager":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        if self.previous_wifi and self.previous_wifi[0] != self.get_current_wifi()[0]:
            self.logger.info("Reconnecting to previous WiFi network")
            self.connect(self.previous_wifi[0])
