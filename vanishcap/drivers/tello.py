"""Tello driver implementation."""

import socket
import time
import threading
from typing import Any, Dict, Optional

from vanishcap.drivers.base import BaseDroneDriver


# pylint: disable=too-many-instance-attributes
class TelloDriver(BaseDroneDriver):
    """Driver implementation for Tello drones using raw socket communication."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the Tello driver.

        Args:
            config: Driver configuration dictionary containing:
                - ip: IP address of the Tello (default: 192.168.10.1)
                - interface: Optional WiFi interface to use for communication
                - log_level: Optional log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        super().__init__(config)
        self.ip = config.get("ip", "192.168.10.1")
        self.command_port = 8889
        self.state_port = 8890
        self.interface = config.get("interface")

        self.logger.debug("Tello driver initialized with config: %s", config)

        # Socket connections
        self.command_socket: Optional[socket.socket] = None
        self.state_socket: Optional[socket.socket] = None

        # State tracking
        self.last_state: Dict[str, Any] = {}
        self.state_thread: Optional[threading.Thread] = None
        self.running = False

    def connect(self) -> None:
        """Connect to the Tello."""
        try:
            # Create command socket
            self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if self.interface:
                self.logger.debug("Binding command socket to interface: %s", self.interface)
                self.command_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, self.interface.encode())
            self.command_socket.bind(("", self.command_port))
            self.command_socket.settimeout(20.0)  # 5 second timeout

            # Send command mode
            self._send_command("command")

            # Start state thread
            self.running = True
            self.state_thread = threading.Thread(target=self._state_thread)
            self.state_thread.daemon = True
            self.state_thread.start()

            self.logger.info(
                "Connected to Tello at %s%s", self.ip, f" on interface {self.interface}" if self.interface else ""
            )

        except (socket.error, OSError) as e:
            self.logger.error("Failed to connect to Tello: %s", e)
            self.disconnect()
            raise

    def disconnect(self) -> None:
        """Disconnect from the Tello."""
        self.running = False
        if self.state_thread:
            self.state_thread.join()

        for sock in [self.command_socket, self.state_socket]:
            if sock:
                try:
                    sock.close()
                except (socket.error, OSError):
                    pass
                sock = None

    def _send_command(self, command: str) -> str:
        """Send a command to the Tello and wait for response.

        Args:
            command: Command to send

        Returns:
            Response from Tello

        Raises:
            RuntimeError: If command fails or times out
        """
        if not self.command_socket:
            raise RuntimeError("Not connected to Tello")

        try:
            self.logger.debug("Sending command: %s", command)
            response_expected = not (command in ["emergency", "reboot"] or command.startswith("rc "))
            self.command_socket.sendto(command.encode(), (self.ip, self.command_port))
            if not response_expected:
                return None

            response, _ = self.command_socket.recvfrom(1024)
            self.logger.debug("Received response to command '%s': %s", command, response)
            return response.decode("ASCII").strip()

        except socket.timeout as exc:
            raise RuntimeError(f"Command '{command}' timed out") from exc
        except (socket.error, OSError) as exc:
            raise RuntimeError(f"Command '{command}' failed: {exc}") from exc

    def _state_thread(self) -> None:
        """Thread that continuously receives state updates."""
        # Create state socket
        self.state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if self.interface:
            self.state_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, self.interface.encode())
        self.state_socket.bind(("", self.state_port))
        self.state_socket.settimeout(5.0)

        try:
            while self.running:
                self._process_state_update()
        except (socket.error, OSError) as exc:
            self.logger.error("State thread error: %s", exc)
        finally:
            if self.state_socket:
                self.state_socket.close()

    def _process_state_update(self) -> None:
        """Process a single state update from the Tello."""
        try:
            data = self.state_socket.recv(1024)
            if not data:
                return

            # Parse state data
            state = {}
            for line in data.decode("ASCII").split(";"):
                if ":" in line:
                    key, value = line.split(":")
                    try:
                        state[key] = float(value)
                    except ValueError:
                        state[key] = value

            self.last_state = state

        except (socket.error, OSError) as exc:
            self.logger.error("Error receiving state: %s", exc)
            time.sleep(0.1)

    def takeoff(self) -> None:
        """Take off."""
        response = self._send_command("takeoff")
        if response != "ok":
            raise RuntimeError(f"Takeoff failed: {response}")

    def land(self) -> None:
        """Land."""
        response = self._send_command("land")
        if response != "ok":
            raise RuntimeError(f"Land failed: {response}")

    def _send_rc_control(self, left_right: int, forward_back: int, up_down: int, yaw: int) -> None:
        """Send RC control commands to the Tello.

        Args:
            left_right: Left/right velocity [-100, 100]
            forward_back: Forward/backward velocity [-100, 100]
            up_down: Up/down velocity [-100, 100]
            yaw: Yaw velocity [-100, 100]
        """
        self.logger.debug(
            "Sending RC control: left_right=%d, forward_back=%d, up_down=%d, yaw=%d",
            left_right,
            forward_back,
            up_down,
            yaw,
        )

        # Clamp values to [-100, 100]
        left_right = max(min(left_right, 100), -100)
        forward_back = max(min(forward_back, 100), -100)
        up_down = max(min(up_down, 100), -100)
        yaw = max(min(yaw, 100), -100)

        command = f"rc {left_right} {forward_back} {up_down} {yaw}"
        self._send_command(command)

    def get_current_state(self) -> Dict[str, Any]:
        """Get the current state of the Tello.

        Returns:
            Dict[str, Any]: Current state dictionary
        """
        return self.last_state.copy()

    def streamon(self) -> None:
        """Start video streaming."""
        response = self._send_command("streamon")
        if response != "ok":
            raise RuntimeError(f"Streamon failed: {response}")

    def streamoff(self) -> None:
        """Stop video streaming."""
        response = self._send_command("streamoff")
        if response != "ok":
            raise RuntimeError(f"Streamoff failed: {response}")
