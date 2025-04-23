"""Tello driver implementation."""

from typing import Any, Dict

from djitellopy import Tello

from vanishcap.drivers.base import BaseDroneDriver


class TelloDriver(BaseDroneDriver):
    """Driver implementation for Tello drones."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the Tello driver.

        Args:
            config: Driver configuration dictionary
        """
        super().__init__(config)
        self.tello = Tello(config.get("ip", "192.168.10.1"))

    def connect(self) -> None:
        """Connect to the Tello."""
        self.tello.connect()

    def disconnect(self) -> None:
        """Disconnect from the Tello."""
        self.tello.end()

    def takeoff(self) -> None:
        """Take off."""
        self.tello.takeoff()

    def land(self) -> None:
        """Land."""
        self.tello.land()

    def _send_rc_control(self, left_right: int, forward_back: int, up_down: int, yaw: int) -> None:
        """Send RC control commands to the Tello.

        Args:
            left_right: Left/right velocity [-100, 100]
            forward_back: Forward/backward velocity [-100, 100]
            up_down: Up/down velocity [-100, 100]
            yaw: Yaw velocity [-100, 100]
        """
        self.tello.send_rc_control(left_right, forward_back, up_down, yaw)

    def get_current_state(self) -> Dict[str, Any]:
        """Get the current state of the Tello.

        Returns:
            Dict[str, Any]: Current state dictionary
        """
        return self.tello.get_current_state()

    def streamon(self) -> None:
        """Start video streaming."""
        self.tello.streamon()

    def streamoff(self) -> None:
        """Stop video streaming."""
        self.tello.streamoff()
